import os
import sys
import logging
from datetime import datetime
import traceback

if sys.platform == 'win32' and os.environ.get('PYTHONUTF8') != '1':
    print("Restarting Python with UTF-8 encoding...")
    os.environ['PYTHONUTF8'] = '1'
    import subprocess
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=os.environ).returncode)

import torch
import psutil
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    pipeline,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer


log_dir = r"E:\logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"fine_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logger.info("CUDA_LAUNCH_BLOCKING=1 enabled for precise error reporting.")

logger.info("="*60)
logger.info("Starting Qwen 3.5 Fine-tuning Process")
logger.info("="*60)


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
logger.info(f"Using device: {DEVICE}")


model_path = r"E:\Qwen3.5-2B"
data_path  = r"E:\qiskit_quantumkatas.jsonl"
output_dir = r"E:\Qwen3.5-2B-qiskit-lora"

logger.info(f"Model path: {model_path}")
logger.info(f"Data path: {data_path}")
logger.info(f"Output directory: {output_dir}")


logger.info("\n" + "="*60)
logger.info("Step 1: Loading Tokenizer and Model")
logger.info("="*60)

logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Set pad_token to eos_token")

if USE_CUDA:
    logger.info("CUDA available – loading model in bfloat16.")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": DEVICE},   
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    logger.info("Model loaded on GPU with bfloat16.")
    model.gradient_checkpointing_enable()
else:
    logger.info("No CUDA detected – loading model in full precision on CPU.")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    logger.info("Model loaded on CPU.")

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
logger.info("Applying LoRA adapters...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


logger.info("\n" + "="*60)
logger.info("Step 2: Loading and Formatting Dataset")
logger.info("="*60)

def format_chat(example):
    messages = [
        {"role": "system", "content": "You are a Qiskit quantum computing expert."},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["canonical_solution"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

logger.info(f"Loading dataset from {data_path}...")
dataset = load_dataset("json", data_files=data_path, split="train")
logger.info(f"Loaded {len(dataset)} examples")

logger.info("Formatting dataset with chat template...")
dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
logger.info(f"Formatted dataset size: {len(dataset)}")
logger.info(f"Sample (first 200 chars): {dataset[0]['text'][:200]}")

logger.info("Splitting into train/eval (95%/5%)...")
dataset = dataset.train_test_split(test_size=0.05)
train_data = dataset["train"]
eval_data = dataset["test"]
logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

sample_tok = tokenizer(train_data[:5]["text"], truncation=True, max_length=2048, padding=False)
lengths = [len(ids) for ids in sample_tok["input_ids"]]
logger.info(f"Sample token lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")


logger.info("\n" + "="*60)
logger.info("Step 3: Setting Up Training Configuration")
logger.info("="*60)

if USE_CUDA:
    ds_config_path = "ds_config.json"
    if not os.path.exists(ds_config_path):
        logger.warning(f"DeepSpeed config file '{ds_config_path}' not found! Creating a default one.")
        import json
        ds_config = {
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu"
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6
            },
            "bf16": {
                "enabled": True
            },
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_allow_untested_optimizer": True,
            "wall_clock_breakdown": False
        }
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f, indent=4)
        logger.info(f"Default DeepSpeed config written to {ds_config_path}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,     # still useful for memory
        bf16=True,                       # will be overridden by DeepSpeed config
        deepspeed=ds_config_path,        # use DeepSpeed
        report_to="none",
        run_name="qwen3.5-2b-qiskit",
    )
else:
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        bf16=False,
        fp16=False,
        report_to="none",
        run_name="qwen3.5-2b-qiskit-cpu",
    )

logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
logger.info(f"Epochs: {training_args.num_train_epochs}, LR: {training_args.learning_rate}")
if USE_CUDA:
    logger.info(f"DeepSpeed enabled with config: {ds_config_path}")


logger.info("\n" + "="*60)
logger.info("Step 4: Initializing SFT Trainer")
logger.info("="*60)

class DetailedLogCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        logger.info(f"  >>> Step begin: {state.global_step}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:
            loss = state.log_history[-1].get("loss", "N/A")
        else:
            loss = "N/A"
        logger.info(f"  <<< Step end: {state.global_step}  loss={loss}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_str = "  LOG: " + " | ".join(f"{k}={v}" for k, v in logs.items() if not k.startswith("_"))
            logger.info(log_str)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        logger.info(f"  EVALUATION at step {state.global_step}: {metrics}")

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Training has started (callback).")

    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Training has ended (callback).")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    callbacks=[DetailedLogCallback()],
)

logger.info("Trainer initialized successfully.")
logger.info(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")


logger.info("\n" + "="*60)
logger.info("Step 5: Starting Training")
logger.info("="*60)

if USE_CUDA:
    logger.info("GPU memory before training:")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: allocated={torch.cuda.memory_allocated(i)/1e9:.2f} GB, reserved={torch.cuda.memory_reserved(i)/1e9:.2f} GB")
else:
    mem = psutil.virtual_memory()
    logger.info(f"CPU RAM: used={mem.used/1e9:.2f} GB, available={mem.available/1e9:.2f} GB")

try:
    if USE_CUDA:
        torch.cuda.synchronize()  
    training_result = trainer.train()
    logger.info("trainer.train() completed.")
except Exception as e:
    logger.error(f"Training failed: {e}")
    logger.error(traceback.format_exc())
    raise

logger.info("\n" + "="*60)
logger.info("Training completed successfully!")
logger.info("="*60)
logger.info(f"Final training loss: {training_result.training_loss:.4f}")


logger.info("\n" + "="*60)
logger.info("Step 6: Saving Model and Testing")
logger.info("="*60)

logger.info(f"Saving LoRA adapter to {output_dir}...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info("LoRA adapter saved.")

logger.info("\nMerging LoRA adapter into base model...")
merged_output = r"E:\Qwen3.5-2B-qiskit-merged"
try:
    if USE_CUDA:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": DEVICE},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    logger.info("Base model loaded for merging.")
    merged_model = PeftModel.from_pretrained(base_model, output_dir)
    logger.info("LoRA adapter loaded.")
    merged_model = merged_model.merge_and_unload()
    logger.info("Adapter merged.")
    merged_model.save_pretrained(merged_output)
    tokenizer.save_pretrained(merged_output)
    logger.info(f"Merged model saved to {merged_output}")

    logger.info("\nRunning a quick test...")
    pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer, device=0 if USE_CUDA else -1)
    messages = [
        {"role": "system", "content": "You are a Qiskit quantum computing expert."},
        {"role": "user", "content": "How do I create a Bell state in Qiskit?"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = pipe(prompt, max_new_tokens=256, temperature=0.7, do_sample=True)
    logger.info("Test response:\n" + output[0]["generated_text"])
    logger.info("="*60)
except Exception as e:
    logger.error(f"Error during merging/testing: {e}")
    logger.error(traceback.format_exc())
    raise

logger.info("\n" + "="*60)
logger.info("Fine-tuning Process Completed Successfully!")
logger.info("="*60)
