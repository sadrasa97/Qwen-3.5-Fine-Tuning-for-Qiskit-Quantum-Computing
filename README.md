
---

# Qwen 3.5 Fine‑Tuning for Qiskit Quantum Computing

This repository contains a **complete training pipeline for fine‑tuning Qwen‑3.5‑2B using LoRA adapters on Qiskit Quantum Katas datasets**.  
The script implements a **robust, production‑style fine‑tuning workflow** with detailed logging, DeepSpeed integration, dataset formatting, LoRA training, evaluation, and model merging.

The goal of this project is to produce a **specialized Qwen model capable of answering quantum computing questions and generating Qiskit code solutions.**

---

# Project Overview

This training pipeline performs the following steps:

1. Loads the **Qwen 3.5 base model**
2. Applies **LoRA parameter‑efficient fine‑tuning**
3. Formats the dataset into a **chat-style prompt format**
4. Trains the model using **TRL SFTTrainer**
5. Uses **DeepSpeed optimization** for GPU efficiency
6. Logs detailed training diagnostics
7. Saves the **LoRA adapter**
8. Merges the adapter into the base model
9. Runs a **post‑training inference test**

The final model becomes a **Qiskit‑aware assistant capable of explaining quantum circuits and writing Qiskit code.**

---

# Key Features

• Fine‑tunes **Qwen‑3.5‑2B**  
• Uses **LoRA (Low Rank Adaptation)** for efficient training  
• Supports **GPU and CPU training modes**  
• Integrated **DeepSpeed ZeRO Stage 2 optimization**  
• Automatic **dataset formatting to chat templates**  
• Detailed **training logs with callbacks**  
• Automatic **LoRA merging into base model**  
• Built‑in **inference testing pipeline**  
• Memory‑efficient training using **gradient checkpointing**

---

# Architecture

Training pipeline:

```
Dataset (Qiskit Katas JSONL)
        │
        ▼
Chat Template Formatting
        │
        ▼
Tokenizer (Qwen)
        │
        ▼
Qwen 3.5 Base Model
        │
        ▼
LoRA Adapter Injection
        │
        ▼
SFTTrainer (TRL)
        │
        ▼
DeepSpeed Optimization
        │
        ▼
Training & Evaluation
        │
        ▼
Save LoRA Adapter
        │
        ▼
Merge Adapter + Base Model
        │
        ▼
Final Fine‑Tuned Model
```

---

# Repository Structure

```
project/
│
├── train.py
├── ds_config.json
├── qiskit_quantumkatas.jsonl
│
├── logs/
│   └── fine_tune_TIMESTAMP.log
│
├── Qwen3.5-2B-qiskit-lora/
│   └── LoRA adapter weights
│
└── Qwen3.5-2B-qiskit-merged/
    └── final merged model
```

---

# Requirements

Python 3.10+

Install dependencies:

```
pip install torch transformers datasets peft trl accelerate deepspeed psutil
```

Optional but recommended:

```
pip install bitsandbytes
```

---

# Hardware Requirements

Recommended configuration:

GPU training

```
GPU: 16GB+ VRAM (RTX 3090 / A100 / H100 recommended)
RAM: 32GB
```

CPU training (slow but supported)

```
RAM: 64GB+
```

The script automatically detects GPU availability.

---

# Dataset Format

Training data must be a **JSONL file** with the following structure:

```
{
  "prompt": "How do I create a Bell state in Qiskit?",
  "canonical_solution": "from qiskit import QuantumCircuit..."
}
```

Each entry becomes a **chat conversation**:

```
System: You are a Qiskit quantum computing expert
User: <prompt>
Assistant: <canonical_solution>
```

This format aligns with **instruction‑tuned LLM chat training.**

---

# Configuration

Important paths inside the script:

```
model_path = "E:\\Qwen3.5-2B"
data_path  = "E:\\qiskit_quantumkatas.jsonl"
output_dir = "E:\\Qwen3.5-2B-qiskit-lora"
```

After training the merged model will be saved to:

```
E:\Qwen3.5-2B-qiskit-merged
```

---

# LoRA Configuration

The model uses **LoRA adapters on transformer attention and MLP layers.**

```
r = 16
alpha = 32
dropout = 0.05
```

Target modules:

```
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Only **~1–2% of parameters are trained**, dramatically reducing memory usage.

---

# Training Configuration

GPU Mode

```
Batch size: 2
Gradient accumulation: 4
Effective batch size: 8
Epochs: 3
Learning rate: 2e‑4
Scheduler: cosine
Precision: bfloat16
DeepSpeed: ZeRO Stage 2
```

CPU Mode

```
Batch size: 1
Gradient accumulation: 2
Precision: FP32
```

---

# Logging System

The script creates a **timestamped log file**:

```
E:\logs\fine_tune_YYYYMMDD_HHMMSS.log
```

Logs include:

• step‑level training loss  
• evaluation metrics  
• GPU memory usage  
• dataset statistics  
• training progress  
• error tracebacks  

This makes debugging and experiment tracking much easier.

---

# Training

Run the script:

```
python train.py
```

The training pipeline automatically executes:

1. Model loading
2. Dataset preprocessing
3. LoRA injection
4. Trainer initialization
5. Training
6. Evaluation
7. Adapter saving
8. Model merging
9. Inference test

---

# Model Testing

After merging, the script automatically runs a test prompt:

```
How do I create a Bell state in Qiskit?
```

Using a HuggingFace generation pipeline:

```
pipeline("text-generation")
```

This confirms the fine‑tuned model works correctly.

---

# Output

Two model outputs are generated:

### LoRA Adapter

```
Qwen3.5-2B-qiskit-lora/
```

Small adapter weights used with the base model.

### Fully Merged Model

```
Qwen3.5-2B-qiskit-merged/
```

Standalone model containing both base weights and fine‑tuned LoRA parameters.

---

# Example Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_path = "Qwen3.5-2B-qiskit-merged"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "How do I create a Bell state in Qiskit?"

print(pipe(prompt, max_new_tokens=200))
```

---

# Future Improvements

Possible extensions for this project:

• multi‑GPU distributed training  
• QLoRA support  
• dataset scaling to thousands of Qiskit problems  
• evaluation benchmarks  
• automated hyperparameter search  
• integration with HuggingFace Hub  
• reinforcement learning fine‑tuning (RLHF)

---

# Author

**Sadra Saremi**

GitHub  
https://github.com/sadrasa97

LinkedIn  
https://linkedin.com/in/sadra-saremi

---
