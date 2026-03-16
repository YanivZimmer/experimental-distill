# SOC Alert Distillation Pipeline

## Overview

This codebase implements **step-by-step knowledge distillation** to transfer expertise from large frontier models (GPT-4, Claude) to a smaller, deployable Qwen 3.5 MoE model for security alert triage. The system learns both the reasoning process and classification decisions from expert demonstrations.

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [Usage](#usage)
  - [Local Training](#local-training)
  - [Cloud Training](#cloud-training)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Architecture

### Distillation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  1. DATA PREPARATION (prepare_dataset.py)               │
│  ─────────────────────────────────────────────────────  │
│  Input:  langfuse_test.json (raw alerts + reasoning)    │
│  Output: train_distill.json (formatted for training)    │
│                                                          │
│  Combines:                                               │
│    - System prompt (baseline.txt)                        │
│    - Alert data (JSON)                                   │
│    - Reasoning trace (frontier model output)             │
│    - Classification (final decision)                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  2. MODEL TRAINING (train.py)                           │
│  ─────────────────────────────────────────────────────  │
│  Framework: Unsloth + LoRA                              │
│  Base Model: Qwen 2.5-32B (proxy for Qwen 3.5-35B-3A)  │
│                                                          │
│  Optimizations:                                          │
│    ✓ 4-bit quantization (QLoRA)                         │
│    ✓ Flash Attention 2                                  │
│    ✓ Gradient checkpointing                             │
│    ✓ LoRA adapters (memory efficient)                   │
│                                                          │
│  Output: Fine-tuned model weights                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  3. DEPLOYMENT (cloud_train.py + Vertex AI)             │
│  ─────────────────────────────────────────────────────  │
│  Cloud Platform: Google Cloud Vertex AI                 │
│  GPU Options: L4 / A100 / V100                          │
│                                                          │
│  Workflow:                                               │
│    1. Download data from GCS                            │
│    2. Execute training                                  │
│    3. Upload model artifacts to GCS                     │
└─────────────────────────────────────────────────────────┘
```

### Why Step-by-Step Distillation?

Traditional distillation only transfers final predictions. Step-by-step distillation transfers the **reasoning process**:

- **Input**: Security alert
- **Teacher Output**: "This appears suspicious because... [reasoning steps]... → Classification: MALICIOUS"
- **Student Learns**: Both reasoning AND classification

Benefits:
- Better generalization with less data
- Interpretable outputs (explains decisions)
- More robust to edge cases

---

## Prerequisites

### Hardware Requirements

| Configuration      | VRAM  | Batch Size | Training Time (1000 samples) |
|--------------------|-------|------------|------------------------------|
| **Minimum**        | 24GB  | 1-2        | ~4 hours                     |
| **Recommended**    | 40GB  | 4-8        | ~2 hours                     |
| **Optimal**        | 80GB  | 8-16       | ~1 hour                      |

Supported GPUs:
- NVIDIA L4 (24GB) - Google Cloud G2
- NVIDIA A100 (40/80GB) - Google Cloud A2
- NVIDIA V100 (16/32GB) - Older, slower
- RTX 4090 (24GB) - Local development

### Software Requirements

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- Docker (for cloud deployment)
- Google Cloud SDK (for Vertex AI)

---

## Installation

### Local Setup

```bash
# Clone repository
cd /path/to/simple_distill

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

### Cloud Setup (Vertex AI)

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create GCS bucket
gsutil mb -l us-central1 gs://your-distill-bucket

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

---

## Dataset Format

### Input Format (langfuse_test.json)

Your dataset should contain items with this structure:

```json
[
  {
    "id": "unique-alert-id",
    "input": {
      "case_id": "...",
      "raw_alert": {
        "name": "SuspiciousProcess",
        "device": {...},
        "process": {...}
      }
    },
    "reasoning": "**Event Comprehension:**\nThe alert shows PowerShell executing encoded commands...\n\n**TTP Mapping:**\nT1059.001 - Command and Scripting Interpreter: PowerShell...\n\n**Decision:**\nThis represents genuine malicious activity.",
    "classification": "True Positive - Malicious (Severity: High)"
  }
]
```

### Output Format (train_distill.json)

The preprocessing script converts this to:

```json
[
  {
    "instruction": "<System prompt from baseline.txt>\n\n**Event:**\n```json\n{alert data}\n```",
    "output": "<reasoning trace>\n\nFinal Classification: <classification>"
  }
]
```

### Generating Reasoning Traces

If you don't have reasoning traces yet, generate them using a frontier model:

```python
import anthropic

client = anthropic.Anthropic(api_key="...")
prompt_template = open("baseline.txt").read()

for alert in alerts:
    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"{prompt_template}\n{json.dumps(alert)}"
        }]
    )
    alert["reasoning"] = message.content[0].text
    alert["classification"] = extract_classification(message.content[0].text)
```

---

## Usage

### Step 1: Prepare Dataset

```bash
python prepare_dataset.py
```

**What it does:**
- Loads `data/langfuse_test.json`
- Reads system prompt from `baseline.txt`
- Filters items with reasoning traces
- Formats for training
- Saves to `data/train_distill.json`

**Expected output:**
```
Loading dataset from data/langfuse_test.json...
Converting 1247 examples...
Prepared 1247 training examples
Saved to data/train_distill.json
```

### Step 2A: Local Training

```bash
python train.py
```

**Training process:**
1. Loads base model (Qwen 2.5-32B) with 4-bit quantization
2. Applies LoRA adapters to attention and MLP layers
3. Trains for 3 epochs with cosine learning rate schedule
4. Saves model to `outputs/distilled_model/`

**Expected output:**
```
Loading model...
Loading dataset...
Training on 1247 examples
Starting training...
{'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 0.5}
{'loss': 0.856, 'learning_rate': 0.00015, 'epoch': 1.0}
...
Saving model...
Training complete! Model saved to outputs/distilled_model
```

### Step 2B: Cloud Training (Vertex AI)

#### Upload Data to GCS

```bash
export GCS_BUCKET=your-distill-bucket
gsutil cp data/train_distill.json gs://${GCS_BUCKET}/data/
gsutil cp baseline.txt gs://${GCS_BUCKET}/
```

#### Edit Configuration

Open `vertex_ai_submit.sh` and set:

```bash
PROJECT_ID="your-gcp-project-id"
BUCKET_NAME="your-distill-bucket"
REGION="us-central1"

# Choose GPU type
MACHINE_TYPE="g2-standard-8"  # 1x L4 (24GB)
# OR
# MACHINE_TYPE="a2-highgpu-1g"  # 1x A100 (40GB)
```

#### Submit Job

```bash
bash vertex_ai_submit.sh
```

**What happens:**
1. Builds Docker image with training code
2. Pushes to Google Container Registry
3. Submits custom training job to Vertex AI
4. Downloads data from GCS
5. Trains model
6. Uploads results back to GCS

#### Monitor Training

```bash
# List jobs
gcloud ai custom-jobs list --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

---

## Configuration

### Training Hyperparameters

Edit `CONFIG` in `train.py`:

```python
CONFIG = {
    # Model selection
    "model_name": "unsloth/Qwen2.5-32B-Instruct",

    # Memory management
    "max_seq_length": 4096,        # Max tokens per example
    "load_in_4bit": True,          # Use QLoRA (reduces VRAM)

    # LoRA parameters
    "lora_r": 16,                  # Rank (higher = more capacity, more memory)
    "lora_alpha": 16,              # Scaling factor
    "lora_dropout": 0.05,          # Regularization

    # Training
    "learning_rate": 2e-4,         # Peak learning rate
    "num_epochs": 3,               # Training epochs
    "batch_size": 2,               # Per-device batch size
    "gradient_accumulation_steps": 4,  # Effective batch = 2 * 4 = 8

    # Output
    "output_dir": "outputs/distilled_model"
}
```

### Tuning Guidelines

**If you have more VRAM:**
- Increase `batch_size` to 4 or 8
- Increase `lora_r` to 32 or 64 (more model capacity)
- Set `load_in_4bit: False` for faster training

**If you run out of memory:**
- Decrease `batch_size` to 1
- Increase `gradient_accumulation_steps` to 8
- Decrease `max_seq_length` to 2048
- Decrease `lora_r` to 8

**If training is too slow:**
- Use `bf16=True` instead of `fp16` (on A100)
- Enable `packing=True` in SFTTrainer (shorter examples)
- Use smaller model variant

**If quality is poor:**
- Increase `num_epochs` to 5
- Decrease `learning_rate` to 1e-4
- Add validation split to detect overfitting
- Check reasoning trace quality in dataset

---

## File Structure

```
simple_distill/
├── data/
│   ├── langfuse_test.json       # Raw dataset (input + reasoning + classification)
│   └── train_distill.json       # Processed training data (generated)
│
├── outputs/
│   └── distilled_model/         # Trained model artifacts (generated)
│       ├── adapter_config.json
│       ├── adapter_model.bin
│       └── tokenizer files
│
├── baseline.txt                 # SOC analyst system prompt
├── prepare_dataset.py           # Dataset preprocessing script
├── train.py                     # Local training script
├── cloud_train.py              # Cloud training wrapper (GCS integration)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container for cloud training
├── vertex_ai_submit.sh         # Vertex AI job submission script
│
└── README.md                   # This file
```

---

## Performance Tuning

### Training Speed Optimization

1. **Use BF16 on A100 GPUs:**
   ```python
   training_args = TrainingArguments(
       bf16=True,  # Instead of fp16
       ...
   )
   ```

2. **Enable Flash Attention:**
   Already enabled in Unsloth by default. Verify with:
   ```python
   print(model.config.use_flash_attention_2)  # Should be True
   ```

3. **Multi-GPU Training:**
   ```bash
   # Local
   torchrun --nproc_per_node=4 train.py

   # Vertex AI
   # Set replica-count=1,accelerator-count=4 in vertex_ai_submit.sh
   ```

### Memory Optimization

1. **Gradient Checkpointing:**
   Already enabled via `use_gradient_checkpointing="unsloth"`

2. **Mixed Precision:**
   Automatically uses BF16/FP16 based on GPU capability

3. **Reduce Batch Size, Increase Accumulation:**
   ```python
   "batch_size": 1,
   "gradient_accumulation_steps": 16,  # Effective batch = 16
   ```

### Quality Optimization

1. **Learning Rate Schedule:**
   Cosine annealing with warmup (already configured)

2. **Add Validation Set:**
   ```python
   dataset = dataset.train_test_split(test_size=0.1)
   trainer = SFTTrainer(
       train_dataset=dataset["train"],
       eval_dataset=dataset["test"],
       ...
   )
   ```

3. **Early Stopping:**
   ```python
   from transformers import EarlyStoppingCallback

   trainer.add_callback(EarlyStoppingCallback(
       early_stopping_patience=2
   ))
   ```

---

## Troubleshooting

### Out of Memory (OOM)

**Error:** `CUDA out of memory`

**Solutions:**
1. Decrease batch size: `"batch_size": 1`
2. Increase gradient accumulation: `"gradient_accumulation_steps": 8`
3. Reduce sequence length: `"max_seq_length": 2048`
4. Enable CPU offload (slow): `device_map="auto"`

### Model Quality Issues

**Problem:** Model outputs gibberish or doesn't follow format

**Solutions:**
1. Check reasoning traces in `train_distill.json` - ensure they're high quality
2. Increase training epochs: `"num_epochs": 5`
3. Verify prompt template is included in training data
4. Add more diverse examples to dataset

### Slow Training

**Problem:** Training takes too long

**Solutions:**
1. Use larger GPU (A100 instead of L4)
2. Reduce `max_seq_length` if examples are shorter
3. Enable `packing=True` in SFTTrainer
4. Use multi-GPU setup

### Vertex AI Issues

**Error:** `Permission denied` or `Service account error`

**Solutions:**
```bash
# Create service account with permissions
gcloud iam service-accounts create vertex-training
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:vertex-training@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:vertex-training@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
```

**Error:** `GPU quota exceeded`

**Solutions:**
1. Request quota increase: https://console.cloud.google.com/iam-admin/quotas
2. Use different region with available quota
3. Use smaller GPU type (L4 instead of A100)

---

## Next Steps

1. **Evaluate Model:**
   - Test on held-out validation set
   - Compare classifications with frontier model
   - Measure reasoning quality (ROUGE, BERTScore)

2. **Deploy Model:**
   - Merge LoRA weights: `model.merge_and_unload()`
   - Quantize for inference: GGUF, AWQ, or GPTQ
   - Deploy to production (vLLM, TGI, or Vertex AI Prediction)

3. **Iterate:**
   - Collect failure cases
   - Regenerate reasoning traces for difficult examples
   - Fine-tune on corrected data

---

## Support

For issues or questions:
- Check Unsloth docs: https://github.com/unslothai/unsloth
- Vertex AI docs: https://cloud.google.com/vertex-ai/docs
- Open an issue in this repository

## License

MIT
