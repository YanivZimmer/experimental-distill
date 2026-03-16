# Distillation Training - Complete Guide

## Quick Start

### 1. Setup Environment

```bash
# Run setup script
./setup.sh

# Or manually
uv sync
```

### 2. Prepare Data

```bash
# Filter dataset (keep only Gemini agreements)
python scripts/prepare_dataset.py

# Split into train/val/test
python scripts/split_dataset.py
```

### 3. Train Model

```bash
# Local training (Mac) - 0.5B model, 512 tokens
python training/local_entry.py

# Cloud training (GPU) - 3B model, 32K tokens
python training/cloud_entry.py
```

## Or Use Makefile

```bash
make setup           # Setup UV environment
make prepare         # Prepare dataset
make split           # Split dataset
make train-local     # Train locally
make train-cloud     # Train on cloud
make all             # Complete pipeline
```

## Project Structure

```
experimental-distill/
├── training/                   # Training module (SOLID architecture)
│   ├── base_trainer.py            # Abstract base class
│   ├── local_trainer.py           # Local implementation
│   ├── cloud_trainer.py           # Cloud implementation
│   ├── config.py                  # Configuration classes
│   ├── local_entry.py             # Local entry point
│   └── cloud_entry.py             # Cloud entry point
│
├── scripts/                    # Data preparation
│   ├── prepare_dataset.py         # Filter by Gemini agreement
│   └── split_dataset.py           # Split train/val/test
│
├── prompts/                    # Prompt templates
│   └── baseline.txt               # Default prompt
│
├── data/                       # Data files
│   ├── langfuse_test.json         # Input alerts
│   ├── teacher_output.json        # Teacher outputs
│   ├── train_distill.json         # Prepared data
│   └── splits/                    # Train/val/test splits
│
├── outputs/                    # Generated during training
│   ├── local_model/               # Local training output
│   ├── distilled_model/           # Cloud training output
│   └── checkpoints/               # Training checkpoints
│
├── Makefile                    # Build commands
├── pyproject.toml              # UV dependencies
├── setup.sh                    # Setup script
└── README.md                   # This guide
```

## Architecture

### SOLID Principles Design

```
BaseTrainer (Abstract)
├── Abstract Methods:
│   ├── load_model()         # Environment-specific
│   └── create_trainer()     # Environment-specific
│
├── Concrete Methods (shared):
│   ├── load_datasets()      # Common dataset loading
│   ├── evaluate_before_training()
│   ├── train()
│   ├── evaluate_after_training()
│   ├── save_model()
│   └── run_full_training()  # Template method
│
├── LocalTrainer             # Mac implementation
│   ├── load_model()         # Standard transformers
│   └── create_trainer()     # Standard Trainer
│
└── CloudTrainer             # GPU implementation
    ├── load_model()         # Unsloth + 4-bit
    └── create_trainer()     # SFTTrainer
```

## Configurations

### Local Training (Mac)
```python
LocalTrainingConfig:
- Model: Qwen2.5-0.5B-Instruct
- Context: 512 tokens
- Device: CPU/MPS
- Batch: 1 × 16 grad accum = 16 effective
- Epochs: 1
```

### Cloud Training (GPU)
```python
CloudTrainingConfig:
- Model: Qwen2.5-3B-Instruct
- Context: 32,000 tokens
- Device: GPU + Unsloth
- Batch: 2 × 4 grad accum = 8 effective
- Epochs: 1
- Features: 4-bit quant, Flash Attention
```

## Data Pipeline

```
1. Raw Data
   ├── langfuse_test.json (alerts)
   └── teacher_output.json (Gemini outputs + labels)
          ↓
2. prepare_dataset.py
   → Filters to keep only Gemini agreements (hit == 1)
   → Outputs: data/train_distill.json
          ↓
3. split_dataset.py
   → Splits into train/val/test (70/15/15)
   → Prevents data leakage
   → Outputs: data/splits/{train,val,test}.json
          ↓
4. Training
   → Loads from data/splits/
   → Trains model
   → Outputs: outputs/{local_model,distilled_model}/
```

## Usage Examples

### Basic Usage

```bash
# Setup
./setup.sh

# Prepare and split data
python scripts/prepare_dataset.py
python scripts/split_dataset.py

# Train locally
python training/local_entry.py
```

### Using Make

```bash
# One command does it all
make all
```

### Custom Configuration

```python
from training import LocalTrainer, LocalTrainingConfig

# Customize config
config = LocalTrainingConfig(
    num_epochs=2,
    batch_size=2,
    learning_rate=1e-4,
)

# Train with custom config
trainer = LocalTrainer(config)
trainer.run_full_training()
```

### Running with UV (without activation)

```bash
uv run python training/local_entry.py
uv run python scripts/prepare_dataset.py
```

## Output

Training produces:

```
outputs/
├── local_model/ or distilled_model/
│   ├── adapter_model.safetensors    # LoRA weights
│   ├── adapter_config.json
│   ├── tokenizer files
│   └── evaluation_results.json      # Metrics
│
└── checkpoints/
    └── checkpoint-*/                # Training checkpoints
```

## Makefile Commands

```bash
# Setup
make setup           # Initialize UV environment

# Data
make prepare         # Filter dataset
make split           # Split train/val/test

# Training
make train-local     # Train on Mac (0.5B)
make train-cloud     # Train on GPU (3B)

# Docker
make docker-build    # Build images
make docker-train-mac    # Train in Docker

# Quality
make lint            # Run linting
make format          # Format code
make clean           # Clean artifacts

# Workflows
make all             # Complete pipeline
```

## Requirements

### System Requirements

**Local (Mac):**
- Python 3.9+
- 8GB+ RAM
- Apple Silicon (for MPS) or Intel

**Cloud (GPU):**
- Python 3.9+
- CUDA-capable GPU
- 16GB+ VRAM (for 3B model with 4-bit)

### Dependencies

Managed via `pyproject.toml`:

**Core:**
- torch
- transformers
- datasets
- peft
- accelerate

**Cloud only:**
- unsloth (GPU optimizations)
- trl (SFT trainer)
- xformers (efficient attention)

## Extending the Architecture

Want to add a new training environment?

```python
# 1. Create config
@dataclass
class AWSTrainingConfig(TrainingConfig):
    s3_bucket: str = "my-bucket"
    # ... AWS-specific settings

# 2. Create trainer
class AWSTrainer(BaseTrainer):
    def load_model(self):
        # AWS-specific model loading
        pass

    def create_trainer(self, train_dataset, eval_dataset):
        # AWS-specific trainer creation
        pass

    # load_datasets() inherited from base!

# 3. Create entry point
# aws_entry.py
def main():
    config = AWSTrainingConfig()
    trainer = AWSTrainer(config)
    trainer.run_full_training()
```

Just 2 methods to implement!

## Troubleshooting

### UV Setup Issues

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reinstall environment
rm -rf .venv
uv sync
```

### Import Errors

```bash
# Activate environment
source .venv/bin/activate

# Or run with uv
uv run python training/local_entry.py
```

### GPU Not Detected

```bash
# Check CUDA
nvidia-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### MPS Not Available (Mac)

```bash
# Check MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, use CPU
# Edit config: use_cpu_only=True
```

## Documentation

- **training/ARCHITECTURE.md** - Detailed architecture docs
- **SETUP.md** - Setup instructions
- **UV_SETUP.md** - UV-specific setup
- **DATA_FILES.md** - Expected data formats

## Summary

**Three simple commands:**

```bash
./setup.sh                          # 1. Setup
make prepare && make split          # 2. Prepare data
make train-local                    # 3. Train
```

That's it! 🎉

For cloud training, just use `make train-cloud` instead.
