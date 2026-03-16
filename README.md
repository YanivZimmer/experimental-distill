# Distillation Training Pipeline

Knowledge distillation from Gemini 2.0 to smaller Qwen models for security alert classification.

## Quick Start

```bash
# 1. Setup
./setup.sh

# 2. Prepare data
make prepare && make split

# 3. Train
make train-local    # Local (Mac): 0.5B model, 512 tokens
make train-cloud    # Cloud (GPU): 3B model, 32K tokens
```

## Project Structure

```
experimental-distill/
├── training/              # SOLID architecture training module
│   ├── base_trainer.py       # Abstract base class
│   ├── local_trainer.py      # Mac implementation
│   ├── cloud_trainer.py      # GPU implementation
│   ├── config.py             # Configuration classes
│   ├── local_entry.py        # Local entry point
│   └── cloud_entry.py        # Cloud entry point
│
├── scripts/               # Data preparation
│   ├── prepare_dataset.py    # Filter by Gemini agreement
│   └── split_dataset.py      # Split train/val/test
│
├── prompts/               # Prompt templates
├── data/                  # Data files
└── outputs/               # Generated models
```

## Architecture

Clean, SOLID-principles-based design:

- **S**ingle Responsibility - Each class has one job
- **O**pen/Closed - Open for extension, closed for modification
- **L**iskov Substitution - Implementations are interchangeable
- **I**nterface Segregation - Clean, focused interfaces
- **D**ependency Inversion - Depend on abstractions

```
BaseTrainer (Abstract)
├── load_model()        [abstract - implement per environment]
├── create_trainer()    [abstract - implement per environment]
├── load_datasets()     [concrete - shared implementation]
└── run_full_training() [concrete - template method]
    ├── LocalTrainer    (Mac: 0.5B model, CPU/MPS)
    └── CloudTrainer    (GPU: 3B model, Unsloth)
```

## Commands

### Setup
```bash
./setup.sh              # Initialize UV environment
make setup              # Same as above
```

### Data Preparation
```bash
make prepare            # Filter dataset (Gemini agreements only)
make split              # Split into train/val/test
```

### Training
```bash
make train-local        # Train on Mac (validation)
make train-cloud        # Train on GPU (production)
```

### Complete Pipeline
```bash
make all                # prepare → split → train-local
```

### Docker
```bash
make docker-build       # Build images
make docker-train-mac   # Train in Docker (Mac)
```

## Requirements

- Python 3.9+
- UV package manager
- Mac: 8GB+ RAM
- Cloud: CUDA GPU with 16GB+ VRAM

## Configuration

### Local (Mac)
- Model: Qwen2.5-0.5B-Instruct
- Context: 512 tokens
- Device: CPU/MPS
- Purpose: Code validation

### Cloud (GPU)
- Model: Qwen2.5-3B-Instruct
- Context: 32,000 tokens
- Device: GPU + Unsloth + 4-bit
- Purpose: Production training

## Documentation

- **GUIDE.md** - Complete guide with examples
- **training/ARCHITECTURE.md** - Architecture details
- **SETUP.md** - Setup instructions
- **DATA_FILES.md** - Data format reference

## Key Features

✅ SOLID principles design
✅ Zero code duplication
✅ Easy to extend (just 2 methods to implement)
✅ Consistent interface (local & cloud)
✅ UV-based dependency management
✅ Before/after evaluation
✅ Checkpoint saving
✅ Make-based workflow

## Output

Training produces:
```
outputs/
├── local_model/ or distilled_model/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer files
│   └── evaluation_results.json
└── checkpoints/
```

## Example Usage

```python
from training import LocalTrainer, LocalTrainingConfig

# Create and run trainer
config = LocalTrainingConfig()
trainer = LocalTrainer(config)
trainer.run_full_training()
```

## Extending

Add a new environment in 3 steps:

```python
# 1. Config
class AWSConfig(TrainingConfig): ...

# 2. Trainer
class AWSTrainer(BaseTrainer):
    def load_model(self): ...
    def create_trainer(self): ...
    # load_datasets() inherited!

# 3. Entry point
python aws_entry.py
```

## License

[Your license]
