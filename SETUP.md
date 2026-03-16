# Setup Guide

## Quick Start with UV

UV is a fast Python package installer. This project uses UV for dependency management.

### 1. Install UV

```bash
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 2. Setup Environment

```bash
# For Mac (local training)
make install-mac

# For Cloud/GPU (production training)
make install-cloud

# For development
make install-dev
```

### 3. Run Training

```bash
# Local training (0.5B model)
make train-local

# Cloud training (3B model)
make train-cloud
```

## Manual Setup (without Make)

### Local (Mac) Setup

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Mac/Linux

# Install dependencies
uv pip install -e ".[mac]"
```

### Cloud (GPU) Setup

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies (includes Unsloth)
uv pip install -e ".[cloud]"
```

## Complete Workflow

### Prepare Data

```bash
# 1. Prepare dataset (filters by Gemini agreement)
make prepare

# 2. Split into train/val/test
make split
```

### Train Model

```bash
# Local (Mac) - 0.5B model for validation
make train-local

# Cloud (GPU) - 3B model for production
make train-cloud
```

### Run Everything

```bash
# Prepare, split, and train locally
make all
```

## Makefile Commands

### Installation
- `make install` - Install base dependencies
- `make install-mac` - Install Mac dependencies
- `make install-cloud` - Install cloud/GPU dependencies
- `make install-dev` - Install development tools

### Data Preparation
- `make prepare` - Prepare and filter dataset
- `make split` - Split into train/val/test

### Training
- `make train-local` - Local training (Mac)
- `make train-cloud` - Cloud training (GPU)

### Testing & Quality
- `make test` - Run tests
- `make lint` - Run linting
- `make format` - Format code
- `make validate` - Lint + test

### Docker
- `make docker-build` - Build all Docker images
- `make docker-build-mac` - Build Mac image only
- `make docker-build-cloud` - Build cloud image only
- `make docker-train-mac` - Train in Docker (Mac)
- `make docker-train-cloud` - Train in Docker (cloud)
- `make docker-clean` - Clean Docker artifacts

### Cleanup
- `make clean` - Clean build artifacts

### Workflows
- `make all` - Complete pipeline (prepare → split → train-local)
- `make dev-setup` - Setup development environment
- `make validate` - Run linting and tests

## Project Structure

```
experimental-distill/
├── training/                 # Training module (SOLID architecture)
│   ├── base_trainer.py          # Abstract interface
│   ├── local_trainer.py         # Local implementation
│   ├── cloud_trainer.py         # Cloud implementation
│   ├── config.py                # Configuration classes
│   ├── local_entry.py           # Local entry point
│   └── cloud_entry.py           # Cloud entry point
│
├── scripts/                  # Data preparation scripts
│   ├── prepare_dataset.py       # Filter & prepare data
│   └── split_dataset.py         # Split train/val/test
│
├── prompts/                  # Prompt templates
├── data/                     # Data files
├── outputs/                  # Generated models
│
├── pyproject.toml            # UV dependencies
├── Makefile                  # Build commands
└── .python-version           # Python version (3.11)
```

## UV Benefits

### Why UV?

1. **Fast**: 10-100x faster than pip
2. **Reliable**: Consistent dependency resolution
3. **Modern**: Built on Rust, designed for 2024+
4. **Compatible**: Works with pip requirements

### UV Commands

```bash
# Install package
uv pip install package-name

# Install from requirements
uv pip install -r requirements.txt

# Install editable package
uv pip install -e .

# Install with extras
uv pip install -e ".[cloud,dev]"

# Create venv
uv venv

# Sync dependencies
uv pip sync requirements.txt
```

## Configuration

### Local Training Config

Edit `training/config.py`:

```python
@dataclass
class LocalTrainingConfig(TrainingConfig):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 512
    num_epochs: int = 1
    batch_size: int = 1
    # ... customize as needed
```

### Cloud Training Config

```python
@dataclass
class CloudTrainingConfig(TrainingConfig):
    model_name: str = "unsloth/Qwen2.5-3B-Instruct"
    max_seq_length: int = 32000
    num_epochs: int = 1
    batch_size: int = 2
    # ... customize as needed
```

## Python Version

This project uses **Python 3.11**. The version is specified in `.python-version`.

UV will automatically use the correct Python version if you have it installed.

## Dependencies

### Base (all environments)
- torch
- transformers
- datasets
- peft
- accelerate

### Cloud-only
- unsloth (GPU optimizations)
- trl (RLHF/SFT)
- xformers (memory efficient attention)
- bitsandbytes (quantization)

### Mac-only
- Standard packages without CUDA

### Development
- pytest (testing)
- black (formatting)
- ruff (linting)
- mypy (type checking)

## Troubleshooting

### UV not found

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### Wrong Python version

```bash
# Install Python 3.11
# Mac:
brew install python@3.11

# Then UV will use it automatically
uv venv --python 3.11
```

### GPU not detected (cloud)

```bash
# Check CUDA
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### MPS not available (Mac)

```bash
# Check MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, set use_cpu_only=True in LocalTrainingConfig
```

## Next Steps

1. **Setup environment**: `make install-mac` or `make install-cloud`
2. **Prepare data**: `make prepare && make split`
3. **Train model**: `make train-local` or `make train-cloud`
4. **Check results**: See `outputs/` folder

For architecture details, see `training/ARCHITECTURE.md`.
