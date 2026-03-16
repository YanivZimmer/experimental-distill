# UV Environment Setup - Fixed!

## What Was Fixed

1. **Python version requirement** - Changed from `>=3.10` to `>=3.9` (compatible with your system)
2. **Package discovery** - Added explicit package configuration to `pyproject.toml`
3. **Setuptools configuration** - Specified which packages to include (`training`, `scripts`)

## Quick Setup (One Command)

```bash
./setup.sh
```

This will:
- Check UV installation
- Create virtual environment
- Install all dependencies
- Show next steps

## Manual Setup

### 1. Initialize UV Environment

```bash
uv sync
```

This creates `.venv/` and installs all dependencies.

### 2. Activate Environment (Optional)

```bash
source .venv/bin/activate
```

Or run commands directly with UV without activating:

```bash
uv run python training/local_entry.py
```

### 3. Verify Installation

```bash
# With activated environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Or with UV
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Updated pyproject.toml

The key fixes:

```toml
[project]
requires-python = ">=3.9"  # Changed from >=3.10

[tool.setuptools]
packages = ["training", "scripts"]  # Explicitly specify packages

[tool.setuptools.package-data]
training = ["*.txt"]
prompts = ["*.txt"]
```

## Usage After Setup

### Option 1: With Make (Recommended)

```bash
# Make handles activation automatically
make prepare       # Prepare dataset
make split         # Split dataset
make train-local   # Train locally
```

### Option 2: With UV (No activation needed)

```bash
uv run python scripts/prepare_dataset.py
uv run python scripts/split_dataset.py
uv run python training/local_entry.py
```

### Option 3: With Activated Environment

```bash
source .venv/bin/activate
python scripts/prepare_dataset.py
python scripts/split_dataset.py
python training/local_entry.py
```

## What UV Created

```
.venv/                  # Virtual environment
├── bin/
│   └── python         # Python interpreter
├── lib/
│   └── python3.9/
│       └── site-packages/  # Installed packages
└── pyvenv.cfg
```

## Installed Packages

Core packages:
- ✅ torch (PyTorch)
- ✅ transformers
- ✅ datasets
- ✅ peft (LoRA)
- ✅ accelerate
- ✅ sentencepiece

Total: 54 packages installed

## Common Commands

```bash
# Setup
uv sync                          # Install/update dependencies

# Run scripts
uv run python script.py         # Run without activation

# Add dependency
uv pip install package-name     # Add new package

# Update dependencies
uv sync --upgrade               # Update all packages

# Check environment
uv pip list                     # List installed packages
uv pip show package-name        # Show package info
```

## Troubleshooting

### Issue: "uv: command not found"

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or with Homebrew (Mac)
brew install uv
```

### Issue: "Multiple top-level packages discovered"

**Fixed!** The pyproject.toml now explicitly specifies which packages to include.

### Issue: "Python version not found"

The project now requires Python >=3.9 (your system has 3.9.6) ✓

### Issue: Package import errors after setup

```bash
# Reinstall
rm -rf .venv
uv sync
```

## Next Steps

After successful setup:

```bash
# 1. Prepare data
make prepare

# 2. Split data
make split

# 3. Train
make train-local
```

Or run the complete pipeline:

```bash
make all
```

## Makefile Integration

The Makefile now has a `setup` target:

```bash
make setup      # Same as: uv sync
```

All other Make commands work with the UV environment automatically!

## Benefits of UV

✅ **Fast** - 10-100x faster than pip
✅ **Reliable** - Consistent dependency resolution
✅ **Simple** - One command setup
✅ **Compatible** - Works with existing Python packages

## Summary

**Fixed issues:**
1. ✅ Python version compatibility (3.9+)
2. ✅ Package discovery configuration
3. ✅ Setuptools build configuration

**Setup is now:**
```bash
./setup.sh    # Or: uv sync
```

**Run training:**
```bash
make train-local    # Or: uv run python training/local_entry.py
```

That's it! 🎉
