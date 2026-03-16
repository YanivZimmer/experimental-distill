# Repository Organization (March 2026)

## 📁 New Folder Structure

The repository has been reorganized for better maintainability:

```
experimental-distill/
├── scripts/              # Data preparation and testing
│   ├── prepare_dataset.py    # Filter & prepare training data
│   ├── split_dataset.py      # Split into train/val/test
│   ├── test_dataset.py       # Test dataset loading
│   ├── test_model.py         # Test model inference
│   └── test_pipeline.py      # End-to-end pipeline test
│
├── training/             # Training scripts
│   ├── train.py              # Main training (Qwen 3B, 32K tokens)
│   ├── train_mac.py          # Mac-optimized training (0.5B model)
│   └── cloud_train.py        # Google Cloud wrapper
│
├── prompts/              # Prompt templates
│   ├── baseline.txt          # Standard classification prompt
│   ├── gepa_gemini_v1.txt    # GEPA framework prompt
│   └── chat.txt              # Conversational prompt
│
├── data/                 # Data files
│   ├── langfuse_test.json              # Input alerts
│   ├── baseline_benchmark_*.json       # Teacher outputs
│   ├── train_distill.json              # Prepared training data
│   ├── splits/                         # Train/val/test splits
│   └── notes.txt                       # User notes
│
├── outputs/              # Generated during training
│   ├── checkpoints/          # Training checkpoints
│   └── distilled_model/      # Final trained model
│
├── docs/                 # Documentation
│   ├── QUICK_START.md            # Quick start guide
│   ├── IMPLEMENTATION_SUMMARY.md # Recent changes
│   └── ... (other documentation)
│
├── requirements.txt      # Python dependencies
└── requirements.mac.txt  # Mac-specific dependencies
```

## 🔧 Configuration Changes

### Max Token Size: 4096 → 32000

All training scripts now default to **32,000 tokens** (increased from 4,096):

**training/train.py:**
```python
CONFIG = {
    "max_seq_length": 32000,  # Was 4096
    ...
}
```

**training/train_mac.py:**
- Kept at 512 tokens for Mac RAM constraints (0.5B validation model)

### Updated Paths

All scripts now reference the new folder structure:

- Prompt templates: `prompts/baseline.txt` (was `baseline.txt`)
- Training scripts: `training/train.py` (was `train.py`)
- Data preparation: `scripts/prepare_dataset.py` (was `prepare_dataset.py`)

## 🚀 How to Run

### From Repository Root

```bash
# Prepare dataset
cd scripts
python prepare_dataset.py
python split_dataset.py

# Train model
cd ../training
python train.py
```

### Or Using Relative Paths

```bash
# From anywhere
python scripts/prepare_dataset.py
python scripts/split_dataset.py
python training/train.py
```

## 📝 Benefits of New Structure

1. **Clearer Organization**: Related files grouped together
2. **Easier Navigation**: Find files by purpose (scripts, training, prompts)
3. **Better Separation**: Training logic separate from data prep
4. **Scalable**: Easy to add new scripts in appropriate folders
5. **Documentation**: Centralized in docs/ folder

## 🔄 Migration Notes

If you have existing scripts or notebooks:

**Old paths → New paths:**
- `prepare_dataset.py` → `scripts/prepare_dataset.py`
- `train.py` → `training/train.py`
- `baseline.txt` → `prompts/baseline.txt`
- `*.md` docs → `docs/*.md`

**Update your imports/references:**
```python
# Old
from train import train, CONFIG

# New
from training.train import train, CONFIG
```

## ✅ What Changed

1. ✅ Files organized into logical folders
2. ✅ Max token size increased to 32,000
3. ✅ Prompt paths updated to `prompts/` folder
4. ✅ Documentation moved to `docs/` folder
5. ✅ Repository root kept clean

## 📖 Next Steps

See **docs/QUICK_START.md** for updated commands using the new structure.
