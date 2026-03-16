# Quick Start Guide

## Run the Complete Pipeline

### Option 1: Run all steps sequentially

```bash
# 1. Prepare dataset (filters to keep only samples where Gemini agreed with ground truth)
cd scripts
python prepare_dataset.py

# 2. Split into train/val/test
python split_dataset.py

# 3. Train Qwen 3B for 1 epoch with before/after evaluation
cd ../training
python train.py
```

### Option 2: Run as a single command (from repo root)

```bash
python scripts/prepare_dataset.py && python scripts/split_dataset.py && python training/train.py
```

## What Each Step Does

### scripts/prepare_dataset.py
- ✅ **Filters samples**: Only keeps data where Gemini agreed with ground truth (`hit == 1`)
- Outputs to `data/train_distill.json`

### scripts/split_dataset.py
- Splits filtered data into train/val/test (70/15/15)
- Outputs to `data/splits/`

### training/train.py
- ✅ **Uses Qwen 3B model** (3B parameters - faster training)
- ✅ **32K token context** (increased from 4K)
- ✅ **Evaluates BEFORE training** on validation set
- ✅ **Trains for 1 epoch**
- ✅ **Evaluates AFTER training** on validation set
- ✅ **Evaluates on test set**
- ✅ **Saves checkpoints** to `outputs/checkpoints/`
- ✅ **Saves final weights** to `outputs/distilled_model/`
- ✅ **Saves evaluation results** to `outputs/distilled_model/evaluation_results.json`

## Expected Results Structure

```
outputs/
├── checkpoints/                          # Training checkpoints
│   └── checkpoint-XXX/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
└── distilled_model/                      # Final model weights
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── evaluation_results.json           # Before/after/test metrics
```

## Configuration Highlights

### Key Settings (training/train.py)
```python
CONFIG = {
    "model_name": "unsloth/Qwen2.5-3B-Instruct",  # 3B student model
    "max_seq_length": 32000,                      # 32K context (was 4K)
    "num_epochs": 1,                              # Single epoch
    "prompt_template_path": "prompts/baseline.txt"
}
```

## Requirements Met ✅

1. ✅ Only keep samples where Gemini agreed with ground truth
2. ✅ Distill on train set for one epoch
3. ✅ Evaluate on validation set before and after distillation
4. ✅ Write weights to checkpoint file
5. ✅ Use Qwen 3B model as the student
6. ✅ Use 32K token context window (increased from 4K)

## Repository Structure

See **PROJECT_STRUCTURE.md** for details on the new folder organization.
