# Classification Evaluation Refactor

## Changes Made

### 1. **Extracted Classification Logic** → `training/classification_evaluator.py`

Created a dedicated `ClassificationEvaluator` class that handles:
- Loading prompt templates
- Generating outputs from the model
- Extracting classifications using keyword matching
- Calculating accuracy metrics

**Benefits:**
- ✅ Reusable across local, cloud, and future trainer implementations
- ✅ Separates generation logic from trainer classes
- ✅ Easier to test and debug
- ✅ Single source of truth for evaluation logic

**Usage:**
```python
from training.classification_evaluator import ClassificationEvaluator

evaluator = ClassificationEvaluator(
    model=model,
    tokenizer=tokenizer,
    prompt_template_path="prompts/baseline.txt",
    max_seq_length=32000,
)

# Evaluate on dataset
results = evaluator.evaluate_dataset(dataset, max_examples=5)

# Test single example (for debugging)
result = evaluator.test_single_example(example)
```

### 2. **Added Pre-Training Evaluation**

The pipeline now evaluates classification accuracy **both before and after training**:

```
1. Pre-training loss evaluation
2. 🆕 Pre-training classification accuracy (5 examples)
3. Training
4. Post-training loss evaluation
5. Post-training classification accuracy (full validation set)
6. Test set loss evaluation
7. Save model
8. Save results
```

**Why 5 examples for pre-training?**
- Pre-training model is untrained/random → likely 0-20% accuracy
- Generating outputs is slow
- 5 examples is enough to see baseline performance
- Post-training uses full validation set for accurate measurement

### 3. **Updated All Trainers**

**LocalTrainer** (training/local_trainer.py:180-193)
```python
def evaluate_classification_accuracy(self, dataset, max_examples=None):
    evaluator = ClassificationEvaluator(...)
    return evaluator.evaluate_dataset(dataset, max_examples)
```

**CloudTrainer** (training/cloud_trainer.py:97-110)
```python
def evaluate_classification_accuracy(self, dataset, max_examples=None):
    FastLanguageModel.for_inference(self.model)  # Unsloth optimization
    evaluator = ClassificationEvaluator(...)
    return evaluator.evaluate_dataset(dataset, max_examples)
```

**MockTrainer** (training/mock_trainer.py:155-208)
```python
def evaluate_classification_accuracy(self, dataset, max_examples=None):
    # Returns simulated results for testing
    return {...}
```

### 4. **Enhanced Results Tracking**

Results now include both pre and post classification accuracy:

**`outputs/*/evaluation_results.json`:**
```json
{
  "pre_training": {
    "eval_loss": 8.234
  },
  "pre_classification_accuracy": {
    "accuracy": 0.20,
    "hits": 1,
    "total": 5,
    "by_category": {...},
    "sample_misses": [...],
    "sample_hits": [...]
  },
  "post_training": {
    "eval_loss": 2.456
  },
  "post_classification_accuracy": {
    "accuracy": 0.75,
    "hits": 12,
    "total": 16,
    "by_category": {...},
    "sample_misses": [...],
    "sample_hits": [...]
  },
  "test": {
    "eval_loss": 2.512
  }
}
```

### 5. **Added Test Scripts**

**`scripts/test_label_extraction.py`** - Tests keyword matching without a model
```bash
python scripts/test_label_extraction.py
```

Tests:
- Label normalization (True Positive - Malicious → malicious)
- JSON extraction from various formats (markdown, plain, with text)
- Hit-miss evaluation
- Keyword matching for benign/malicious

**`scripts/test_classification_eval.py`** - Tests full evaluation with model
```bash
python scripts/test_classification_eval.py
```

Tests:
- Loading model
- Generating outputs
- Extracting classifications
- Evaluating on single example
- Evaluating on multiple examples

## File Structure

```
training/
├── evaluation.py                  # Keyword matching logic
├── classification_evaluator.py    # 🆕 Generation + evaluation logic
├── base_trainer.py               # Updated to eval before & after
├── local_trainer.py              # Uses ClassificationEvaluator
├── cloud_trainer.py              # Uses ClassificationEvaluator
└── mock_trainer.py               # Uses ClassificationEvaluator

scripts/
├── test_label_extraction.py      # 🆕 Test keyword matching
└── test_classification_eval.py   # 🆕 Test full evaluation
```

## Terminal Output

When running training, you'll now see:

```
============================================================
DISTILLATION TRAINING - LocalTrainer
Model: Qwen/Qwen2.5-0.5B-Instruct
Max sequence length: 32000
Epochs: 1
============================================================

Loading model...
   ✓ Model loaded

Loading datasets...
   ✓ Train: 82 examples
   ✓ Val: 16 examples

============================================================
EVALUATION BEFORE TRAINING
============================================================
Pre-training validation loss: 8.2340

============================================================
EVALUATING CLASSIFICATION ACCURACY (PRE-TRAINING)
============================================================
   Limiting evaluation to 5 examples (out of 16)
   Generating outputs for 5 examples...
Evaluating: 100%|████████████| 5/5 [00:08<00:00,  1.61s/it]
Pre-training accuracy: 20.00%
Hits: 1/5

============================================================
STARTING TRAINING
============================================================
[training progress...]

============================================================
EVALUATION AFTER TRAINING
============================================================
Post-training validation loss: 2.4560

============================================================
EVALUATING CLASSIFICATION ACCURACY (POST-TRAINING)
============================================================
   Generating outputs for 16 examples...
Evaluating: 100%|████████████| 16/16 [00:25<00:00,  1.59s/it]
Post-training accuracy: 75.00%
Hits: 12/16
By category: {'benign': {'accuracy': 0.8, 'hits': 4, 'total': 5}, ...}

============================================================
SAVING MODEL
============================================================
Model saved to outputs/local_training

Evaluation results saved to outputs/local_training/evaluation_results.json

Validation loss improvement: 5.7780
Classification accuracy improvement: +55.00%
  Pre-training:  20.00%
  Post-training: 75.00%

============================================================
TRAINING COMPLETE!
============================================================
```

## Key Benefits

1. **Baseline comparison** - See how much training improved classification
2. **Faster debugging** - Test label extraction without training
3. **Cleaner code** - Logic extracted to dedicated class
4. **Better visibility** - Clear improvement metrics in output
5. **Reusable** - Same evaluator works for local, cloud, and future trainers

## Testing

### Test 1: Label Extraction (No Model Required)
```bash
source .venv/bin/activate
python scripts/test_label_extraction.py
```

**Expected output:**
- ✓ All label normalization tests pass
- ✓ JSON extraction works for all formats
- ✓ Keyword matching correctly identifies benign/malicious
- ✓ Hit-miss evaluation matches expectations

### Test 2: Full Evaluation (Requires Model)
```bash
source .venv/bin/activate
python scripts/test_classification_eval.py
```

**Expected output:**
- Model loads successfully
- Generates output for first example
- Extracts classification fields
- Shows hit/miss result
- Evaluates on 3 examples
- Displays accuracy metrics

### Test 3: Full Training Pipeline
```bash
make train-local
```

**Expected behavior:**
- Evaluates classification accuracy before training (~20% baseline)
- Trains for 1 epoch
- Evaluates classification accuracy after training (~60-80%)
- Shows improvement metrics
- Saves results with both pre and post accuracy

## Next Steps

To test that labels are being extracted correctly:

1. **Quick test (no model):**
   ```bash
   python scripts/test_label_extraction.py
   ```

2. **Full test (with model):**
   ```bash
   python scripts/test_classification_eval.py
   ```

3. **Run full training:**
   ```bash
   make train-local
   ```

All tests should pass and show label extraction working correctly.
