# Loss and Evaluation Explained

## Overview

The training uses **Causal Language Modeling (CLM)** loss, which trains the model to predict the next token given previous tokens.

## Loss Calculation

### 1. What is Causal Language Modeling Loss?

For each position in the sequence, the model:
- Sees tokens [0, 1, 2, ..., i-1]
- Predicts token at position i
- Compares prediction with actual token i
- Calculates cross-entropy loss

### 2. How It Works in Our Code

**In `local_trainer.py` (tokenize_function):**
```python
def tokenize_function(examples):
    result = self.tokenizer(
        examples["text"],
        truncation=True,
        max_length=self.config.max_seq_length,
        padding=False,
    )
    # IMPORTANT: Set labels = input_ids
    result["labels"] = result["input_ids"].copy()
    return result
```

**Key points:**
- `input_ids`: The tokenized input sequence
- `labels`: What the model should predict (same as input_ids for CLM)
- The Trainer automatically shifts labels by 1 position internally

### 3. Loss Formula

```
For each token position i:
    prediction = model(input_ids[0:i])  # Predict based on previous tokens
    target = labels[i]                    # Actual next token
    loss += CrossEntropy(prediction, target)

Final Loss = Average loss across all tokens
```

### 4. Example

Input text: `"The cat sat on the mat"`
Tokenized: `[464, 3857, 3724, 319, 262, 2603]`

```
Position 0: Predict token 1 (3857) given [464]
Position 1: Predict token 2 (3724) given [464, 3857]
Position 2: Predict token 3 (319) given [464, 3857, 3724]
...
```

Loss = Average of all prediction errors

## Evaluation Process

### 1. When Evaluation Happens

**In `base_trainer.py` (run_full_training):**
```python
# 1. Before training
pre_results = self.evaluate_before_training()

# 2. Train model
self.train()

# 3. After training
post_results = self.evaluate_after_training()

# 4. On test set (optional)
test_results = self.evaluate_test_set(datasets['test'])
```

### 2. What evaluate() Does

**Code path:**
```python
def evaluate_before_training(self):
    results = self.trainer.evaluate()  # Calls Hugging Face Trainer.evaluate()
    return results
```

**Trainer.evaluate() internally:**
1. Sets model to eval mode (`model.eval()`)
2. Disables gradient computation (`torch.no_grad()`)
3. Iterates through eval dataset in batches
4. For each batch:
   - Forward pass through model
   - Calculate loss (same formula as training)
   - Accumulate metrics
5. Returns averaged metrics

### 3. Metrics Returned

```python
{
    'eval_loss': 1.234,           # Average loss on eval set
    'eval_runtime': 12.5,         # Time taken (seconds)
    'eval_samples_per_second': 6.4,  # Throughput
    'eval_steps_per_second': 0.8,    # Steps throughput
}
```

### 4. Why We Evaluate Multiple Times

```
Before Training → Baseline performance (how bad is random model?)
After Training → Final performance (did we improve?)
Test Set → Generalization (does it work on unseen data?)
```

## Loss Interpretation

### Good vs Bad Loss

**For our use case (alert classification with reasoning):**
- **Initial loss (untrained):** ~8-12
  - Random predictions, model doesn't know anything

- **Good final loss:** ~1.5-3.0
  - Model learned patterns, can generate coherent text

- **Excellent final loss:** <1.0
  - Model is very confident and accurate
  - Rare for complex tasks like ours

- **Too low loss (<0.1):** ⚠️ Warning!
  - Might be overfitting
  - Check if it generalizes to test set

### What Affects Loss

1. **Model size**: Larger models → lower loss potential
2. **Data quality**: Better examples → lower loss
3. **Sequence length**: Longer sequences → harder to predict
4. **Training time**: More epochs → lower loss (until overfitting)
5. **Learning rate**: Too high → unstable, too low → slow convergence

## In Our Pipeline

### Training Configuration

```python
# local_trainer.py
TrainingArguments(
    eval_strategy="epoch",          # Evaluate after each epoch
    metric_for_best_model="loss",   # Use loss to pick best model
    load_best_model_at_end=True,    # Load checkpoint with lowest loss
)
```

### What We Track

**Saved to `outputs/*/evaluation_results.json`:**
```json
{
  "pre_training": {
    "eval_loss": 8.234,
    "eval_runtime": 2.1
  },
  "post_training": {
    "eval_loss": 2.456,
    "eval_runtime": 2.3
  },
  "test": {
    "eval_loss": 2.512,
    "eval_runtime": 1.8
  }
}
```

**Improvement calculation:**
```python
improvement = pre_results['eval_loss'] - post_results['eval_loss']
# Example: 8.234 - 2.456 = 5.778 improvement ✓
```

## Data Collator's Role

```python
# local_trainer.py
data_collator = DataCollatorForLanguageModeling(
    tokenizer=self.tokenizer,
    mlm=False,  # NOT masked language modeling (that's BERT-style)
                # We're doing causal LM (GPT-style)
)
```

**What it does:**
1. Pads sequences to same length in batch
2. Creates attention mask (1 for real tokens, 0 for padding)
3. Sets labels to -100 for padding tokens (ignored in loss)

## Advanced: Loss Masking

Only real tokens contribute to loss:

```python
input_ids:  [464, 3857, 3724,  PAD,  PAD]
labels:     [464, 3857, 3724, -100, -100]  # -100 = ignore in loss
```

This prevents the model from "cheating" by learning padding patterns.

## Summary

**Loss calculation:**
- Cross-entropy between predicted and actual next tokens
- Averaged across all tokens in sequence
- Padding tokens ignored (labels = -100)

**Evaluation:**
- Same loss calculation, but without training (no gradient updates)
- Done before training (baseline), after training (final), and on test set (generalization)
- Model in eval mode, dropout disabled

**Goal:**
- Minimize loss = Model better at predicting next tokens
- Lower loss = Better understanding of patterns
- For our task: Lower loss = Better at generating reasoning + classification

**In practice:**
- Pre-training loss: ~8-12 (random)
- Post-training loss: ~2-4 (after 1 epoch)
- Improvement: ~4-8 points
- Test loss should be close to validation loss (good generalization)

---

**Key Insight:** We're not directly training on classification accuracy. We're training the model to generate the full reasoning + classification text. The loss measures how well it predicts each word in that output. Good loss = good text generation = good reasoning = good classification!
