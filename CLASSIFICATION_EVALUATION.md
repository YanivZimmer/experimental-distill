# Classification Accuracy Evaluation

This document explains how the classification accuracy evaluation works in the distillation pipeline.

## Overview

In addition to the standard loss-based evaluation, the pipeline now evaluates **classification accuracy** by:

1. **Generating outputs** - Model generates JSON responses for each validation example
2. **Extracting classifications** - Parse `primary_assessment` and `final_decision` fields
3. **Comparing predictions** - Use keyword matching to determine benign vs malicious
4. **Calculating accuracy** - Compare predicted vs expected labels

This provides a direct measure of how well the model classifies alerts, beyond just next-token prediction loss.

## When Evaluation Happens

Classification accuracy is evaluated **after training** on the validation set, as part of the `run_full_training()` pipeline:

```
1. Pre-training loss evaluation
2. Training
3. Post-training loss evaluation
4. Test set loss evaluation
5. ✨ Classification accuracy evaluation (NEW)
6. Save model
7. Save all results
```

## How It Works

### 1. Generate Outputs

For each validation example:
- Build the prompt using the alert data
- Generate model output (JSON response)
- Extract the assistant's response

```python
prompt = f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# Generate with model
outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)
```

### 2. Extract Classification

Parse the JSON output and extract key fields:

```python
from .evaluation import extract_classification_from_output

classification = extract_classification_from_output(generated_text)

# Returns:
{
    "primary_assessment": "High-Confidence Suspicious",
    "final_decision": "Escalate for Review",
    "severity": "Medium",
    "justification": "..."
}
```

### 3. Infer Category

Use keyword matching to determine the predicted category:

**Benign keywords:**
- `primary_assessment`: "anomalous but benign", "expected noise", "false positive"
- `final_decision`: "close"

**Malicious keywords:**
- `primary_assessment`: "confirmed malicious", "high-confidence suspicious"
- `final_decision`: "escalate immediately", "escalate for review"

```python
from .evaluation import is_benign_classification, is_malicious_classification

if is_benign_classification(classification):
    predicted = "benign"
elif is_malicious_classification(classification):
    predicted = "malicious"
else:
    predicted = "unknown"
```

### 4. Compare and Score

```python
from .evaluation import evaluate_hit_miss

hit, eval_details = evaluate_hit_miss(generated_text, expected_label)

# Returns:
# hit: 1 if match, 0 if miss
# eval_details: {
#     "expected": "malicious",
#     "predicted": "malicious",
#     "classification": {...}
# }
```

## Label Normalization

Expected labels are normalized to standard categories:

| Input Label | Normalized |
|-------------|------------|
| "True Positive - Malicious" | "malicious" |
| "True Positive - Benign" | "benign" |
| "False Positive" | "false_positive" → treated as "benign" |

## Metrics Returned

The `evaluate_classification_accuracy()` method returns:

```json
{
  "accuracy": 0.75,          // Overall accuracy
  "hits": 12,                // Number of correct predictions
  "total": 16,               // Total examples
  "by_category": {           // Per-category breakdown
    "benign": {
      "accuracy": 0.80,
      "hits": 4,
      "total": 5
    },
    "malicious": {
      "accuracy": 0.70,
      "hits": 7,
      "total": 10
    }
  },
  "sample_misses": [         // First 5 misses for debugging
    {
      "index": 2,
      "expected_label": "True Positive - Malicious",
      "expected_normalized": "malicious",
      "predicted": "benign",
      "hit": 0,
      "generated_text": "...",
      "classification": {...}
    }
  ],
  "sample_hits": [           // First 5 hits for validation
    {
      "index": 0,
      "expected_label": "True Positive - Benign",
      "expected_normalized": "benign",
      "predicted": "benign",
      "hit": 1,
      "generated_text": "...",
      "classification": {...}
    }
  ]
}
```

## Where Results Are Saved

Results are saved to `outputs/*/evaluation_results.json`:

```json
{
  "pre_training": {
    "eval_loss": 8.234
  },
  "post_training": {
    "eval_loss": 2.456
  },
  "test": {
    "eval_loss": 2.512
  },
  "classification_accuracy": {
    "accuracy": 0.75,
    "hits": 12,
    "total": 16,
    "by_category": {...},
    "sample_misses": [...],
    "sample_hits": [...]
  }
}
```

## Terminal Output

During evaluation, you'll see:

```
============================================================
EVALUATING CLASSIFICATION ACCURACY
============================================================
   Generating outputs for 16 examples...
Evaluating: 100%|████████████| 16/16 [00:15<00:00,  1.03it/s]
Accuracy: 75.00%
Hits: 12/16
By category: {'benign': {'accuracy': 0.8, 'hits': 4, 'total': 5}, 'malicious': {...}}
```

## Implementation Details

### Local Trainer (CPU/MPS)

Uses standard transformers generation:

```python
def evaluate_classification_accuracy(self, dataset):
    self.model.eval()

    for example in dataset:
        # Generate with model.generate()
        outputs = self.model.generate(...)

        # Evaluate
        hit, details = evaluate_hit_miss(output, expected)
```

### Cloud Trainer (GPU)

Uses Unsloth's optimized inference:

```python
def evaluate_classification_accuracy(self, dataset):
    # Enable Unsloth inference mode
    FastLanguageModel.for_inference(self.model)

    for example in dataset:
        # Generate on GPU
        outputs = self.model.generate(...)

        # Evaluate
        hit, details = evaluate_hit_miss(output, expected)
```

### Mock Trainer (Testing)

Returns simulated results:

```python
def evaluate_classification_accuracy(self, dataset):
    # Simulate 60-85% accuracy
    hits = int(len(dataset) * random.uniform(0.6, 0.85))

    return {
        "accuracy": hits / len(dataset),
        "hits": hits,
        "total": len(dataset),
        ...
    }
```

## Key Points

1. **Generation is slow** - This evaluation generates full outputs, so it takes longer than loss evaluation
2. **Not used for training** - This is an evaluation metric only, not a loss function
3. **Complements loss** - Low loss doesn't always mean good classification; this metric measures actual task performance
4. **Keyword-based** - Classification extraction relies on predefined keywords (can be extended)
5. **Samples included** - The results include sample misses and hits for debugging

## Example Evaluation Flow

Given a validation set with 16 examples:

```
1. Load raw validation data (with expected labels)
2. For each example:
   - Build prompt with alert JSON
   - Generate model output (512 max tokens)
   - Parse JSON response
   - Extract primary_assessment and final_decision
   - Infer category (benign/malicious/unknown)
   - Compare with expected label
   - Record hit (1) or miss (0)
3. Calculate metrics:
   - Overall accuracy: 12/16 = 75%
   - Benign accuracy: 4/5 = 80%
   - Malicious accuracy: 7/10 = 70%
4. Save results with sample misses/hits
```

## Why This Matters

**Loss tells you:** How well the model predicts the next token in the training format

**Classification accuracy tells you:** How well the model actually classifies alerts

A model can have low loss but still classify incorrectly if:
- It generates fluent but wrong classifications
- The reasoning is good but the conclusion is wrong
- The JSON format is correct but the content is incorrect

This metric directly measures the **task performance** we care about.

## Future Enhancements

Potential improvements:

1. **Add more keywords** - Expand benign/malicious keyword lists
2. **Use regex patterns** - More robust extraction beyond keywords
3. **Evaluate reasoning quality** - Check if justifications make sense
4. **Confusion matrix** - Show which categories are commonly confused
5. **Confidence scores** - Extract severity levels and match against thresholds
