# Three-Category Classification System

## Overview

The evaluation system now correctly handles **three distinct categories**, not two:

1. **True Positive - Malicious** (`malicious`) - Real security threat
2. **True Positive - Benign** (`benign`) - Real activity but authorized/legitimate
3. **False Positive** (`false_positive`) - No actual security event (noise)

## Category Definitions

### 1. Malicious (True Positive - Malicious)
**Real threat that needs escalation**

Model should output:
- `primary_assessment`: "Confirmed Malicious", "High-Confidence Suspicious", "Policy Violation"
- `final_decision`: "Escalate Immediately", "Escalate for Review"

Dataset labels:
- `"Escalate Immediately (Assessment: Confirmed Malicious, ...)"`
- `"Escalate Immediately (Assessment: High-Confidence Suspicious, ...)"`
- `"Escalate Immediately (Assessment: Policy Violation, ...)"`

### 2. Benign (True Positive - Benign)
**Real activity but authorized/legitimate**

Model should output:
- `primary_assessment`: "Anomalous but Benign"
- `final_decision`: "Close"

Dataset labels:
- `"Close (Assessment: Anomalous but Benign, ...)"`

**Key distinction:** This is REAL activity (not noise), just authorized. Examples:
- Admin using diagnostic tools
- Authorized scripts/updates
- Legitimate but unusual behavior

### 3. False Positive
**No actual security event - just noise**

Model should output:
- `primary_assessment`: "Expected Noise", "False Positive"
- `final_decision`: "Close"

Dataset labels:
- `"Close (Assessment: Expected Noise, ...)"`
- `"Close (Assessment: False Positive, ...)"`

**Key distinction:** Nothing actually happened - detection error or normal system noise.

## Validation Set Distribution

Current validation set (`data/splits/val.json`):
- **Benign:** 5 examples (31%)
  - All are "Close (Assessment: Anomalous but Benign, ...)"
- **Malicious:** 11 examples (69%)
  - Confirmed Malicious, High-Confidence Suspicious, Policy Violation
- **False Positive:** 0 examples (0%)
  - None in validation set (likely in train/test sets)

## How Evaluation Works

### Step 1: Normalize Expected Label

```python
from training.evaluation import normalize_label

# Dataset format → normalized category
normalize_label("Close (Assessment: Anomalous but Benign, Severity: Low)")
# → "benign"

normalize_label("Escalate Immediately (Assessment: Confirmed Malicious, Severity: High)")
# → "malicious"

normalize_label("Close (Assessment: Expected Noise, Severity: Informational)")
# → "false_positive"
```

### Step 2: Extract Model Output

```python
from training.evaluation import extract_classification_from_output

output = """{
  "primary_assessment": "High-Confidence Suspicious",
  "final_decision": "Escalate for Review",
  "severity": "Medium"
}"""

classification = extract_classification_from_output(output)
# → {"primary_assessment": "High-Confidence Suspicious", "final_decision": "Escalate for Review", ...}
```

### Step 3: Classify Model Prediction

```python
from training.evaluation import (
    is_malicious_classification,
    is_benign_classification,
    is_false_positive_classification
)

# Check which category the model predicted
if is_malicious_classification(classification):
    predicted = "malicious"
elif is_benign_classification(classification):
    predicted = "benign"
elif is_false_positive_classification(classification):
    predicted = "false_positive"
else:
    predicted = "unknown"
```

### Step 4: Compare (Exact Match)

```python
from training.evaluation import evaluate_hit_miss

hit, details = evaluate_hit_miss(output, expected_label)

# Hit = 1 only if predicted exactly matches expected
# All three categories must match exactly:
# - malicious ≠ benign
# - benign ≠ false_positive
# - malicious ≠ false_positive
```

## Keyword Matching Logic

### Malicious Keywords
- **Assessments:** "confirmed malicious", "high-confidence suspicious", "policy violation"
- **Decisions:** "escalate immediately", "escalate for review"

### Benign Keywords
- **Assessments:** "anomalous but benign"
- Must explicitly say "benign" (not just "close")

### False Positive Keywords
- **Assessments:** "expected noise", "false positive"
- **Decisions:** "close" (without "benign" in assessment)

## Example Mappings

| Model Output | Category | Reasoning |
|--------------|----------|-----------|
| `primary_assessment: "High-Confidence Suspicious"<br>final_decision: "Escalate for Review"` | **malicious** | "suspicious" + "escalate" → threat |
| `primary_assessment: "Anomalous but Benign"<br>final_decision: "Close"` | **benign** | Explicitly says "benign" → authorized activity |
| `primary_assessment: "Expected Noise"<br>final_decision: "Close"` | **false_positive** | "noise" → no real event |
| `primary_assessment: "Confirmed Malicious"<br>final_decision: "Escalate Immediately"` | **malicious** | "malicious" + "escalate" → threat |
| `primary_assessment: "Policy Violation"<br>final_decision: "Escalate Immediately"` | **malicious** | Policy violations treated as threats |

## Testing

### Test All Three Categories

```bash
python scripts/test_label_extraction.py
```

**Expected output:**
```
✓ 'True Positive - Malicious' → 'malicious'
✓ 'True Positive - Benign' → 'benign'
✓ 'False Positive' → 'false_positive'

Test 1: Malicious → Malicious (HIT)
Test 2: Benign → Benign (HIT)
Test 3: False Positive → False Positive (HIT)
Test 4: Malicious → Benign (MISS)
Test 5: Benign → False Positive (MISS)
```

### Accuracy Calculation

With three categories, accuracy is:
```
accuracy = (malicious_hits + benign_hits + fp_hits) / total_examples
```

Per-category accuracy:
```python
{
    "malicious": {
        "accuracy": 0.80,  # 8/10 malicious correctly predicted
        "hits": 8,
        "total": 10
    },
    "benign": {
        "accuracy": 0.60,  # 3/5 benign correctly predicted
        "hits": 3,
        "total": 5
    },
    "false_positive": {
        "accuracy": 1.00,  # 1/1 FP correctly predicted
        "hits": 1,
        "total": 1
    }
}
```

## Why Three Categories Matter

**Before (2 categories):**
- Benign and False Positive were conflated
- Model could close legitimate activity and get credit for "detecting" false positives
- No distinction between "authorized but unusual" vs "complete noise"

**After (3 categories):**
- Model must distinguish real activity (benign) from noise (FP)
- More accurate measure of model understanding
- Better alignment with SOC analyst workflow:
  - Malicious → Escalate (threat)
  - Benign → Close (authorized)
  - False Positive → Close (noise)

## Files Updated

1. **training/evaluation.py**
   - `normalize_label()` - Returns malicious/benign/false_positive
   - `is_false_positive_classification()` - New function
   - `is_benign_classification()` - Updated to exclude FP
   - `evaluate_hit_miss()` - Three-way comparison

2. **training/classification_evaluator.py**
   - Tracks three categories in `category_stats`

3. **scripts/test_label_extraction.py**
   - Tests all three categories
   - Includes FP test cases

## Summary

✅ **Three distinct categories:** malicious, benign, false_positive
✅ **Exact matching required:** No category conflation
✅ **Clear keyword logic:** Each category has specific indicators
✅ **Validated on dataset:** Correctly maps all validation labels
✅ **All tests passing:** Label extraction, JSON parsing, hit-miss evaluation
