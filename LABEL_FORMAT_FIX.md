# Label Format Fix

## Issues Fixed

### Issue 1: Dataset Uses Different Label Format

**Problem:**
The dataset has labels in the format:
```
"Escalate for Review (Assessment: High-Confidence Suspicious, Severity: Medium)"
"Close (Assessment: Anomalous but Benign, Severity: Low)"
"Escalate Immediately (Assessment: Confirmed Malicious, Severity: High)"
```

But the evaluation logic expected:
```
"True Positive - Malicious"
"True Positive - Benign"
"False Positive"
```

This caused all evaluations to return `expected_normalized: "unknown"` and `hit: 0`.

**Solution:**
Updated `normalize_label()` in `training/evaluation.py` to parse both formats:

1. **Legacy format** (from original labels): "True Positive - Malicious" → "malicious"
2. **Structured format** (from Gemini output): Parses decision and assessment

**Mapping logic:**
- `"Close"` → **benign** (regardless of assessment)
- `"Escalate for Review"` or `"Escalate Immediately"` with:
  - "Confirmed Malicious" or "High-Confidence Suspicious" → **malicious**
  - "Policy Violation" → **malicious** (treated as security issue)
  - "Anomalous but Benign" → **benign**

### Issue 2: Generated Text Included Full Prompt

**Problem:**
The `generated_text` field showed the entire conversation including:
- All prompt instructions (SOC analyst guidelines, investigation procedure, etc.)
- The full alert JSON
- Only at the end, the model's actual response

This made it impossible to extract the classification JSON.

**Solution:**
Updated `_generate_output()` in `training/classification_evaluator.py` with three extraction methods:

1. **Method 1**: Look for `<|im_start|>assistant` marker and extract after it
2. **Method 2**: If prompt includes ``` blocks, skip past them
3. **Method 3**: If text starts with "user" or prompt text, find where analysis actually starts (after line 50, looking for JSON or analysis keywords)

## Current Dataset Label Distribution

From `data/splits/val.json` (16 examples):

**Benign (Close):**
- Close (Assessment: Anomalous but Benign, Severity: High) - 1
- Close (Assessment: Anomalous but Benign, Severity: Informational) - 1
- Close (Assessment: Anomalous but Benign, Severity: Low) - 1
- Close (Assessment: Anomalous but Benign, Severity: Medium) - 2

**Malicious (Escalate):**
- Escalate Immediately (Assessment: Confirmed Malicious, Severity: High) - 1
- Escalate Immediately (Assessment: Confirmed Malicious, Severity: Low) - 1
- Escalate Immediately (Assessment: Confirmed Malicious, Severity: Medium) - 1
- Escalate Immediately (Assessment: High-Confidence Suspicious, Severity: High) - 1
- Escalate Immediately (Assessment: Policy Violation, Severity: High) - 1
- Escalate Immediately (Assessment: Policy Violation, Severity: Low) - 1
- Escalate Immediately (Assessment: Policy Violation, Severity: Medium) - 2
- Escalate for Review (Assessment: High-Confidence Suspicious, Severity: Medium) - 3

**Split:**
- Benign: 5 examples (31%)
- Malicious: 11 examples (69%)

## Testing

### Test 1: Label Normalization ✅

```bash
python scripts/test_label_extraction.py
```

**Results:**
- ✓ All legacy format tests pass
- ✓ All structured format tests pass
- ✓ JSON extraction works
- ✓ Hit-miss evaluation correct

### Test 2: Full Evaluation (with model)

```bash
python scripts/test_classification_eval.py
```

**Expected behavior:**
- Model generates JSON output
- `expected_normalized` shows "benign" or "malicious" (not "unknown")
- `classification` dict contains extracted fields
- `generated_text` shows only the model's response (not the full prompt)

## Label Normalization Examples

```python
from training.evaluation import normalize_label

# Structured format (from dataset)
normalize_label("Close (Assessment: Anomalous but Benign, Severity: Low)")
# → "benign"

normalize_label("Escalate for Review (Assessment: High-Confidence Suspicious, Severity: Medium)")
# → "malicious"

normalize_label("Escalate Immediately (Assessment: Policy Violation, Severity: High)")
# → "malicious"

# Legacy format (still supported)
normalize_label("True Positive - Malicious")
# → "malicious"

normalize_label("False Positive")
# → "benign"
```

## What the Model Should Generate

The model should generate JSON like:

```json
{
  "primary_assessment": "High-Confidence Suspicious",
  "final_decision": "Escalate for Review",
  "severity": "Medium",
  "justification": "..."
}
```

The evaluation will:
1. Extract `primary_assessment` and `final_decision`
2. Match keywords:
   - "High-Confidence Suspicious" + "Escalate" → **malicious**
   - "Anomalous but Benign" + "Close" → **benign**
3. Compare with expected category from dataset label

## Files Modified

1. **training/evaluation.py**
   - Updated `normalize_label()` to parse structured format
   - Handles both legacy and new label formats

2. **training/classification_evaluator.py**
   - Enhanced `_generate_output()` to extract only model response
   - Three-method extraction to handle various output formats

3. **scripts/test_label_extraction.py**
   - Added tests for structured label format
   - Validates both legacy and new formats

## Next Steps

To verify the fixes work:

1. **Test label parsing (fast):**
   ```bash
   python scripts/test_label_extraction.py
   ```
   Should show all ✓ for both legacy and structured formats

2. **Test with model (slower):**
   ```bash
   python scripts/test_classification_eval.py
   ```
   Should show:
   - `expected_normalized`: "benign" or "malicious" (not "unknown")
   - `classification`: Extracted JSON fields
   - `generated_text`: Model response only (not full prompt)
   - `hit`: 1 if match, 0 if mismatch

3. **Run full training:**
   ```bash
   make train-local
   ```
   Should evaluate classification accuracy before and after training with correct label matching.
