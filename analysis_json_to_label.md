# Classification Extraction from LLM Output

This document explains how the benchmark script (`scripts/05_benchmark.py`) extracts and evaluates the final classification from the LLM's JSON output.

## Overview

The classification extraction process consists of several steps:

1. **Generate Response**: The LLM produces a JSON response with structured fields
2. **Parse JSON**: Extract key classification fields from the JSON output
3. **Infer Category**: Use keyword matching to determine if the response indicates benign or malicious activity
4. **Compare**: Match the inferred category against the expected label
5. **Score**: Return a hit (1) or miss (0) based on the match

## Valid Expected Labels

The dataset uses three valid expected labels (from `data/langfuse_test.json`):

- **"True Positive - Malicious"**: Confirmed malicious activity
- **"True Positive - Benign"**: Legitimate/authorized activity
- **"False Positive"**: No actual security concern (treated as benign for matching)

## Step-by-Step Process

### Step 1: LLM Response Generation

In `scripts/05_benchmark.py`, the `run_benchmark()` function generates the LLM response:

```python
# Build prompt using template
prompt = build_prompt(prompt_template, input_data)

# Generate response
response = client.generate(prompt, max_output_tokens=max_output_tokens)
output = response["content"]
metadata = response["metadata"]
```

The `output` is a JSON string containing fields like:
- `severity`: "Low", "Medium", "High", or "Critical"
- `event_summary`: Brief description of the alert
- `primary_assessment`: Key field for classification (e.g., "High-Confidence Suspicious", "Anomalous but Benign")
- `final_decision`: Action to take (e.g., "Escalate for Review", "Close")
- `justification`: Reasoning for the decision
- Supporting fields like `event_timeline`, `suggested_actions`, etc.

### Step 2: Evaluate Hit-Miss

The output is passed to the evaluation function:

```python
from src.prompt_distill.evaluation import evaluate_hit_miss

# Evaluate hit-miss
hit, eval_details = evaluate_hit_miss(output, expected_label)

results.append({
    "item_id": item_id,
    "expected_label": expected_label,
    "expected_normalized": eval_details.get("expected"),
    "predicted": eval_details.get("predicted"),
    "hit": hit,
    "output": output,
    "classification": eval_details.get("classification", {}),
    "usage": metadata.get("usage", {}),
})
```

### Step 3: Extract Classification from JSON

The `evaluate_hit_miss()` function in `src/prompt_distill/evaluation.py` first extracts the classification:

```python
def evaluate_hit_miss(output: str, expected_label: str) -> tuple[int, dict]:
    """
    Evaluate if model output matches expected label.

    Args:
        output: Model's JSON output string
        expected_label: Expected classification label (e.g., "True Positive - Malicious")

    Returns:
        Tuple of (hit: 0 or 1, details: dict with classification info)
    """
    # Normalize expected label
    expected_normalized = normalize_label(expected_label)

    # Extract classification from output
    classification = extract_classification_from_output(output)

    if not classification:
        return 0, {"error": "failed to parse output", "expected": expected_normalized}

    # ... (continued below)
```

The `extract_classification_from_output()` function parses the JSON:

```python
def extract_classification_from_output(output: str) -> dict[str, Any]:
    """
    Extract classification information from model JSON output.

    Looks for key fields:
    - primary_assessment
    - final_decision
    - severity

    Returns dict with extracted fields or empty dict if parsing fails.
    """
    if not output:
        return {}

    try:
        # Try to extract JSON from the output
        # Handle markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', output)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {}

        data = json.loads(json_str)

        # Handle nested "properties" structure (some outputs wrap in properties)
        if "properties" in data and isinstance(data["properties"], dict):
            data = data["properties"]

        return {
            "primary_assessment": data.get("primary_assessment"),
            "final_decision": data.get("final_decision"),
            "severity": data.get("severity"),
            "justification": data.get("justification"),
        }
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}
```

### Step 4: Infer Benign vs Malicious

The extracted classification is then analyzed using keyword matching:

```python
def is_benign_classification(classification: dict) -> bool:
    """
    Determine if a classification indicates benign activity.

    Based on primary_assessment and final_decision fields.
    """
    assessment = (classification.get("primary_assessment") or "").lower()
    decision = (classification.get("final_decision") or "").lower()

    benign_assessments = [
        "anomalous but benign",
        "expected noise",
        "false positive",
    ]

    benign_decisions = [
        "close",
    ]

    for ba in benign_assessments:
        if ba in assessment:
            return True

    for bd in benign_decisions:
        if bd in decision:
            return True

    return False


def is_malicious_classification(classification: dict) -> bool:
    """
    Determine if a classification indicates malicious activity.

    Based on primary_assessment and final_decision fields.
    """
    assessment = (classification.get("primary_assessment") or "").lower()
    decision = (classification.get("final_decision") or "").lower()

    malicious_assessments = [
        "confirmed malicious",
        "high-confidence suspicious",
    ]

    malicious_decisions = [
        "escalate immediately",
        "escalate for review",
    ]

    for ma in malicious_assessments:
        if ma in assessment:
            return True

    for md in malicious_decisions:
        if md in decision:
            return True

    return False
```

### Step 5: Compare and Score

Finally, the inferred category is compared to the expected label:

```python
# (continuation of evaluate_hit_miss)

    # Determine predicted category
    if is_benign_classification(classification):
        predicted_normalized = "benign"
    elif is_malicious_classification(classification):
        predicted_normalized = "malicious"
    else:
        predicted_normalized = "unknown"

    # Compare - treat "false_positive" same as "benign" for matching
    # Since "False Positive" in model output maps to "benign" category
    expected_for_comparison = "benign" if expected_normalized == "false_positive" else expected_normalized
    hit = 1 if predicted_normalized == expected_for_comparison else 0

    return hit, {
        "expected": expected_normalized,
        "predicted": predicted_normalized,
        "classification": classification,
    }
```

## Label Normalization

The `normalize_label()` function handles various label formats:

```python
LABEL_MAPPINGS = {
    # Benign variants
    "true positive - benign": "benign",
    "true positive – benign": "benign",
    "true positive benign": "benign",
    "tp benign": "benign",
    "benign": "benign",
    "anomalous but benign": "benign",
    "expected noise": "benign",
    # Malicious variants
    "true positive - malicious": "malicious",
    "true positive – malicious": "malicious",
    "true positive malicious": "malicious",
    "tp malicious": "malicious",
    "malicious": "malicious",
    "confirmed malicious": "malicious",
    "high-confidence suspicious": "malicious",
    # False positive
    "false positive": "false_positive",
    "fp": "false_positive",
}
```

## Example Flow

Given this LLM output:

```json
{
  "severity": "Medium",
  "event_summary": "PowerShell execution with encoded command...",
  "primary_assessment": "High-Confidence Suspicious",
  "final_decision": "Escalate for Review",
  "justification": "The combination of obfuscated PowerShell and suspicious file name..."
}
```

And expected label: `"True Positive - Benign"`

**Extraction Process:**
1. ✅ Parse JSON successfully
2. ✅ Extract `primary_assessment: "High-Confidence Suspicious"`
3. ✅ Extract `final_decision: "Escalate for Review"`
4. ✅ Check benign keywords: No match
5. ✅ Check malicious keywords: "high-confidence suspicious" matches → **predicted = "malicious"**
6. ✅ Normalize expected: "True Positive - Benign" → **expected = "benign"**
7. ❌ Compare: "malicious" ≠ "benign" → **hit = 0**

**Result:**
```python
{
    "expected": "benign",
    "predicted": "malicious",
    "hit": 0
}
```

## Key Points

1. **JSON Parsing**: Handles both markdown code blocks (```json) and raw JSON objects
2. **Keyword Matching**: Uses predefined keywords in `primary_assessment` and `final_decision` fields
3. **False Positive Handling**: "False Positive" expected labels are treated as "benign" for matching
4. **Failure Handling**: Returns hit=0 if JSON parsing fails or keywords don't match any category
5. **Case Insensitive**: All keyword matching is case-insensitive
6. **Partial Matching**: Keywords can appear anywhere in the field (substring match)
