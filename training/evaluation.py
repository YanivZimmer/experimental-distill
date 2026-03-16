"""
Classification evaluation for alert analysis.
Extracts and evaluates classifications from LLM JSON outputs.
"""
import json
import re
from typing import Dict, Any, Tuple


# Label normalization mappings
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


def normalize_label(label: str) -> str:
    """
    Normalize label to standard format.

    Handles two formats:
    1. Legacy: "True Positive - Malicious", "False Positive", etc.
    2. Structured: "Escalate Immediately (Assessment: Confirmed Malicious, Severity: High)"

    Args:
        label: Raw label string

    Returns:
        Normalized label: "malicious", "benign", "false_positive", or "unknown"
    """
    if not label:
        return "unknown"

    normalized = label.lower().strip()

    # Try legacy mapping first
    if normalized in LABEL_MAPPINGS:
        return LABEL_MAPPINGS[normalized]  # Keep all three categories distinct

    # Parse structured format: "Decision (Assessment: X, Severity: Y)"
    # Extract decision and assessment
    import re

    # Try to extract decision (Close, Escalate for Review, Escalate Immediately)
    decision_match = re.search(r'^(close|escalate for review|escalate immediately)', normalized)
    decision = decision_match.group(1) if decision_match else None

    # Try to extract assessment
    assessment_match = re.search(r'assessment:\s*([^,)]+)', normalized)
    assessment = assessment_match.group(1).strip() if assessment_match else None

    # Determine category based on decision + assessment
    # Three categories:
    # 1. True Positive - Malicious: Real threat
    # 2. True Positive - Benign: Real activity but authorized/legitimate
    # 3. False Positive: No actual security event (noise)

    if decision == "close":
        if assessment:
            # "Expected Noise" or "False Positive" → false_positive
            if "noise" in assessment or "false positive" in assessment:
                return "false_positive"
            # "Anomalous but Benign" → benign (real activity but safe)
            elif "benign" in assessment:
                return "benign"
        # Default close without clear assessment
        return "false_positive"

    elif decision in ["escalate for review", "escalate immediately"]:
        # Escalations are typically malicious threats
        if assessment:
            if "malicious" in assessment or "suspicious" in assessment:
                return "malicious"
            elif "policy violation" in assessment:
                # Policy violations are real issues (not FP), treated as malicious
                return "malicious"
            elif "benign" in assessment:
                # Edge case: escalating something benign (authorized but needs review)
                return "benign"
        return "malicious"  # Default for escalations

    return "unknown"


def extract_classification_from_output(output: str) -> Dict[str, Any]:
    """
    Extract classification information from model JSON output.

    Looks for key fields:
    - primary_assessment
    - final_decision
    - severity
    - justification

    Args:
        output: Model output string (may contain markdown, JSON, or text)

    Returns:
        Dict with extracted fields or empty dict if parsing fails
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


def is_benign_classification(classification: Dict[str, Any]) -> bool:
    """
    Determine if a classification indicates benign activity (True Positive - Benign).

    This means real activity that is authorized/legitimate, NOT false positives.

    Based on primary_assessment and final_decision fields.

    Args:
        classification: Extracted classification dict

    Returns:
        True if classification indicates benign activity (authorized/legitimate)
    """
    assessment = (classification.get("primary_assessment") or "").lower()
    decision = (classification.get("final_decision") or "").lower()

    # Benign = real activity but authorized
    benign_assessments = [
        "anomalous but benign",
    ]

    for ba in benign_assessments:
        if ba in assessment:
            return True

    return False


def is_false_positive_classification(classification: Dict[str, Any]) -> bool:
    """
    Determine if a classification indicates a false positive (no real security event).

    Based on primary_assessment and final_decision fields.

    Args:
        classification: Extracted classification dict

    Returns:
        True if classification indicates false positive/noise
    """
    assessment = (classification.get("primary_assessment") or "").lower()
    decision = (classification.get("final_decision") or "").lower()

    # False positive = no actual security event (expected noise)
    fp_assessments = [
        "expected noise",
        "false positive",
    ]

    for fp in fp_assessments:
        if fp in assessment:
            return True

    # Close with no clear benign justification might be FP
    if decision == "close" and not any(b in assessment for b in ["benign", "authorized", "legitimate"]):
        return True

    return False


def is_malicious_classification(classification: Dict[str, Any]) -> bool:
    """
    Determine if a classification indicates malicious activity.

    Based on primary_assessment and final_decision fields.

    Args:
        classification: Extracted classification dict

    Returns:
        True if classification indicates malicious activity
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


def evaluate_hit_miss(output: str, expected_label: str) -> Tuple[int, Dict[str, Any]]:
    """
    Evaluate if model output matches expected label.

    Three possible labels:
    1. malicious - True Positive - Malicious (real threat)
    2. benign - True Positive - Benign (real activity but authorized)
    3. false_positive - False Positive (no actual security event)

    Args:
        output: Model's JSON output string
        expected_label: Expected classification label

    Returns:
        Tuple of (hit: 0 or 1, details: dict with classification info)
    """
    # Normalize expected label (returns: malicious, benign, false_positive, or unknown)
    expected_normalized = normalize_label(expected_label)

    # Extract classification from output
    classification = extract_classification_from_output(output)

    if not classification:
        return 0, {
            "error": "failed to parse output",
            "expected": expected_normalized,
            "predicted": "unknown",
            "classification": {},
        }

    # Determine predicted category (check all three categories)
    if is_malicious_classification(classification):
        predicted_normalized = "malicious"
    elif is_benign_classification(classification):
        predicted_normalized = "benign"
    elif is_false_positive_classification(classification):
        predicted_normalized = "false_positive"
    else:
        predicted_normalized = "unknown"

    # Compare - exact match required across all three categories
    hit = 1 if predicted_normalized == expected_normalized else 0

    return hit, {
        "expected": expected_normalized,
        "predicted": predicted_normalized,
        "classification": classification,
    }
