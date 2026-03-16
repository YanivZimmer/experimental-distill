"""
Test label extraction from JSON outputs.
This tests the keyword matching logic without requiring a model.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.evaluation import (
    extract_classification_from_output,
    evaluate_hit_miss,
    normalize_label,
    is_benign_classification,
    is_malicious_classification,
    is_false_positive_classification,
)


def test_label_normalization():
    """Test label normalization."""
    print("\n" + "="*60)
    print("TESTING LABEL NORMALIZATION")
    print("="*60)

    print("\nLegacy format (3 categories):")
    legacy_cases = [
        ("True Positive - Malicious", "malicious"),
        ("True Positive - Benign", "benign"),
        ("False Positive", "false_positive"),  # Keep as distinct category
        ("true positive - benign", "benign"),
        ("TP Malicious", "malicious"),
    ]

    for input_label, expected in legacy_cases:
        result = normalize_label(input_label)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_label}' → '{result}' (expected: '{expected}')")

    print("\nStructured format (from dataset):")
    structured_cases = [
        ("Close (Assessment: Anomalous but Benign, Severity: Low)", "benign"),
        ("Close (Assessment: Expected Noise, Severity: Informational)", "false_positive"),
        ("Escalate for Review (Assessment: High-Confidence Suspicious, Severity: Medium)", "malicious"),
        ("Escalate Immediately (Assessment: Confirmed Malicious, Severity: High)", "malicious"),
        ("Escalate Immediately (Assessment: Policy Violation, Severity: Medium)", "malicious"),
    ]

    for input_label, expected in structured_cases:
        result = normalize_label(input_label)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_label[:70]}...' → '{result}' (expected: '{expected}')")


def test_json_extraction():
    """Test JSON extraction from model outputs."""
    print("\n" + "="*60)
    print("TESTING JSON EXTRACTION")
    print("="*60)

    # Test case 1: Markdown JSON
    output1 = """Here's my analysis:

```json
{
  "primary_assessment": "High-Confidence Suspicious",
  "final_decision": "Escalate for Review",
  "severity": "Medium"
}
```
"""

    print("\nTest 1: Markdown JSON block (Malicious)")
    classification = extract_classification_from_output(output1)
    print(f"  Extracted: {classification}")
    print(f"  Is malicious: {is_malicious_classification(classification)}")
    print(f"  Is benign: {is_benign_classification(classification)}")
    print(f"  Is false positive: {is_false_positive_classification(classification)}")

    # Test case 2: Plain JSON (Benign)
    output2 = """{
  "primary_assessment": "Anomalous but Benign",
  "final_decision": "Close",
  "severity": "Low"
}"""

    print("\nTest 2: Plain JSON (Benign)")
    classification = extract_classification_from_output(output2)
    print(f"  Extracted: {classification}")
    print(f"  Is malicious: {is_malicious_classification(classification)}")
    print(f"  Is benign: {is_benign_classification(classification)}")
    print(f"  Is false positive: {is_false_positive_classification(classification)}")

    # Test case 3: JSON with extra text (Malicious)
    output3 = """Based on the analysis, here's my assessment:

{
  "primary_assessment": "Confirmed Malicious",
  "final_decision": "Escalate Immediately",
  "severity": "Critical",
  "justification": "This is clearly malicious behavior"
}

This requires immediate attention.
"""

    print("\nTest 3: JSON with surrounding text (Malicious)")
    classification = extract_classification_from_output(output3)
    print(f"  Extracted: {classification}")
    print(f"  Is malicious: {is_malicious_classification(classification)}")
    print(f"  Is benign: {is_benign_classification(classification)}")
    print(f"  Is false positive: {is_false_positive_classification(classification)}")

    # Test case 4: False Positive
    output4 = """{
  "primary_assessment": "Expected Noise",
  "final_decision": "Close",
  "severity": "Informational"
}"""

    print("\nTest 4: False Positive")
    classification = extract_classification_from_output(output4)
    print(f"  Extracted: {classification}")
    print(f"  Is malicious: {is_malicious_classification(classification)}")
    print(f"  Is benign: {is_benign_classification(classification)}")
    print(f"  Is false positive: {is_false_positive_classification(classification)}")


def test_hit_miss_evaluation():
    """Test complete hit-miss evaluation."""
    print("\n" + "="*60)
    print("TESTING HIT-MISS EVALUATION")
    print("="*60)

    test_cases = [
        {
            "output": """{
  "primary_assessment": "High-Confidence Suspicious",
  "final_decision": "Escalate for Review"
}""",
            "expected": "True Positive - Malicious",
            "should_hit": True,
            "description": "Malicious → Malicious (HIT)",
        },
        {
            "output": """{
  "primary_assessment": "Anomalous but Benign",
  "final_decision": "Close"
}""",
            "expected": "True Positive - Benign",
            "should_hit": True,
            "description": "Benign → Benign (HIT)",
        },
        {
            "output": """{
  "primary_assessment": "Expected Noise",
  "final_decision": "Close"
}""",
            "expected": "False Positive",
            "should_hit": True,
            "description": "False Positive → False Positive (HIT)",
        },
        {
            "output": """{
  "primary_assessment": "High-Confidence Suspicious",
  "final_decision": "Escalate for Review"
}""",
            "expected": "True Positive - Benign",
            "should_hit": False,
            "description": "Malicious → Benign (MISS)",
        },
        {
            "output": """{
  "primary_assessment": "Anomalous but Benign",
  "final_decision": "Close"
}""",
            "expected": "False Positive",
            "should_hit": False,
            "description": "Benign → False Positive (MISS)",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        hit, details = evaluate_hit_miss(test_case["output"], test_case["expected"])

        status = "✓" if (hit == 1) == test_case["should_hit"] else "✗"
        print(f"  {status} Expected: {test_case['expected']} → {details['expected']}")
        print(f"     Predicted: {details['predicted']}")
        print(f"     Hit: {hit} (should_hit: {test_case['should_hit']})")
        print(f"     Classification: {details['classification']}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("LABEL EXTRACTION AND CLASSIFICATION TESTS")
    print("="*70)

    test_label_normalization()
    test_json_extraction()
    test_hit_miss_evaluation()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
