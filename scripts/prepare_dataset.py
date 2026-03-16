"""
Convert langfuse + baseline_benchmark datasets to training format.
Stores only alert data (not full prompt) for efficiency.
"""
import json
from pathlib import Path

def load_datasets(langfuse_path, benchmark_path):
    """Load and index both datasets."""
    with open(langfuse_path) as f:
        langfuse_data = json.load(f)

    with open(benchmark_path) as f:
        benchmark_data = json.load(f)

    # Index langfuse by id
    langfuse_by_id = {item["id"]: item for item in langfuse_data}

    return langfuse_by_id, benchmark_data["items"]

def parse_teacher_output(benchmark_item, use_full_reasoning=False):
    """Extract reasoning and classification from teacher."""
    classification = benchmark_item["classification"]

    if use_full_reasoning:
        try:
            full_output = json.loads(benchmark_item["output"])
            reasoning_parts = [
                f"**Event Summary:**\n{full_output.get('event_summary', '')}",
                f"\n**Primary Assessment:**\n{full_output.get('primary_summary', '')}",
                f"\n**Supporting Evidence:**\n" + "\n".join(f"- {e}" for e in full_output.get('supporting_evidence', [])),
                f"\n**Alternative Hypotheses:**\n" + "\n".join(f"- {h}" for h in full_output.get('alternative_hypotheses', [])),
                f"\n**Justification:**\n{full_output.get('justification', '')}",
            ]
            reasoning = "\n".join(reasoning_parts)
        except:
            reasoning = classification["justification"]
    else:
        reasoning = classification["justification"]

    decision = classification.get("final_decision", "Unknown")
    severity = classification.get("severity", "Unknown")
    assessment = classification.get("primary_assessment", "Unknown")

    classification_text = f"{decision} (Assessment: {assessment}, Severity: {severity})"

    return reasoning, classification_text

def format_training_example(langfuse_item, benchmark_item, use_full_reasoning=False):
    """
    Convert to training format.
    Stores only alert data (not full prompt) for efficiency.
    """
    # Extract reasoning and classification
    reasoning, classification = parse_teacher_output(benchmark_item, use_full_reasoning)

    return {
        "alert": langfuse_item["input"],  # Just the alert JSON
        "reasoning": reasoning,
        "classification": classification,
        "metadata": {
            "item_id": benchmark_item["item_id"],
            "expected_label": benchmark_item.get("expected_label"),
            "teacher_model": ""
        }
    }

def prepare_dataset(
    langfuse_path="data/langfuse_test.json",
    benchmark_path="data/teacher_outputs.json",
    output_path="data/train_distill.json",
    use_full_reasoning=True,
    filter_by_agreement=True
):
    """Main conversion function."""

    print(f"Loading datasets...")
    langfuse_by_id, benchmark_items = load_datasets(langfuse_path, benchmark_path)

    print(f"Found {len(langfuse_by_id)} langfuse items, {len(benchmark_items)} benchmark items")

    # Join datasets
    training_data = []
    missing_ids = []
    filtered_disagreement = 0

    for benchmark_item in benchmark_items:
        item_id = benchmark_item["item_id"]

        if item_id not in langfuse_by_id:
            missing_ids.append(item_id)
            continue

        # Filter: only keep samples where Gemini agreed with ground truth
        if filter_by_agreement and benchmark_item.get("hit") != 1:
            filtered_disagreement += 1
            continue

        langfuse_item = langfuse_by_id[item_id]
        example = format_training_example(langfuse_item, benchmark_item, use_full_reasoning)
        training_data.append(example)

    print(f"Successfully joined {len(training_data)} examples")

    if filter_by_agreement:
        print(f"Filtered out {filtered_disagreement} examples where Gemini disagreed with ground truth")

    if missing_ids:
        print(f"Warning: {len(missing_ids)} benchmark items not found in langfuse data")

    # Save
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"Saved to {output_path}")

    # Show example
    if training_data:
        print("\n=== Example (alert stored, prompt loaded at training time) ===")
        print(f"Alert size: {len(json.dumps(training_data[0]['alert']))} chars")
        print(f"Reasoning size: {len(training_data[0]['reasoning'])} chars")
        print(f"Classification: {training_data[0]['classification']}")

    return training_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full-reasoning", action="store_true",
                       help="Use full structured output instead of just justification")
    parser.add_argument("--no-filter", action="store_true",
                       help="Include all samples (don't filter by Gemini agreement)")
    args = parser.parse_args()

    prepare_dataset(
        use_full_reasoning=args.full_reasoning,
        filter_by_agreement=not args.no_filter
    )
