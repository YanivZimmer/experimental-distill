"""
Convert langfuse dataset to training format for step-by-step distillation.
"""
import json
from pathlib import Path

def load_prompt_template(path="baseline.txt"):
    """Load the SOC analyst prompt template."""
    return Path(path).read_text()

def format_training_example(item, prompt_template):
    """
    Convert a single langfuse item to training format.

    Expected item structure:
    {
        "input": {...},  # Raw alert data
        "reasoning": "...",  # Frontier model's step-by-step analysis
        "classification": "..."  # Final verdict
    }
    """
    # Format the alert data as JSON string
    alert_json = json.dumps(item["input"], indent=2)

    # Construct the full prompt (instruction + alert)
    instruction = f"{prompt_template}\n{alert_json}\n```"

    # Combine reasoning + classification as the response
    response = f"{item['reasoning']}\n\nFinal Classification: {item['classification']}"

    return {
        "instruction": instruction,
        "output": response
    }

def prepare_dataset(input_path, output_path, prompt_template_path="baseline.txt"):
    """Main conversion function."""

    # Load data
    print(f"Loading dataset from {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    prompt_template = load_prompt_template(prompt_template_path)

    # Convert each item
    print(f"Converting {len(data)} examples...")
    training_data = [
        format_training_example(item, prompt_template)
        for item in data
        if "reasoning" in item and "classification" in item  # Filter items with teacher outputs
    ]

    print(f"Prepared {len(training_data)} training examples")

    # Save
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"Saved to {output_path}")
    return training_data

if __name__ == "__main__":
    prepare_dataset(
        input_path="data/langfuse_test.json",
        output_path="data/train_distill.json"
    )
