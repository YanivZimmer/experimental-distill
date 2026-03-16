"""
Classification accuracy evaluator.
Generates outputs and evaluates classification accuracy.
"""
import torch
from pathlib import Path
from typing import Any, Dict, Optional
from tqdm import tqdm

from .evaluation import evaluate_hit_miss


class ClassificationEvaluator:
    """
    Evaluates classification accuracy by generating outputs and comparing predictions.
    """

    def __init__(self, model, tokenizer, prompt_template_path: str, max_seq_length: int = 32000):
        """
        Initialize evaluator.

        Args:
            model: The language model
            tokenizer: The tokenizer
            prompt_template_path: Path to prompt template file
            max_seq_length: Maximum sequence length for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template_path = prompt_template_path
        self.max_seq_length = max_seq_length

        # Load prompt template
        self.prompt_template = Path(prompt_template_path).read_text()

    def evaluate_dataset(self, dataset: Any, max_examples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate classification accuracy on a dataset.

        Args:
            dataset: Dataset with 'alert', 'reasoning', 'classification' fields
            max_examples: Optional limit on number of examples to evaluate

        Returns:
            Dict with classification metrics
        """
        # Limit dataset size if requested
        if max_examples is not None and len(dataset) > max_examples:
            print(f"   Limiting evaluation to {max_examples} examples (out of {len(dataset)})")
            dataset = dataset.select(range(max_examples))

        print(f"   Generating outputs for {len(dataset)} examples...")

        # Set model to eval mode
        self.model.eval()

        # Track results
        results = []
        hits = 0
        total = len(dataset)

        # Track by category (three possible labels)
        category_stats = {
            "malicious": {"hits": 0, "total": 0},           # True Positive - Malicious
            "benign": {"hits": 0, "total": 0},              # True Positive - Benign
            "false_positive": {"hits": 0, "total": 0},      # False Positive
        }

        # Process each example
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                # Generate output
                generated_text = self._generate_output(example)

                # Evaluate hit-miss
                expected_label = example['classification']
                hit, eval_details = evaluate_hit_miss(generated_text, expected_label)

                # Update stats
                hits += hit
                expected_category = eval_details.get("expected", "unknown")
                if expected_category in category_stats:
                    category_stats[expected_category]["total"] += 1
                    category_stats[expected_category]["hits"] += hit

                # Store result
                result = {
                    "index": idx,
                    "expected_label": expected_label,
                    "expected_normalized": eval_details.get("expected"),
                    "predicted": eval_details.get("predicted"),
                    "hit": hit,
                    "generated_text": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
                    "classification": eval_details.get("classification", {}),
                }
                results.append(result)

            except Exception as e:
                print(f"   Warning: Failed to evaluate example {idx}: {e}")
                # Record as miss
                result = {
                    "index": idx,
                    "expected_label": example.get('classification', 'unknown'),
                    "expected_normalized": "unknown",
                    "predicted": "error",
                    "hit": 0,
                    "generated_text": f"Error: {str(e)}",
                    "classification": {},
                }
                results.append(result)

        # Calculate accuracy
        accuracy = hits / total if total > 0 else 0.0

        # Calculate per-category accuracy
        by_category = {}
        for category, stats in category_stats.items():
            if stats["total"] > 0:
                by_category[category] = {
                    "accuracy": stats["hits"] / stats["total"],
                    "hits": stats["hits"],
                    "total": stats["total"],
                }

        # Sample results (first 5 misses and first 5 hits)
        misses = [r for r in results if r["hit"] == 0][:5]
        hits_examples = [r for r in results if r["hit"] == 1][:5]

        return {
            "accuracy": accuracy,
            "hits": hits,
            "total": total,
            "by_category": by_category,
            "sample_misses": misses,
            "sample_hits": hits_examples,
        }

    def _generate_output(self, example: Dict[str, Any]) -> str:
        """
        Generate output for a single example.

        Args:
            example: Dataset example with 'alert' field

        Returns:
            Generated text (assistant's response only)
        """
        # Build prompt
        alert_json = example['alert']
        instruction = f"{self.prompt_template}\n{alert_json}\n```"
        prompt = f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response (after the prompt)
        # The model may include the full conversation, so we need to extract just the response

        # Method 1: Try to find the assistant marker
        if "<|im_start|>assistant" in generated_text:
            # Split on assistant marker and take the last part
            parts = generated_text.split("<|im_start|>assistant")
            if len(parts) > 1:
                generated_text = parts[-1].strip()
                # Remove end marker if present
                if "<|im_end|>" in generated_text:
                    generated_text = generated_text.split("<|im_end|>")[0].strip()

        # Method 2: If that didn't work, try to find where the prompt ends
        # The prompt ends with the event JSON, so look for the end of the JSON block
        elif "```" in generated_text:
            # Find the last ``` in the prompt (end of schema or event)
            parts = generated_text.split("```")
            if len(parts) > 2:
                # Take everything after the last ``` from the prompt
                # The response should start after that
                # Look for the next JSON block or content
                response_parts = parts[2:]  # Skip first two ``` blocks (schema and event)
                generated_text = "```".join(response_parts).strip()

        # Method 3: If the text starts with "user", it means we have the full prompt
        # Try to find where the actual response starts
        if generated_text.startswith("user") or "You are a SOC Tier" in generated_text[:200]:
            # The response likely starts after the event JSON
            # Look for the end of the JSON event block and take everything after
            lines = generated_text.split('\n')
            # Find where the response actually starts (after all the prompt instructions)
            response_start_idx = 0
            for i, line in enumerate(lines):
                # Look for the start of the actual analysis response
                # Usually starts with something like "Based on", "{", or a JSON block
                if i > 50 and (line.strip().startswith('{') or
                               'based on' in line.lower() or
                               'analysis' in line.lower()):
                    response_start_idx = i
                    break

            if response_start_idx > 0:
                generated_text = '\n'.join(lines[response_start_idx:]).strip()

        return generated_text

    def test_single_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test evaluation on a single example (for debugging).

        Args:
            example: Dataset example with 'alert' and 'classification' fields

        Returns:
            Dict with evaluation results
        """
        print("\n" + "="*60)
        print("TESTING SINGLE EXAMPLE")
        print("="*60)

        # Generate
        print("\n1. Generating output...")
        generated_text = self._generate_output(example)
        print(f"   Generated {len(generated_text)} characters")
        print(f"\n   Output preview:\n   {generated_text[:300]}...")

        # Evaluate
        print("\n2. Evaluating classification...")
        expected_label = example['classification']
        hit, eval_details = evaluate_hit_miss(generated_text, expected_label)

        print(f"   Expected: {expected_label}")
        print(f"   Expected (normalized): {eval_details.get('expected')}")
        print(f"   Predicted: {eval_details.get('predicted')}")
        print(f"   Hit: {'✓' if hit else '✗'}")
        print(f"   Classification extracted: {eval_details.get('classification')}")

        return {
            "expected_label": expected_label,
            "expected_normalized": eval_details.get("expected"),
            "predicted": eval_details.get("predicted"),
            "hit": hit,
            "generated_text": generated_text,
            "classification": eval_details.get("classification", {}),
        }
