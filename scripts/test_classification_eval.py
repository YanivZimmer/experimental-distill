"""
Test script for classification evaluation.
Tests label extraction on a single example before running full training.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from training.config import LocalTrainingConfig
from training.local_trainer import LocalTrainer


def main():
    print("\n" + "="*60)
    print("TESTING CLASSIFICATION EVALUATION")
    print("="*60)

    # Create config
    config = LocalTrainingConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        prompt_template_path="prompts/baseline.txt",  # Uses baseline prompt with JSON format instructions
        train_data_path="data/splits/train.json",
        val_data_path="data/splits/val.json",
        test_data_path="data/splits/test.json",
        output_dir="outputs/test_classification",
        checkpoint_dir="outputs/test_classification/checkpoints",
        max_seq_length=2048,  # Smaller for testing
        num_epochs=1,
        batch_size=1,
        use_cpu_only=True,  # Force CPU for testing
    )

    print("\n1. Loading model...")
    trainer = LocalTrainer(config)
    trainer.load_model()

    print("\n2. Loading validation dataset...")
    raw_val_dataset = load_dataset("json", data_files={"validation": config.val_data_path})["validation"]
    print(f"   Loaded {len(raw_val_dataset)} validation examples")

    # Test on first example
    print("\n3. Testing on first example...")
    example = raw_val_dataset[0]

    print(f"\n   Alert preview:")
    alert_text = example['alert']
    print(f"   {alert_text[:200]}...")

    print(f"\n   Expected label: {example['classification']}")

    # Create evaluator and test single example
    from training.classification_evaluator import ClassificationEvaluator

    evaluator = ClassificationEvaluator(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        prompt_template_path=config.prompt_template_path,
        max_seq_length=config.max_seq_length,
    )

    result = evaluator.test_single_example(example)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

    print(f"\nResult summary:")
    print(f"  Expected: {result['expected_label']} → {result['expected_normalized']}")
    print(f"  Predicted: {result['predicted']}")
    print(f"  Match: {'✓ YES' if result['hit'] else '✗ NO'}")

    print(f"\nExtracted classification:")
    for key, value in result['classification'].items():
        print(f"  {key}: {value}")

    print(f"\nFull generated text:")
    print("-" * 60)
    print(result['generated_text'])
    print("-" * 60)

    # Test on 3 examples
    print("\n\n" + "="*60)
    print("TESTING ON 3 EXAMPLES")
    print("="*60)

    results = evaluator.evaluate_dataset(raw_val_dataset, max_examples=3)

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Hits: {results['hits']}/{results['total']}")
    print(f"  By category: {results['by_category']}")

    print(f"\nSample results:")
    for i, r in enumerate(results['sample_hits'][:2]):
        print(f"\n  Hit {i+1}:")
        print(f"    Expected: {r['expected_normalized']}")
        print(f"    Predicted: {r['predicted']}")
        print(f"    Text: {r['generated_text'][:100]}...")

    for i, r in enumerate(results['sample_misses'][:2]):
        print(f"\n  Miss {i+1}:")
        print(f"    Expected: {r['expected_normalized']}")
        print(f"    Predicted: {r['predicted']}")
        print(f"    Text: {r['generated_text'][:100]}...")


if __name__ == "__main__":
    main()
