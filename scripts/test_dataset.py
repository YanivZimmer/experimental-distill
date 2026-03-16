"""
Test suite to validate dataset quality before training.
Run this BEFORE wasting compute resources.
"""
import json
import sys
from pathlib import Path
from collections import Counter

class DatasetValidator:
    """Comprehensive dataset validation."""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.errors = []
        self.warnings = []

    def load_data(self):
        """Load and validate JSON format."""
        print("=" * 60)
        print("TEST 1: Loading dataset...")
        print("=" * 60)

        try:
            with open(self.dataset_path) as f:
                self.data = json.load(f)
            print(f"✅ Successfully loaded {len(self.data)} examples")
            return True
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {e}")
            print(f"❌ JSON parsing error: {e}")
            return False
        except FileNotFoundError:
            self.errors.append(f"File not found: {self.dataset_path}")
            print(f"❌ File not found: {self.dataset_path}")
            return False

    def validate_schema(self):
        """Validate each example has required fields."""
        print("\n" + "=" * 60)
        print("TEST 2: Validating schema...")
        print("=" * 60)

        required_fields = ["alert", "reasoning", "classification"]

        for i, example in enumerate(self.data):
            missing = [field for field in required_fields if field not in example]
            if missing:
                self.errors.append(f"Example {i}: missing fields {missing}")

        if self.errors:
            print(f"❌ Schema validation failed: {len(self.errors)} errors")
            return False

        print(f"✅ All {len(self.data)} examples have required fields")
        return True

    def validate_alert_structure(self):
        """Validate alert data structure."""
        print("\n" + "=" * 60)
        print("TEST 3: Validating alert structure...")
        print("=" * 60)

        valid_count = 0
        for i, example in enumerate(self.data):
            alert = example.get("alert", {})

            # Check for raw_alert
            if "raw_alert" not in alert:
                self.warnings.append(f"Example {i}: no raw_alert field")
                continue

            # Check for basic fields
            raw_alert = alert["raw_alert"]
            if not raw_alert.get("name"):
                self.warnings.append(f"Example {i}: no alert name")

            valid_count += 1

        print(f"✅ {valid_count}/{len(self.data)} examples have valid alert structure")
        if self.warnings:
            print(f"⚠️  {len(self.warnings)} warnings")
        return True

    def validate_reasoning_quality(self):
        """Validate reasoning traces are non-empty and reasonable."""
        print("\n" + "=" * 60)
        print("TEST 4: Validating reasoning quality...")
        print("=" * 60)

        reasoning_lengths = []
        empty_count = 0
        short_count = 0

        for i, example in enumerate(self.data):
            reasoning = example.get("reasoning", "")
            length = len(reasoning)
            reasoning_lengths.append(length)

            if length == 0:
                self.errors.append(f"Example {i}: empty reasoning")
                empty_count += 1
            elif length < 50:
                self.warnings.append(f"Example {i}: very short reasoning ({length} chars)")
                short_count += 1

        if reasoning_lengths:
            avg_length = sum(reasoning_lengths) / len(reasoning_lengths)
            min_length = min(reasoning_lengths)
            max_length = max(reasoning_lengths)

            print(f"Reasoning statistics:")
            print(f"  - Average length: {avg_length:.0f} chars")
            print(f"  - Min length: {min_length} chars")
            print(f"  - Max length: {max_length} chars")

        if empty_count > 0:
            print(f"❌ {empty_count} examples with empty reasoning")
            return False

        if short_count > 0:
            print(f"⚠️  {short_count} examples with very short reasoning")

        print(f"✅ All examples have non-empty reasoning")
        return True

    def validate_classification_format(self):
        """Validate classification format."""
        print("\n" + "=" * 60)
        print("TEST 5: Validating classification format...")
        print("=" * 60)

        classifications = [ex.get("classification", "") for ex in self.data]
        empty_count = sum(1 for c in classifications if not c)

        if empty_count > 0:
            self.errors.append(f"{empty_count} examples with empty classification")
            print(f"❌ {empty_count} examples with empty classification")
            return False

        # Show classification distribution
        unique_classifications = Counter(classifications)
        print(f"Classification distribution:")
        for cls, count in unique_classifications.most_common():
            print(f"  - {cls[:50]}{'...' if len(cls) > 50 else ''}: {count}")

        print(f"✅ All examples have classification")
        return True

    def validate_metadata(self):
        """Validate metadata fields."""
        print("\n" + "=" * 60)
        print("TEST 6: Validating metadata...")
        print("=" * 60)

        has_metadata = sum(1 for ex in self.data if "metadata" in ex)
        has_item_id = sum(1 for ex in self.data if ex.get("metadata", {}).get("item_id"))

        print(f"Metadata presence:")
        print(f"  - Examples with metadata: {has_metadata}/{len(self.data)}")
        print(f"  - Examples with item_id: {has_item_id}/{len(self.data)}")

        if has_metadata == len(self.data):
            print(f"✅ All examples have metadata")
        else:
            print(f"⚠️  Some examples missing metadata")

        return True

    def validate_size(self):
        """Validate dataset is not too small or suspiciously large."""
        print("\n" + "=" * 60)
        print("TEST 7: Validating dataset size...")
        print("=" * 60)

        n = len(self.data)

        if n < 5:
            self.warnings.append(f"Very small dataset: {n} examples")
            print(f"⚠️  Dataset is very small ({n} examples)")
            print(f"   Recommendation: Need at least 50+ for meaningful training")
        elif n < 50:
            self.warnings.append(f"Small dataset: {n} examples")
            print(f"⚠️  Dataset is small ({n} examples)")
            print(f"   Recommendation: 100+ examples recommended")
        else:
            print(f"✅ Dataset size is adequate ({n} examples)")

        return True

    def check_duplicates(self):
        """Check for duplicate examples."""
        print("\n" + "=" * 60)
        print("TEST 8: Checking for duplicates...")
        print("=" * 60)

        # Check item_id duplicates
        item_ids = [ex.get("metadata", {}).get("item_id") for ex in self.data if ex.get("metadata", {}).get("item_id")]
        duplicate_ids = [id for id, count in Counter(item_ids).items() if count > 1]

        if duplicate_ids:
            self.warnings.append(f"{len(duplicate_ids)} duplicate item_ids")
            print(f"⚠️  Found {len(duplicate_ids)} duplicate item_ids")
            print(f"   First few: {duplicate_ids[:3]}")
        else:
            print(f"✅ No duplicate item_ids")

        return True

    def estimate_memory(self):
        """Estimate memory requirements."""
        print("\n" + "=" * 60)
        print("TEST 9: Estimating memory requirements...")
        print("=" * 60)

        # Load prompt template
        try:
            prompt_template = Path("baseline.txt").read_text()
            prompt_len = len(prompt_template)
        except:
            prompt_len = 7000  # Estimate

        # Calculate average example size
        avg_alert_size = sum(len(json.dumps(ex["alert"])) for ex in self.data) / len(self.data)
        avg_reasoning_size = sum(len(ex["reasoning"]) for ex in self.data) / len(self.data)
        avg_total_size = prompt_len + avg_alert_size + avg_reasoning_size

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        avg_tokens = avg_total_size / 4

        print(f"Average example size:")
        print(f"  - Prompt: {prompt_len:,} chars")
        print(f"  - Alert: {avg_alert_size:,.0f} chars")
        print(f"  - Reasoning: {avg_reasoning_size:,.0f} chars")
        print(f"  - Total: {avg_total_size:,.0f} chars (~{avg_tokens:,.0f} tokens)")

        # Check if fits in 4096 token limit
        if avg_tokens > 4096:
            self.warnings.append(f"Average example ({avg_tokens:.0f} tokens) exceeds 4096 token limit")
            print(f"⚠️  Average example may exceed max_seq_length (4096 tokens)")
            print(f"   Recommendation: Use --max-seq-length 8192 or truncate examples")
        else:
            print(f"✅ Average example fits within 4096 token limit")

        # Estimate training memory (very rough)
        # 32B model in 4-bit ≈ 8GB, batch_size=2, gradients+optimizer ≈ 3x
        estimated_vram = 8 + (avg_tokens / 1000 * 0.5)  # MB per example
        print(f"\nEstimated VRAM usage (rough):")
        print(f"  - Model (4-bit): ~8 GB")
        print(f"  - Per example: ~{estimated_vram:.1f} MB")
        print(f"  - Batch size 2: ~{8 + 2 * estimated_vram / 1000:.1f} GB")
        print(f"  - With gradients: ~{(8 + 2 * estimated_vram / 1000) * 1.5:.1f} GB")

        if (8 + 2 * estimated_vram / 1000) * 1.5 > 24:
            print(f"⚠️  May exceed 24GB VRAM limit")
            print(f"   Recommendation: Use batch_size=1 or smaller model")
        else:
            print(f"✅ Should fit in 24GB VRAM")

        return True

    def run_all_tests(self):
        """Run all validation tests."""
        print("\n" + "🔬" * 30)
        print("DATASET VALIDATION TEST SUITE")
        print("🔬" * 30 + "\n")

        tests = [
            self.load_data,
            self.validate_schema,
            self.validate_alert_structure,
            self.validate_reasoning_quality,
            self.validate_classification_format,
            self.validate_metadata,
            self.validate_size,
            self.check_duplicates,
            self.estimate_memory,
        ]

        # Reset error/warning counts
        self.errors = []
        self.warnings = []

        # Run tests
        passed = 0
        for test in tests:
            if test():
                passed += 1

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Tests passed: {passed}/{len(tests)}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors[:10]:  # Show first 10
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        print("\n" + "=" * 60)
        if len(self.errors) == 0:
            print("✅ ALL TESTS PASSED - Safe to proceed with training")
            print("=" * 60)
            return True
        else:
            print("❌ TESTS FAILED - Fix errors before training")
            print("=" * 60)
            return False


def test_splits(splits_dir="data/splits"):
    """Test train/val/test splits for leakage."""
    print("\n" + "🔬" * 30)
    print("SPLIT VALIDATION TEST SUITE")
    print("🔬" * 30 + "\n")

    splits_path = Path(splits_dir)

    if not splits_path.exists():
        print(f"⚠️  Splits directory not found: {splits_dir}")
        print(f"   Run: python split_dataset.py --strategy conservative")
        return False

    # Load splits
    try:
        with open(splits_path / "train.json") as f:
            train_data = json.load(f)
        with open(splits_path / "val.json") as f:
            val_data = json.load(f)
        with open(splits_path / "test.json") as f:
            test_data = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Missing split file: {e}")
        return False

    print(f"Loaded splits:")
    print(f"  - Train: {len(train_data)}")
    print(f"  - Val: {len(val_data)}")
    print(f"  - Test: {len(test_data)}")

    # Check for item_id overlap
    train_ids = {ex.get("metadata", {}).get("item_id") for ex in train_data}
    val_ids = {ex.get("metadata", {}).get("item_id") for ex in val_data}
    test_ids = {ex.get("metadata", {}).get("item_id") for ex in test_data}

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap:
        print(f"❌ {len(train_val_overlap)} items appear in both train and val")
        return False
    if train_test_overlap:
        print(f"❌ {len(train_test_overlap)} items appear in both train and test")
        return False
    if val_test_overlap:
        print(f"❌ {len(val_test_overlap)} items appear in both val and test")
        return False

    print(f"✅ No item_id overlap between splits")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate dataset before training")
    parser.add_argument("--dataset", default="data/train_distill.json", help="Dataset path")
    parser.add_argument("--test-splits", action="store_true", help="Also test splits")
    args = parser.parse_args()

    # Test main dataset
    validator = DatasetValidator(args.dataset)
    dataset_valid = validator.run_all_tests()

    # Test splits if requested
    splits_valid = True
    if args.test_splits:
        splits_valid = test_splits()

    # Exit code
    if dataset_valid and splits_valid:
        print("\n✅ All validation tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed - review errors above")
        sys.exit(1)
