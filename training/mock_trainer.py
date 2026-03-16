"""
Mock trainer implementation for testing and validation.
Uses mock functions instead of actual model training.
Perfect for CI/CD, testing, and development.
"""
import time
import random
from typing import Any, Dict
from pathlib import Path

from .base_trainer import BaseTrainer
from .config import MockTrainingConfig


class MockModel:
    """Mock model class."""
    def __init__(self, name: str):
        self.name = name
        self.config = type('Config', (), {'pad_token_id': 0, 'eos_token_id': 1})()

    def num_parameters(self):
        return 1_000_000  # Mock 1M parameters

    def get_nb_trainable_parameters(self):
        return 100_000  # Mock 100K trainable

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "mock_model.txt").write_text(f"Mock model: {self.name}")


class MockTokenizer:
    """Mock tokenizer class."""
    def __init__(self, name: str):
        self.name = name
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "mock_tokenizer.txt").write_text(f"Mock tokenizer: {self.name}")


class MockTrainer:
    """Mock trainer that simulates training."""
    def __init__(self, model, args, train_dataset, eval_dataset=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.current_epoch = 0

    def train(self):
        """Simulate training."""
        print(f"   Starting mock training for {self.args.num_train_epochs} epoch(s)...")

        for epoch in range(int(self.args.num_train_epochs)):
            self.current_epoch = epoch + 1
            num_steps = len(self.train_dataset) // self.args.per_device_train_batch_size

            print(f"   Epoch {self.current_epoch}/{self.args.num_train_epochs}")

            for step in range(num_steps):
                if hasattr(self.args, 'training_delay_per_step'):
                    time.sleep(self.args.training_delay_per_step)

                # Simulate decreasing loss
                loss = 2.0 - (step / num_steps) * 0.5 - (epoch * 0.2)

                if step % 10 == 0:
                    print(f"      Step {step}/{num_steps} - Loss: {loss:.4f}")

            # Save checkpoint
            if hasattr(self.args, 'checkpoint_dir'):
                checkpoint_dir = Path(self.args.checkpoint_dir) / f"checkpoint-{self.current_epoch}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                print(f"   ✓ Saved checkpoint: {checkpoint_dir}")

        print(f"   ✓ Mock training complete!")

    def evaluate(self, eval_dataset=None):
        """Simulate evaluation."""
        dataset = eval_dataset or self.eval_dataset

        if dataset is None:
            return {}

        # Simulate evaluation metrics
        # Loss decreases over training
        base_loss = 1.5 - (self.current_epoch * 0.2)
        loss = base_loss + random.uniform(-0.1, 0.1)

        return {
            'eval_loss': loss,
            'eval_runtime': random.uniform(1.0, 3.0),
            'eval_samples_per_second': len(dataset) / random.uniform(1.0, 3.0),
        }


class MockTrainerImplementation(BaseTrainer):
    """
    Mock trainer for testing and validation.
    Simulates training without actual model loading or GPU usage.
    """

    def __init__(self, config: MockTrainingConfig):
        super().__init__(config)
        self.config: MockTrainingConfig = config  # Type hint for IDE

    def load_model(self) -> None:
        """Load mock model (no actual model loading)."""
        print(f"\nLoading mock model: {self.config.model_name}")
        print(f"   This is a mock - no actual model loaded")

        # Create mock model and tokenizer
        self.model = MockModel(self.config.model_name)
        self.tokenizer = MockTokenizer(self.config.model_name)

        print(f"   ✓ Mock model created: {self.model.num_parameters():,} parameters")
        print(f"   ✓ Mock trainable parameters: {self.model.get_nb_trainable_parameters():,}")

    # load_datasets() inherited from BaseTrainer - no duplication!

    def create_trainer(self, train_dataset: Any, eval_dataset: Any) -> None:
        """Create mock trainer."""
        print(f"\nCreating mock trainer...")

        # Create mock training arguments
        class MockArgs:
            def __init__(self, config):
                self.output_dir = config.checkpoint_dir
                self.per_device_train_batch_size = config.batch_size
                self.per_device_eval_batch_size = config.batch_size
                self.gradient_accumulation_steps = config.gradient_accumulation_steps
                self.num_train_epochs = config.num_epochs
                self.learning_rate = config.learning_rate
                self.checkpoint_dir = config.checkpoint_dir
                self.training_delay_per_step = config.training_delay_per_step if config.simulate_training_time else 0

        mock_args = MockArgs(self.config)

        # Create mock trainer
        self.trainer = MockTrainer(
            model=self.model,
            args=mock_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        print(f"   ✓ Mock trainer created")
        print(f"   ✓ Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"   ✓ Training delay: {self.config.training_delay_per_step if self.config.simulate_training_time else 0}s per step")

    def evaluate_classification_accuracy(self, dataset: Any, max_examples: int = None) -> Dict[str, Any]:
        """
        Mock classification accuracy evaluation.
        Returns simulated metrics instead of actually generating outputs.

        Args:
            dataset: Raw dataset (not used in mock)

        Returns:
            Dict with simulated classification metrics
        """
        print(f"   Simulating classification accuracy for {len(dataset)} examples...")

        # Simulate accuracy (random but reasonable)
        total = len(dataset)
        hits = int(total * random.uniform(0.6, 0.85))  # 60-85% accuracy
        accuracy = hits / total if total > 0 else 0.0

        # Mock category stats
        by_category = {
            "benign": {
                "accuracy": random.uniform(0.5, 0.9),
                "hits": random.randint(5, 10),
                "total": random.randint(10, 15),
            },
            "malicious": {
                "accuracy": random.uniform(0.5, 0.9),
                "hits": random.randint(5, 10),
                "total": random.randint(10, 15),
            },
        }

        # Mock sample results
        sample_misses = [
            {
                "index": i,
                "expected_label": "True Positive - Malicious",
                "expected_normalized": "malicious",
                "predicted": "benign",
                "hit": 0,
                "generated_text": "Mock generated text (miss)...",
            }
            for i in range(min(3, total))
        ]

        sample_hits = [
            {
                "index": i,
                "expected_label": "True Positive - Benign",
                "expected_normalized": "benign",
                "predicted": "benign",
                "hit": 1,
                "generated_text": "Mock generated text (hit)...",
            }
            for i in range(min(3, total))
        ]

        print(f"   ✓ Mock accuracy: {accuracy:.2%}")

        return {
            "accuracy": accuracy,
            "hits": hits,
            "total": total,
            "by_category": by_category,
            "sample_misses": sample_misses,
            "sample_hits": sample_hits,
        }
