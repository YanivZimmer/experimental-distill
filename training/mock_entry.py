"""
Mock training entry point.
Minimal bootstrap code for testing/validation without actual training.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.mock_trainer import MockTrainerImplementation
from training.config import MockTrainingConfig


def main():
    """Main entry point for mock training."""

    # Create config
    config = MockTrainingConfig()

    print("="*60)
    print("MOCK TRAINING (TESTING/VALIDATION)")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Simulate timing: {config.simulate_training_time}")
    print(f"This is a MOCK - no actual training occurs")
    print("="*60)

    # Create trainer
    trainer = MockTrainerImplementation(config)

    # Run training
    trainer.run_full_training()


if __name__ == "__main__":
    main()
