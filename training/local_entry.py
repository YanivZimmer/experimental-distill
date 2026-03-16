"""
Local training entry point.
Minimal bootstrap code for running training on Mac.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.local_trainer import LocalTrainer
from training.config import LocalTrainingConfig


def main():
    """Main entry point for local training."""

    # Create config (can be customized via command line args if needed)
    config = LocalTrainingConfig()

    print("="*60)
    print("LOCAL TRAINING ON MAC")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Device: {'MPS' if config.use_mps else 'CPU'}")
    print("="*60)

    # Create trainer
    trainer = LocalTrainer(config)

    # Run training
    trainer.run_full_training()


if __name__ == "__main__":
    main()
