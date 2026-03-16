"""
Cloud training entry point.
Minimal bootstrap code for running training on cloud GPU.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.cloud_trainer import CloudTrainer
from training.config import CloudTrainingConfig


def main():
    """Main entry point for cloud training."""

    # Create config (can be customized via command line args if needed)
    config = CloudTrainingConfig()

    print("="*60)
    print("CLOUD GPU TRAINING")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Max sequence length: {config.max_seq_length:,}")
    print(f"4-bit quantization: {config.load_in_4bit}")
    print(f"Unsloth optimizations: {config.use_unsloth}")
    print("="*60)

    # Create trainer
    trainer = CloudTrainer(config)

    # Run training
    trainer.run_full_training()


if __name__ == "__main__":
    main()
