"""
Configuration classes for training.
Follows Single Responsibility Principle - each config for specific environment.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Base configuration for all training environments."""
    model_name: str
    max_seq_length: int
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    output_dir: str
    checkpoint_dir: str
    prompt_template_path: str
    use_splits: bool = True

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Data paths
    train_data_path: str = "data/splits/train.json"
    val_data_path: str = "data/splits/val.json"
    test_data_path: str = "data/splits/test.json"


@dataclass
class LocalTrainingConfig(TrainingConfig):
    """Configuration for local Mac training (tiny model for validation)."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 512
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    output_dir: str = "outputs/local_model"
    checkpoint_dir: str = "outputs/local_checkpoints"
    prompt_template_path: str = "prompts/baseline.txt"

    # Local-specific settings
    use_mps: bool = True  # Use Metal Performance Shaders on Mac
    use_cpu_only: bool = False


@dataclass
class CloudTrainingConfig(TrainingConfig):
    """Configuration for cloud GPU training (production model)."""
    model_name: str = "unsloth/Qwen2.5-3B-Instruct"
    max_seq_length: int = 32000
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    output_dir: str = "outputs/distilled_model"
    checkpoint_dir: str = "outputs/checkpoints"
    prompt_template_path: str = "prompts/baseline.txt"

    # Cloud-specific settings
    load_in_4bit: bool = True
    use_unsloth: bool = True
    use_flash_attention: bool = True

    # Optional GCS paths
    gcs_bucket: Optional[str] = None
    gcs_data_path: Optional[str] = None
    gcs_output_prefix: Optional[str] = None


@dataclass
class MockTrainingConfig(TrainingConfig):
    """Configuration for mock training (testing/validation only)."""
    model_name: str = "mock-model"
    max_seq_length: int = 128
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    output_dir: str = "outputs/mock_model"
    checkpoint_dir: str = "outputs/mock_checkpoints"
    prompt_template_path: str = "prompts/baseline.txt"

    # Mock-specific settings
    simulate_training_time: bool = True
    training_delay_per_step: float = 0.1  # seconds
