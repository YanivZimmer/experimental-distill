"""
Training module for distillation.
Provides interface-based architecture with local, cloud, and mock implementations.
"""
from .base_trainer import BaseTrainer
from .local_trainer import LocalTrainer
from .config import TrainingConfig, LocalTrainingConfig, CloudTrainingConfig, MockTrainingConfig

# Conditionally import CloudTrainer (requires unsloth)
try:
    from .cloud_trainer import CloudTrainer
    _cloud_available = True
except ImportError:
    CloudTrainer = None
    _cloud_available = False

# Import MockTrainer
from .mock_trainer import MockTrainer

__all__ = [
    'BaseTrainer',
    'LocalTrainer',
    'CloudTrainer',
    'MockTrainer',
    'TrainingConfig',
    'LocalTrainingConfig',
    'CloudTrainingConfig',
    'MockTrainingConfig',
]

def is_cloud_available():
    """Check if cloud training is available (unsloth installed)."""
    return _cloud_available
