# Training Architecture

## Overview

This training system follows **SOLID principles** with a clean interface-based architecture that separates local and cloud training implementations.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    BaseTrainer (ABC)                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Abstract Interface:                                  │  │
│  │  - load_model()                                       │  │
│  │  - load_datasets()                                    │  │
│  │  - create_trainer()                                   │  │
│  │                                                        │  │
│  │  Template Method:                                     │  │
│  │  - run_full_training()   (orchestrates everything)   │  │
│  │  - evaluate_before_training()                         │  │
│  │  - train()                                            │  │
│  │  - evaluate_after_training()                          │  │
│  │  - save_model()                                       │  │
│  │  - save_results()                                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ (inherits)
          ┌─────────────────┴─────────────────┐
          │                                   │
┌─────────┴─────────┐             ┌──────────┴──────────┐
│  LocalTrainer     │             │  CloudTrainer       │
├───────────────────┤             ├─────────────────────┤
│ - 0.5B model      │             │ - 3B model          │
│ - 512 tokens      │             │ - 32K tokens        │
│ - CPU/MPS         │             │ - GPU + Unsloth     │
│ - Standard libs   │             │ - Flash Attention   │
│ - Mac optimized   │             │ - 4-bit quant       │
└───────────────────┘             └─────────────────────┘
          ▲                                   ▲
          │                                   │
┌─────────┴─────────┐             ┌──────────┴──────────┐
│ local_entry.py    │             │ cloud_entry.py      │
│ (minimal code)    │             │ (minimal code)      │
└───────────────────┘             └─────────────────────┘
```

## SOLID Principles Applied

### 1. Single Responsibility Principle (SRP)
Each class has one clear responsibility:
- **config.py**: Configuration management
- **base_trainer.py**: Training interface and template
- **local_trainer.py**: Local training implementation
- **cloud_trainer.py**: Cloud training implementation
- **local_entry.py**: Local bootstrap
- **cloud_entry.py**: Cloud bootstrap

### 2. Open/Closed Principle (OCP)
- `BaseTrainer` is **closed for modification** but **open for extension**
- New training environments can be added by extending `BaseTrainer`
- No need to modify existing code

Example:
```python
# Want to add Azure training? Just extend BaseTrainer:
class AzureTrainer(BaseTrainer):
    def load_model(self):
        # Azure-specific model loading
        pass

    def load_datasets(self):
        # Azure-specific data loading
        pass

    def create_trainer(self):
        # Azure-specific trainer creation
        pass
```

### 3. Liskov Substitution Principle (LSP)
Any `BaseTrainer` implementation can be substituted without breaking code:

```python
def train_model(trainer: BaseTrainer):
    trainer.run_full_training()  # Works with ANY implementation

# Both work identically from caller's perspective
train_model(LocalTrainer(LocalTrainingConfig()))
train_model(CloudTrainer(CloudTrainingConfig()))
```

### 4. Interface Segregation Principle (ISP)
- Clean, focused interface in `BaseTrainer`
- Implementations only override what they need
- No fat interfaces with unused methods

### 5. Dependency Inversion Principle (DIP)
- High-level code depends on `BaseTrainer` abstraction, not concrete implementations
- Entry points depend on interfaces, not implementations
- Configuration injected via dependency injection

```python
# Depend on abstraction (BaseTrainer), not concrete class
def run_training(trainer: BaseTrainer):
    trainer.run_full_training()
```

## File Structure

```
training/
├── __init__.py              # Module exports
├── config.py                # Configuration classes
├── base_trainer.py          # Abstract base class (interface)
├── local_trainer.py         # Local implementation
├── cloud_trainer.py         # Cloud implementation
├── local_entry.py           # Local entry point (minimal)
├── cloud_entry.py           # Cloud entry point (minimal)
└── ARCHITECTURE.md          # This file
```

## Usage

### Local Training (Mac)

```bash
# Simple - just run the entry point
python training/local_entry.py
```

**What happens:**
1. Loads 0.5B model (fast, fits in Mac RAM)
2. Uses CPU or MPS (Metal Performance Shaders)
3. Trains with 512 token context
4. Evaluates before/after/test
5. Saves model to `outputs/local_model/`

### Cloud Training (GPU)

```bash
# Simple - just run the entry point
python training/cloud_entry.py
```

**What happens:**
1. Loads 3B model with Unsloth optimizations
2. Uses GPU with 4-bit quantization
3. Trains with 32K token context
4. Flash Attention 2 for speed
5. Evaluates before/after/test
6. Saves model to `outputs/distilled_model/`

### Custom Configuration

```python
from training import CloudTrainer, CloudTrainingConfig

# Customize config
config = CloudTrainingConfig(
    model_name="unsloth/Qwen2.5-7B-Instruct",  # Bigger model
    max_seq_length=16000,                       # Different context
    num_epochs=2,                               # More epochs
    batch_size=4,                               # Bigger batch
)

# Run with custom config
trainer = CloudTrainer(config)
trainer.run_full_training()
```

## Benefits of This Architecture

### 1. Maintainability
- Clear separation of concerns
- Easy to understand and modify
- Each file has single responsibility

### 2. Extensibility
- Add new training environments without modifying existing code
- Just extend `BaseTrainer` and implement 3 methods

### 3. Testability
- Each component can be tested independently
- Mock implementations for testing
- Clear interfaces make testing easier

### 4. Consistency
- All implementations follow same interface
- Consistent behavior across environments
- Template method ensures consistent workflow

### 5. Flexibility
- Easy to swap implementations
- Configuration-driven behavior
- Dependency injection for customization

## Adding a New Training Environment

Want to add AWS SageMaker training? Here's how:

```python
# 1. Create config
@dataclass
class SageMakerTrainingConfig(TrainingConfig):
    model_name: str = "unsloth/Qwen2.5-3B-Instruct"
    s3_bucket: str = "my-bucket"
    s3_data_path: str = "data/"
    # ... other AWS-specific settings

# 2. Create trainer
class SageMakerTrainer(BaseTrainer):
    def load_model(self):
        # AWS SageMaker model loading
        pass

    def load_datasets(self):
        # Load from S3
        pass

    def create_trainer(self):
        # SageMaker-specific trainer
        pass

# 3. Create entry point
# sagemaker_entry.py
def main():
    config = SageMakerTrainingConfig()
    trainer = SageMakerTrainer(config)
    trainer.run_full_training()
```

That's it! No modifications to existing code needed.

## Template Method Pattern

`BaseTrainer.run_full_training()` uses the **Template Method** pattern:

```python
def run_full_training(self):
    # Template - defines the algorithm structure
    self.load_model()           # Hook - implemented by subclass
    datasets = self.load_datasets()  # Hook - implemented by subclass
    self.create_trainer(datasets['train'], datasets['val'])  # Hook

    pre_results = self.evaluate_before_training()  # Concrete method
    self.train()                                    # Concrete method
    post_results = self.evaluate_after_training()  # Concrete method
    test_results = self.evaluate_test()            # Concrete method

    self.save_model()           # Concrete method
    self.save_results(...)      # Concrete method
```

**Benefits:**
- Algorithm structure defined once
- Subclasses implement specific steps
- Consistent workflow across all implementations

## Comparison: Old vs New

### Old Approach
```
train.py          (3B model, monolithic)
train_mac.py      (0.5B model, duplicated code)
cloud_train.py    (wrapper, tight coupling)
```

**Problems:**
- Code duplication
- Hard to test
- Tight coupling
- No clear interface
- Difficult to extend

### New Approach
```
base_trainer.py    (interface + template)
  ├── local_trainer.py   (local impl)
  └── cloud_trainer.py   (cloud impl)

local_entry.py     (minimal bootstrap)
cloud_entry.py     (minimal bootstrap)
```

**Benefits:**
- ✅ No duplication
- ✅ Easy to test
- ✅ Loose coupling
- ✅ Clear interface
- ✅ Easy to extend

## Design Patterns Used

1. **Template Method**: `run_full_training()` orchestrates workflow
2. **Strategy Pattern**: Different implementations for different environments
3. **Dependency Injection**: Configuration injected via constructor
4. **Factory Pattern** (implicit): Entry points act as simple factories

## Testing Example

```python
# Easy to test with clear interfaces
def test_local_trainer():
    config = LocalTrainingConfig(num_epochs=1)
    trainer = LocalTrainer(config)

    # Test individual methods
    trainer.load_model()
    assert trainer.model is not None

    datasets = trainer.load_datasets()
    assert 'train' in datasets

    # Or test full pipeline
    trainer.run_full_training()
    assert Path(config.output_dir).exists()
```

## Summary

This architecture provides:
- ✅ **Clean separation** of concerns
- ✅ **SOLID principles** throughout
- ✅ **Easy extension** without modification
- ✅ **Minimal entry points** for simplicity
- ✅ **Consistent interface** across environments
- ✅ **Testable** components
- ✅ **Production-ready** design

The same approach can train locally or on cloud with just:
```python
python training/local_entry.py   # Local
python training/cloud_entry.py   # Cloud
```

**No complex configuration, no duplicate code, just clean architecture.**
