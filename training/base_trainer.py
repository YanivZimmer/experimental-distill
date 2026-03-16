"""
Abstract base class for training.
Follows Interface Segregation and Dependency Inversion principles.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import json

from datasets import load_dataset

from .config import TrainingConfig


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    Defines the contract that all implementations must follow.

    Concrete methods (shared by all subclasses):
    - load_datasets() - Common dataset loading logic
    - evaluate_before_training() - Pre-training evaluation
    - train() - Execute training
    - evaluate_after_training() - Post-training evaluation
    - save_model() - Save trained model
    - save_results() - Save evaluation metrics
    - run_full_training() - Complete training pipeline (Template Method)

    Abstract methods (must be implemented by subclasses):
    - load_model() - Environment-specific model loading
    - create_trainer() - Environment-specific trainer creation
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration (LocalTrainingConfig or CloudTrainingConfig)
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load and configure the model for training.
        Must be implemented by subclasses for environment-specific loading.
        """
        pass

    @abstractmethod
    def create_trainer(self, train_dataset: Any, eval_dataset: Any) -> None:
        """
        Create the trainer object with datasets.
        Must be implemented by subclasses for environment-specific trainers.

        Args:
            train_dataset: Training dataset
            eval_dataset: Validation dataset
        """
        pass

    def load_datasets(self) -> Dict[str, Any]:
        """
        Load and prepare datasets.
        This is a concrete method - shared by all implementations.

        Returns:
            Dict with 'train', 'val', 'test' datasets
        """
        print(f"\nLoading datasets...")

        # Load prompt template
        prompt_template = Path(self.config.prompt_template_path).read_text()
        print(f"   ✓ Loaded {len(prompt_template)} chars from {self.config.prompt_template_path}")

        # Create format function
        def format_prompt(example):
            """Format example for training."""
            # Alert is already a JSON string (stored as text to avoid schema conflicts)
            alert_json = example['alert']
            instruction = f"{prompt_template}\n{alert_json}\n```"
            output = f"{example['reasoning']}\n\n**Final Classification:** {example['classification']}"

            return {
                "text": f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
            }

        # Load and format datasets
        dataset = load_dataset("json", data_files={
            "train": self.config.train_data_path,
            "validation": self.config.val_data_path,
            "test": self.config.test_data_path
        })

        train_dataset = dataset["train"].map(format_prompt, remove_columns=dataset["train"].column_names)
        val_dataset = dataset["validation"].map(format_prompt, remove_columns=dataset["validation"].column_names)
        test_dataset = dataset["test"].map(format_prompt, remove_columns=dataset["test"].column_names)

        print(f"   ✓ Train: {len(train_dataset)} examples")
        print(f"   ✓ Val: {len(val_dataset)} examples")
        print(f"   ✓ Test: {len(test_dataset)} examples")

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

    def evaluate_before_training(self) -> Dict[str, float]:
        """
        Evaluate model before training.

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call create_trainer() first.")

        print("\n" + "="*60)
        print("EVALUATION BEFORE TRAINING")
        print("="*60)

        results = self.trainer.evaluate()

        if 'eval_loss' in results:
            print(f"Pre-training validation loss: {results['eval_loss']:.4f}")
        print(f"Full results: {results}")

        return results

    def train(self) -> None:
        """Execute training."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call create_trainer() first.")

        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)

        self.trainer.train()

    def evaluate_after_training(self) -> Dict[str, float]:
        """
        Evaluate model after training.

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call create_trainer() first.")

        print("\n" + "="*60)
        print("EVALUATION AFTER TRAINING")
        print("="*60)

        results = self.trainer.evaluate()

        if 'eval_loss' in results:
            print(f"Post-training validation loss: {results['eval_loss']:.4f}")
        print(f"Full results: {results}")

        return results

    def save_model(self) -> None:
        """Save the trained model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        print(f"Model saved to {self.config.output_dir}")

    def save_results(self, pre_results: Dict[str, float],
                    post_results: Dict[str, float],
                    test_results: Optional[Dict[str, float]] = None) -> None:
        """
        Save evaluation results to JSON.

        Args:
            pre_results: Pre-training evaluation metrics
            post_results: Post-training evaluation metrics
            test_results: Test set evaluation metrics (optional)
        """
        results_path = Path(self.config.output_dir) / "evaluation_results.json"

        results_data = {
            "pre_training": pre_results,
            "post_training": post_results,
        }

        if test_results:
            results_data["test"] = test_results

        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"Evaluation results saved to {results_path}")

        # Print improvement
        if 'eval_loss' in pre_results and 'eval_loss' in post_results:
            improvement = pre_results['eval_loss'] - post_results['eval_loss']
            print(f"\nValidation loss improvement: {improvement:.4f}")

    def run_full_training(self) -> None:
        """
        Template method - runs the complete training pipeline.
        This is the main public interface.
        """
        print("\n" + "="*60)
        print(f"DISTILLATION TRAINING - {self.__class__.__name__}")
        print(f"Model: {self.config.model_name}")
        print(f"Max sequence length: {self.config.max_seq_length}")
        print(f"Epochs: {self.config.num_epochs}")
        print("="*60)

        # Load model
        self.load_model()

        # Load datasets
        datasets = self.load_datasets()

        # Create trainer
        self.create_trainer(datasets['train'], datasets['val'])

        # Evaluate before training
        pre_results = self.evaluate_before_training()

        # Train
        self.train()

        # Evaluate after training
        post_results = self.evaluate_after_training()

        # Evaluate on test set if available
        test_results = None
        if datasets.get('val') is not None:
            print("\n" + "="*60)
            print("EVALUATING ON VALIDATION SET")
            print("="*60)

            # Use evaluate_test_set if available (handles tokenization)
            if hasattr(self, 'evaluate_test_set'):
                test_results = self.evaluate_test_set(datasets['val'])
            else:
                # Fallback for trainers without evaluate_test_set method
                test_results = self.trainer.evaluate(datasets['val'])

            print(f"Test loss: {test_results.get('eval_loss', 'N/A'):.4f}")
            print(f"Full test results: {test_results}")

        # Save model
        self.save_model()

        # Save results
        self.save_results(pre_results, post_results, test_results)

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
