"""
Local trainer implementation for Mac.
Uses standard transformers library (no Unsloth).
Optimized for CPU/MPS with tiny model.
"""
import torch
from typing import Any, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

from .base_trainer import BaseTrainer
from .config import LocalTrainingConfig
from .classification_evaluator import ClassificationEvaluator


class LocalTrainer(BaseTrainer):
    """
    Local trainer for Mac with standard transformers.
    Follows Open/Closed Principle - extends BaseTrainer without modifying it.
    """

    def __init__(self, config: LocalTrainingConfig):
        super().__init__(config)
        self.config: LocalTrainingConfig = config  # Type hint for IDE

    def load_model(self) -> None:
        """Load model with standard transformers (CPU/MPS compatible)."""
        print(f"\nLoading model: {self.config.model_name}")
        print(f"   Using: {'MPS' if self.config.use_mps and torch.backends.mps.is_available() else 'CPU'}")

        # Determine device
        if self.config.use_cpu_only:
            device = "cpu"
        elif self.config.use_mps and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Apply LoRA
        print(f"   Applying LoRA (r={self.config.lora_r}, alpha={self.config.lora_alpha})")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)

        print(f"   ✓ Model loaded with {self.model.num_parameters():,} parameters")
        trainable_params = self.model.get_nb_trainable_parameters()
        # Handle case where it returns tuple (trainable_params, all_params)
        if isinstance(trainable_params, tuple):
            trainable_params = trainable_params[0]
        print(f"   ✓ Trainable parameters: {trainable_params:,}")

    # load_datasets() inherited from BaseTrainer - no duplication!

    def create_trainer(self, train_dataset: Any, eval_dataset: Any) -> None:
        """Create standard Trainer."""
        print(f"\nCreating trainer...")

        # Store datasets for later use (test set evaluation)
        self._raw_datasets = {
            'train': train_dataset,
            'val': eval_dataset
        }

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            fp16=False,  # Use FP32 for CPU/MPS
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            optim="adamw_torch",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            remove_unused_columns=False,
        )

        # Tokenize function
        def tokenize_function(examples):
            # Tokenize the text field
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,  # Will pad in data collator
            )
            # Set labels to input_ids for causal LM
            result["labels"] = result["input_ids"].copy()
            return result

        # Tokenize datasets
        print(f"   Tokenizing datasets...")
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )

        # Store tokenize function for test set later
        self._tokenize_function = tokenize_function

        # Data collator for padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        print(f"   ✓ Trainer created")
        print(f"   ✓ Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

    def evaluate_test_set(self, test_dataset: Any) -> dict:
        """Evaluate on test set (tokenizes first if needed)."""
        # Check if test dataset is already tokenized
        if 'input_ids' not in test_dataset.column_names:
            # Tokenize test dataset
            test_dataset = test_dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=test_dataset.column_names
            )

        return self.trainer.evaluate(test_dataset)

    def evaluate_classification_accuracy(self, dataset: Any, max_examples: int = None) -> Dict[str, Any]:
        """
        Evaluate classification accuracy by generating outputs and comparing predictions.

        Args:
            dataset: Raw dataset with 'alert', 'reasoning', 'classification' fields
            max_examples: Optional limit on number of examples to evaluate

        Returns:
            Dict with classification metrics (accuracy, hits, total, by_category, examples)
        """
        evaluator = ClassificationEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_template_path=self.config.prompt_template_path,
            max_seq_length=self.config.max_seq_length,
        )

        return evaluator.evaluate_dataset(dataset, max_examples=max_examples)
