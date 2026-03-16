"""
Cloud trainer implementation.
Uses Unsloth optimizations for fast GPU training.
"""
from typing import Any, Dict

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

from .base_trainer import BaseTrainer
from .config import CloudTrainingConfig
from .classification_evaluator import ClassificationEvaluator


class CloudTrainer(BaseTrainer):
    """
    Cloud trainer with Unsloth optimizations.
    Follows Open/Closed Principle - extends BaseTrainer without modifying it.
    """

    def __init__(self, config: CloudTrainingConfig):
        super().__init__(config)
        self.config: CloudTrainingConfig = config  # Type hint for IDE

    def load_model(self) -> None:
        """Load model with Unsloth optimizations."""
        print(f"\nLoading model with Unsloth: {self.config.model_name}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=None,
        )

        print(f"   Applying LoRA with Unsloth optimizations")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        print(f"   ✓ Model loaded")
        print(f"   ✓ Max sequence length: {self.config.max_seq_length:,}")
        print(f"   ✓ 4-bit quantization: {self.config.load_in_4bit}")

    # load_datasets() inherited from BaseTrainer - no duplication!

    def create_trainer(self, train_dataset: Any, eval_dataset: Any) -> None:
        """Create SFTTrainer with Unsloth."""
        print(f"\nCreating Unsloth SFTTrainer...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            optim="adamw_8bit",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
        )

        # Create SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=False,
        )

        print(f"   ✓ SFTTrainer created")
        print(f"   ✓ Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"   ✓ Using: {'bfloat16' if is_bfloat16_supported() else 'float16'}")

    def evaluate_classification_accuracy(self, dataset: Any, max_examples: int = None) -> Dict[str, Any]:
        """
        Evaluate classification accuracy by generating outputs and comparing predictions.

        Args:
            dataset: Raw dataset with 'alert', 'reasoning', 'classification' fields
            max_examples: Optional limit on number of examples to evaluate

        Returns:
            Dict with classification metrics (accuracy, hits, total, by_category, examples)
        """
        # Enable inference mode for Unsloth
        FastLanguageModel.for_inference(self.model)

        evaluator = ClassificationEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_template_path=self.config.prompt_template_path,
            max_seq_length=self.config.max_seq_length,
        )

        return evaluator.evaluate_dataset(dataset, max_examples=max_examples)
