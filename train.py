"""
Step-by-step distillation training with Unsloth.
Optimized for Qwen 3.5 MoE models.
"""
import json
from pathlib import Path
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration
CONFIG = {
    "model_name": "unsloth/Qwen2.5-32B-Instruct",
    "max_seq_length": 4096,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "output_dir": "outputs/distilled_model",
    "use_splits": True,
    "prompt_template_path": "baseline.txt"
}

def load_prompt_template(path):
    """Load prompt template once (not per example)."""
    return Path(path).read_text()

def create_format_function(prompt_template):
    """
    Create formatting function with prompt template in closure.
    Template is loaded once, not duplicated per example.
    """
    def format_prompt(example):
        """Dynamically combine prompt + alert + reasoning at batch time."""
        # Format alert as JSON
        alert_json = json.dumps(example['alert'], indent=2)

        # Combine prompt + alert
        instruction = f"{prompt_template}\n{alert_json}\n```"

        # Combine reasoning + classification
        output = f"{example['reasoning']}\n\n**Final Classification:** {example['classification']}"

        # Format for chat template
        return {
            "text": f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        }

    return format_prompt

def load_model_and_tokenizer(config):
    """Load model with Unsloth optimizations."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model, tokenizer

def train(data_path="data/train_distill.json", config=CONFIG):
    """Main training loop."""

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(config)

    print("Loading prompt template...")
    prompt_template = load_prompt_template(config["prompt_template_path"])
    print(f"Prompt template loaded: {len(prompt_template)} chars")

    # Create format function with template in closure
    format_fn = create_format_function(prompt_template)

    print("Loading dataset...")

    if config["use_splits"]:
        # Load from split files
        dataset = load_dataset("json", data_files={
            "train": "data/splits/train.json",
            "validation": "data/splits/val.json",
            "test": "data/splits/test.json"
        })
        train_dataset = dataset["train"].map(format_fn, remove_columns=dataset["train"].column_names)
        eval_dataset = dataset["validation"].map(format_fn, remove_columns=dataset["validation"].column_names)

        print(f"Training on {len(train_dataset)} examples")
        print(f"Validating on {len(eval_dataset)} examples")
    else:
        # Load single file
        dataset = load_dataset("json", data_files=data_path, split="train")
        dataset = dataset.map(format_fn, remove_columns=dataset.column_names)

        if len(dataset) > 50:
            dataset = dataset.train_test_split(test_size=0.15, seed=42)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
            print(f"Training on {len(train_dataset)} examples")
            print(f"Validating on {len(eval_dataset)} examples")
        else:
            train_dataset = dataset
            eval_dataset = None
            print(f"Training on {len(train_dataset)} examples (no validation split)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        optim="adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print(f"Training complete! Model saved to {config['output_dir']}")

    # Evaluate on test set if available
    if config["use_splits"]:
        print("\nEvaluating on test set...")
        test_dataset = load_dataset("json", data_files="data/splits/test.json", split="train")
        test_dataset = test_dataset.map(format_fn, remove_columns=test_dataset.column_names)
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")

if __name__ == "__main__":
    train()
