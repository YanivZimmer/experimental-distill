"""
Step-by-step distillation training with Unsloth.
Optimized for Qwen 3.5 MoE models.
"""
import json
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration
CONFIG = {
    "model_name": "unsloth/Qwen2.5-32B-Instruct",  # Closest to Qwen3.5-35B-3A
    "max_seq_length": 4096,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "output_dir": "outputs/distilled_model"
}

def format_prompt(example):
    """Format example into chat template."""
    return {
        "text": f"""<|im_start|>user
{example['instruction']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
    }

def load_model_and_tokenizer(config):
    """Load model with Unsloth optimizations."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=None,  # Auto-detect
    )

    # Apply LoRA for efficient fine-tuning
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

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    print(f"Training on {len(dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=False,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print("Saving model...")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print(f"Training complete! Model saved to {config['output_dir']}")

if __name__ == "__main__":
    train()
