"""
Training script optimized for Mac with LIMITED RAM (8GB available).
Uses tiny model (Qwen2.5-0.5B) for validation only.

WARNING: This is for CODE VALIDATION only, not production.
Production training requires cloud GPU with 32B model.
"""
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for Mac with 8GB AVAILABLE RAM
CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",  # Tiny model for validation
    "max_seq_length": 512,  # Short to fit in RAM
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_epochs": 1,  # Just validate it works
    "batch_size": 1,  # Minimal
    "gradient_accumulation_steps": 16,  # Effective batch = 16
    "output_dir": "outputs/distilled_model_mac",
    "use_splits": True,
    "prompt_template_path": "prompts/baseline.txt"
}

def load_prompt_template(path):
    """Load prompt template once."""
    return Path(path).read_text()

def create_format_function(prompt_template, max_length=512):
    """
    Create formatting function with prompt in closure.
    Truncates examples to fit in memory.
    """
    def format_prompt(example):
        # Format alert (truncate if needed)
        alert_json = json.dumps(example['alert'], indent=2)

        # Truncate alert if too long
        if len(alert_json) > 4000:
            alert_json = alert_json[:4000] + "\n... [truncated]"

        # Combine prompt + alert
        instruction = f"{prompt_template}\n{alert_json}\n```"

        # Truncate instruction if too long
        if len(instruction) > max_length * 3:  # ~3 chars per token
            instruction = instruction[:max_length * 3]

        # Combine reasoning + classification
        reasoning = example.get('reasoning', '')[:500]  # Truncate reasoning
        classification = example.get('classification', '')
        output = f"{reasoning}\n\n**Final Classification:** {classification}"

        # Format for chat template
        return {
            "text": f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        }

    return format_prompt

def load_model_and_tokenizer(config):
    """Load tiny model for Mac with limited RAM."""
    logger.info(f"Loading model: {config['model_name']}")
    logger.info("NOTE: Using 0.5B model for validation. Production uses 32B on cloud GPU.")

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS (GPU acceleration)")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA")
    else:
        device = "cpu"
        logger.info("WARNING: Using CPU only (very slow)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (small, fits in 8GB)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    return model, tokenizer

def train(data_path="data/train_distill.json", config=CONFIG):
    """Main training loop for Mac."""

    print("\n" + "=" * 70)
    print("🚀 VALIDATION TRAINING ON MAC (LIMITED RAM)")
    print("=" * 70)
    print(f"📊 Available RAM: ~8GB")
    print(f"🤖 Model: {config['model_name']} (0.5B params)")
    print(f"⚠️  Purpose: CODE VALIDATION ONLY")
    print(f"☁️  Production: Use cloud GPU with 32B model")
    print("=" * 70)
    print()

    # Load prompt template
    print("📄 Loading prompt template...")
    prompt_template = load_prompt_template(config["prompt_template_path"])
    print(f"   ✓ Loaded {len(prompt_template)} chars from {config['prompt_template_path']}")
    format_fn = create_format_function(prompt_template, config["max_seq_length"])

    # Load model
    print("\n🔄 Loading model and tokenizer...")
    print(f"   Model: {config['model_name']}")
    print(f"   This may take 1-2 minutes on first run (downloading ~2GB)...")
    model, tokenizer = load_model_and_tokenizer(config)
    print("   ✓ Model loaded successfully")

    # Load dataset
    print("\n📂 Loading dataset...")
    if config["use_splits"]:
        try:
            print("   Using train/val splits from data/splits/")
            dataset = load_dataset("json", data_files={
                "train": "data/splits/train.json",
                "validation": "data/splits/val.json",
            })
            print(f"   ✓ Loaded train split: {len(dataset['train'])} examples")
            print(f"   ✓ Loaded val split: {len(dataset['validation'])} examples")

            print("\n🔄 Formatting examples (adding prompt template)...")
            train_dataset = dataset["train"].map(format_fn, remove_columns=dataset["train"].column_names)
            eval_dataset = dataset["validation"].map(format_fn, remove_columns=dataset["validation"].column_names)
            print(f"   ✓ Formatted {len(train_dataset)} training examples")
            print(f"   ✓ Formatted {len(eval_dataset)} validation examples")
        except FileNotFoundError:
            print("   ⚠️  Split files not found, using full dataset")
            dataset = load_dataset("json", data_files=data_path, split="train")
            dataset = dataset.map(format_fn, remove_columns=dataset.column_names)
            train_dataset = dataset
            eval_dataset = None
            print(f"   ✓ Loaded {len(train_dataset)} examples")
    else:
        dataset = load_dataset("json", data_files=data_path, split="train")
        dataset = dataset.map(format_fn, remove_columns=dataset.column_names)
        train_dataset = dataset
        eval_dataset = None
        print(f"   ✓ Loaded {len(train_dataset)} examples")

    # Tokenize
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["max_seq_length"],
            padding=False,  # Dynamic padding
        )
        result["labels"] = result["input_ids"].copy()
        return result

    print("\n🔤 Tokenizing dataset...")
    print(f"   Max sequence length: {config['max_seq_length']} tokens")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="   Tokenizing train"
    )
    print(f"   ✓ Tokenized {len(train_dataset)} training examples")

    if eval_dataset:
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="   Tokenizing val"
        )
        print(f"   ✓ Tokenized {len(eval_dataset)} validation examples")

    # Data collator (dynamic padding)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # Training arguments
    print("\n⚙️  Configuring training parameters...")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Total training steps: {len(train_dataset) // (config['batch_size'] * config['gradient_accumulation_steps']) * config['num_epochs']}")

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        fp16=False,  # MPS doesn't support FP16 well
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=1,  # Keep only latest checkpoint
        load_best_model_at_end=False,
        disable_tqdm=False,  # Show progress bar
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n" + "=" * 70)
    print("🎯 STARTING TRAINING")
    print("=" * 70)
    print("This validates the code pipeline. For production, use cloud GPU.")
    print()
    print("📊 Progress will show:")
    print("   - Loss value (should decrease over time)")
    print("   - Learning rate")
    print("   - Training speed (examples/sec)")
    print()
    print("⏱️  Estimated time: ~10-30 minutes for 6 examples")
    print("=" * 70)
    print()

    try:
        # Training with progress tracking
        trainer.train()

        print("\n" + "=" * 70)
        print("💾 SAVING MODEL")
        print("=" * 70)
        print(f"Saving to: {config['output_dir']}")
        model.save_pretrained(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])
        print("✓ Model weights saved")
        print("✓ Tokenizer saved")
        print("✓ Config saved")

        print("\n" + "=" * 70)
        print("✅ VALIDATION COMPLETE!")
        print("=" * 70)
        print(f"Model saved to: {config['output_dir']}")
        print()
        print("This proves the pipeline works end-to-end! 🎉")
        print()
        print("📍 Next steps:")
        print("   1. Check outputs: ls -lh outputs/distilled_model_mac/")
        print("   2. For production training:")
        print("      → bash vertex_ai_submit.sh (32B model on cloud GPU)")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TRAINING FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        print("💡 Common fixes:")
        print("   1. Out of memory:")
        print("      - Close other applications")
        print("      - Increase Docker memory (Settings → Resources → 7 GB)")
        print("      - Edit train_mac.py: reduce max_seq_length to 256")
        print()
        print("   2. Docker issues:")
        print("      - Restart Docker Desktop")
        print("      - docker-compose down && docker-compose build train-mac")
        print()
        print("   3. Dataset issues:")
        print("      - Run: python test_dataset.py --test-splits")
        print("=" * 70)
        raise

if __name__ == "__main__":
    train()
