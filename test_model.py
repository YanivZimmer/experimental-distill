"""
Test model loading and inference before training.
Validates that the model can be loaded and used without errors.
"""
import json
import torch
import sys
from pathlib import Path

def test_model_loading():
    """Test if we can load the base model."""
    print("\n" + "=" * 60)
    print("TEST: Model Loading")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-7B-Instruct"  # Smaller for testing
        print(f"Loading {model_name}...")
        print(f"(Using 7B for testing; 32B will be used in actual training)")

        # Load with low memory
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        print(f"✅ Model loaded successfully")
        print(f"   - Parameters: {model.num_parameters() / 1e9:.1f}B")
        print(f"   - Device: {next(model.parameters()).device}")

        return True, model, tokenizer

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False, None, None


def test_tokenization(tokenizer):
    """Test tokenization of example data."""
    print("\n" + "=" * 60)
    print("TEST: Tokenization")
    print("=" * 60)

    # Load example
    try:
        with open("data/train_distill.json") as f:
            data = json.load(f)

        if not data:
            print("❌ No examples in dataset")
            return False

        example = data[0]

        # Load prompt
        prompt_template = Path("baseline.txt").read_text()
        alert_json = json.dumps(example["alert"], indent=2)
        instruction = f"{prompt_template}\n{alert_json}\n```"

        # Tokenize
        tokens = tokenizer(instruction, return_tensors="pt")
        num_tokens = tokens.input_ids.shape[1]

        print(f"Example tokenization:")
        print(f"  - Text length: {len(instruction):,} chars")
        print(f"  - Token count: {num_tokens:,} tokens")

        if num_tokens > 4096:
            print(f"⚠️  Example exceeds 4096 token limit")
            print(f"   Recommendation: Use max_seq_length=8192 or truncate")
        else:
            print(f"✅ Example fits within 4096 token limit")

        return True

    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return False


def test_inference(model, tokenizer):
    """Test a single inference pass."""
    print("\n" + "=" * 60)
    print("TEST: Inference")
    print("=" * 60)

    try:
        # Simple prompt
        prompt = "Analyze this alert: PowerShell executed with encoded command."
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        print(f"Running inference...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Inference successful")
        print(f"   Output: {response[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False


def test_memory():
    """Test memory availability."""
    print("\n" + "=" * 60)
    print("TEST: Memory Check")
    print("=" * 60)

    import psutil

    # RAM
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024**3)
    ram_available_gb = ram.available / (1024**3)

    print(f"RAM:")
    print(f"  - Total: {ram_total_gb:.1f} GB")
    print(f"  - Available: {ram_available_gb:.1f} GB")
    print(f"  - Used: {(ram_total_gb - ram_available_gb):.1f} GB")

    if ram_total_gb < 16:
        print(f"⚠️  Low RAM ({ram_total_gb:.0f} GB)")
        print(f"   Recommendation: 32GB+ for comfortable training")

    # GPU/MPS
    if torch.cuda.is_available():
        print(f"\nGPU (CUDA):")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  - Device {i}: {props.name}")
            print(f"    VRAM: {props.total_memory / (1024**3):.1f} GB")
    elif torch.backends.mps.is_available():
        print(f"\nGPU (Apple MPS): Available")
        print(f"  - Shared memory with RAM")
    else:
        print(f"\n⚠️  No GPU detected (CPU only)")
        print(f"   Training will be very slow")

    print(f"\n✅ Memory check complete")
    return True


def run_all_tests():
    """Run all model tests."""
    print("\n" + "🧪" * 30)
    print("MODEL VALIDATION TEST SUITE")
    print("🧪" * 30)

    # Test memory first
    test_memory()

    # Test model loading
    success, model, tokenizer = test_model_loading()
    if not success:
        print("\n❌ Model loading failed - cannot proceed with other tests")
        return False

    # Test tokenization
    if not test_tokenization(tokenizer):
        print("\n❌ Tokenization test failed")
        return False

    # Test inference
    if not test_inference(model, tokenizer):
        print("\n❌ Inference test failed")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL MODEL TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
