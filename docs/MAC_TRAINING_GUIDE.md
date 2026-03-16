# Mac M5 Pro Training Guide (8GB Available RAM)

## ⚠️ Important Limitations

Your Mac has **8GB available RAM** (after OS and apps), which severely limits what you can do:

| Model Size | RAM Needed | Fits in 8GB? | Use Case |
|------------|------------|--------------|----------|
| **0.5B** | ~2-3 GB | ✅ Yes | Code validation |
| **1.5B** | ~6-8 GB | ⚠️ Tight | Testing |
| **7B** | ~15-20 GB | ❌ No | N/A |
| **32B** | ~60-80 GB | ❌ No | Cloud GPU only |

**Recommendation:** Use 0.5B model on Mac for code validation only. Train 32B on cloud GPU.

---

## Quick Start (3 Commands)

### Step 1: Validate Everything Works

```bash
# Run comprehensive tests (30 seconds)
./run_tests.sh
```

Expected output:
```
✅ ALL TESTS PASSED - SAFE TO TRAIN
```

### Step 2: Build Docker Image

```bash
# Build image for Mac M5 (one time, ~5 minutes)
docker-compose build train-mac
```

### Step 3: Run Validation Training

```bash
# Train on 0.5B model (validates pipeline works)
docker-compose up train-mac
```

**What happens:**
- Uses Qwen2.5-0.5B-Instruct (tiny model)
- Trains for 1 epoch (~10-30 minutes for 6 examples)
- Saves to `outputs/distilled_model_mac/`
- **Purpose:** Proves code works, NOT production training

---

## Detailed Usage

### Check Docker Resources

```bash
# 1. Open Docker Desktop
open /Applications/Docker.app

# 2. Go to Settings → Resources
# 3. Set:
#    - CPUs: 8-10
#    - Memory: 6 GB (leave 2GB for macOS)
#    - Swap: 2 GB
#    - Disk: 50 GB
```

### Run Training with Live Logs

```bash
# Start training (foreground, see live logs)
docker-compose up train-mac

# You'll see:
# - Model loading
# - Dataset tokenization
# - Training progress (loss values)
# - Final save
```

### Run Training in Background

```bash
# Start in background
docker-compose up -d train-mac

# View logs
docker-compose logs -f train-mac

# Stop training
docker-compose down
```

### Interactive Shell (for debugging)

```bash
# Enter container
docker-compose run --rm train-mac /bin/bash

# Inside container:
python test_dataset.py --test-splits
python train_mac.py
exit
```

---

## What You're Training

### Model Comparison

| Aspect | Mac Training | Cloud Training |
|--------|--------------|----------------|
| **Model** | Qwen2.5-0.5B | Qwen2.5-32B |
| **Parameters** | 0.5 billion | 32 billion |
| **Quality** | Poor (too small) | Production-ready |
| **Speed** | Slow (no CUDA) | 10-20x faster |
| **Cost** | Free | $3-10 per run |
| **Purpose** | Code validation | Production model |

### Example Output

**Mac (0.5B model):**
```
Input: Analyze this PowerShell alert
Output: Suspicious activity detected. Escalate.
Quality: ⭐⭐ (basic, not nuanced)
```

**Cloud (32B model):**
```
Input: Analyze this PowerShell alert
Output: The combination of a low-confidence ML detection,
        PowerShell writing an executable, a suspicious file name
        'Game Loader All Rh.exe', and the unusual parent process
        (dllhost.exe) indicates potentially malicious activity...
Quality: ⭐⭐⭐⭐⭐ (detailed, expert-level)
```

---

## Monitoring Training

### Key Metrics to Watch

```bash
# While training runs, you'll see:

Epoch 1/1:  50%|████████     | 3/6 [00:45<00:30,  0.10it/s]
{'loss': 2.456, 'learning_rate': 0.00015, 'epoch': 0.5}
```

**What to look for:**
- ✅ **Loss decreasing:** 3.0 → 2.5 → 2.0 (good)
- ❌ **Loss increasing:** 2.0 → 2.5 → 3.0 (bad - something wrong)
- ❌ **Loss NaN:** Training crashed
- ✅ **No OOM errors:** Fits in memory

### Check Outputs

```bash
# After training completes
ls -lh outputs/distilled_model_mac/

# You should see:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer files
```

---

## Troubleshooting

### Out of Memory (OOM)

**Error:**
```
RuntimeError: MPS backend out of memory
```

**Solutions:**

**Option 1: Increase Docker Memory**
```bash
# Docker Desktop → Settings → Resources → Memory → 7 GB
# Restart Docker
```

**Option 2: Reduce Sequence Length**
```python
# Edit train_mac.py CONFIG:
"max_seq_length": 256,  # Reduce from 512
```

**Option 3: Close Other Apps**
```bash
# Free up RAM
# Close Chrome, Slack, etc.
# Check available memory:
activity monitor
```

### Very Slow Training

**Expected Speed:**
- 0.5B model: ~5-10 seconds per example
- 6 examples: ~10-30 minutes total

**If much slower:**
```bash
# Check if using MPS (GPU) or CPU
docker run --rm simple-distill:mac-train python -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
"

# If both False, training uses CPU (very slow)
# This is okay for validation, just takes longer
```

### Training Hangs

**If stuck at "Loading model..." for >5 minutes:**
```bash
# Model download might be slow
# Check progress:
docker-compose logs train-mac

# If downloading, wait (first time only)
# Model is ~2GB
```

### Docker Build Fails

**Error:** `Cannot connect to Docker daemon`
```bash
# Start Docker Desktop
open /Applications/Docker.app

# Wait for it to start
# Try again:
docker-compose build train-mac
```

---

## What Success Looks Like

### Successful Training Run

```bash
$ docker-compose up train-mac

Creating distill-train-mac ... done
Attaching to distill-train-mac

============================================================
VALIDATION TRAINING ON MAC (LIMITED RAM)
============================================================
Available RAM: ~8GB
Model: Qwen/Qwen2.5-0.5B-Instruct (0.5B params)
Purpose: CODE VALIDATION ONLY
Production: Use cloud GPU with 32B model
============================================================

Loading prompt template...
Prompt template loaded: 7123 chars

Loading model...
NOTE: Using 0.5B model for validation. Production uses 32B on cloud GPU.
Using Apple MPS (GPU acceleration)
Trainable params: 4,194,304 (0.83%)

Loading dataset...
Training on 6 examples
Validating on 1 examples

Tokenizing...
Tokenizing train: 100%|██████████| 6/6
Tokenizing val: 100%|██████████| 1/1

Starting training...
Epoch 1/1:  17%|███       | 1/6 [00:12<01:00, 12s/it]
{'loss': 2.345, 'learning_rate': 0.0002, 'epoch': 0.17}
...
Epoch 1/1: 100%|██████████| 6/6 [01:15<00:00, 12.5s/it]

Saving model...
✅ Validation complete! Model saved to outputs/distilled_model_mac
This proves the pipeline works. For production:
  → Use bash vertex_ai_submit.sh (32B model on cloud GPU)
```

### Validation Complete Checklist

After successful run, verify:
- [x] Training completed without errors
- [x] Loss decreased (even slightly)
- [x] Model saved to `outputs/distilled_model_mac/`
- [x] Files present: `adapter_model.safetensors`, `adapter_config.json`

**Next step:** Deploy to cloud GPU for production training!

---

## Docker Commands Reference

```bash
# Build image
docker-compose build train-mac

# Run training (foreground)
docker-compose up train-mac

# Run training (background)
docker-compose up -d train-mac

# View logs (live)
docker-compose logs -f train-mac

# Stop training
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Interactive shell
docker-compose run --rm train-mac /bin/bash

# Remove all Docker images (clean slate)
docker system prune -a
```

---

## Cost & Time Comparison

### Mac M5 Pro (Free, Slow)

| Task | Time | Cost | Quality |
|------|------|------|---------|
| Code validation (6 examples, 1 epoch) | 10-30 min | Free | N/A (validation only) |
| Testing (100 examples, 1 epoch) | 2-4 hours | Free | Poor (0.5B model) |

### Cloud GPU L4 (Fast, Cheap)

| Task | Time | Cost | Quality |
|------|------|------|---------|
| Production (1000 examples, 3 epochs, 32B) | 6 hours | $4.38 | Excellent |
| Testing (100 examples, 1 epoch, 32B) | 1 hour | $0.73 | Excellent |

**Recommendation:** Use Mac for quick validation, cloud for everything else.

---

## Summary

✅ **Mac training validates your code works**
✅ **Uses tiny 0.5B model (fits in 8GB RAM)**
✅ **Takes 10-30 minutes for 6 examples**
⚠️ **NOT for production - model too small**
🚀 **For production: Use cloud GPU (32B model)**

**Next steps:**
1. Run `./run_tests.sh` to validate data
2. Run `docker-compose up train-mac` to validate code
3. When ready: `bash vertex_ai_submit.sh` for production
