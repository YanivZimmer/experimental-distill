# Testing & Docker Setup Summary

## What Was Created

### 1. **Comprehensive Test Suite** ✅

#### `test_dataset.py` - Dataset Validation
Tests run BEFORE training to catch issues early:

| Test | Checks | Why Important |
|------|--------|---------------|
| **Loading** | JSON format valid | Prevents parse errors during training |
| **Schema** | Required fields present | Ensures data structure is correct |
| **Alert structure** | raw_alert exists | Validates input data quality |
| **Reasoning quality** | Non-empty, reasonable length | Catches missing teacher outputs |
| **Classification format** | Valid classifications | Ensures labels are present |
| **Metadata** | item_id, teacher_model | Needed for tracking/debugging |
| **Size** | Dataset not too small | Warns if <100 examples |
| **Duplicates** | No duplicate item_ids | Prevents data leakage |
| **Memory estimation** | Token counts, VRAM usage | Warns if won't fit in GPU |
| **Split validation** | No leakage between splits | Critical for valid evaluation |

**Usage:**
```bash
# Test dataset
python test_dataset.py --test-splits

# Expected: ✅ ALL TESTS PASSED
```

#### `test_model.py` - Model Validation
Tests model loading and inference:

| Test | Checks | Why Important |
|------|--------|---------------|
| **Model loading** | Can load base model | Catches download/permission issues |
| **Tokenization** | Examples fit in context | Prevents truncation problems |
| **Inference** | Model generates output | Validates model works |
| **Memory check** | RAM/VRAM available | Warns about resource constraints |

**Usage:**
```bash
# Test model (requires transformers)
python test_model.py

# NOTE: Downloads ~14GB model (Qwen 2.5-7B)
```

### 2. **Docker Configuration for Mac M5 Pro** 🐳

#### Key Limitations of Mac M5
- **No CUDA support** - Unsloth won't work
- **32GB RAM limit** - Can't fit 32B models
- **MPS backend** - Slower than NVIDIA GPUs

#### Files Created

| File | Purpose |
|------|---------|
| `Dockerfile.mac` | ARM64 image for Mac M5 |
| `requirements.mac.txt` | Dependencies without Unsloth |
| `docker-compose.yml` | Multi-service orchestration |
| `train_mac.py` | Training script for Mac (7B model) |
| `DOCKER_USAGE.md` | Complete Docker guide |
| `run_tests.sh` | Automated test runner |

#### Docker Services

```yaml
# 1. Testing (always run first)
docker-compose run --rm test

# 2. Mac M5 Training (validation only)
docker-compose up train-mac

# 3. Cloud GPU Training (production)
docker-compose up train-cloud
```

### 3. **Automated Test Runner** 🚀

**`run_tests.sh`** - One command to test everything:

```bash
# Run all tests
./run_tests.sh
```

**What it tests:**
1. ✅ Docker installed and running
2. ✅ Required files exist
3. ✅ Dataset validation (Python)
4. ✅ Docker image builds successfully
5. ✅ Dataset validation (Docker)

**Output:**
```
✅ ALL TESTS PASSED - SAFE TO TRAIN

Next steps:
  1. Train on Mac M5 (testing):  docker-compose up train-mac
  2. Train on cloud GPU (prod):  bash vertex_ai_submit.sh
```

---

## Workflow

### Before Training (ALWAYS)

```bash
# Step 1: Run comprehensive tests
./run_tests.sh

# If any test fails, fix it before proceeding
```

### Mac M5 Pro Workflow (Testing/Validation)

```bash
# Step 1: Validate
./run_tests.sh

# Step 2: Quick training test (1 epoch, 7B model)
docker-compose up train-mac

# Step 3: Check outputs
ls -lh outputs/distilled_model_mac/
```

**Use Mac M5 for:**
- ✅ Data validation
- ✅ Code testing
- ✅ Quick sanity checks
- ❌ NOT for production training (too slow, model too small)

### Cloud GPU Workflow (Production)

```bash
# Step 1: Validate locally
./run_tests.sh

# Step 2: Deploy to cloud
bash vertex_ai_submit.sh

# Step 3: Monitor
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

---

## Test Results (Current Dataset)

### Dataset Validation ✅

```
Total examples: 8
Strategy: conservative

✅ All 8 examples have required fields
✅ All examples have non-empty reasoning
✅ No duplicate item_ids
✅ No item_id overlap between splits

⚠️  Warnings:
  - Small dataset (8 examples) - Need 100+ for production
  - Average 4413 tokens exceeds 4096 limit - Use max_seq_length=8192
```

### Memory Estimation

```
Average example: 4413 tokens
Estimated VRAM (32B model, 4-bit, batch=2):
  - Model: 8 GB
  - Training: ~12 GB
  - ✅ Fits in 24GB GPU
  - ❌ Too large for Mac M5 (32GB RAM)
```

---

## Why Tests Are Critical

### Without Tests (Bad)

```
❌ Start training → 2 hours later → OOM error
❌ Dataset has duplicates → Invalid evaluation
❌ Examples too long → Truncated, poor quality
❌ Missing reasoning → Crashes mid-training
❌ No split validation → Data leakage → 95% test accuracy, 60% production
```

**Cost:** Wasted compute + time + false confidence

### With Tests (Good)

```
✅ Run tests (10 seconds) → Catch all issues
✅ Fix problems before training
✅ Confident training will succeed
✅ Valid evaluation metrics
```

**Savings:** Hours of debugging + $10-100 in wasted cloud compute

---

## Mac M5 Pro Specifications

| Spec | Value | Training Impact |
|------|-------|-----------------|
| **CPU** | 14-core (10 performance, 4 efficiency) | Good for preprocessing |
| **RAM** | 32 GB unified memory | **Tight for 7B, impossible for 32B** |
| **GPU** | 16-core (MPS) | **~10x slower than NVIDIA A100** |
| **Bandwidth** | 400 GB/s | Excellent for data loading |

### What Fits in 32GB?

| Model | Full Precision | 8-bit | 4-bit | Fits? |
|-------|----------------|-------|-------|-------|
| 7B | 28 GB | 14 GB | 7 GB | ✅ Yes |
| 13B | 52 GB | 26 GB | 13 GB | ⚠️ Tight |
| 32B | 128 GB | 64 GB | 32 GB | ❌ No |

**Recommendation:** Use 7B for testing, 32B on cloud GPU for production.

---

## Cost Analysis

### Mac M5 Pro (Local)

| Task | Time | Cost | Use For |
|------|------|------|---------|
| **Tests** | 30 seconds | Free | Always |
| **7B training (1 epoch)** | 1-2 hours | Free | Validation |
| **7B training (3 epochs)** | 3-6 hours | Free | Quick experiments |

### Cloud GPU (L4, $0.73/hour)

| Task | Time | Cost | Use For |
|------|------|------|---------|
| **32B training (1 epoch, 1000 examples)** | 2 hours | $1.46 | Production |
| **32B training (3 epochs)** | 6 hours | $4.38 | Production |
| **Hyperparameter search (5 runs)** | 10 hours | $7.30 | Optimization |

**Strategy:**
1. Test/validate on Mac (free, fast feedback)
2. Train production model on cloud (cheap, 10x faster)

---

## Next Steps

### When You Have Full Dataset (100+ examples)

```bash
# 1. Regenerate dataset
python prepare_dataset.py --full-reasoning

# 2. Split dataset
python split_dataset.py --strategy conservative

# 3. Run tests
./run_tests.sh

# 4. Train on cloud GPU
bash vertex_ai_submit.sh
```

### For Now (8 examples)

```bash
# 1. Validate current setup
./run_tests.sh

# 2. Test training on Mac (optional)
docker-compose up train-mac

# 3. Wait for full benchmark data before production training
```

---

## Troubleshooting

### Test Failures

**"Dataset too small (8 examples)"**
- This is a warning, not an error
- Wait for full benchmark dataset before production

**"Average example exceeds 4096 tokens"**
- Update `train.py` CONFIG: `"max_seq_length": 8192`
- Or update `train_mac.py` CONFIG: `"max_seq_length": 2048` (Mac only)

**"Docker daemon not running"**
```bash
# Start Docker Desktop
open /Applications/Docker.app

# Verify
docker ps
```

### Docker Issues

**"Out of memory" on Mac**
```python
# Edit train_mac.py CONFIG:
"batch_size": 1,
"max_seq_length": 1024,  # Reduce
```

**"Cannot allocate memory"**
- Close other applications
- Increase Docker memory: Docker Desktop → Settings → Resources → Memory (28 GB)

---

## Summary

| ✅ What You Have | 📋 What to Do |
|------------------|---------------|
| Comprehensive test suite | Run `./run_tests.sh` before training |
| Docker for Mac M5 | Use for validation only |
| Docker for cloud GPU | Use for production training |
| Dataset validation | Caught warnings (small dataset, long examples) |
| Split validation | No data leakage ✅ |

**Status:** Ready for testing. Wait for full dataset (100+ examples) before production training.

**Recommended next action:** Run `./run_tests.sh` to validate everything works on your Mac M5.
