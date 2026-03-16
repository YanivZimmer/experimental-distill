# Docker Usage Guide

## Overview

This project provides three Docker configurations:

1. **Testing** - Validate dataset before training
2. **Mac M5 Training** - Train on Mac M5 Pro (CPU/MPS, testing only)
3. **Cloud GPU Training** - Production training on NVIDIA GPUs

---

## Prerequisites

### Mac M5 Pro

```bash
# Install Docker Desktop for Mac
# Download from: https://www.docker.com/products/docker-desktop

# Verify installation
docker --version
docker-compose --version
```

### Resource Allocation

**Docker Desktop → Settings → Resources:**
- CPUs: 10 cores
- Memory: 28 GB
- Swap: 2 GB
- Disk: 100 GB

---

## Quick Start

### Step 1: Run Tests (ALWAYS DO THIS FIRST)

```bash
# Validate dataset before wasting compute
docker-compose run --rm test
```

**Expected output:**
```
✅ ALL TESTS PASSED - Safe to proceed with training
```

If tests fail, fix errors before proceeding.

### Step 2: Train on Mac M5 (Testing/Validation)

```bash
# Build image
docker-compose build train-mac

# Run training
docker-compose up train-mac
```

**Expected behavior:**
- Uses 7B model (32B won't fit in 32GB RAM)
- Very slow (no CUDA support)
- For validation only, NOT production

### Step 3: Train on Cloud GPU (Production)

```bash
# On Google Cloud / AWS / Azure with NVIDIA GPU
docker-compose up train-cloud
```

---

## Detailed Commands

### Testing

```bash
# Run all tests
docker-compose run --rm test

# Run dataset tests only
docker run --rm -v $(pwd)/data:/workspace/data simple-distill:mac-test \
    python test_dataset.py

# Run model tests
docker run --rm -v $(pwd)/data:/workspace/data simple-distill:mac-test \
    python test_model.py

# Interactive shell for debugging
docker-compose run --rm test /bin/bash
```

### Mac M5 Training

```bash
# Full training run
docker-compose up train-mac

# With custom config
docker run --rm \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -e BATCH_SIZE=1 \
    -e NUM_EPOCHS=1 \
    simple-distill:mac-train \
    python train_mac.py

# Monitor logs
docker-compose logs -f train-mac

# Stop training
docker-compose down
```

### Cloud GPU Training

```bash
# Ensure NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Build for GPU
docker-compose build train-cloud

# Train with GPU
docker-compose up train-cloud

# Multi-GPU training
docker run --rm --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    simple-distill:cloud-train \
    torchrun --nproc_per_node=4 train.py
```

---

## Workflows

### Local Development (Mac M5)

```bash
# 1. Validate data
docker-compose run --rm test

# 2. Quick training test (1 epoch on 7B model)
docker-compose up train-mac

# 3. Inspect outputs
ls -lh outputs/distilled_model_mac/
```

### Production Training (Cloud GPU)

```bash
# 1. Validate locally
docker-compose run --rm test

# 2. Push data to cloud
gsutil cp -r data/ gs://your-bucket/
gsutil cp baseline.txt gs://your-bucket/

# 3. Run on Vertex AI
bash vertex_ai_submit.sh

# OR run manually on GPU instance
docker-compose up train-cloud
```

### Continuous Integration

```bash
# In CI/CD pipeline (GitHub Actions, GitLab CI, etc.)
docker-compose run --rm test || exit 1
echo "✅ Tests passed, safe to deploy"
```

---

## Troubleshooting

### Mac M5 Issues

**Problem:** "Out of memory" error

**Solution:**
```python
# Edit train_mac.py CONFIG:
"batch_size": 1,
"max_seq_length": 1024,  # Reduce from 2048
"gradient_accumulation_steps": 16,
```

**Problem:** "MPS backend not available"

**Solution:**
```bash
# Check PyTorch MPS support
docker run --rm simple-distill:mac-test python -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
"
```

If False, training will use CPU (very slow).

**Problem:** Training is extremely slow

**Expected:** Mac M5 without CUDA is ~10-50x slower than GPU.

**Solution:** Use cloud GPU for actual training. Mac is for testing only.

### Docker Issues

**Problem:** "Cannot connect to Docker daemon"

**Solution:**
```bash
# Start Docker Desktop
open /Applications/Docker.app

# Verify
docker ps
```

**Problem:** "No space left on device"

**Solution:**
```bash
# Clean up old images
docker system prune -a

# Increase disk in Docker Desktop Settings
```

**Problem:** Permission errors with volumes

**Solution:**
```bash
# Fix ownership
sudo chown -R $(whoami) data/ outputs/

# Or run with user
docker run --user $(id -u):$(id -g) ...
```

---

## Resource Usage

### Mac M5 Pro (32GB RAM)

| Model Size | RAM Usage | Training Time (1 epoch) | Recommended |
|------------|-----------|-------------------------|-------------|
| 7B | ~15 GB | ~2 hours (6 examples) | ✅ Testing only |
| 13B | ~25 GB | ~4 hours | ⚠️ Tight fit |
| 32B | **Out of Memory** | N/A | ❌ Use cloud |

### Cloud GPU (24GB VRAM)

| Model Size | VRAM Usage | Training Time (1 epoch) | Recommended |
|------------|------------|-------------------------|-------------|
| 7B | ~8 GB | ~10 min (1000 examples) | ✅ Fast testing |
| 32B (4-bit) | ~18 GB | ~2 hours (1000 examples) | ✅ Production |

---

## File Structure

```
simple_distill/
├── Dockerfile.mac          # Mac M5 Pro (ARM64, MPS)
├── Dockerfile              # Cloud GPU (CUDA)
├── docker-compose.yml      # Orchestration
├── requirements.mac.txt    # Mac dependencies
├── requirements.txt        # GPU dependencies
├── train_mac.py           # Mac training script
├── train.py               # GPU training script (Unsloth)
├── test_dataset.py        # Data validation
├── test_model.py          # Model validation
└── data/
    ├── train_distill.json
    └── splits/
        ├── train.json
        ├── val.json
        └── test.json
```

---

## Best Practices

### ✅ DO

1. **Always run tests first**
   ```bash
   docker-compose run --rm test
   ```

2. **Use Mac for validation only**
   - Quick sanity check
   - Data format validation
   - 1-2 epoch tests

3. **Use cloud GPU for production**
   - Full training runs
   - Large models (32B)
   - Hyperparameter tuning

4. **Version your outputs**
   ```bash
   docker run --rm \
       -v $(pwd)/outputs:/workspace/outputs \
       ... \
       && mv outputs/distilled_model outputs/model_v1
   ```

### ❌ DON'T

1. **Don't train 32B models on Mac M5**
   - Will OOM or swap to disk (extremely slow)

2. **Don't skip tests**
   - Wasted hours on bad data

3. **Don't use Mac for production**
   - 10-50x slower than GPU

---

## Summary

| Task | Command | Time | Cost |
|------|---------|------|------|
| **Validate data** | `docker-compose run --rm test` | 10 sec | Free |
| **Test training (Mac)** | `docker-compose up train-mac` | 1-2 hours | Free |
| **Production training (GPU)** | `docker-compose up train-cloud` | 2-4 hours | $3-10 |

**Recommended workflow:**
1. Test locally on Mac (free, 10 min)
2. Deploy to cloud GPU for production (fast, cheap)
