# Command Reference - Copy & Paste

## 🚀 Quick Commands (Mac M5 Pro, 8GB RAM)

### Step 1: Validate Everything (30 seconds)

```bash
./run_tests.sh
```

### Step 2: Build Docker Image (one time, ~5 minutes)

```bash
docker-compose build train-mac
```

### Step 3: Run Validation Training (10-30 minutes)

```bash
docker-compose up train-mac
```

**That's it!** These 3 commands validate your entire pipeline.

---

## 📋 Full Workflow

### Initial Setup (One Time)

```bash
# 1. Prepare dataset (when you have full benchmark)
python prepare_dataset.py --full-reasoning

# 2. Create splits (prevents data leakage)
python split_dataset.py --strategy conservative

# 3. Validate
./run_tests.sh
```

### Mac Validation

```bash
# Build image (one time)
docker-compose build train-mac

# Run training
docker-compose up train-mac

# View outputs
ls -lh outputs/distilled_model_mac/
```

### Cloud Production Training

```bash
# Upload to GCS
export GCS_BUCKET=your-bucket-name
gsutil cp data/splits/*.json gs://${GCS_BUCKET}/data/splits/
gsutil cp baseline.txt gs://${GCS_BUCKET}/

# Submit to Vertex AI
bash vertex_ai_submit.sh

# Monitor
gcloud ai custom-jobs list --region=us-central1
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

---

## 🐳 Docker Commands

### Basic Operations

```bash
# Build
docker-compose build train-mac

# Run (foreground, see logs)
docker-compose up train-mac

# Run (background)
docker-compose up -d train-mac

# View logs (live)
docker-compose logs -f train-mac

# Stop
docker-compose down
```

### Debugging

```bash
# Interactive shell
docker-compose run --rm train-mac /bin/bash

# Inside container:
python test_dataset.py
python train_mac.py
exit

# Check Docker status
docker ps
docker stats
```

### Cleanup

```bash
# Remove containers
docker-compose down

# Remove containers + volumes
docker-compose down -v

# Remove all images (fresh start)
docker system prune -a
```

---

## 🔧 Troubleshooting Commands

### Check Available Memory

```bash
# macOS
vm_stat | grep "Pages free"

# Docker memory limit
docker info | grep Memory
```

### Increase Docker Memory

```bash
# 1. Open Docker Desktop
open /Applications/Docker.app

# 2. Settings → Resources → Memory → 6-7 GB
# 3. Apply & Restart
```

### Check Training Progress

```bash
# View live logs
docker-compose logs -f train-mac

# Check if model saved
ls -lh outputs/distilled_model_mac/

# Expected files:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer files
```

### Fix Out of Memory

```bash
# Option 1: Reduce sequence length
# Edit train_mac.py CONFIG:
# "max_seq_length": 256  # Was 512

# Option 2: Close other apps
# Free RAM, then retry

# Option 3: Increase Docker memory
# Docker Desktop → Settings → Resources → 7 GB
```

---

## 📊 Testing Commands

### Dataset Validation

```bash
# Quick test
python test_dataset.py

# With split validation
python test_dataset.py --test-splits

# In Docker
docker-compose run --rm test
```

### Model Validation

```bash
# Test model loading (downloads ~2GB first time)
python test_model.py
```

### Complete Test Suite

```bash
# All tests (recommended)
./run_tests.sh
```

---

## 🔍 Inspection Commands

### Check Dataset

```bash
# Count examples
python -c "import json; print(len(json.load(open('data/train_distill.json'))))"

# View first example
python -c "import json; print(json.dumps(json.load(open('data/train_distill.json'))[0], indent=2)[:500])"

# Check splits
ls -lh data/splits/
```

### Check Outputs

```bash
# List saved models
ls -lh outputs/

# Check model size
du -sh outputs/distilled_model_mac/

# View adapter config
cat outputs/distilled_model_mac/adapter_config.json
```

---

## ⚡ One-Liners

```bash
# Full validation pipeline
./run_tests.sh && docker-compose up train-mac

# Clean and rebuild
docker-compose down -v && docker-compose build train-mac

# Check if Docker running
docker ps || echo "Docker not running - open Docker Desktop"

# Free disk space
docker system prune -a -f

# View training speed (tokens/sec)
docker-compose logs train-mac | grep "it/s"

# Check last saved model
ls -lt outputs/ | head -5
```

---

## 🎯 Common Workflows

### Fresh Start

```bash
# Clean everything
docker-compose down -v
docker system prune -a -f

# Rebuild
docker-compose build train-mac

# Test
./run_tests.sh

# Train
docker-compose up train-mac
```

### Quick Validation

```bash
# Just test, don't train
./run_tests.sh
```

### Monitor Running Training

```bash
# Terminal 1: Start training
docker-compose up train-mac

# Terminal 2: Monitor
watch -n 5 'docker stats --no-stream'
```

### Compare Models

```bash
# List all saved models
find outputs -name "adapter_*.safetensors" -ls

# Check sizes
du -sh outputs/*/
```

---

## 📦 Installation Commands (if needed)

### Install Docker

```bash
# Download Docker Desktop for Mac
open https://www.docker.com/products/docker-desktop

# Verify installation
docker --version
docker-compose --version
```

### Install Python Dependencies (for local testing)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS

# Install
pip install -r requirements.mac.txt

# Test
python test_dataset.py
```

---

## Summary Table

| Task | Command | Time |
|------|---------|------|
| **Test data** | `./run_tests.sh` | 30 sec |
| **Build Docker** | `docker-compose build train-mac` | 5 min |
| **Train on Mac** | `docker-compose up train-mac` | 10-30 min |
| **Train on cloud** | `bash vertex_ai_submit.sh` | 6 hours |

Copy these commands and paste them into your terminal!
