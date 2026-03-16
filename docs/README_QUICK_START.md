# Quick Start Guide - SOC Alert Distillation

## 🎯 TL;DR

```bash
# 1. Validate data (30 seconds)
./run_tests.sh

# 2. Validate code on Mac (10-30 minutes, uses 0.5B model)
docker-compose up train-mac

# 3. Train production model on cloud (6 hours, uses 32B model)
bash vertex_ai_submit.sh
```

---

## ⚠️ Critical Info: Mac M5 Pro Limitations

**You have ~8GB available RAM**, which limits what you can train:

| Model | RAM Needed | Mac Compatible? | Purpose |
|-------|------------|-----------------|---------|
| **0.5B** | ~3 GB | ✅ Yes | Code validation only |
| **7B** | ~15 GB | ❌ No | Cloud GPU |
| **32B** | ~60 GB | ❌ No | Cloud GPU (production) |

**Mac can ONLY validate code, NOT train production models.**

---

## 🚀 Complete Workflow

### Phase 1: Validate Dataset (30 seconds)

```bash
./run_tests.sh
```

**What it checks:**
- ✅ Dataset format correct
- ✅ No data leakage between splits
- ✅ Reasoning traces present
- ✅ Docker builds successfully

**Expected output:**
```
✅ ALL TESTS PASSED - SAFE TO TRAIN
```

### Phase 2: Validate Code on Mac (10-30 minutes)

**Why:** Proves your code/data pipeline works before spending $$ on cloud.

```bash
# Build Docker image (one time, ~5 minutes)
docker-compose build train-mac

# Run validation training
docker-compose up train-mac
```

**What happens:**
- Uses **Qwen2.5-0.5B** (tiny model, fits in 8GB RAM)
- Trains for 1 epoch on 6 examples
- Saves to `outputs/distilled_model_mac/`
- **Takes:** 10-30 minutes
- **Purpose:** Validation only (model too small for production)

**See:** `MAC_TRAINING_GUIDE.md` for detailed Mac usage.

### Phase 3: Train Production Model on Cloud (6 hours)

**Why:** 32B model needs 24GB+ VRAM, 10-20x faster than Mac.

```bash
# Upload data to Google Cloud Storage
export GCS_BUCKET=your-distill-bucket
gsutil cp data/splits/*.json gs://${GCS_BUCKET}/data/splits/
gsutil cp baseline.txt gs://${GCS_BUCKET}/

# Submit training job to Vertex AI
bash vertex_ai_submit.sh
```

**What happens:**
- Uses **Qwen2.5-32B** (production model)
- Trains on L4/A100 GPU
- 3 epochs on 1000 examples: ~6 hours, ~$4
- Saves to GCS

---

## 📊 Current Status

### Your Dataset ✅

```
8 examples validated
├── Train: 6 examples
├── Val: 1 example
└── Test: 1 example

✅ No errors
⚠️  Warnings:
  - Small dataset (need 100+ for production)
  - Long examples (4413 tokens avg)
    → Solution: Use max_seq_length=8192
```

### What Works Now

- ✅ All tests pass
- ✅ Data format correct
- ✅ Splits validated (no leakage)
- ✅ Docker configured for Mac
- ✅ Ready for code validation

### What You Need Before Production

- [ ] Full benchmark dataset (100+ examples)
- [ ] Cloud GPU access (Vertex AI / AWS / Azure)
- [ ] GCS bucket for data storage

---

## 🎬 What to Do Right Now

### Option 1: Validate Setup (Recommended)

```bash
# Prove everything works
./run_tests.sh
docker-compose up train-mac

# This takes 10-30 minutes, proves code works
# Then wait for full dataset before cloud training
```

### Option 2: Wait for Full Dataset

```bash
# When you have 100+ examples:
python prepare_dataset.py --full-reasoning
python split_dataset.py --strategy conservative
./run_tests.sh
bash vertex_ai_submit.sh  # Cloud training
```

---

## 📁 Key Files

### Start Here
- **`README_QUICK_START.md`** - This file
- **`MAC_TRAINING_GUIDE.md`** - Mac usage details
- **`DOCKER_USAGE.md`** - Cloud GPU training

### Run These
- **`./run_tests.sh`** - Validate everything (run first!)
- **`docker-compose up train-mac`** - Mac training
- **`bash vertex_ai_submit.sh`** - Cloud training

### Documentation
- `TESTING_SUMMARY.md` - Test details
- `SPLITTING_STRATEGY.md` - Preventing data leakage
- `OPTIMIZATION_SUMMARY.md` - Storage optimization

---

## 💡 Mac vs Cloud Comparison

| Aspect | Mac M5 Pro (8GB) | Cloud GPU (L4) |
|--------|------------------|----------------|
| **Model** | 0.5B (tiny) | 32B (production) |
| **Purpose** | Code validation | Production training |
| **Time** | 10-30 min (6 examples) | 6 hours (1000 examples) |
| **Cost** | Free | ~$4-10 |
| **Quality** | Poor (too small) | Excellent |
| **Use for** | Testing pipeline | Final model |

---

## 🔧 Troubleshooting

### Tests Fail

```bash
# See detailed errors
python test_dataset.py --test-splits

# Fix issues, then re-run
./run_tests.sh
```

### Mac Training Out of Memory

```bash
# 1. Increase Docker memory
# Docker Desktop → Settings → Resources → Memory → 7 GB

# 2. Close other apps to free RAM

# 3. If still fails, reduce sequence length:
# Edit train_mac.py CONFIG: "max_seq_length": 256
```

### Mac Training Very Slow

**Expected:** 5-10 seconds per example (0.5B model)

**If much slower:** Likely using CPU instead of MPS (GPU). This is okay for validation, just takes longer.

---

## ✅ Success Checklist

Before cloud training:
- [x] Tests pass: `./run_tests.sh`
- [x] Mac validation works: `docker-compose up train-mac`
- [ ] Full dataset available (100+ examples)
- [ ] Cloud GPU configured
- [ ] GCS bucket created

---

## 🚀 Next Steps

**Right now:**
```bash
./run_tests.sh
docker-compose up train-mac
```

**When you have full dataset:**
```bash
python prepare_dataset.py --full-reasoning
python split_dataset.py --strategy conservative
./run_tests.sh
bash vertex_ai_submit.sh
```

---

## Summary

**Your Mac M5 Pro (8GB RAM):**
- ✅ Can validate code works (0.5B model)
- ❌ Cannot train production model (too small)
- ⏱️ Takes 10-30 minutes for validation

**Cloud GPU (L4/A100):**
- ✅ Trains production model (32B)
- ⏱️ Takes 6 hours for 1000 examples
- 💰 Costs ~$4-10 per run

**Recommendation:** Validate on Mac (free, 30 min) → Train on cloud (fast, cheap).
