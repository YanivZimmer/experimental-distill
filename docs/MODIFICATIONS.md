# Dataset Update: Modifications Required

## Overview

The dataset structure has changed. Instead of having reasoning traces directly in `langfuse_test.json`, we now have two separate files that need to be joined:

1. **langfuse_test.json**: Raw security alerts
2. **baseline_benchmark_flash_gepa_v1.json**: Teacher model (Gemini 2.0 Flash) reasoning and classifications

## Data Structure

### Input Files

**langfuse_test.json:**
```json
[
  {
    "id": "af24e5a7-ac15-5009-967d-f5c4f6ef0a59",
    "status": "ACTIVE",
    "input": {
      "case_id": "...",
      "raw_alert": { ... }
    }
  }
]
```

**baseline_benchmark_flash_gepa_v1.json:**
```json
{
  "metadata": {
    "model": "gemini-2.0-flash-001",
    "test_items": 143,
    "items_processed": 8
  },
  "items": [
    {
      "item_id": "af24e5a7-ac15-5009-967d-f5c4f6ef0a59",
      "output": "{...full JSON reasoning...}",
      "classification": {
        "primary_assessment": "High-Confidence Suspicious",
        "final_decision": "Escalate for Review",
        "severity": "Low",
        "justification": "The combination of a low-confidence ML detection..."
      }
    }
  ]
}
```

## Required Modifications

### 1. Data Preparation (prepare_dataset.py) ✅ UPDATED

**Changes Made:**

1. **Join datasets by ID:**
   - `langfuse_test.json[].id` ↔ `baseline_benchmark[].item_id`

2. **Extract teacher reasoning:**
   - **Option A (Concise):** Use `classification.justification` field
   - **Option B (Rich):** Parse full `output` JSON for complete reasoning chain

3. **Extract classification:**
   - Use `classification.final_decision` + `severity` + `primary_assessment`

4. **Training format:**
   ```python
   {
     "instruction": "<prompt_template> + <alert_data>",
     "output": "<reasoning> + <classification>",
     "metadata": {
       "item_id": "...",
       "teacher_model": "gemini-2.0-flash-001"
     }
   }
   ```

**New Features:**

- `--full-reasoning` flag to use rich structured output
- Automatic validation (reports missing IDs)
- Example preview after processing

### 2. Learning Method (train.py) ⚠️ NO CHANGES NEEDED

**Why no changes?**

The distillation approach remains identical:
- Still step-by-step distillation
- Still learning: `instruction → reasoning → classification`
- `justification` field **IS** the Chain-of-Thought from teacher
- Model architecture unchanged
- Training loop unchanged
- Hyperparameters unchanged

The only difference is **where** the reasoning comes from (separate file vs. embedded), not **how** we learn from it.

### 3. Cloud Training (cloud_train.py) ⚠️ MINOR UPDATE NEEDED

Need to upload **both** files to GCS:

```bash
# Upload both datasets
gsutil cp data/langfuse_test.json gs://${GCS_BUCKET}/data/
gsutil cp data/baseline_benchmark_flash_gepa_v1.json gs://${GCS_BUCKET}/data/
```

The `cloud_train.py` script will call `prepare_dataset.py` which handles the join automatically.

## Usage

### Step 1: Prepare Dataset (NEW)

**Concise reasoning (recommended for speed):**
```bash
python prepare_dataset.py
```

**Full reasoning (recommended for quality):**
```bash
python prepare_dataset.py --full-reasoning
```

### Step 2: Train (UNCHANGED)

```bash
python train.py
```

### Step 3: Deploy (UNCHANGED)

```bash
bash vertex_ai_submit.sh
```

## Comparison: Concise vs. Full Reasoning

### Concise (`classification.justification` only)

**Example:**
```
The combination of a low-confidence ML detection, PowerShell writing an
executable, a suspicious file name, and the unusual parent process (dllhost.exe)
indicates potentially malicious activity. Further investigation is necessary.

Final Classification: Escalate for Review (Assessment: High-Confidence Suspicious, Severity: Low)
```

**Pros:**
- Shorter training examples (faster training)
- Less memory usage
- Clearer, more focused reasoning

**Cons:**
- Less detailed context
- Missing supporting evidence, alternatives, TTPs

### Full Reasoning (`output` JSON parsed)

**Example:**
```
**Event Summary:**
The alert shows PowerShell executing encoded commands...

**Primary Assessment:**
The low-confidence ML detection of PowerShell writing an executable...

**Supporting Evidence:**
- Low-confidence ML detection of file write by PowerShell
- Suspicious file name 'Game Loader All Rh.exe'
- PowerShell command attempts to set location to 'C:\'
- Parent process is 'dllhost.exe'

**Alternative Hypotheses:**
- The detection could be a false positive if legitimate application
- The file could be part of software update process

**Justification:**
The combination of a low-confidence ML detection...

Final Classification: Escalate for Review (Assessment: High-Confidence Suspicious, Severity: Low)
```

**Pros:**
- Richer training signal
- Learns structured reasoning (evidence → hypothesis → decision)
- Better generalization
- Includes MITRE TTPs, timelines, etc.

**Cons:**
- Longer sequences (more VRAM, slower training)
- May need to increase `max_seq_length` to 6144 or 8192

## Recommendation

**Start with concise reasoning:**
- Faster iteration
- Easier to debug
- Lower resource requirements

**Upgrade to full reasoning if:**
- Model quality is insufficient
- You want richer explanations
- You have sufficient GPU memory (40GB+)

## Expected Performance Impact

| Metric | Concise | Full |
|--------|---------|------|
| **Avg sequence length** | ~2500 tokens | ~4500 tokens |
| **Training speed** | Baseline | 60% slower |
| **VRAM usage** | 24GB (L4) | 40GB (A100) |
| **Quality (expected)** | 90% of teacher | 93% of teacher |
| **Reasoning depth** | Good | Excellent |

## Files Modified

- ✅ `prepare_dataset.py`: Complete rewrite to join datasets
- ⚠️ `train.py`: No changes
- ⚠️ `cloud_train.py`: No changes (auto-handles join)
- ⚠️ `vertex_ai_submit.sh`: No changes
- ✅ `MODIFICATIONS.md`: This file (documentation)

## Testing

```bash
# Test data preparation
python prepare_dataset.py

# Expected output:
# Loading datasets...
# Found 41420 langfuse items, 8 benchmark items
# Successfully joined 8 examples
# Saved to data/train_distill.json
#
# === Example Training Item ===
# Instruction length: 8764 chars
# Output length: 523 chars
```

## Migration Checklist

- [x] Update `prepare_dataset.py` with join logic
- [x] Add `--full-reasoning` flag
- [x] Test on sample data (8 items from benchmark file)
- [ ] Run full preparation when complete benchmark available
- [ ] Verify training data quality manually (spot-check)
- [ ] Run training with updated dataset
- [ ] Compare quality vs. previous approach (if applicable)
