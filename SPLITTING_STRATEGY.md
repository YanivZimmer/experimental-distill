# Dataset Splitting Strategy: Preventing Data Leakage

## Problem: Why Naive Random Split Fails

Security alerts contain **multiple dimensions of similarity** that cause data leakage if split randomly:

### Similarity Dimensions

| Dimension | Example | Leakage Risk |
|-----------|---------|--------------|
| **File Hash** | Same malware on 10 hosts | HIGH - Model memorizes hash, not behavior |
| **Hostname** | 5 alerts from "LAPTOP-XYZ" | HIGH - Model memorizes host patterns |
| **Alert Type** | 20 "MLSensor-Lowest" alerts | MEDIUM - Model learns alert-specific patterns |
| **User** | Admin "john.doe" in 8 alerts | MEDIUM - Model learns user behavior |
| **Process Chain** | PowerShell → encoded command | LOW - Legitimate pattern to learn |
| **Temporal** | Incident campaign (3 days) | MEDIUM - Related attack sequence |

### Example Leakage Scenario

**Naive random split:**
```
Train:
  - Alert 1: Host "LAPTOP-A", hash "abc123", "MLSensor-High"
  - Alert 2: Host "LAPTOP-B", hash "def456", "SuspiciousPrivEsc"

Test:
  - Alert 3: Host "LAPTOP-A", hash "abc123", "MLSensor-High" ← LEAKAGE!
```

Model learns: "LAPTOP-A + abc123 = malicious" instead of general threat patterns.

**Result:** Inflated test accuracy (95%) that collapses in production (65%).

---

## Solution: Similarity-Aware Grouping

### Strategy 1: Strict (Most Conservative)

**Groups by:** Hash **OR** Hostname **OR** Alert Type

**Logic:**
```python
if same_hash OR same_host OR same_alert_type:
    keep_in_same_split()
```

**Pros:**
- Zero leakage across all dimensions
- Best generalization to new hosts/hashes/alert types

**Cons:**
- Smallest effective dataset (groups are large)
- May not have enough diversity in small datasets (<100 examples)
- Alert types never overlap between splits

**Use when:**
- Dataset is large (>500 examples)
- Production will see completely new hosts/hashes
- Maximum generalization is critical

### Strategy 2: Conservative (Recommended)

**Groups by:** Hash **OR** (Hostname + Alert Type)

**Logic:**
```python
if same_hash:
    keep_in_same_split()
elif same_host AND same_alert_type:
    keep_in_same_split()
else:
    can_split()
```

**Pros:**
- Prevents hash leakage (strongest signal)
- Allows same host in train/test with different alert types
- Alert types can overlap (model learns alert-agnostic patterns)
- Good balance of safety and dataset size

**Cons:**
- Same host may appear in train/test (if different alerts)
- Requires sufficient diversity in alert types

**Use when:**
- Dataset is medium (100-500 examples)
- You want to generalize to new hashes but may see same hosts
- Recommended default strategy

### Strategy 3: Moderate

**Groups by:** Hash only

**Logic:**
```python
if same_hash:
    keep_in_same_split()
else:
    can_split()  # Even if same host or alert type
```

**Pros:**
- Prevents strongest leakage (file hash memorization)
- Larger effective dataset
- Same host/alert can appear across splits

**Cons:**
- Host patterns may leak
- Alert type patterns may leak

**Use when:**
- Dataset is small (<100 examples)
- File hash is the primary signal of interest
- Hosts/users are expected to repeat in production

### Strategy 4: Label-Only (Stratified)

**Groups by:** Expected label (benign/malicious) for stratification only

**Logic:**
```python
stratify_by_label()  # Standard stratified split
```

**Pros:**
- Maximum dataset utilization
- Balanced label distribution

**Cons:**
- **High leakage risk** - same hash/host/alert can appear in train/test
- Only use for baseline comparison

**Use when:**
- You want to measure worst-case leakage
- Baseline comparison only (NOT for final model)

---

## Implementation

### Basic Usage

```bash
# Conservative (recommended)
python split_dataset.py --strategy conservative

# Strict (maximum safety)
python split_dataset.py --strategy strict

# Moderate (small datasets)
python split_dataset.py --strategy moderate

# Custom ratios
python split_dataset.py --strategy conservative --train 0.8 --val 0.1 --test 0.1
```

### Validation Output

The script validates splits and reports leakage:

```
Creating similarity groups...
Created 45 groups
Group sizes: min=1, max=8, avg=3.2
Largest groups: [8, 6, 5, 4, 3]

Splitting into train/val/test...
Train: 98 examples (70.0%)
Val:   21 examples (15.0%)
Test:  21 examples (15.0%)

Validating splits...
✅ No leakage detected!
ℹ️  12 alert types appear in both train and test (expected)

✅ Dataset split complete!
```

### Leakage Warnings

If leakage detected:

```
⚠️  3 file hashes appear in both train and test
⚠️  5 hosts appear in both train and val
ℹ️  15 alert types appear in both train and test (expected)
```

**Action:** Use stricter strategy or manually review groups.

---

## Advanced: Semantic Similarity (Optional)

For very large datasets (>1000 examples), consider **embedding-based clustering**:

### Approach

1. Encode alerts using embedding model (e.g., sentence-transformers)
2. Cluster similar alerts (K-means, DBSCAN, HDBSCAN)
3. Assign entire clusters to splits

### Example (pseudo-code)

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

# Encode alert descriptions
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([alert["event_summary"] for alert in alerts])

# Cluster
clusterer = HDBSCAN(min_cluster_size=3)
cluster_labels = clusterer.fit_predict(embeddings)

# Group by cluster
groups = defaultdict(list)
for i, label in enumerate(cluster_labels):
    groups[f"cluster_{label}"].append(i)

# Split groups (same as before)
train_idx, val_idx, test_idx = stratified_split(groups)
```

**Pros:**
- Captures semantic similarity beyond exact features
- Finds "similar incidents" automatically

**Cons:**
- Slower (requires embedding model)
- May over-cluster (splits become very strict)
- Requires tuning clustering parameters

**Implementation:** Can be added to `split_dataset.py` if needed.

---

## Strategy Selection Guide

| Dataset Size | Hash Diversity | Host Diversity | Recommended Strategy | Rationale |
|--------------|----------------|----------------|---------------------|-----------|
| <50 | Low | Low | Moderate | Need maximum data, hash is key signal |
| 50-100 | Medium | Low | Moderate | Hash-only grouping sufficient |
| 100-300 | High | Medium | Conservative | Balance safety and size |
| 300-500 | High | High | Conservative | Standard best practice |
| >500 | High | High | Strict | Can afford strictest grouping |

### Special Cases

**Same malware campaign:**
- Multiple alerts from same incident
- Use Strict or add temporal grouping

**Penetration testing data:**
- Controlled environment, same tools reused
- Use Strict or group by test_id

**Mixed benign/malicious:**
- Heavily imbalanced classes
- Use Conservative + ensure label stratification

---

## Integration with Training

### Update `train.py`

The updated `train.py` already supports splits:

```python
CONFIG = {
    ...
    "use_splits": True,  # Enable split mode
}
```

### Workflow

```bash
# Step 1: Prepare full dataset
python prepare_dataset.py --full-reasoning

# Step 2: Split dataset
python split_dataset.py --strategy conservative

# Step 3: Train with splits
python train.py  # Reads from data/splits/
```

### Output Structure

```
data/splits/
├── train.json          # 70% of data
├── val.json            # 15% of data
├── test.json           # 15% of data
└── split_metadata.json # Split statistics
```

---

## Validation Checklist

Before training, verify:

- [ ] No file hashes overlap between train/test
- [ ] Minimal or no hostname overlap
- [ ] Alert type overlap is acceptable (expected)
- [ ] Label distribution is balanced across splits
- [ ] Group sizes are reasonable (not too large)
- [ ] Validation warnings are reviewed and understood

---

## Future Enhancements

1. **Temporal awareness:** Group alerts within N-day windows
2. **Campaign detection:** Auto-detect related incidents
3. **Cross-validation:** K-fold with group preservation
4. **Active learning:** Iteratively select most informative examples
5. **Embedding-based clustering:** Semantic similarity grouping

---

## References

- **Time Series Cross-Validation:** https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
- **Group K-Fold:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- **Data Leakage in ML:** https://machinelearningmastery.com/data-leakage-machine-learning/

## Summary

**Use `conservative` strategy by default.** It prevents critical leakage (file hashes) while allowing reasonable dataset utilization. Adjust based on your dataset size and diversity.
