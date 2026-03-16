"""
Split dataset into train/val/test with similarity-aware deduplication.
Updated to work with new efficient format (alert-only storage).
"""
import json
from collections import defaultdict
from pathlib import Path
import random

def extract_similarity_features(item):
    """Extract features that indicate alert similarity."""
    alert_data = item["alert"]
    metadata = item.get("metadata", {})

    features = {
        "item_id": metadata.get("item_id"),
        "expected_label": metadata.get("expected_label"),
    }

    # Extract from alert data
    raw_alert = alert_data.get("raw_alert", {})

    features["alert_name"] = raw_alert.get("name", "")
    features["alert_type"] = raw_alert.get("type", "")

    device = raw_alert.get("device", {})
    features["hostname"] = device.get("hostname", "")
    features["device_id"] = device.get("device_id", "")

    features["file_sha256"] = raw_alert.get("sha256", "")
    features["file_md5"] = raw_alert.get("md5", "")

    features["user_name"] = raw_alert.get("user_name", "")
    features["process_name"] = raw_alert.get("file_name", "")
    features["parent_process"] = raw_alert.get("parent_process_name", "")

    return features

def create_similarity_groups(data, strategy="conservative"):
    """Group similar alerts that should not split across train/test."""

    if strategy == "strict":
        return _group_strict(data)
    elif strategy == "conservative":
        return _group_conservative(data)
    elif strategy == "moderate":
        return _group_moderate(data)
    elif strategy == "label_only":
        return _group_by_label(data)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def _group_strict(data):
    """Group by hash OR hostname OR alert_type."""
    groups = defaultdict(list)

    for i, item in enumerate(data):
        features = extract_similarity_features(item)

        key_parts = []
        if features["file_sha256"]:
            key_parts.append(f"hash:{features['file_sha256']}")
        if features["hostname"]:
            key_parts.append(f"host:{features['hostname']}")
        if features["alert_name"]:
            key_parts.append(f"alert:{features['alert_name']}")

        key = key_parts[0] if key_parts else f"unique:{features['item_id']}"
        groups[key].append(i)

    return groups

def _group_conservative(data):
    """Group by hash OR (hostname + alert_type)."""
    groups = defaultdict(list)

    for i, item in enumerate(data):
        features = extract_similarity_features(item)

        if features["file_sha256"]:
            key = f"hash:{features['file_sha256']}"
        elif features["hostname"] and features["alert_name"]:
            key = f"host_alert:{features['hostname']}:{features['alert_name']}"
        elif features["hostname"]:
            key = f"host:{features['hostname']}"
        else:
            key = f"unique:{features['item_id']}"

        groups[key].append(i)

    return groups

def _group_moderate(data):
    """Group only by file hash."""
    groups = defaultdict(list)

    for i, item in enumerate(data):
        features = extract_similarity_features(item)

        if features["file_sha256"]:
            key = f"hash:{features['file_sha256']}"
        else:
            key = f"unique:{features['item_id']}"

        groups[key].append(i)

    return groups

def _group_by_label(data):
    """Group only by expected label (weak)."""
    groups = defaultdict(list)

    for i, item in enumerate(data):
        features = extract_similarity_features(item)
        label = features["expected_label"] or "unknown"
        groups[f"label:{label}"].append(i)

    return groups

def stratified_split(groups, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split groups into train/val/test."""
    random.seed(seed)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    total_items = sum(len(groups[k]) for k in group_keys)
    train_target = int(total_items * train_ratio)
    val_target = int(total_items * val_ratio)

    train_indices = []
    val_indices = []
    test_indices = []

    train_count = 0
    val_count = 0

    for group_key in group_keys:
        indices = groups[group_key]

        if train_count < train_target:
            train_indices.extend(indices)
            train_count += len(indices)
        elif val_count < val_target:
            val_indices.extend(indices)
            val_count += len(indices)
        else:
            test_indices.extend(indices)

    return train_indices, val_indices, test_indices

def validate_no_leakage(data, train_idx, val_idx, test_idx):
    """Validate no feature overlap between splits."""
    warnings = []

    def get_features(indices):
        features = {
            "hashes": set(),
            "hosts": set(),
            "alert_types": set(),
            "users": set()
        }
        for i in indices:
            f = extract_similarity_features(data[i])
            if f["file_sha256"]:
                features["hashes"].add(f["file_sha256"])
            if f["hostname"]:
                features["hosts"].add(f["hostname"])
            if f["alert_name"]:
                features["alert_types"].add(f["alert_name"])
            if f["user_name"]:
                features["users"].add(f["user_name"])
        return features

    train_features = get_features(train_idx)
    val_features = get_features(val_idx)
    test_features = get_features(test_idx)

    hash_leak_val = train_features["hashes"] & val_features["hashes"]
    hash_leak_test = train_features["hashes"] & test_features["hashes"]
    if hash_leak_val:
        warnings.append(f"⚠️  {len(hash_leak_val)} file hashes in both train and val")
    if hash_leak_test:
        warnings.append(f"⚠️  {len(hash_leak_test)} file hashes in both train and test")

    host_leak_val = train_features["hosts"] & val_features["hosts"]
    host_leak_test = train_features["hosts"] & test_features["hosts"]
    if host_leak_val:
        warnings.append(f"⚠️  {len(host_leak_val)} hosts in both train and val")
    if host_leak_test:
        warnings.append(f"⚠️  {len(host_leak_test)} hosts in both train and test")

    return warnings

def split_and_save(
    input_path="data/train_distill.json",
    output_dir="data/splits",
    strategy="conservative",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """Main function to split and save dataset."""

    print(f"Loading dataset from {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    print(f"Total examples: {len(data)}")
    print(f"Strategy: {strategy}")

    print(f"\nCreating similarity groups...")
    groups = create_similarity_groups(data, strategy=strategy)
    print(f"Created {len(groups)} groups")

    group_sizes = sorted([len(g) for g in groups.values()], reverse=True)
    print(f"Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, avg={sum(group_sizes)/len(group_sizes):.1f}")

    print(f"\nSplitting into train/val/test...")
    train_idx, val_idx, test_idx = stratified_split(
        groups, train_ratio, val_ratio, test_ratio, seed
    )

    print(f"Train: {len(train_idx)} examples ({len(train_idx)/len(data)*100:.1f}%)")
    print(f"Val:   {len(val_idx)} examples ({len(val_idx)/len(data)*100:.1f}%)")
    print(f"Test:  {len(test_idx)} examples ({len(test_idx)/len(data)*100:.1f}%)")

    print(f"\nValidating splits...")
    warnings = validate_no_leakage(data, train_idx, val_idx, test_idx)
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("✅ No leakage detected!")

    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    splits = {
        "train": [data[i] for i in train_idx],
        "val": [data[i] for i in val_idx],
        "test": [data[i] for i in test_idx]
    }

    for split_name, split_data in splits.items():
        output_path = output_dir / f"{split_name}.json"
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name}: {output_path}")

    print(f"\n✅ Dataset split complete!")
    return splits

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train_distill.json")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--strategy", choices=["strict", "conservative", "moderate", "label_only"],
                       default="conservative")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    split_and_save(
        input_path=args.input,
        output_dir=args.output_dir,
        strategy=args.strategy,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
