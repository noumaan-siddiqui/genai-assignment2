"""
Data splitting and manifest generation for TrialGPT Criterion Annotations dataset.
Strictly implements the assignment protocol:
- Sort unique patient IDs.
- Shuffle using NumPy default_rng(42).
- Assign first 37 patients to training, next 8 to validation, remaining 8 to testing.
- Verify zero patient leakage between splits.
- Save split manifest and partitioned jsonl datasets.
- Ensure all GPT-4 predictions and explanations are excluded from model inputs.
"""

import json
import os
import sys
import ast
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from datasets import load_dataset


def parse_sentences(sentence_val):
    """Safely parse sentence list representation into list of ints."""
    if sentence_val is None or pd.isna(sentence_val):
        return []
    if isinstance(sentence_val, list):
        return [int(x) for x in sentence_val]
    if isinstance(sentence_val, str):
        try:
            parsed = ast.literal_eval(sentence_val.strip())
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except Exception:
            pass
    return []


def create_splits(output_dir="data", seed=42):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading dataset 'ncbi/TrialGPT-Criterion-Annotations'...")
    ds = load_dataset("ncbi/TrialGPT-Criterion-Annotations", split="train")
    df = ds.to_pandas()

    # Step 1: Sort unique patient IDs
    unique_pids = sorted(df["patient_id"].unique().tolist())
    print(f"Total unique patients found: {len(unique_pids)} (Total rows: {len(df)})")
    assert len(unique_pids) == 53, f"Expected 53 patients, found {len(unique_pids)}"

    # Step 2: Shuffle using NumPy default_rng(seed)
    rng = np.random.default_rng(seed)
    shuffled_pids = list(unique_pids)
    rng.shuffle(shuffled_pids)

    # Step 3: Assign 37 train, 8 val, 8 test
    train_pids = shuffled_pids[:37]
    val_pids = shuffled_pids[37:45]
    test_pids = shuffled_pids[45:53]

    # Step 4: Strict patient leakage verification
    assert len(set(train_pids) & set(val_pids)) == 0, "Patient leakage between train and val!"
    assert len(set(train_pids) & set(test_pids)) == 0, "Patient leakage between train and test!"
    assert len(set(val_pids) & set(test_pids)) == 0, "Patient leakage between val and test!"
    assert len(train_pids) + len(val_pids) + len(test_pids) == 53, "Split patient counts sum mismatch!"

    train_df = df[df["patient_id"].isin(train_pids)].copy()
    val_df = df[df["patient_id"].isin(val_pids)].copy()
    test_df = df[df["patient_id"].isin(test_pids)].copy()

    manifest = {
        "dataset_name": "ncbi/TrialGPT-Criterion-Annotations",
        "random_seed": seed,
        "patient_split_protocol": "Sorted unique patient IDs, shuffled with np.random.default_rng(42); 37 train / 8 val / 8 test",
        "patient_counts": {
            "total_patients": len(unique_pids),
            "train_patients": len(train_pids),
            "val_patients": len(val_pids),
            "test_patients": len(test_pids),
        },
        "annotation_counts": {
            "total_annotations": len(df),
            "train_annotations": len(train_df),
            "val_annotations": len(val_df),
            "test_annotations": len(test_df),
        },
        "patient_ids": {
            "train": train_pids,
            "val": val_pids,
            "test": test_pids,
        },
        "label_distribution": {
            "train": train_df["expert_eligibility"].value_counts().to_dict(),
            "val": val_df["expert_eligibility"].value_counts().to_dict(),
            "test": test_df["expert_eligibility"].value_counts().to_dict(),
        },
        "leakage_checks_passed": True
    }

    manifest_path = os.path.join(output_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved split manifest to {manifest_path}")

    # Process each partition to clean records
    # Crucial: GPT-4 predictions and explanations are strictly excluded from input
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        cleaned_records = []
        for _, row in split_df.iterrows():
            record = {
                "annotation_id": int(row["annotation_id"]),
                "patient_id": str(row["patient_id"]),
                "note": str(row["note"]),
                "trial_id": str(row["trial_id"]),
                "trial_title": str(row.get("trial_title", "")),
                "criterion_type": str(row["criterion_type"]),  # 'inclusion' or 'exclusion'
                "criterion_text": str(row["criterion_text"]),
                "expert_eligibility": str(row["expert_eligibility"]),
                "expert_sentences": parse_sentences(row.get("expert_sentences")),
                # Keep gold references for evaluation only (NEVER input to model)
            }
            cleaned_records.append(record)

        split_file = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(split_file, "w", encoding="utf-8") as f:
            for r in cleaned_records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(cleaned_records)} records to {split_file}")

    return manifest


if __name__ == "__main__":
    create_splits()
