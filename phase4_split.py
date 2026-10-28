#!/usr/bin/env python3
"""Create and validate the frozen Phase 4 stratified holdout."""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def create_split(dataset_name="nyu-mll/glue", dataset_config="sst2", seed=20261028, fraction=0.1):
    raw = load_dataset(dataset_name, dataset_config)
    train = raw["train"]
    labels = list(train["label"])
    ids = list(train["idx"])
    target = round(len(labels) * fraction)
    by_label = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(index)
    allocations = {}
    remainders = []
    for label, indices in by_label.items():
        exact = len(indices) * fraction
        allocations[label] = int(exact)
        remainders.append((exact - int(exact), label))
    for _, label in sorted(remainders, reverse=True)[: target - sum(allocations.values())]:
        allocations[label] += 1
    rng = random.Random(seed)
    holdout_indices = []
    for label in sorted(by_label):
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        holdout_indices.extend(candidates[:allocations[label]])
    holdout_indices.sort()
    holdout_set = set(holdout_indices)
    train_indices = [i for i in range(len(labels)) if i not in holdout_set]
    payload = {
        "dataset": f"{dataset_name}/{dataset_config}",
        "source_split": "train",
        "dataset_fingerprint": getattr(train, "_fingerprint", None),
        "split_seed": seed,
        "holdout_fraction_requested": fraction,
        "source_size": len(labels),
        "holdout_size": len(holdout_indices),
        "train_size": len(train_indices),
        "holdout_indices": holdout_indices,
        "train_indices": train_indices,
        "holdout_ids": [ids[i] for i in holdout_indices],
        "train_ids": [ids[i] for i in train_indices],
        "holdout_labels": [labels[i] for i in holdout_indices],
        "train_label_counts": {str(label): sum(labels[i] == label for i in train_indices) for label in sorted(by_label)},
        "holdout_label_counts": {str(label): sum(labels[i] == label for i in holdout_indices) for label in sorted(by_label)},
    }
    validate_split(payload, labels, ids)
    return payload


def validate_split(payload, labels=None, ids=None):
    train_indices = set(payload["train_indices"])
    holdout_indices = set(payload["holdout_indices"])
    source_size = payload["source_size"]
    if train_indices & holdout_indices:
        raise ValueError("train and holdout indices overlap")
    if train_indices | holdout_indices != set(range(source_size)):
        raise ValueError("train and holdout indices do not partition the source split")
    if len(payload["holdout_ids"]) != len(holdout_indices) or len(set(payload["holdout_ids"])) != len(holdout_indices):
        raise ValueError("holdout IDs are missing or duplicated")
    if labels is not None and ids is not None:
        if [ids[i] for i in payload["holdout_indices"]] != payload["holdout_ids"]:
            raise ValueError("holdout IDs do not match source indices")
        if [labels[i] for i in payload["holdout_indices"]] != payload["holdout_labels"]:
            raise ValueError("holdout labels do not match source indices")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("phase4_split.json"))
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--fraction", type=float, default=0.1)
    args = parser.parse_args()
    payload = create_split(seed=args.seed, fraction=args.fraction)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}: train={payload['train_size']} holdout={payload['holdout_size']} fingerprint={payload['dataset_fingerprint']}")


if __name__ == "__main__":
    main()
