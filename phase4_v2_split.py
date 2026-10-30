#!/usr/bin/env python3
"""Generate and validate the additive Phase 4 v2 data protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


EXPECTED_FINGERPRINT = "e0f6962d602757cb"
EXPECTED_SOURCE_SIZE = 67349
LEGACY_PROBE_SEED = 42
LEGACY_PROBE_SIZE = 512
LEGACY_PROBE_ALGORITHM = "sorted(random.Random(seed).sample(range(source_size), probe_size))"


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_payload(payload):
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def reconstruct_legacy_probe_indices(source_size=EXPECTED_SOURCE_SIZE, seed=LEGACY_PROBE_SEED,
                                     probe_size=LEGACY_PROBE_SIZE):
    if source_size < probe_size:
        raise ValueError("source split is smaller than the legacy probe")
    return sorted(random.Random(seed).sample(range(source_size), probe_size))


def _validate_legacy_loader_procedure(source_code):
    required = (
        "random.Random(probe_seed)",
        "rng.sample(range(len(encoded[\"train\"])), probe_size)",
        "encoded[\"train\"].select(indices)",
    )
    missing = [fragment for fragment in required if fragment not in source_code]
    if missing:
        raise ValueError(f"legacy probe implementation differs; missing fragments={missing}")


def load_legacy_source_mapping(path=Path("phase4_split.json"), expected_fingerprint=EXPECTED_FINGERPRINT,
                               expected_source_size=EXPECTED_SOURCE_SIZE):
    payload = json.loads(Path(path).read_text())
    if payload.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError("legacy split fingerprint does not match expected source")
    if payload.get("source_size") != expected_source_size:
        raise ValueError("legacy split source size does not match expected source")
    all_indices = list(payload.get("train_indices", [])) + list(payload.get("holdout_indices", []))
    all_ids = list(payload.get("train_ids", [])) + list(payload.get("holdout_ids", []))
    if len(all_indices) != expected_source_size or len(all_ids) != expected_source_size:
        raise ValueError("legacy split does not provide a complete source index/ID mapping")
    if sorted(all_indices) != list(range(expected_source_size)):
        raise ValueError("legacy split indices do not cover the source exactly")
    if len(set(all_ids)) != expected_source_size:
        raise ValueError("legacy split example IDs are missing or duplicated")
    index_to_id = dict(zip(all_indices, all_ids))
    index_to_label = {}
    for index, label in zip(payload["holdout_indices"], payload["holdout_labels"]):
        index_to_label[index] = int(label)
    if set(payload.get("holdout_indices", [])) & set(payload.get("train_indices", [])):
        raise ValueError("legacy train and holdout indices overlap")
    return payload, index_to_id, index_to_label


def verify_legacy_probe_reconstruction(source_code, split_payload, index_to_id):
    _validate_legacy_loader_procedure(source_code)
    indices = reconstruct_legacy_probe_indices(split_payload["source_size"])
    ids = [index_to_id[index] for index in indices]
    if len(ids) != LEGACY_PROBE_SIZE or len(set(ids)) != LEGACY_PROBE_SIZE:
        raise ValueError("reconstructed legacy probe IDs are invalid")
    return {"indices": indices, "ids": ids, "seed": LEGACY_PROBE_SEED,
            "size": LEGACY_PROBE_SIZE, "algorithm": LEGACY_PROBE_ALGORITHM,
            "standalone_manifest_available": False}


def _largest_remainder_counts(indices, labels, target_size):
    if target_size < 0 or target_size > len(indices):
        raise ValueError("invalid stratified target size")
    by_label = {}
    for index in indices:
        by_label.setdefault(int(labels[index]), []).append(index)
    total = len(indices)
    allocations = {label: int(len(group) * target_size / total) for label, group in by_label.items()}
    remainder = target_size - sum(allocations.values())
    ranked = sorted(
        by_label,
        key=lambda label: (-(len(by_label[label]) * target_size / total - allocations[label]), label),
    )
    for label in ranked[:remainder]:
        allocations[label] += 1
    return by_label, allocations


def _stratified_take(indices, labels, target_size, seed):
    by_label, allocations = _largest_remainder_counts(indices, labels, target_size)
    rng = random.Random(seed)
    selected = []
    for label in sorted(by_label):
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        selected.extend(candidates[:allocations[label]])
    return sorted(selected)


def _class_counts(indices, labels):
    counts = {}
    for index in indices:
        counts[str(int(labels[index]))] = counts.get(str(int(labels[index])), 0) + 1
    return dict(sorted(counts.items()))


def _assert_disjoint(pools):
    names = list(pools)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            overlap = set(pools[left_name]) & set(pools[right_name])
            if overlap:
                raise ValueError(f"split overlap: {left_name} and {right_name}: {sorted(overlap)[:5]}")


def create_v2_split(labels, ids, dataset_fingerprint, legacy_payload, legacy_probe,
                    train_size=47640, validation_size=6000, probe_size=512, final_test_size=6000,
                    allocation_seed=20261029, validation_seed=20261030,
                    probe_seed=20261031, test_seed=20261032):
    if dataset_fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError("dataset fingerprint differs from expected source")
    if len(labels) != EXPECTED_SOURCE_SIZE or len(ids) != EXPECTED_SOURCE_SIZE:
        raise ValueError("dataset source size differs from expected source")
    if len(set(ids)) != len(ids):
        raise ValueError("dataset example IDs are missing or duplicated")
    quarantine = sorted(set(legacy_payload["holdout_indices"]) | set(legacy_probe["indices"]))
    quarantine_set = set(quarantine)
    eligible = [i for i in range(len(labels)) if i not in quarantine_set]
    requested = train_size + validation_size + probe_size + final_test_size
    if requested != len(eligible):
        raise ValueError(f"split sizes total {requested}, eligible pool is {len(eligible)}")
    final_test = _stratified_take(eligible, labels, final_test_size, test_seed)
    final_test_set = set(final_test)
    remaining = [i for i in eligible if i not in final_test_set]
    validation = _stratified_take(remaining, labels, validation_size, validation_seed)
    validation_set = set(validation)
    remaining = [i for i in remaining if i not in validation_set]
    selection_probe = _stratified_take(remaining, labels, probe_size, probe_seed)
    selection_probe_set = set(selection_probe)
    train = [i for i in remaining if i not in selection_probe_set]
    pools = {"train": sorted(train), "validation": validation,
             "selection_probe": selection_probe, "final_test": final_test}
    _assert_disjoint(pools)
    if set().union(*map(set, pools.values())) != set(eligible):
        raise ValueError("new pools do not partition the eligible source")
    if set().union(*map(set, pools.values())) & set(quarantine):
        raise ValueError("quarantined example entered a v2 pool")
    expected_sizes = {"train": train_size, "validation": validation_size,
                      "selection_probe": probe_size, "final_test": final_test_size}
    for name, expected in expected_sizes.items():
        if len(pools[name]) != expected:
            raise ValueError(f"{name} has size {len(pools[name])}, expected {expected}")
    payload = {
        "protocol_version": "phase4_v2",
        "dataset": "nyu-mll/glue/sst2",
        "source_split": "train",
        "dataset_fingerprint": dataset_fingerprint,
        "source_size": len(labels),
        "legacy_quarantine": {
            "holdout_size": len(legacy_payload["holdout_indices"]),
            "probe_size": len(legacy_probe["indices"]),
            "union_size": len(quarantine),
            "holdout_indices": legacy_payload["holdout_indices"],
            "holdout_ids": legacy_payload["holdout_ids"],
            "probe_indices": legacy_probe["indices"],
            "probe_ids": legacy_probe["ids"],
            "probe_algorithm": legacy_probe["algorithm"],
            "probe_standalone_manifest_available": False,
        },
        "allocation": {
            "allocation_seed": allocation_seed,
            "validation_seed": validation_seed,
            "probe_seed": probe_seed,
            "test_seed": test_seed,
            "algorithm": "largest-remainder stratified sequential sampling",
        },
        "pools": {
            name: {
                "size": len(indices),
                "indices": indices,
                "ids": [ids[i] for i in indices],
                "labels": [int(labels[i]) for i in indices],
                "class_counts": _class_counts(indices, labels),
            } for name, indices in pools.items()
        },
    }
    payload["pool_hashes"] = {name: sha256_payload(data) for name, data in payload["pools"].items()}
    payload["manifest_sha256"] = sha256_payload(payload)
    return payload


def load_dataset_source(dataset_name="nyu-mll/glue", dataset_config="sst2"):
    from datasets import load_dataset
    raw = load_dataset(dataset_name, dataset_config)
    train = raw["train"]
    return list(train["label"]), list(train["idx"]), getattr(train, "_fingerprint", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("phase4_v2_split.json"))
    parser.add_argument("--legacy-split", type=Path, default=Path("phase4_split.json"))
    parser.add_argument("--legacy-source", type=Path, default=Path("roberta/data.py"))
    args = parser.parse_args()
    labels, ids, fingerprint = load_dataset_source()
    legacy_payload, index_to_id, _ = load_legacy_source_mapping(args.legacy_split)
    legacy_probe = verify_legacy_probe_reconstruction(args.legacy_source.read_text(), legacy_payload, index_to_id)
    payload = create_v2_split(labels, ids, fingerprint, legacy_payload, legacy_probe)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "manifest_sha256": payload["manifest_sha256"],
                      "quarantine_size": payload["legacy_quarantine"]["union_size"],
                      "pool_sizes": {name: data["size"] for name, data in payload["pools"].items()}}, indent=2))


if __name__ == "__main__":
    main()
