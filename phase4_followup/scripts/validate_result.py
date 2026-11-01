#!/usr/bin/env python3
"""Strict validator for completed Phase 4 v2 follow-up artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase4_v2_io import run_paths

EXPECTED_ADAPTER = 73728
EXPECTED_TRAINABLE = 665858
EXPECTED_FINGERPRINT = "e0f6962d602757cb"
EXPECTED_SPLIT = "bb3db736dead8440aa74c4863e61a1646ddf4f5fc3c7c5d5da60813ad70a3fae"


def read(path):
    return json.loads(Path(path).read_text())


def validate_run(root, expected, manifest):
    run_id = expected["run_id"]
    paths = run_paths(root, run_id)
    for path in (paths["status"], paths["result"], paths["predictions"], paths["complete"]):
        if not path.exists():
            raise ValueError(f"missing artifact: {path}")
    status, result, predictions, complete = map(read, (paths["status"], paths["result"], paths["predictions"], paths["complete"]))
    if status.get("status") != "completed" or complete.get("status") != "completed" or result.get("status") != "artifact_ready":
        raise ValueError("run is not completed")
    if any(payload.get("run_id") != run_id for payload in (status, result, predictions, complete)):
        raise ValueError("run ID mismatch")
    cfg = result.get("configuration", {})
    checks = {
        "configuration name": cfg.get("id") == expected["configuration_id"],
        "seed": result.get("seed") == expected["seed"],
        "layer indices": cfg.get("layers") == expected["layers"],
        "human-readable layers": cfg.get("reported_layers") == expected["reported_layers"],
        "rank": cfg.get("rank") == expected["rank"],
        "alpha": cfg.get("alpha") == expected["alpha"],
        "adapter parameters": result.get("n_lora_adapter_params") == EXPECTED_ADAPTER,
        "total trainable parameters": result.get("n_trainable_params") == EXPECTED_TRAINABLE,
        "dataset fingerprint": result.get("dataset_fingerprint") == EXPECTED_FINGERPRINT,
        "split hash": result.get("split_manifest_sha256") == EXPECTED_SPLIT,
        "manifest hash": result.get("manifest_sha256") == manifest.get("manifest_sha256"),
        "validation size": result.get("validation_metrics", {}).get("n") == 6000,
        "prediction count": len(predictions.get("predictions", [])) == 6000 and len(predictions.get("labels", [])) == 6000,
        "validation-only": result.get("evaluation_mode") == "validation_only" and not any("test" in key.lower() for key in result),
        "environment metadata": all(key in result.get("runtime_metadata", {}) for key in
                                     ("python", "software", "cuda_version", "gpu_name", "device", "precision_dtype")),
        "configuration hash": cfg.get("configuration_sha256") == expected.get("configuration_sha256"),
        "finite accuracy": isinstance(result.get("validation_metrics", {}).get("accuracy"), (int, float)) and
                           result["validation_metrics"]["accuracy"] == result["validation_metrics"]["accuracy"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("; ".join(failed))
    expected_targets = [f"base_model.model.roberta.encoder.layer.{layer}.attention.self.{target}"
                        for layer in expected["layers"] for target in ("query", "value")]
    if sorted(result.get("resolved_lora_target_names", [])) != sorted(expected_targets):
        raise ValueError("resolved LoRA target names mismatch")
    return result


def validate(root, manifest_path):
    root, manifest_path = Path(root), Path(manifest_path)
    manifest = read(manifest_path)
    runs = manifest.get("runs", [])
    run_ids = [run.get("run_id") for run in runs]
    config_seeds = [(run.get("configuration_id"), run.get("seed")) for run in runs]
    if len(run_ids) != len(set(run_ids)) or len(config_seeds) != len(set(config_seeds)):
        raise ValueError("manifest contains duplicate run IDs or configuration/seed combinations")
    results = []
    for expected in runs:
        results.append(validate_run(root, expected, manifest))
    actual = {path.name for path in root.iterdir() if path.is_dir() and ".incomplete." not in path.name and not path.name.startswith("." )}
    expected_ids = set(run_ids)
    unexpected = actual - expected_ids
    if unexpected:
        raise ValueError(f"unexpected run directories: {sorted(unexpected)}")
    print(f"PASS: {len(results)} run(s) validated")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate(args.input_dir, args.manifest)
    except Exception as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
