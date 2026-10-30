#!/usr/bin/env python3
"""No-training preparation and validation for the nine-run Phase 4 v2 study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from phase4_v2_manifest import load_json, validate_selection
from phase4_v2_split import (EXPECTED_FINGERPRINT, EXPECTED_SOURCE_SIZE,
                             create_v2_split, load_dataset_source,
                             load_legacy_source_mapping,
                             verify_legacy_probe_reconstruction)
from phase4_v2_io import sha256_payload


def validate_split_manifest(split, config):
    if split.get("protocol_version") != config["protocol_version"]:
        raise ValueError("split protocol version mismatch")
    if split.get("dataset_fingerprint") != EXPECTED_FINGERPRINT:
        raise ValueError("split fingerprint mismatch")
    if split.get("source_size") != EXPECTED_SOURCE_SIZE:
        raise ValueError("split source size mismatch")
    split_content = dict(split)
    recorded_hash = split_content.pop("manifest_sha256", None)
    if recorded_hash is None:
        raise ValueError("split manifest is not hashed")
    from phase4_v2_io import sha256_payload
    if sha256_payload(split_content) != recorded_hash:
        raise ValueError("split manifest hash mismatch")
    pools = split.get("pools", {})
    expected = {"train": 47640, "validation": 6000, "selection_probe": 512, "final_test": 6000}
    if set(pools) != set(expected):
        raise ValueError("split pool names mismatch")
    all_ids = []
    for name, size in expected.items():
        pool = pools[name]
        if pool.get("size") != size or len(pool.get("indices", [])) != size or len(pool.get("ids", [])) != size:
            raise ValueError(f"split size mismatch for {name}")
        if len(set(pool["indices"])) != size or len(set(pool["ids"])) != size:
            raise ValueError(f"duplicate split entries for {name}")
        all_ids.extend(pool["ids"])
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("v2 pools overlap by example ID")
    quarantine = split["legacy_quarantine"]
    quarantine_ids = set(quarantine["holdout_ids"]) | set(quarantine["probe_ids"])
    if quarantine_ids & set(all_ids):
        raise ValueError("quarantined example entered a v2 pool")
    if quarantine["union_size"] != 7197:
        raise ValueError("unexpected quarantine union size")


def validate_final_manifest(manifest, config, split, selection):
    if manifest.get("protocol_version") != config["protocol_version"]:
        raise ValueError("manifest protocol version mismatch")
    if manifest.get("split_manifest_sha256") != split.get("manifest_sha256"):
        raise ValueError("manifest split hash mismatch")
    if manifest.get("selection_artifact_sha256") != selection.get("content_sha256"):
        raise ValueError("manifest selection hash mismatch")
    manifest_content = dict(manifest)
    recorded_hash = manifest_content.pop("manifest_sha256", None)
    if recorded_hash is None:
        raise ValueError("manifest is not hashed")
    from phase4_v2_io import sha256_payload
    if sha256_payload(manifest_content) != recorded_hash:
        raise ValueError("manifest hash mismatch")
    if manifest.get("run_count") != 9 or len(manifest.get("runs", [])) != 9:
        raise ValueError("v2 primary manifest must contain nine runs")
    run_ids = [run["run_id"] for run in manifest["runs"]]
    if len(set(run_ids)) != 9:
        raise ValueError("manifest run IDs are not unique")
    if set(run["seed"] for run in manifest["runs"]) != {0, 1, 2}:
        raise ValueError("manifest training seeds mismatch")
    required = {
        "gradient_top3_r8": (8, 16),
        "random_a_r8": (8, 16),
        "uniform_all_r2": (2, 4),
    }
    if {entry["id"] for entry in manifest["configurations"]} != set(required):
        raise ValueError("primary manifest must contain exactly the three required configurations")
    for entry in manifest["configurations"]:
        if entry["expected_trainable_parameters"] != 665858:
            raise ValueError("primary parameter budget mismatch")
        if (entry["rank"], entry["alpha"]) != required[entry["id"]]:
            raise ValueError(f"rank/alpha mismatch for {entry['id']}")
    by_id = {entry["id"]: entry for entry in manifest["configurations"]}
    if sorted(by_id["random_a_r8"]["layers"]) != [8, 9, 11]:
        raise ValueError("random_a_r8 layers are not frozen as predeclared")
    if sorted(by_id["uniform_all_r2"]["layers"]) != list(range(12)):
        raise ValueError("uniform_all_r2 layers are invalid")
    for run in manifest["runs"]:
        entry = by_id[run["configuration_id"]]
        if (run["layers"], run["rank"], run["alpha"]) != (entry["layers"], entry["rank"], entry["alpha"]):
            raise ValueError(f"run configuration does not match manifest entry: {run['run_id']}")


def _write_new_json(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite preflight artifact: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_preflight(config_path=Path("phase4_v2_config.json"), output_dir=Path("phase4_v2_preflight"), model_check=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"preflight output directory is not empty: {output_dir}")

    # Prevent any Hugging Face network fallback for all downstream functions.
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = load_json(config_path)
    statuses = {name: "NOT RUN" for name in ("split", "selection", "manifest", "parameters")}
    artifacts = {}
    report = {"protocol_version": "phase4_v2_preflight",
              "status": statuses, "artifacts": artifacts,
              "network_policy": "offline environment variables enabled",
              "training_started": False, "final_test_evaluated": False}
    report_path = output_dir / "preflight_report.json"

    def save_report():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        labels, ids, fingerprint = load_dataset_source(
            config["dataset"]["name"], config["dataset"]["config"])
        legacy, index_to_id, _ = load_legacy_source_mapping("phase4_split.json")
        legacy_probe = verify_legacy_probe_reconstruction(
            Path("roberta/data.py").read_text(), legacy, index_to_id)
        split = create_v2_split(labels, ids, fingerprint, legacy, legacy_probe)
        validate_split_manifest(split, config)
        split_path = output_dir / "phase4_v2_split.json"
        _write_new_json(split_path, split)
        statuses["split"] = "PASS"
        artifacts["split"] = {"path": str(split_path), "manifest_sha256": split["manifest_sha256"]}

        from phase4_v2_selection import build_selection_artifact, _score_from_local_dataset
        scores = _score_from_local_dataset(config, split, local_files_only=True)
        selection = build_selection_artifact(
            config, split, scores,
            sha256_payload(split["pools"]["selection_probe"]),
        )
        selection_path = output_dir / "phase4_v2_selection.json"
        _write_new_json(selection_path, selection)
        statuses["selection"] = "PASS"
        artifacts["selection"] = {"path": str(selection_path), "content_sha256": selection["content_sha256"]}

        from phase4_v2_manifest import build_manifest
        manifest = build_manifest(config, split, selection)
        manifest_path = output_dir / "phase4_v2_manifest.json"
        _write_new_json(manifest_path, manifest)
        validate_final_manifest(manifest, config, split, selection)
        statuses["manifest"] = "PASS"
        artifacts["manifest"] = {"path": str(manifest_path), "manifest_sha256": manifest["manifest_sha256"]}

        if model_check:
            from phase4_v2_parameter_check import actual_parameter_count
            counts = {}
            for entry in manifest["configurations"]:
                counts[entry["id"]] = actual_parameter_count(
                    config["model"]["name"], config["model"]["resolved_revision"],
                    entry["layers"], entry["rank"], entry["alpha"],
                    tuple(config["lora"]["target_modules"]))
                if counts[entry["id"]] != entry["expected_trainable_parameters"]:
                    raise ValueError(f"parameter mismatch for {entry['id']}: {counts[entry['id']]}")
            artifacts["parameters"] = counts
        statuses["parameters"] = "PASS" if model_check else "NOT RUN"

        # Independently recompute all artifact hashes and split references.
        for path, field in ((split_path, "manifest_sha256"), (selection_path, "content_sha256"), (manifest_path, "manifest_sha256")):
            payload = load_json(path); recorded = payload.pop(field)
            assert sha256_payload(payload) == recorded
        statuses["artifacts"] = "PASS"
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        report["status"] = statuses
        save_report()
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("phase4_v2_config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase4_v2_preflight"))
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args()
    report = run_preflight(args.config, args.output_dir, model_check=not args.skip_model_check)
    raise SystemExit(0 if all(value in {"PASS", "NOT RUN"} for value in report["status"].values()) and not report.get("error") else 1)


if __name__ == "__main__":
    main()
