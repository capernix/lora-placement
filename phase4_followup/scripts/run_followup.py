#!/usr/bin/env python3
"""Run additive Phase 4 v2 follow-up jobs using the frozen runner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase4_v2_io import CompletedRunError
from run_phase4_v2 import run
from build_followup_manifest import build, read


def manifest_for(names, split_path, output):
    path = Path(output)
    if path.exists():
        return path
    payload = build(names, read(split_path), read(ROOT / "phase4_v2_config.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run_jobs(names, split_path, manifest_path, output_dir, seed_filter=None):
    manifest = read(manifest_path)
    base_config = ROOT / "phase4_v2_config.json"
    requested = set(names)
    entries = [entry for entry in manifest["runs"]
               if entry["configuration_id"] in requested and
               (seed_filter is None or entry["seed"] in seed_filter)]
    if not entries:
        raise ValueError("no matching jobs in manifest")
    for entry in entries:
        print(f"Starting {entry['run_id']}", flush=True)
        try:
            result = run(read(base_config), Path(split_path), Path(manifest_path),
                         entry["configuration_id"], entry["seed"], Path(output_dir))
            print(json.dumps({"run_id": result["run_id"],
                              "accuracy": result["validation_metrics"]["accuracy"],
                              "adapter_parameters": result["n_lora_adapter_params"],
                              "trainable_parameters": result["n_trainable_params"]}, indent=2), flush=True)
        except CompletedRunError as exc:
            print(f"REFUSING TO OVERWRITE COMPLETED RUN: {exc}", flush=True)
            raise


def verify(split_path, artifacts_dir, output_dir, reference=None):
    manifest_path = Path(artifacts_dir) / "verification_manifest.json"
    manifest_for(["extra_seeds"], split_path, manifest_path)
    # Verification is intentionally a separately hashed one-run manifest so it
    # cannot be mistaken for one of the ten planned follow-up runs.
    payload = build(["extra_seeds"], read(split_path), read(ROOT / "phase4_v2_config.json"))
    payload["configurations"] = [entry for entry in payload["configurations"] if entry["id"] == "utility_top3_r8"]
    payload["runs"] = [entry for entry in payload["runs"] if entry["configuration_id"] == "utility_top3_r8" and entry["seed"] == 3]
    # Use the reproducibility seed 0 and recompute the manifest hash.
    entry = dict(payload["runs"][0]); entry["seed"] = 0; entry["run_id"] = "utility_top3_r8_seed0"
    payload["runs"] = [entry]; payload["run_count"] = 1
    from phase4_v2_io import sha256_payload
    payload["manifest_sha256"] = sha256_payload({k: v for k, v in payload.items() if k != "manifest_sha256"})
    manifest_path = Path(artifacts_dir) / "verification_manifest.json"
    if manifest_path.exists() and read(manifest_path).get("manifest_sha256") != payload["manifest_sha256"]:
        raise FileExistsError(f"verification manifest collision: {manifest_path}")
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    run_jobs(["utility_top3_r8"], split_path, manifest_path, output_dir, {0})
    result_path = Path(output_dir) / "utility_top3_r8_seed0" / "result.json"
    result = read(result_path)
    reference_path = Path(reference) if reference else ROOT / "phase4-18runs/utility_top3_r8_seed0/result.json"
    report = {"status": "REFERENCE_NOT_FOUND", "reproduced_accuracy": result["validation_metrics"]["accuracy"],
              "reference": str(reference_path), "runtime_metadata": result["runtime_metadata"]}
    if reference_path.exists():
        ref = read(reference_path)
        ref_accuracy = ref["validation_metrics"]["accuracy"]
        difference = abs(result["validation_metrics"]["accuracy"] - ref_accuracy)
        report.update({"status": "PASS" if difference <= 0.002 else "WARN",
                       "reference_accuracy": ref_accuracy, "absolute_difference": difference,
                       "warning_threshold": 0.002})
        if difference > 0.002:
            print("WARNING: reproducibility difference exceeds 0.2 percentage points", flush=True)
    report_path = Path(output_dir) / "reproducibility_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] != "WARN" else 2


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    verify_p = sub.add_parser("verify")
    verify_p.add_argument("--split", type=Path, required=True)
    verify_p.add_argument("--artifacts-dir", type=Path, required=True)
    verify_p.add_argument("--output-dir", type=Path, required=True)
    verify_p.add_argument("--reference", type=Path)
    run_p = sub.add_parser("run")
    run_p.add_argument("--config", nargs="+", required=True)
    run_p.add_argument("--seeds", nargs="+", type=int)
    run_p.add_argument("--split", type=Path, required=True)
    run_p.add_argument("--manifest", type=Path, required=True)
    run_p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        raise SystemExit(verify(args.split, args.artifacts_dir, args.output_dir, args.reference))
    run_jobs(args.config, args.split, args.manifest, args.output_dir,
             set(args.seeds) if args.seeds else None)


if __name__ == "__main__":
    main()
