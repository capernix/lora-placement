#!/usr/bin/env python3
"""Manifest-authoritative aggregation for completed Phase 4 v2 runs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from phase4_v2_io import read_json, run_paths


def load_completed(output_dir, manifest):
    expected = {run["run_id"]: run for run in manifest["runs"]}
    root = Path(output_dir)
    records = []
    if not root.exists():
        raise ValueError(f"missing output directory: {root}")
    for run_id, expected_run in sorted(expected.items()):
        paths = run_paths(root, run_id)
        required = (paths["status"], paths["result"], paths["predictions"], paths["complete"])
        if not all(path.exists() for path in required):
            raise ValueError(f"missing or incomplete run artifacts: {run_id}")
        status, result, predictions, marker = map(read_json, required)
        if any(not payload.get("protocol_version", "").startswith("phase4_v2")
               for payload in (status, result, predictions, marker) if "protocol_version" in payload):
            raise ValueError(f"protocol version mismatch: {run_id}")
        if status.get("status") != "completed" or marker.get("status") != "completed" or result.get("status") != "artifact_ready":
            raise ValueError(f"run is not completed: {run_id}")
        if any(payload.get("run_id") != run_id for payload in (status, result, predictions, marker)):
            raise ValueError(f"run-ID mismatch: {run_id}")
        if result.get("configuration", {}).get("id", result.get("configuration_id")) != expected_run["configuration_id"]:
            raise ValueError(f"configuration mismatch: {run_id}")
        if result.get("seed") != expected_run["seed"]:
            raise ValueError(f"seed mismatch: {run_id}")
        if result.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise ValueError(f"manifest hash mismatch: {run_id}")
        if result.get("configuration", {}).get("layers") != expected_run["layers"]:
            raise ValueError(f"layer mismatch: {run_id}")
        if result.get("configuration", {}).get("rank") != expected_run["rank"]:
            raise ValueError(f"rank mismatch: {run_id}")
        if result.get("configuration", {}).get("alpha") != expected_run["alpha"]:
            raise ValueError(f"alpha mismatch: {run_id}")
        if result.get("n_trainable_params") != expected_run["expected_trainable_parameters"]:
            raise ValueError(f"parameter count mismatch: {run_id}")
        records.append(result)
    actual_dirs = {path.name for path in root.iterdir() if path.is_dir() and ".incomplete." not in path.name}
    unexpected = actual_dirs - set(expected)
    if unexpected:
        raise ValueError(f"unexpected run directories: {sorted(unexpected)}")
    return records


def aggregate_from_manifest(input_dir, manifest_path, output_dir):
    manifest = json.loads(Path(manifest_path).read_text())
    records = load_completed(input_dir, manifest)
    write_aggregate(records, output_dir)
    return records


def write_aggregate(records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        metrics = record.get("final_test_metrics", record.get("validation_metrics", record.get("holdout_metrics", {})))
        rows.append({"run_id": record["run_id"], "configuration_id": record["configuration"]["id"],
                     "seed": record["seed"], "layers": ",".join(str(x) for x in record["configuration"]["layers"]),
                     "rank": record["configuration"]["rank"], "alpha": record["configuration"]["alpha"],
                     "n_trainable_params": record["n_trainable_params"],
                     "validation_accuracy": metrics.get("accuracy"), "validation_loss": metrics.get("loss"),
                     "frozen_validation_accuracy": record.get("frozen_val_accuracy"),
                     "utility": record.get("utility"),
                     "runtime_sec": record.get("runtime_sec")})
    with (output_dir / "phase4_v2_master.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader(); writer.writerows(rows)
    summaries = []
    by_configuration = {}
    for row in rows:
        by_configuration.setdefault(row["configuration_id"], []).append(row)
    for configuration_id, group in sorted(by_configuration.items()):
        summary = {"configuration_id": configuration_id, "n_runs": len(group),
                   "layers": group[0]["layers"], "rank": group[0]["rank"], "alpha": group[0]["alpha"]}
        for field in ("validation_accuracy", "validation_loss", "frozen_validation_accuracy", "utility"):
            values = [float(row[field]) for row in group if row[field] is not None]
            summary[f"mean_{field}"] = statistics.mean(values) if values else None
            summary[f"std_{field}"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        summaries.append(summary)
    with (output_dir / "phase4_v2_summary.csv").open("w", newline="") as handle:
        fields = list(summaries[0]) if summaries else ["configuration_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summaries)
    (output_dir / "phase4_v2_aggregation_status.json").write_text(
        json.dumps({"protocol_version": "phase4_v2", "completed_runs": len(rows),
                    "completed_configurations": len(summaries)}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = aggregate_from_manifest(args.input_dir, args.manifest, args.output_dir)
    print(f"Aggregated {len(records)} completed Phase 4 v2 runs")


if __name__ == "__main__":
    main()
