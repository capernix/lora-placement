#!/usr/bin/env python3
"""Aggregate only Phase 4 runs with a valid final completion marker."""
import argparse
import json
from pathlib import Path

import pandas as pd

from phase4_io import read_json, run_paths


def load_completed(output_dir):
    records = []
    ignored = []
    for directory in sorted(Path(output_dir).iterdir()) if Path(output_dir).exists() else []:
        if not directory.is_dir() or ".incomplete." in directory.name:
            continue
        paths = run_paths(output_dir, directory.name)
        if not paths["complete"].exists():
            ignored.append(directory.name)
            continue
        marker = read_json(paths["complete"])
        result = read_json(paths["result"])
        predictions = read_json(paths["predictions"])
        if marker.get("status") != "completed" or result.get("status") != "artifact_ready":
            raise ValueError(f"invalid completion protocol: {directory}")
        if result.get("run_id") != predictions.get("run_id") or result.get("run_id") != marker.get("run_id"):
            raise ValueError(f"run-ID mismatch: {directory}")
        row = {
            "run_id": result["run_id"],
            "configuration_id": result["configuration"]["id"],
            "seed": result["seed"],
            "layers": ",".join(f"L{i + 1}" for i in result["configuration"]["layers"]),
            "rank": result["configuration"]["rank"],
            "n_trainable_params": result["n_trainable_params"],
            "accuracy": result["holdout_metrics"]["accuracy"],
            "loss": result["holdout_metrics"]["loss"],
            "precision_positive": result["holdout_metrics"]["precision_positive"],
            "recall_positive": result["holdout_metrics"]["recall_positive"],
            "f1_positive": result["holdout_metrics"]["f1_positive"],
            "runtime_sec": result["runtime_sec"],
            "selected_epoch": result["selected_epoch"],
            "dataset_fingerprint": result["dataset_fingerprint"],
            "resolved_revision": result["model"]["resolved_revision"],
        }
        records.append(row)
    return pd.DataFrame(records), ignored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/phase4"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/phase4"))
    args = parser.parse_args()
    master, ignored = load_completed(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output_dir / "phase4_master.csv", index=False)
    if not master.empty:
        summary = master.groupby(["configuration_id", "layers", "rank", "n_trainable_params"], as_index=False).agg(
            n_runs=("accuracy", "count"), mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
            mean_loss=("loss", "mean"), std_loss=("loss", "std"), mean_precision_positive=("precision_positive", "mean"),
            mean_recall_positive=("recall_positive", "mean"), mean_f1_positive=("f1_positive", "mean"),
            std_f1_positive=("f1_positive", "std"), mean_runtime_sec=("runtime_sec", "mean"))
    else:
        summary = pd.DataFrame()
    summary.to_csv(args.output_dir / "phase4_summary.csv", index=False)
    (args.output_dir / "phase4_aggregation_status.json").write_text(json.dumps({"completed_runs": len(master), "ignored_incomplete_runs": ignored}, indent=2) + "\n")
    print(f"Aggregated {len(master)} completed runs; ignored incomplete={ignored}")


if __name__ == "__main__":
    main()
