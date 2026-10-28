#!/usr/bin/env python3
"""Generate and validate the deterministic Phase 4 21-run manifest."""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from roberta.config import RobertaSweepConfig


ROOT = Path(__file__).resolve().parent


def load_config(path):
    return json.loads(Path(path).read_text())


def verify_gradient_selection():
    path = ROOT / "analysis" / "roberta" / "roberta_ranking_analysis.csv"
    if not path.exists():
        raise ValueError(f"missing Phase 3 ranking artifact: {path}")
    rows = list(csv.DictReader(path.open()))
    top = sorted((row for row in rows if row["predictor"] == "gradient"), key=lambda row: int(row["gradient_rank"])) if rows and "predictor" in rows[0] else []
    # The current Phase 3 table stores ranks by layer, not one row per predictor.
    if not top:
        rows = list(csv.DictReader(path.open()))
        top_layers = sorted(rows, key=lambda row: int(row["grad_norm_rank"]))[:3]
    else:
        top_layers = top[:3]
    observed = sorted(int(row["layer"]) for row in top_layers)
    expected = [9, 10, 11]
    if observed != expected:
        raise ValueError(f"Phase 3 gradient ranking contradicts required L10-L12 selection: {observed}")
    return observed


def validate_config(config):
    layer = config["layer_convention"]
    if layer["zero_based_indices"] != list(range(12)):
        raise ValueError("layer convention is not zero-based 0..11")
    if layer["reported_labels"] != [f"L{i}" for i in range(1, 13)]:
        raise ValueError("reported layer labels are not L1..L12")
    random_sets = config["random_layer_sets"]
    a, b = random_sets["set_a_zero_based"], random_sets["set_b_zero_based"]
    excluded = {tuple([9, 10, 11]), tuple([0, 1, 2]), tuple([4, 5, 6])}
    if len(a) != 3 or len(b) != 3 or len(set(a)) != 3 or len(set(b)) != 3 or a == b:
        raise ValueError("random layer sets are not distinct valid triples")
    if tuple(a) in excluded or tuple(b) in excluded:
        raise ValueError("random layer set duplicates a fixed configuration")
    required = {"gradient_top3_r8", "random_a_r8", "random_b_r8", "early_r8", "middle_r8", "uniform_all_r2", "all_layer_reference_r8"}
    actual = {entry["id"] for entry in config["configurations"]}
    if actual != required or len(config["configurations"]) != 7:
        raise ValueError(f"configuration IDs differ from required seven: {actual}")
    if config["training"]["training_seeds"] != [0, 1, 2]:
        raise ValueError("training seeds must be [0, 1, 2]")


def make_manifest(config):
    runs = []
    shared = {
        "model": config["model"], "dataset": config["dataset"],
        "training": config["training"], "lora": config["lora"],
        "holdout_manifest": config["dataset"]["split_manifest"],
    }
    for entry in config["configurations"]:
        for seed in config["training"]["training_seeds"]:
            runs.append({
                "run_id": f"{entry['id']}_seed{seed}",
                "configuration_id": entry["id"],
                "layers": entry["layers"],
                "reported_layers": entry["reported_layers"],
                "rank": entry["rank"],
                "seed": seed,
                **shared,
            })
    if len(runs) != 21 or len({run["run_id"] for run in runs}) != 21:
        raise ValueError("manifest does not contain 21 unique runs")
    return {"phase": "phase4", "run_count": len(runs), "gradient_selection_verified": [9, 10, 11], "runs": runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "phase4_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "phase4_manifest.json")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    verify_gradient_selection()
    manifest = make_manifest(config)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}: {manifest['run_count']} runs")


if __name__ == "__main__":
    main()
