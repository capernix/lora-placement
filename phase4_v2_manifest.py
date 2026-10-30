#!/usr/bin/env python3
"""Freeze the nine-run v2 manifest only from a validated selection artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phase4_v2_io import canonical_json, sha256_payload


def load_json(path):
    return json.loads(Path(path).read_text())


def validate_selection(selection, config, split):
    if selection.get("protocol_version") != config["protocol_version"]:
        raise ValueError("selection protocol version mismatch")
    if selection.get("dataset_fingerprint") != config["dataset"]["expected_fingerprint"]:
        raise ValueError("selection dataset fingerprint mismatch")
    if selection.get("split_manifest_sha256") != split.get("manifest_sha256"):
        raise ValueError("selection split hash mismatch")
    if selection.get("collision") is not False:
        raise ValueError("selection is collided or collision status is unknown")
    if "content_sha256" not in selection:
        raise ValueError("selection artifact is not hashed")
    content = dict(selection)
    expected_hash = content.pop("content_sha256")
    if sha256_payload(content) != expected_hash:
        raise ValueError("selection artifact hash mismatch")
    selected = sorted(selection.get("selected_layers", []))
    if len(selected) != 3 or len(set(selected)) != 3 or any(layer < 0 or layer >= 12 for layer in selected):
        raise ValueError("selection must contain three unique valid layers")
    random_layers = sorted(config["selection"]["random_a_layers"])
    if selected == random_layers:
        raise RuntimeError("gradient selection collides with random_a_r8; protocol decision required")


def build_manifest(config, split, selection):
    validate_selection(selection, config, split)
    gradient_layers = sorted(selection["selected_layers"])
    entries = [
        {"id": "gradient_top3_r8", "layers": gradient_layers, "rank": 8, "alpha": 16, "expected_trainable_parameters": 665858},
        {"id": "random_a_r8", "layers": sorted(config["selection"]["random_a_layers"]), "rank": 8, "alpha": 16, "expected_trainable_parameters": 665858},
        {"id": "uniform_all_r2", "layers": list(range(12)), "rank": 2, "alpha": 4, "expected_trainable_parameters": 665858},
    ]
    runs = []
    for entry in entries:
        for seed in config["training"]["training_seeds"]:
            runs.append({"run_id": f"{entry['id']}_seed{seed}", "configuration_id": entry["id"],
                         "layers": entry["layers"], "rank": entry["rank"], "alpha": entry["alpha"], "seed": seed,
                         "expected_trainable_parameters": entry["expected_trainable_parameters"]})
    manifest = {
        "protocol_version": config["protocol_version"], "manifest_type": "phase4_v2_primary",
        "selection_artifact_sha256": selection["content_sha256"],
        "split_manifest_sha256": split["manifest_sha256"],
        "model": config["model"], "dataset": config["dataset"], "training": config["training"],
        "lora": config["lora"], "configurations": entries, "run_count": len(runs), "runs": runs,
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("phase4_v2_config.json"))
    parser.add_argument("--split", type=Path, default=Path("phase4_v2_split.json"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("phase4_v2_manifest.json"))
    args = parser.parse_args()
    config, split, selection = load_json(args.config), load_json(args.split), load_json(args.selection)
    manifest = build_manifest(config, split, selection)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "manifest_sha256": manifest["manifest_sha256"],
                      "run_count": manifest["run_count"]}, indent=2))


if __name__ == "__main__":
    main()
