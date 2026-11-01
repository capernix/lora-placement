#!/usr/bin/env python3
"""Build a hashed additive Phase 4 v2 follow-up manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase4_v2_io import sha256_payload

EXPECTED_SPLIT_HASH = "bb3db736dead8440aa74c4863e61a1646ddf4f5fc3c7c5d5da60813ad70a3fae"
EXPECTED_FINGERPRINT = "e0f6962d602757cb"


def read(path):
    return json.loads(Path(path).read_text())


def configuration_files(names):
    paths = []
    if names == ["all"]:
        paths = sorted((ROOT / "phase4_followup/configs").glob("*.json"))
    else:
        for name in names:
            path = ROOT / "phase4_followup/configs" / name
            if path.suffix != ".json":
                path = path.with_suffix(".json")
            paths.append(path)
    entries = []
    for path in paths:
        payload = read(path)
        entries.extend(payload.get("configurations", [payload]))
    return entries


def build(config_names, split, base_config):
    if split.get("manifest_sha256") != EXPECTED_SPLIT_HASH:
        raise ValueError("wrong Phase 4 v2 split hash; do not train on a regenerated split")
    if split.get("dataset_fingerprint") != EXPECTED_FINGERPRINT:
        raise ValueError("wrong dataset fingerprint")
    requested_entries = configuration_files(config_names)
    if not requested_entries:
        raise ValueError("no follow-up configurations selected")
    # Preserve the exact historical definition (and its configuration hash) for
    # extra seeds. The adapter-budget field is run-level follow-up metadata.
    historical = {entry["id"]: entry for entry in base_config.get("concentration_configurations", [])}
    entries = [historical.get(entry["id"], dict(entry)) for entry in requested_entries]
    runs = []
    frozen = {
        "protocol_version": base_config["protocol_version"],
        "manifest_type": "phase4_v2_followup",
        "model": base_config["model"],
        "dataset": base_config["dataset"],
        "training": base_config["training"],
        "lora": base_config["lora"],
        "split_manifest_sha256": split["manifest_sha256"],
        "configurations": entries,
    }
    for source_entry, entry in zip(requested_entries, entries):
        required = {"id", "layers", "reported_layers", "rank", "alpha",
                    "expected_trainable_parameters"}
        missing = required - set(entry)
        if missing:
            raise ValueError(f"missing fields in {entry.get('id')}: {sorted(missing)}")
        adapter_budget = source_entry.get("expected_adapter_parameters", 73728)
        if adapter_budget != 73728 or entry["expected_trainable_parameters"] != 665858:
            raise ValueError(f"budget mismatch in {entry['id']}")
        if len(entry["layers"]) != len(entry["reported_layers"]) or len(set(entry["layers"])) != len(entry["layers"]):
            raise ValueError(f"invalid layer definition in {entry['id']}")
        definition_hash = sha256_payload(entry)
        for seed in source_entry["seeds"]:
            runs.append({"run_id": f"{entry['id']}_seed{seed}", "configuration_id": entry["id"],
                         "layers": entry["layers"], "reported_layers": entry["reported_layers"],
                         "rank": entry["rank"], "alpha": entry["alpha"], "seed": seed,
                         "expected_adapter_parameters": adapter_budget,
                         "expected_trainable_parameters": entry["expected_trainable_parameters"],
                         "configuration_sha256": definition_hash})
    frozen["run_count"] = len(runs)
    frozen["runs"] = runs
    frozen["manifest_sha256"] = sha256_payload(frozen)
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=["all"])
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=ROOT / "phase4_v2_config.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.config, read(args.split), read(args.base_config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and read(args.output).get("manifest_sha256") != manifest["manifest_sha256"]:
        raise FileExistsError(f"refusing to replace different manifest: {args.output}")
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.output), "manifest_sha256": manifest["manifest_sha256"], "runs": manifest["run_count"]}, indent=2))


if __name__ == "__main__":
    main()
