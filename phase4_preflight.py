#!/usr/bin/env python3
"""Lightweight Phase 4 validation; never trains or writes research results."""
import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

from transformers import RobertaConfig, RobertaForSequenceClassification

from phase4_manifest import load_config, validate_config, verify_gradient_selection
from phase4_split import validate_split
from roberta.config import RobertaSweepConfig
from roberta.model import inject_lora_layers, load_model, trainable_parameters


ROOT = Path(__file__).resolve().parent


def parameter_check(config, actual_model=False):
    if actual_model:
        base = None
    counts = {}
    for entry in config["configurations"]:
        sweep_config = replace(RobertaSweepConfig(), lora_rank=entry["rank"])
        if actual_model:
            model = load_model(sweep_config, seed=0)
        else:
            tiny = RobertaConfig(
                vocab_size=50265, max_position_embeddings=514, num_hidden_layers=12,
                hidden_size=768, intermediate_size=3072, num_attention_heads=12,
                num_labels=2, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            )
            model = RobertaForSequenceClassification(tiny)
        adapted = inject_lora_layers(model, sweep_config, entry["layers"])
        trainable = [(name, parameter) for name, parameter in trainable_parameters(adapted)]
        names = [name for name, _ in trainable]
        if not all("classifier" in name or any(f"encoder.layer.{layer}." in name for layer in entry["layers"]) for name in names):
            raise ValueError(f"unexpected trainable parameter in {entry['id']}")
        measured = sum(parameter.numel() for _, parameter in trainable)
        lora_only = sum(parameter.numel() for name, parameter in trainable if "lora_" in name)
        counts[entry["id"]] = {"rank": entry["rank"], "layers": len(entry["layers"]), "n_trainable_params": measured, "lora_params": lora_only, "classifier_params": measured - lora_only}
        del adapted, model
        gc.collect()
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "phase4_config.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "phase4_manifest.json")
    parser.add_argument("--split", type=Path, default=ROOT / "phase4_split.json")
    parser.add_argument("--model-check", action="store_true", help="also instantiate the exact pretrained model for every configuration")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    gradient = verify_gradient_selection()
    manifest = json.loads(args.manifest.read_text())
    if manifest["run_count"] != 21 or len({run["run_id"] for run in manifest["runs"]}) != 21:
        raise ValueError("manifest must contain 21 unique runs")
    split = json.loads(args.split.read_text())
    validate_split(split)
    counts = parameter_check(config, actual_model=args.model_check)
    print(json.dumps({"gradient_selection": gradient, "run_count": manifest["run_count"], "split": {"train": split["train_size"], "holdout": split["holdout_size"]}, "parameter_counts": counts}, indent=2))
    print("Phase 4 preflight passed; no training was run.")


if __name__ == "__main__":
    main()
