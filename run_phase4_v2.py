#!/usr/bin/env python3
"""Run one frozen Phase 4 v2 configuration; never changes legacy outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

from phase4_v2_io import (CompletedRunError, IncompleteRunError, atomic_json,
                          mark_failed, reserve_run, sha256_payload,
                          write_completed_artifacts)
from phase4_v2_manifest import build_concentration_manifest
from phase4_v2_preflight import validate_concentration_manifest


ROOT = Path(__file__).resolve().parent


def set_seed(seed):
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    return json.loads(Path(path).read_text())


def make_sweep_config(config, rank, alpha=None):
    from roberta.config import RobertaSweepConfig
    if alpha is None:
        raise ValueError("LoRA alpha must be supplied by the frozen run manifest")
    return RobertaSweepConfig(
        model_name=config["model"]["name"], model_revision=config["model"]["resolved_revision"],
        dataset_name=config["dataset"]["name"], dataset_config=config["dataset"]["config"],
        max_length=config["dataset"]["max_length"], lora_rank=rank,
        lora_alpha=alpha,
        target_modules=tuple(config["lora"]["target_modules"]),
        learning_rate=config["training"]["learning_rate"], batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
    )


def load_v2_data(config, split_path, rank):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    split = load_json(split_path)
    if split["dataset_fingerprint"] != config["dataset"]["expected_fingerprint"]:
        raise ValueError("v2 split fingerprint mismatch")
    raw = load_dataset(config["dataset"]["name"], config["dataset"]["config"])
    source = raw[config["dataset"]["source_split"]]
    if getattr(source, "_fingerprint", None) != split["dataset_fingerprint"]:
        raise ValueError("current dataset fingerprint mismatch")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], revision=config["model"]["tokenizer_revision"])

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=config["dataset"]["max_length"])

    pools = split["pools"]
    # Screening/replication intentionally materialize only train and validation.
    # The final-test pool is never selected, tokenized, or returned here.
    selected = {}
    for name in ("train", "validation"):
        subset = raw[config["dataset"]["source_split"]].select(pools[name]["indices"])
        subset = subset.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
        subset = subset.rename_column("label", "labels")
        subset.set_format("torch")
        selected[name] = subset
    return selected, tokenizer, split


def loader(dataset, batch_size, shuffle, seed, collator):
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=collator, num_workers=0)


def evaluate(model, dataset, collator, batch_size):
    import torch
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    predictions, labels_all = [], []
    with torch.no_grad():
        for batch in loader(dataset, batch_size, False, 0, collator):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            output = model(**batch, labels=labels)
            total_loss += output.loss.item() * len(labels)
            predictions.extend(output.logits.argmax(-1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    correct = sum(int(prediction == label) for prediction, label in zip(predictions, labels_all))
    n = len(labels_all)
    return {"accuracy": correct / n, "loss": total_loss / n, "correct": correct,
            "n": n, "predictions": predictions, "labels": labels_all}


def checkpoint_hash(directory):
    files = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({"path": str(path.relative_to(directory)), "sha256": digest})
    return sha256_payload(files)


def runtime_metadata():
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets", "tokenizers", "accelerate", "numpy"):
        try:
            packages[name] = version(name)
        except Exception:
            packages[name] = None
    return {"python": platform.python_version(), "software": packages}


def resolve_run_entry(manifest, configuration_id, seed):
    if "runs" in manifest:
        return next(run for run in manifest["runs"]
                    if run["configuration_id"] == configuration_id and run["seed"] == seed)
    if "pairs" in manifest:
        pair = next(run for run in manifest["pairs"]
                    if run["pair_id"] == configuration_id)
        return {"run_id": f"{configuration_id}_seed{seed}", "configuration_id": configuration_id,
                "layers": pair["layers"], "rank": pair["rank"], "alpha": pair["alpha"],
                "seed": seed, "expected_trainable_parameters": pair["expected_trainable_parameters"]}
    raise ValueError("manifest must contain runs or pairs")


def validate_requested_configurations(manifest, requested):
    available = {entry["id"] for entry in manifest.get("configurations", ())}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"unknown concentration configuration(s): {', '.join(unknown)}")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate concentration configuration requested")


def run(config, split_path, manifest_path, configuration_id, seed, output_dir):
    import torch
    from transformers import DataCollatorWithPadding
    from phase4_v2_io import run_paths
    from roberta.model import (inject_lora_layers, load_model, register_frozen_prefix_detach,
                               trainable_parameters)
    manifest = load_json(manifest_path)
    run_entry = resolve_run_entry(manifest, configuration_id, seed)
    set_seed(seed)
    reservation = reserve_run(output_dir, run_entry["run_id"], {
        "configuration_id": configuration_id, "seed": seed,
        "manifest_sha256": manifest["manifest_sha256"],
    })
    try:
        pools, tokenizer, split = load_v2_data(config, split_path, run_entry["rank"])
        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        sweep_config = make_sweep_config(config, run_entry["rank"], run_entry.get("alpha"))
        from roberta.train import _fit_frozen_baseline
        baseline_model = load_model(sweep_config, seed)
        frozen_val_accuracy, _ = _fit_frozen_baseline(
            baseline_model, pools["train"], pools["validation"], sweep_config, seed, collator
        )
        del baseline_model
        model = inject_lora_layers(load_model(sweep_config, seed), sweep_config, run_entry["layers"])
        params = [parameter for _, parameter in trainable_parameters(model)]
        actual_count = sum(parameter.numel() for parameter in params)
        if actual_count != run_entry["expected_trainable_parameters"]:
            raise ValueError(f"unexpected trainable count: {actual_count}")
        optimizer = torch.optim.AdamW(params, lr=config["training"]["learning_rate"],
                                      weight_decay=config["training"]["weight_decay"])
        train_loader = loader(pools["train"], config["training"]["batch_size"], True, seed, collator)
        hook = register_frozen_prefix_detach(model, min(run_entry["layers"]))
        device = next(model.parameters()).device
        epoch_metrics = []
        started = time.time()
        try:
            for epoch in range(1, config["training"]["epochs"] + 1):
                model.train()
                total_loss, count = 0.0, 0
                for batch in train_loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    optimizer.zero_grad()
                    output = model(**batch)
                    output.loss.backward()
                    optimizer.step()
                    total_loss += output.loss.item() * len(batch["labels"])
                    count += len(batch["labels"])
                epoch_metrics.append({"epoch": epoch, "train_loss": total_loss / count})
        finally:
            hook.remove()
        validation_metrics = evaluate(model, pools["validation"], collator, config["training"]["batch_size"])
        paths = reservation.paths
        model.save_pretrained(paths["checkpoint"])
        checkpoint_digest = checkpoint_hash(paths["checkpoint"])
        predictions = {"run_id": run_entry["run_id"], "split": "validation",
                       "predictions": validation_metrics.pop("predictions"),
                       "labels": validation_metrics.pop("labels")}
        record = {
            "status": "artifact_ready", "protocol_version": manifest["protocol_version"], "run_id": run_entry["run_id"],
                "configuration": {"id": configuration_id, "layers": run_entry["layers"],
                                   "reported_layers": run_entry.get("reported_layers", [f"L{x + 1}" for x in run_entry["layers"]]),
                                   "rank": run_entry["rank"], "alpha": run_entry["alpha"],
                                   "configuration_sha256": run_entry.get("configuration_sha256")},
            "model": manifest["model"], "dataset": manifest["dataset"],
            "seed": seed, "manifest_sha256": manifest["manifest_sha256"],
            "selection_artifact_sha256": manifest.get("selection_artifact_sha256"),
            "split_manifest_sha256": split["manifest_sha256"], "dataset_fingerprint": split["dataset_fingerprint"],
            "evaluation_mode": "validation_only",
            "train_size": len(pools["train"]), "validation_size": len(pools["validation"]),
            "n_trainable_params": actual_count,
            "epoch_metrics": epoch_metrics, "validation_metrics": validation_metrics,
            "frozen_val_accuracy": frozen_val_accuracy,
            "utility": validation_metrics["accuracy"] - frozen_val_accuracy,
            "checkpoint_rule": "final_epoch",
            "checkpoint_path": str(paths["checkpoint"]), "runtime_sec": time.time() - started,
            "checkpoint_sha256": checkpoint_digest, "runtime_metadata": runtime_metadata(),
        }
        write_completed_artifacts(reservation, record, predictions, checkpoint_digest)
        return record
    except Exception as exc:
        mark_failed(reservation, exc)
        raise
    finally:
        if not reservation.completed:
            reservation.__exit__(None, None, None)


def run_core(config_path, split_path, manifest_path, output_dir):
    config = load_json(config_path)
    manifest = load_json(manifest_path)
    from phase4_v2_preflight import validate_final_manifest
    split = load_json(split_path)
    selection = load_json(manifest_path.parent / "phase4_v2_selection.json")
    validate_final_manifest(manifest, config, split, selection)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in manifest["runs"]:
        try:
            record = run(config, split_path, manifest_path, entry["configuration_id"],
                         entry["seed"], output_dir)
            records.append(record)
        except IncompleteRunError:
            raise
        except CompletedRunError:
            print(f"Skipping completed run: {entry['run_id']}", flush=True)
    return records


def run_concentration(config_path, artifacts_dir, requested, output_dir):
    config = load_json(config_path)
    split_path = Path(artifacts_dir) / "phase4_v2_split.json"
    selection_path = Path(artifacts_dir) / "phase4_v2_selection.json"
    manifest_path = Path(artifacts_dir) / "phase4_v2_concentration_manifest.json"
    split, selection = load_json(split_path), load_json(selection_path)
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = build_concentration_manifest(config, split, selection)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate_concentration_manifest(manifest, config, split, selection)
    validate_requested_configurations(manifest, requested)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / "phase4_v2_concentration_manifest.json"
    if output_manifest.exists():
        if load_json(output_manifest).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"concentration output manifest collision: {output_manifest}")
    else:
        output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    selected_runs = [entry for entry in manifest["runs"] if entry["configuration_id"] in requested]
    for entry in selected_runs:
        try:
            run(config, split_path, manifest_path, entry["configuration_id"], entry["seed"], output_dir)
        except IncompleteRunError:
            raise
        except CompletedRunError:
            print(f"Skipping completed run: {entry['run_id']}", flush=True)


def run_pairs(config_path, split_path, manifest_path, output_dir):
    config = load_json(config_path)
    manifest = load_json(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["runs"]:
        try:
            run(config, split_path, manifest_path, entry["configuration_id"],
                entry["seed"], output_dir)
        except IncompleteRunError:
            raise
        except CompletedRunError:
            print(f"Skipping completed pair: {entry['run_id']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 v2 execution entry point")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="prepare and validate artifacts without training")
    preflight.add_argument("--config", type=Path, default=ROOT / "phase4_v2_config.json")
    preflight.add_argument("--output-dir", type=Path, default=ROOT / "phase4_v2_preflight")
    preflight.add_argument("--skip-model-check", action="store_true")

    prepare = subparsers.add_parser("prepare", help="download and verify the exact cached dataset/model")
    prepare.add_argument("--config", type=Path, default=ROOT / "phase4_v2_config.json")

    core = subparsers.add_parser("core", help="run the nine primary experiments")
    core.add_argument("--config", type=Path, default=ROOT / "phase4_v2_config.json")
    core.add_argument("--artifacts-dir", type=Path, default=ROOT / "phase4_v2_preflight")
    core.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase4_v2")

    pairs = subparsers.add_parser("pairs", help="run the optional 66-pair screen")
    pairs.add_argument("--config", type=Path, default=ROOT / "phase4_v2_config.json")
    pairs.add_argument("--artifacts-dir", type=Path, default=ROOT / "phase4_v2_preflight")
    pairs.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase4_v2_pairs")

    concentration = subparsers.add_parser("concentration", help="run selected concentration configurations")
    concentration.add_argument("--config", nargs="+", required=True, metavar="CONFIGURATION")
    concentration.add_argument("--artifacts-dir", type=Path, default=ROOT / "phase4_v2_preflight")
    concentration.add_argument("--config-file", type=Path, default=ROOT / "phase4_v2_config.json")
    concentration.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase4_v2_concentration")

    aggregate = subparsers.add_parser("aggregate", help="aggregate all completed primary runs")
    aggregate.add_argument("--artifacts-dir", type=Path, default=ROOT / "phase4_v2_preflight")
    aggregate.add_argument("--input-dir", type=Path, default=ROOT / "results" / "phase4_v2")
    aggregate.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "phase4_v2")
    aggregate.add_argument("--manifest", type=Path)

    args = parser.parse_args()
    if args.command == "preflight":
        from phase4_v2_preflight import run_preflight
        report = run_preflight(args.config, args.output_dir, model_check=not args.skip_model_check)
        raise SystemExit(0 if not report.get("error") else 1)
    if args.command == "prepare":
        from phase4_v2_preflight import prepare_cache
        prepare_cache(args.config)
        return
    if args.command == "core":
        run_core(args.config, args.artifacts_dir / "phase4_v2_split.json",
                 args.artifacts_dir / "phase4_v2_manifest.json", args.output_dir)
        return
    if args.command == "pairs":
        from phase4_v2_pair_manifest import build_pair_manifest
        split = load_json(args.artifacts_dir / "phase4_v2_split.json")
        pair_dir = args.artifacts_dir / "pairs"
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_manifest_path = pair_dir / "phase4_v2_pair_manifest.json"
        if pair_manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite pair manifest: {pair_manifest_path}")
        pair_manifest = build_pair_manifest(split["manifest_sha256"])
        pair_manifest_path.write_text(json.dumps(pair_manifest, indent=2) + "\n")
        run_pairs(args.config, args.artifacts_dir / "phase4_v2_split.json", pair_manifest_path, args.output_dir)
        return
    if args.command == "concentration":
        run_concentration(args.config_file, args.artifacts_dir, args.config, args.output_dir)
        return
    from phase4_v2_aggregate import aggregate_from_manifest
    if args.manifest:
        manifest_path = args.manifest
    elif args.input_dir.name == "phase4_v2_concentration":
        candidate = args.input_dir / "phase4_v2_concentration_manifest.json"
        manifest_path = candidate if candidate.exists() else (args.artifacts_dir / "phase4_v2_concentration_manifest.json")
    else:
        manifest_path = args.artifacts_dir / "phase4_v2_manifest.json"
    aggregate_from_manifest(args.input_dir, manifest_path, args.output_dir)


if __name__ == "__main__":
    main()
