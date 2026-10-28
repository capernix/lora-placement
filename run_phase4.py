#!/usr/bin/env python3
"""Run one Phase 4 configuration, or perform a no-training smoke test."""
import argparse
import json
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding

from phase4_manifest import load_config
from phase4_io import (archive_incomplete, atomic_json, reserve_run,
                       write_completed_artifacts)
from phase4_split import validate_split
from roberta.config import RobertaSweepConfig
from roberta.model import (inject_lora_layers, load_model, load_tokenizer,
                           register_frozen_prefix_detach, trainable_parameters)
from roberta.train import _loader


ROOT = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sweep_config(config, rank):
    model = config.get("model", {})
    revision = model.get("resolved_revision")
    if not revision:
        raise ValueError("phase4_config.json must contain model.resolved_revision")
    dataset = config["dataset"]
    lora = config["lora"]
    training = config["training"]
    return RobertaSweepConfig(
        model_name=model["name"], model_revision=revision,
        dataset_name=dataset["name"], dataset_config=dataset["config"],
        max_length=dataset["max_length"], lora_rank=rank,
        lora_alpha=lora["alpha"], target_modules=tuple(lora["target_modules"]),
        learning_rate=training["learning_rate"], batch_size=training["batch_size"],
        epochs=training["epochs"],
    )


def load_phase4_data(config, entry, split_path):
    split = json.loads(Path(split_path).read_text())
    raw = load_dataset(config["dataset"]["name"], config["dataset"]["config"])
    raw_train = raw[config["dataset"]["source_split"]]
    if getattr(raw_train, "_fingerprint", None) != split.get("dataset_fingerprint"):
        raise ValueError("current SST-2 training split fingerprint does not match phase4_split.json")
    validate_split(split, list(raw_train["label"]), list(raw_train["idx"]))
    tokenizer = load_tokenizer(make_sweep_config(config, entry["rank"]))

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=config["dataset"]["max_length"])

    encoded = raw.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format("torch")
    return encoded["train"].select(split["train_indices"]), encoded["train"].select(split["holdout_indices"]), tokenizer, split


def metrics(logits, labels, loss):
    predictions = logits.argmax(-1).tolist()
    labels = labels.tolist()
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(labels), "loss": loss / len(labels), "precision_positive": precision, "recall_positive": recall, "f1_positive": f1, "n": len(labels), "predictions": predictions, "labels": labels}


def smoke_test(config, configuration_id, seed):
    entry = next(item for item in config["configurations"] if item["id"] == configuration_id)
    set_seed(seed)
    sweep_config = make_sweep_config(config, entry["rank"])
    model = inject_lora_layers(load_model(sweep_config, seed), sweep_config, entry["layers"])
    names = [name for name, parameter in trainable_parameters(model)]
    expected = [f"encoder.layer.{layer}." for layer in entry["layers"]]
    assert all("classifier" in name or any(prefix in name for prefix in expected) for name in names)
    device = next(model.parameters()).device
    batch = {"input_ids": torch.ones((2, 8), dtype=torch.long, device=device), "attention_mask": torch.ones((2, 8), dtype=torch.long, device=device), "labels": torch.tensor([0, 1], device=device)}
    model.train()
    model(**batch).loss.backward()
    print(f"Phase 4 smoke test passed: {configuration_id}, seed={seed}, trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}")


def evaluate(model, dataset, collator, batch_size):
    model.eval()
    device = next(model.parameters()).device
    logits_all, labels_all, loss_sum = [], [], 0.0
    with torch.no_grad():
        for batch in _loader(dataset, batch_size, False, 0, collator):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            labels = batch.pop("labels")
            output = model(**batch, labels=labels)
            loss_sum += output.loss.item() * len(labels)
            logits_all.append(output.logits.cpu())
            labels_all.append(labels.cpu())
    return metrics(torch.cat(logits_all), torch.cat(labels_all), loss_sum)


def run(config, configuration_id, seed, split_path, output_dir):
    entry = next(item for item in config["configurations"] if item["id"] == configuration_id)
    set_seed(seed)
    run_id = f"{configuration_id}_seed{seed}"
    reservation = reserve_run(output_dir, run_id, {"configuration_id": configuration_id, "seed": seed})
    try:
        train_data, holdout_data, tokenizer, split = load_phase4_data(config, entry, split_path)
        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        sweep_config = make_sweep_config(config, entry["rank"])
        model = inject_lora_layers(load_model(sweep_config, seed), sweep_config, entry["layers"])
        params = [p for _, p in trainable_parameters(model)]
        optimizer = torch.optim.AdamW(params, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        loader = _loader(train_data, config["training"]["batch_size"], True, seed, collator)
        hook = register_frozen_prefix_detach(model, min(entry["layers"]))
        device = next(model.parameters()).device
        epoch_metrics = []
        started = time.time()
        try:
            for epoch in range(1, config["training"]["epochs"] + 1):
                model.train()
                total_loss, count = 0.0, 0
                for batch in loader:
                    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                    optimizer.zero_grad()
                    output = model(**batch)
                    output.loss.backward()
                    optimizer.step()
                    total_loss += output.loss.item() * len(batch["labels"])
                    count += len(batch["labels"])
                epoch_metrics.append({"epoch": epoch, "train_loss": total_loss / count})
        finally:
            hook.remove()
        holdout_metrics = evaluate(model, holdout_data, collator, config["training"]["batch_size"])
        checkpoint_dir = reservation.paths["directory"] / "checkpoint"
        model.save_pretrained(checkpoint_dir)
        predictions = {"run_id": run_id, "selected_epoch": config["training"]["epochs"], "checkpoint_rule": config["training"]["checkpoint_rule"], "predictions": holdout_metrics.pop("predictions"), "labels": holdout_metrics.pop("labels")}
        record = {"status": "artifact_ready", "run_id": run_id, "configuration": entry, "model": {"name": config["model"]["name"], "resolved_revision": config["model"]["resolved_revision"]}, "seed": seed, "split_manifest": str(split_path), "dataset_fingerprint": split["dataset_fingerprint"], "train_size": len(train_data), "holdout_size": len(holdout_data), "n_trainable_params": sum(p.numel() for p in params), "epoch_metrics": epoch_metrics, "holdout_metrics": holdout_metrics, "selected_epoch": config["training"]["epochs"], "checkpoint_rule": config["training"]["checkpoint_rule"], "checkpoint_path": str(checkpoint_dir), "runtime_sec": time.time() - started, "software": {name: version(name) for name in ("torch", "transformers", "peft", "datasets")}, "python": platform.python_version()}
        write_completed_artifacts(reservation, record, predictions)
        return record
    except Exception as exc:
        atomic_json(reservation.paths["status"], {"status": "failed", "run_id": run_id, "error": repr(exc)})
        raise
    finally:
        if not reservation.completed:
            reservation.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "phase4_config.json")
    parser.add_argument("--split", type=Path, default=ROOT / "phase4_split.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase4")
    parser.add_argument("--configuration", default="gradient_top3_r8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--archive-incomplete", action="store_true", help="explicitly archive an incomplete run before rerunning it")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke_test:
        smoke_test(config, args.configuration, args.seed)
    else:
        if args.archive_incomplete:
            archive_incomplete(args.output_dir, f"{args.configuration}_seed{args.seed}")
        run(config, args.configuration, args.seed, args.split, args.output_dir)


if __name__ == "__main__":
    main()
