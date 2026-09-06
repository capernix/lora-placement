import platform
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from .config import RobertaSweepConfig
from .data import load_sst2
from .importance import compute_importance
from .io import result_path, write_json_atomic
from .model import (inject_single_layer_lora, load_model, load_tokenizer,
                    register_frozen_prefix_detach, resolved_model_revision,
                    trainable_parameters)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "roberta"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _runtime_metadata():
    packages = {}
    for package in ("torch", "transformers", "peft", "datasets", "tokenizers", "accelerate"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "software": packages,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": commit,
    }


def _loader(dataset, batch_size, shuffle, seed, collate_fn):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, collate_fn=collate_fn,
                      num_workers=2, pin_memory=torch.cuda.is_available(),
                      persistent_workers=True)


def _evaluate(model, dataset, config, seed, collate_fn):
    model.eval()
    device = next(model.parameters()).device
    loader = _loader(dataset, config.batch_size, shuffle=False, seed=seed, collate_fn=collate_fn)
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            correct += (logits.argmax(-1) == labels).sum().item()
            total += len(labels)
    return correct / total


def _fit(model, train_data, validation_data, config, seed, collate_fn, selected_layer=None):
    params = [parameter for _, parameter in trainable_parameters(model)]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    loader = _loader(train_data, config.batch_size, shuffle=True, seed=seed, collate_fn=collate_fn)
    device = next(model.parameters()).device
    losses = []
    detach_hook = None
    if selected_layer is not None:
        # The prefix before the selected block is frozen. Detaching at this
        # boundary preserves gradients for LoRA/head while avoiding backward
        # traversal through frozen earlier blocks.
        detach_hook = register_frozen_prefix_detach(model, selected_layer)
    try:
        for _ in range(config.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                outputs.loss.backward()
                optimizer.step()
                epoch_loss += outputs.loss.detach()
                epoch_batches += 1
            losses.append(round((epoch_loss / epoch_batches).item(), 6))
    finally:
        if detach_hook is not None:
            detach_hook.remove()
    return _evaluate(model, validation_data, config, seed, collate_fn), losses


def _cached_cls_features(model, dataset, config, seed, collate_fn):
    """Extract frozen <s> features once for the head-only baseline."""
    model.roberta.eval()
    # This is inference-only; the locked training batch size remains 32.
    loader = _loader(dataset, max(config.batch_size, 1024), shuffle=False,
                     seed=seed, collate_fn=collate_fn)
    features, labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels.append(batch.pop("labels"))
            batch = {key: value.to(next(model.parameters()).device, non_blocking=True)
                     for key, value in batch.items()}
            features.append(model.roberta(**batch).last_hidden_state[:, 0, :].cpu())
    return torch.cat(features), torch.cat(labels)


def _fit_frozen_baseline(model, train_data, validation_data, config, seed, collate_fn):
    train_features, train_labels = _cached_cls_features(
        model, train_data, config, seed, collate_fn
    )
    val_features, val_labels = _cached_cls_features(
        model, validation_data, config, seed, collate_fn
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable_parameters(model)],
        lr=config.learning_rate,
    )
    loader = DataLoader(
        list(zip(train_features, train_labels)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    losses = []
    device = next(model.parameters()).device
    for _ in range(config.epochs):
        model.classifier.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.classifier(features.unsqueeze(1))
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(round(loss.item(), 6))
    model.classifier.eval()
    with torch.no_grad():
        accuracy = (model.classifier(val_features.to(device).unsqueeze(1)).argmax(-1) ==
                    val_labels.to(device)).float().mean().item()
    return accuracy, losses


def run_experiment(layer=0, seed=0, config=None, output_dir=RESULTS_DIR, filename=None):
    config = config or RobertaSweepConfig()
    tokenizer = load_tokenizer(config)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_data, validation_data, probe = load_sst2(
        tokenizer, config.max_length, config.probe_size, config.probe_seed
    )

    # Both models are initialized with the same seed, so their classifier heads match.
    baseline_model = load_model(config, seed)
    baseline_trainable = [(n, p.numel()) for n, p in trainable_parameters(baseline_model)]
    frozen_val_acc, _ = _fit_frozen_baseline(
        baseline_model, train_data, validation_data, config, seed, collate_fn
    )

    importance_model = load_model(config, seed)
    predictors = compute_importance(importance_model, probe, collate_fn, config.num_layers)
    del importance_model

    lora_model = load_model(config, seed)
    lora_model = inject_single_layer_lora(lora_model, config, layer)
    trainable = [(name, parameter.numel()) for name, parameter in trainable_parameters(lora_model)]
    backbone_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in lora_model.named_parameters()
        if name.startswith("base_model.model.roberta.") and "lora_" not in name
    }
    started = time.time()
    lora_val_acc, loss_curve = _fit(
        lora_model, train_data, validation_data, config, seed, collate_fn,
        selected_layer=layer,
    )
    elapsed = time.time() - started
    backbone_unchanged = all(
        torch.equal(before, dict(lora_model.named_parameters())[name].detach().cpu())
        for name, before in backbone_before.items()
    )

    record = {
        "model": config.model_name,
        "requested_revision": config.model_revision,
        "resolved_model_revision": resolved_model_revision(lora_model.base_model.model),
        "dataset": f"{config.dataset_name}/{config.dataset_config}",
        "layer": layer,
        "reported_layer": f"L{layer + 1}",
        "seed": seed,
        "max_length": config.max_length,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "target_modules": list(config.target_modules),
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "probe_size": config.probe_size,
        "probe_seed": config.probe_seed,
        "baseline_trainable_parameters": baseline_trainable,
        "trainable_parameters": trainable,
        "n_trainable_params": sum(count for _, count in trainable),
        "backbone_unchanged": backbone_unchanged,
        "frozen_val_acc": round(frozen_val_acc, 6),
        "lora_val_acc": round(lora_val_acc, 6),
        "utility_delta": round(lora_val_acc - frozen_val_acc, 6),
        "training_time_sec": round(elapsed, 3),
        "device": str(next(lora_model.parameters()).device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)
        if torch.cuda.is_available() else None,
        "gpu_peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if torch.cuda.is_available() else None,
        "loss_curve": loss_curve,
        "predictors": predictors,
        "runtime_metadata": _runtime_metadata(),
    }
    record["status"] = "completed"
    output = Path(output_dir) / (filename or result_path(output_dir, layer, seed).name)
    write_json_atomic(output, record)
    return record


def run_pilot(layer=0, seed=0, config=None):
    return run_experiment(layer, seed, config, RESULTS_DIR,
                          filename=f"pilot_layer{layer}_seed{seed}.json")
