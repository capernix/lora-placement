#!/usr/bin/env python3
"""Leakage-controlled Phase 4 v2 gradient layer selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_payload(payload):
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def accumulate_probe_gradients(model, batches, target_parameters, total_examples):
    """Return gradients of the complete probe mean loss, accumulated by sum.

    ``outputs.loss`` is the mean loss for one batch. Multiplying by
    ``batch_size / total_examples`` turns each batch contribution into its
    contribution to the complete probe mean loss.
    """
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("gradient scoring requires torch") from exc
    model.eval()
    accumulators = [torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    for parameter in target_parameters]
    seen = 0
    for batch in batches:
        batch_size = int(batch["labels"].shape[0])
        if batch_size <= 0:
            raise ValueError("empty scoring batch")
        seen += batch_size
        outputs = model(**batch)
        scaled_loss = outputs.loss * (batch_size / total_examples)
        gradients = torch.autograd.grad(scaled_loss, target_parameters,
                                        retain_graph=False, allow_unused=False)
        for accumulator, gradient in zip(accumulators, gradients):
            accumulator.add_(gradient.detach())
    if seen != total_examples:
        raise ValueError(f"scored {seen} examples, expected {total_examples}")
    return accumulators


def full_probe_gradients(model, batch, target_parameters):
    """Reference gradient for one fixed, full-probe batch."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("gradient scoring requires torch") from exc
    model.eval()
    outputs = model(**batch)
    return [gradient.detach() for gradient in torch.autograd.grad(
        outputs.loss, target_parameters, retain_graph=False, allow_unused=False)]


def layer_scores_from_gradients(gradients_by_layer):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("gradient scoring requires torch") from exc
    scores = OrderedDict()
    for layer, gradients in sorted(gradients_by_layer.items()):
        if len(gradients) != 2:
            raise ValueError(f"layer {layer} must contain query and value gradients")
        vector = torch.cat([gradient.reshape(-1) for gradient in gradients])
        scores[str(layer)] = float(vector.norm().item())
    return scores


def aggregate_scorer_scores(scores_by_seed, tie_tolerance=1e-12):
    if not scores_by_seed:
        raise ValueError("no scorer scores provided")
    layers = set(scores_by_seed[0])
    if not layers or any(set(scores) != layers for scores in scores_by_seed):
        raise ValueError("scorer score tables have inconsistent layers")
    mean_scores = {
        layer: sum(float(scores[layer]) for scores in scores_by_seed) / len(scores_by_seed)
        for layer in layers
    }

    def compare_key(item):
        return (-item[1], int(item[0]))

    ranking = sorted(mean_scores.items(), key=compare_key)
    # The explicit tolerance is recorded for auditability; ordering remains
    # deterministic even when values are numerically close.
    selected = [int(layer) for layer, _ in ranking[:3]]
    return {
        "mean_scores": {layer: mean_scores[layer] for layer in sorted(mean_scores, key=int)},
        "ranking": [int(layer) for layer, _ in ranking],
        "selected_layers": sorted(selected),
        "tie_tolerance": tie_tolerance,
    }


def validate_target_parameter_names(target_parameters, layer):
    names = [name for name, _ in target_parameters]
    expected = {
        f"roberta.encoder.layer.{layer}.attention.self.query.weight",
        f"roberta.encoder.layer.{layer}.attention.self.value.weight",
    }
    if set(names) != expected:
        raise ValueError(f"unexpected scorer parameters for layer {layer}: {names}")


def prepare_scorer_model(model, layers):
    """Freeze the model, then enable only query/value weights for scoring."""
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    target_parameters = OrderedDict()
    for layer in layers:
        block = model.roberta.encoder.layer[layer]
        query = block.attention.self.query.weight
        value = block.attention.self.value.weight
        query.requires_grad = True
        value.requires_grad = True
        target_parameters[layer] = [(
            f"roberta.encoder.layer.{layer}.attention.self.query.weight", query),
            (f"roberta.encoder.layer.{layer}.attention.self.value.weight", value),
        ]
        validate_target_parameter_names(target_parameters[layer], layer)
    model.eval()
    return target_parameters


def collision_with_random(selected_layers, random_layers=(8, 9, 11)):
    return sorted(selected_layers) == sorted(random_layers)


def score_model_on_probe(model, probe_batches, target_parameters_by_layer, total_examples):
    gradients_by_layer = {}
    flat_targets = []
    for layer in sorted(target_parameters_by_layer):
        params = target_parameters_by_layer[layer]
        validate_target_parameter_names(params, layer)
        flat_targets.extend(parameter for _, parameter in params)
    accumulated = accumulate_probe_gradients(model, probe_batches, flat_targets, total_examples)
    offset = 0
    for layer in sorted(target_parameters_by_layer):
        gradients_by_layer[layer] = accumulated[offset:offset + 2]
        offset += 2
    return layer_scores_from_gradients(gradients_by_layer)


def build_selection_artifact(config, split_manifest, scores_by_seed, probe_hash):
    aggregated = aggregate_scorer_scores(scores_by_seed)
    random_layers = config["selection"]["random_a_layers"]
    if collision_with_random(aggregated["selected_layers"], random_layers):
        raise RuntimeError("gradient selection collides with random_a_r8; protocol decision required")
    artifact = {
        "protocol_version": config["protocol_version"],
        "artifact_type": "phase4_v2_selection",
        "dataset_fingerprint": config["dataset"]["expected_fingerprint"],
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "selection_probe_hash": probe_hash,
        "scorer_seeds": config["selection"]["scorer_seeds"],
        "score_definition": config["selection"]["score"],
        "scores_by_seed": scores_by_seed,
        **aggregated,
        "random_a_layers": random_layers,
        "collision_policy": config["selection"]["collision_policy"],
        "collision": False,
    }
    artifact["content_sha256"] = sha256_payload(artifact)
    return artifact


def _load_json(path):
    return json.loads(Path(path).read_text())


def _set_all_seeds(seed):
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _score_from_local_dataset(config, split, local_files_only=False):
    """Score the clean probe using locally cached model/data dependencies."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    raw = load_dataset(config["dataset"]["name"], config["dataset"]["config"])
    train = raw[config["dataset"]["source_split"]]
    if getattr(train, "_fingerprint", None) != config["dataset"]["expected_fingerprint"]:
        raise ValueError("dataset fingerprint mismatch")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], revision=config["model"]["tokenizer_revision"], local_files_only=local_files_only)
    pool = split["pools"]["selection_probe"]
    encoded = tokenizer([train["sentence"][i] for i in pool["indices"]], truncation=True,
                        padding="max_length", max_length=config["dataset"]["max_length"], return_tensors="pt")
    labels = torch.tensor(pool["labels"], dtype=torch.long)
    batch_size = config["training"]["batch_size"]
    batches = []
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        batch = {key: value[start:end] for key, value in encoded.items()}
        batch["labels"] = labels[start:end]
        batches.append(batch)
    scores_by_seed = []
    for seed in config["selection"]["scorer_seeds"]:
        _set_all_seeds(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model"]["name"], revision=config["model"]["resolved_revision"],
            num_labels=2, local_files_only=local_files_only)
        targets = prepare_scorer_model(model, range(12))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        device_batches = [{key: value.to(device) for key, value in batch.items()} for batch in batches]
        scores_by_seed.append(score_model_on_probe(model, device_batches, targets, len(labels)))
    return scores_by_seed


def main():
    parser = argparse.ArgumentParser(description="Run clean v2 selection; requires model/data dependencies")
    parser.add_argument("--config", type=Path, default=Path("phase4_v2_config.json"))
    parser.add_argument("--split", type=Path, default=Path("phase4_v2_split.json"))
    parser.add_argument("--output", type=Path, default=Path("phase4_v2_selection.json"))
    parser.add_argument("--scores-json", type=Path,
                        help="optional precomputed score JSON; otherwise score locally cached data/model")
    parser.add_argument("--local-files-only", action="store_true",
                        help="forbid model/dataset downloads when scoring")
    args = parser.parse_args()
    config = _load_json(args.config)
    split = _load_json(args.split)
    if split.get("dataset_fingerprint") != config["dataset"]["expected_fingerprint"]:
        raise ValueError("split fingerprint does not match config")
    if args.scores_json:
        scores = _load_json(args.scores_json)
        scores_by_seed = scores["scores_by_seed"]
        probe_hash = scores["selection_probe_hash"]
    else:
        scores_by_seed = _score_from_local_dataset(config, split, args.local_files_only)
        probe_hash = sha256_payload(split["pools"]["selection_probe"])
    artifact = build_selection_artifact(config, split, scores_by_seed, probe_hash)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "content_sha256": artifact["content_sha256"],
                      "selected_layers": artifact["selected_layers"]}, indent=2))


if __name__ == "__main__":
    main()
