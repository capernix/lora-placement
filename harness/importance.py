"""
harness/importance.py

Computes the three cheap, pre-fine-tuning layer-importance predictors:
CKA-input, gradient norm, and an empirical squared-gradient /
diagonal-Fisher-style proxy.
All are computed on a frozen backbone + freshly-initialized head, using
a fixed probe batch -- BEFORE any LoRA training happens.

Usage:
    python3 -m harness.importance --seed 0 --probe-size 256
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from .model import ToyBackbone, NUM_LAYERS
from .data import make_dataset

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two activation matrices (n_samples x dim)."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = (X.T @ Y).norm(p="fro") ** 2
    xtx = (X.T @ X).norm(p="fro")
    yty = (Y.T @ Y).norm(p="fro")
    denom = xtx * yty
    if denom.item() == 0:
        return 0.0
    return (xty / denom).item()


def compute_importance(seed: int, probe_size: int = 256, backbone_seed: int = 0):
    torch.manual_seed(seed)
    X, y = make_dataset(n_samples=probe_size, seed=1_000_000 + seed, backbone_seed=backbone_seed)

    model = ToyBackbone(backbone_seed=backbone_seed)
    # head must be trainable-but-fresh so gradient/Fisher reflect a realistic
    # pre-fine-tuning state, not a fully-converged one
    for p in model.head.parameters():
        p.requires_grad = True
    # backbone weights need requires_grad=True BEFORE the forward pass so
    # autograd actually records them in the graph
    for blk in model.blocks:
        blk.linear.weight.requires_grad = True
        blk.linear.bias.requires_grad = True

    logits, acts = model(X, return_activations=True)
    final_repr = acts[-1].detach()

    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, y)

    # Gradient norm + empirical squared-gradient / diagonal-Fisher-style
    # proxy per block's weight. These are task-aware because they use loss/y.
    grad_norms = {}
    fisher = {}

    grads = torch.autograd.grad(loss, [blk.linear.weight for blk in model.blocks], retain_graph=True)
    for i, g in enumerate(grads):
        grad_norms[i] = g.norm().item()
        fisher[i] = (g ** 2).mean().item()

    # CKA: similarity between each layer's activation and the RAW INPUT
    # representation, on the same probe batch -- i.e. how much this layer's
    # representation has drifted from the input. A legitimate, cheap,
    # pre-fine-tuning signal (no label leakage, no trivial self-match).
    reference_repr = acts[0].detach()
    cka = {}
    for i in range(NUM_LAYERS):
        layer_repr = acts[i + 1].detach()
        cka[i] = linear_cka(layer_repr, reference_repr)

    record = {
        "seed": seed,
        "probe_size": probe_size,
        "cka": cka,
        "grad_norm": grad_norms,
        "fisher": fisher,
    }
    out_path = RESULTS_DIR / f"importance_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--probe-size", type=int, default=256)
    args = ap.parse_args()
    record = compute_importance(args.seed, probe_size=args.probe_size)
    print(f"[importance] seed={args.seed} computed CKA-input/grad-norm/squared-gradient proxy for "
          f"{len(record['cka'])} layers")


if __name__ == "__main__":
    main()
