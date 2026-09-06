"""
harness/train_one_layer.py

Trains ONE LoRA config: a single backbone layer gets a LoRA adapter,
everything else stays frozen except the task head. Logs val accuracy,
loss curve, param count, and wall-clock time to results/<run_id>.json.

Usage:
    python -m harness.train_one_layer --layer 3 --seed 0
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from .model import ToyBackbone, LORA_RANK, LORA_ALPHA
from .data import make_dataset, train_val_split

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        pred = (torch.sigmoid(logits) > 0.5).float()
    return (pred == y).float().mean().item()


def _train(model, X_tr, y_tr, X_val, y_val, epochs, lr, verbose=False, layer=None, seed=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_curve = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        loss_curve.append(round(loss.item(), 5))
        if verbose and epoch % 10 == 0:
            print(f"  layer={layer} seed={seed} epoch={epoch:03d} loss={loss.item():.4f}")
    return {
        "val_acc": _evaluate(model, X_val, y_val),
        "train_acc": _evaluate(model, X_tr, y_tr),
        "loss_curve": loss_curve,
    }


def run(layer: int, seed: int, epochs: int = 60, lr: float = 0.05,
        n_samples: int = 800, backbone_seed: int = 0, verbose: bool = False):
    torch.manual_seed(seed)

    X, y = make_dataset(n_samples=n_samples, seed=seed, backbone_seed=backbone_seed)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, val_frac=0.3, seed=seed)

    # Train a matched frozen-backbone baseline: only the task head learns.
    torch.manual_seed(seed)
    baseline_model = ToyBackbone(backbone_seed=backbone_seed)
    baseline = _train(baseline_model, X_tr, y_tr, X_val, y_val, epochs, lr)

    # Reset initialization so the LoRA run starts from the same head weights.
    torch.manual_seed(seed)
    model = ToyBackbone(backbone_seed=backbone_seed)
    model.inject_lora(layer, rank=LORA_RANK, alpha=LORA_ALPHA)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)

    t0 = time.time()
    lora = _train(model, X_tr, y_tr, X_val, y_val, epochs, lr,
                  verbose=verbose, layer=layer, seed=seed)
    elapsed = time.time() - t0

    lora_val_acc = lora["val_acc"]
    frozen_val_acc = baseline["val_acc"]

    record = {
        "run_id": f"layer{layer}_seed{seed}",
        "layer": layer,
        "seed": seed,
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "epochs": epochs,
        "lr": lr,
        "n_trainable_params": n_trainable,
        "frozen_val_acc": round(frozen_val_acc, 4),
        "lora_val_acc": round(lora_val_acc, 4),
        "utility_delta": round(lora_val_acc - frozen_val_acc, 4),
        "train_acc": round(lora["train_acc"], 4),
        "final_loss": lora["loss_curve"][-1],
        "loss_curve": lora["loss_curve"],
        "elapsed_sec": round(elapsed, 3),
    }

    out_path = RESULTS_DIR / f"{record['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    record = run(args.layer, args.seed, epochs=args.epochs, verbose=args.verbose)
    print(f"[train_one_layer] layer={args.layer} seed={args.seed} "
          f"lora_val_acc={record['lora_val_acc']:.4f} params={record['n_trainable_params']}")


if __name__ == "__main__":
    main()
