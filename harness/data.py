"""
Synthetic classification task for the dummy-layer harness.

Labels are derived from the FROZEN backbone's own representation at
LABEL_SOURCE_LAYER, passed through a fixed random direction. This means
the task is genuinely "easier to solve" using information that survives
to that depth -- giving the toy sweep a real (if synthetic) layer-utility
pattern to recover, instead of pure noise.
"""
import torch
from .model import ToyBackbone, LABEL_SOURCE_LAYER, HIDDEN_DIM


def make_dataset(n_samples: int, seed: int, backbone_seed: int = 0, noise: float = 0.15):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, HIDDEN_DIM, generator=g)

    # use an untouched reference backbone (same weights every call) to derive labels
    ref = ToyBackbone(backbone_seed=backbone_seed)
    with torch.no_grad():
        _, acts = ref(X, return_activations=True)
        source_repr = acts[LABEL_SOURCE_LAYER + 1]  # +1: acts[0] is the raw input

    dir_g = torch.Generator().manual_seed(1000 + backbone_seed)
    direction = torch.randn(HIDDEN_DIM, generator=dir_g)
    score = source_repr @ direction
    score = score + noise * torch.randn(n_samples, generator=g)
    y = (score > score.median()).float()
    return X, y


def train_val_split(X, y, val_frac=0.3, seed=0):
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
