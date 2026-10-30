"""Pair enumeration and deterministic execution ordering."""
from __future__ import annotations
import itertools, random

def enumerate_pairs(layer_count=12):
    if layer_count < 2: raise ValueError("at least two layers required")
    return [(i, j) for i, j in itertools.combinations(range(layer_count), 2)]

def canonical_pair_id(pair):
    i, j = sorted(pair)
    if i == j or i < 0 or j >= 12: raise ValueError("invalid unordered pair")
    return f"layer_{i:02d}_{j:02d}"

def permute_pairs(pairs, seed=20261036):
    result = list(pairs)
    random.Random(seed).shuffle(result)
    return result
