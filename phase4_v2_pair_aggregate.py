"""Descriptive pairwise scores and additive layer-effect analysis."""
from __future__ import annotations
import math

def rank_pairs(records, score_key="validation_utility"):
    return sorted(records, key=lambda r: (-float(r[score_key]), tuple(r["layers"])))

def recovery_metrics(records, score_key, observed_key="validation_utility"):
    ranked = rank_pairs(records, score_key); observed = rank_pairs(records, observed_key)
    best = observed[0]["pair_id"]; n = len(records)
    result = {"top1_recovery": ranked[0]["pair_id"] == best, "observed_best_pair_id": best,
              "observed_screening_percentile": 100.0 * (1 + [r["pair_id"] for r in ranked].index(best)) / n}
    for k in (1, 5, 10): result[f"top{k}_recovery"] = best in [r["pair_id"] for r in ranked[:k]]
    for k in (1, 5, 10): result[f"mean_utility_top{k}"] = sum(float(r[observed_key]) for r in ranked[:min(k, n)]) / min(k, n)
    result["regret"] = float(observed[0][observed_key]) - float(ranked[0][observed_key])
    return result

def add_prespecified_scores(records, gradient_scores, utility_ranks):
    output = []
    for record in records:
        i, j = record["layers"]
        row = dict(record); row.update({"gradient_score": gradient_scores[i] + gradient_scores[j], "phase3_utility_rank_sum": utility_ranks[i] + utility_ranks[j], "depth_score": i + j})
        output.append(row)
    return output

def additive_ols(records, value_key="validation_utility"):
    """OLS with sum-to-zero layer effects; numpy/scipy-free implementation."""
    layers = sorted({x for r in records for x in r["layers"]}); n = len(layers)
    # Reference-layer parameterization is equivalent to sum-to-zero effects.
    import numpy as np
    X = np.array([[1.0] + [float(i in r["layers"]) - float(i == layers[-1] and layers[-1] in r["layers"]) for i in layers[:-1]] for r in records])
    y = np.array([float(r[value_key]) for r in records])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; fitted = X @ beta; residuals = y - fitted
    effects = {layer: float(beta[k + 1]) for k, layer in enumerate(layers[:-1])}
    effects[layers[-1]] = -sum(effects.values())
    return {"model": "U_ij = mu + alpha_i + alpha_j + epsilon_ij", "intercept": float(beta[0]), "layer_effects_sum_to_zero": effects, "residuals": {r["pair_id"]: float(e) for r, e in zip(records, residuals)}, "residual_interpretation": "not definitive interactions"}
