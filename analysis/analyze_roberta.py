#!/usr/bin/env python3
"""Reproducible RQ1/RQ2/RQ3 analysis for completed RoBERTa matrix results."""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {(layer, seed) for layer in range(12) for seed in (0, 1, 2)}
META_KEYS = ("model", "requested_revision", "resolved_model_revision", "dataset", "max_length",
             "lora_rank", "lora_alpha", "target_modules", "learning_rate", "batch_size",
             "epochs", "probe_size", "probe_seed")


def discover(result_dir):
    pattern = re.compile(r"^layer(\d{2})_seed(\d{2})\.json$")
    rows, paths, extras = [], [], []
    for path in sorted(result_dir.glob("*.json")):
        match = pattern.match(path.name)
        if not match:
            extras.append(path)
            continue
        layer, seed = map(int, match.groups())
        paths.append((layer, seed, path))
        with path.open() as handle:
            record = json.load(handle)
        if record.get("status") != "completed":
            raise ValueError(f"{path}: status is not completed")
        row = dict(record)
        predictors = record.get("predictors", {})
        row["cka"] = predictors.get("cka_input", {}).get(str(layer), np.nan)
        row["grad_norm"] = predictors.get("grad_norm", {}).get(str(layer), np.nan)
        row["squared_gradient_proxy"] = predictors.get("squared_gradient_proxy", {}).get(str(layer), np.nan)
        rows.append(row)
    pairs = [(l, s) for l, s, _ in paths]
    duplicates = sorted(pair for pair, count in __import__("collections").Counter(pairs).items() if count > 1)
    missing = sorted(EXPECTED - set(pairs))
    unexpected = sorted(set(pairs) - EXPECTED)
    if duplicates or unexpected:
        raise ValueError(f"Duplicate/unexpected matrix pairs: duplicates={duplicates}, unexpected={unexpected}")
    return rows, missing, extras


def validate(rows, missing, extras, allow_incomplete):
    errors = []
    if missing and not allow_incomplete:
        errors.append(f"missing matrix pairs: {missing}")
    if len({(r["layer"], r["seed"]) for r in rows}) != len(rows):
        errors.append("duplicate layer/seed pairs")
    if len(rows) > 36:
        errors.append("more than 36 canonical rows")
    for key in META_KEYS:
        values = {json.dumps(r.get(key), sort_keys=True) for r in rows}
        if len(values) != 1:
            errors.append(f"inconsistent metadata: {key}={sorted(values)}")
    if any(r.get("n_trainable_params") != rows[0].get("n_trainable_params") for r in rows):
        errors.append("trainable parameter count differs across runs")
    if any(r.get("backbone_unchanged") is not True for r in rows):
        errors.append("one or more runs have backbone_unchanged != true")
    numeric = ["frozen_val_acc", "lora_val_acc", "utility_delta", "cka", "grad_norm", "squared_gradient_proxy"]
    for r in rows:
        for key in numeric:
            if not np.isfinite(r.get(key, np.nan)):
                errors.append(f"non-finite {key} at layer={r.get('layer')} seed={r.get('seed')}")
    if errors:
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors))


def bootstrap_ci(x, y, statistic, seed=42, n=5000):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        ix = rng.integers(0, len(x), len(x))
        values.append(statistic(x[ix], y[ix]))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def rank_desc(series):
    return series.rank(method="min", ascending=False).astype(int)


def make_figures(raw, layer, predictors, out):
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.errorbar(x, layer["mean_utility_delta"], yerr=layer["std_utility_delta"], fmt="o-", capsize=4)
    ax.set(xlabel="Transformer layer", ylabel="Utility delta (LoRA − frozen accuracy)", xticks=x,
           xticklabels=[f"L{i}" for i in x])
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures / "layer_utility.png"); plt.close(fig)
    pivot = raw.pivot(index="reported_layer", columns="seed", values="utility_delta")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    for seed in sorted(pivot.columns):
        ax.plot(range(1, 13), pivot[seed].reindex([f"L{i}" for i in x]), "o-", label=f"seed {seed}")
    ax.set(xlabel="Transformer layer", ylabel="Utility delta", xticks=x, xticklabels=[f"L{i}" for i in x]); ax.legend()
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures / "seed_utility.png"); plt.close(fig)
    for name, label in (("cka", "CKA-input"), ("grad_norm", "Gradient norm"), ("squared_gradient_proxy", "Squared-gradient proxy")):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
        ax.scatter(predictors[name], predictors["mean_utility_delta"], s=45)
        for _, row in predictors.iterrows(): ax.annotate(row.reported_layer, (row[name], row.mean_utility_delta), xytext=(4, 4), textcoords="offset points")
        ax.set(xlabel=label, ylabel="Mean utility delta"); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(figures / f"{name}_vs_utility.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    for name, label in (("cka", "CKA"), ("grad_norm", "Gradient"), ("squared_gradient_proxy", "Squared-gradient")):
        order = predictors.sort_values(name, ascending=False).layer.tolist()
        oracle = int(predictors.loc[predictors.mean_utility_delta.idxmax(), "layer"])
        vals = [int(oracle in order[:k]) for k in (1, 2, 3)]
        ax.plot([1, 2, 3], vals, "o-", label=label)
    ax.set(xlabel="Candidate budget k", ylabel="Recovery of oracle-best layer", xticks=[1, 2, 3], ylim=(-.05, 1.05)); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "predictor_top_k_recovery.png"); plt.close(fig)


def analyze(rows, missing, extras, out):
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows).sort_values(["layer", "seed"])
    raw["reported_layer"] = raw["reported_layer"].fillna(raw["layer"].map(lambda x: f"L{x+1}"))
    columns = ["layer", "reported_layer", "seed", "frozen_val_acc", "lora_val_acc", "utility_delta", "cka", "grad_norm", "squared_gradient_proxy", "n_trainable_params", "training_time_sec", "backbone_unchanged", "model", "resolved_model_revision", "dataset", "max_length", "lora_rank", "lora_alpha", "target_modules", "learning_rate", "batch_size", "epochs", "probe_size", "probe_seed"]
    raw[[c for c in columns if c in raw]].to_csv(out / "roberta_master.csv", index=False)
    layer = raw.groupby(["layer", "reported_layer"], as_index=False).agg(mean_frozen_acc=("frozen_val_acc", "mean"), mean_lora_acc=("lora_val_acc", "mean"), mean_utility_delta=("utility_delta", "mean"), std_utility_delta=("utility_delta", "std"), min_utility_delta=("utility_delta", "min"), max_utility_delta=("utility_delta", "max"), n_runs=("utility_delta", "count"))
    seed_values = raw.pivot(index=["layer", "reported_layer"], columns="seed", values="utility_delta").reset_index()
    seed_values.columns = ["layer", "reported_layer"] + [f"utility_delta_seed{int(c)}" for c in seed_values.columns[2:]]
    layer = layer.merge(seed_values, on=["layer", "reported_layer"], how="left")
    layer.to_csv(out / "roberta_layer_summary.csv", index=False)
    predictors = raw.groupby(["layer", "reported_layer"], as_index=False).agg(mean_utility_delta=("utility_delta", "mean"), cka=("cka", "first"), grad_norm=("grad_norm", "first"), squared_gradient_proxy=("squared_gradient_proxy", "first"))
    predictors["utility_rank"] = rank_desc(predictors.mean_utility_delta)
    for name in ("cka", "grad_norm", "squared_gradient_proxy"): predictors[f"{name}_rank"] = rank_desc(predictors[name])
    predictor_rows = []
    rank_rows = []
    for name in ("cka", "grad_norm", "squared_gradient_proxy"):
        x, y = predictors[name].to_numpy(), predictors.mean_utility_delta.to_numpy()
        pearson = stats.pearsonr(x, y); spearman = stats.spearmanr(x, y)
        ci = bootstrap_ci(x, y, lambda a, b: stats.spearmanr(a, b).statistic)
        predictor_rows.append({"predictor": name, "pearson_r": pearson.statistic, "pearson_p": pearson.pvalue, "spearman_rho": spearman.statistic, "spearman_p": spearman.pvalue, "spearman_bootstrap_ci_low": ci[0], "spearman_bootstrap_ci_high": ci[1], "predictor_top_layer": predictors.loc[predictors[name].idxmax(), "reported_layer"], "utility_top_layer": predictors.loc[predictors.mean_utility_delta.idxmax(), "reported_layer"]})
        rank_rows.append({"predictor": name, "spearman_rank_rho": spearman.statistic, "top1_agreement": int(predictors.loc[predictors[name].idxmax(), "layer"] == predictors.loc[predictors.mean_utility_delta.idxmax(), "layer"]), "top3_overlap": len(set(predictors.nlargest(3, name).layer) & set(predictors.nlargest(3, "mean_utility_delta").layer)), "top5_overlap": len(set(predictors.nlargest(5, name).layer) & set(predictors.nlargest(5, "mean_utility_delta").layer))})
    pd.DataFrame(predictor_rows).to_csv(out / "roberta_predictor_summary.csv", index=False)
    predictors[["layer", "reported_layer", "utility_rank", "cka_rank", "grad_norm_rank", "squared_gradient_proxy_rank"]].rename(columns={"squared_gradient_proxy_rank": "fisher_rank"}).to_csv(out / "roberta_ranking_analysis.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(out / "roberta_ranking_metrics.csv", index=False)
    oracle = int(predictors.loc[predictors.mean_utility_delta.idxmax(), "layer"])
    budget_rows = []
    for name in ("cka", "grad_norm", "squared_gradient_proxy"):
        order = predictors.sort_values(name, ascending=False).layer.tolist()
        oracle_rank = order.index(oracle) + 1
        for k in (1, 2, 3): budget_rows.append({"predictor": name, "k": k, "selected_layers": ",".join(f"L{i+1}" for i in order[:k]), "oracle_best_layer": f"L{oracle+1}", "oracle_best_rank": oracle_rank, "oracle_best_recovered": int(oracle in order[:k]), "interpretation": "top-k recovery only; no joint multi-layer utility observed"})
    budget = pd.DataFrame(budget_rows); budget.to_csv(out / "roberta_budget_selection.csv", index=False)
    make_figures(raw, layer, predictors, out)
    # RQ1: complete cases only; Friedman is appropriate for repeated seeds.
    wide = raw.pivot(index="seed", columns="layer", values="utility_delta").dropna(axis=0)
    if wide.shape[0] >= 2:
        fried = stats.friedmanchisquare(*[wide[c] for c in wide.columns]); w = fried.statistic / (wide.shape[0] * (len(wide.columns) - 1))
        rq1_text = f"Friedman repeated-measures test on complete seeds (n={len(wide)}, layers={len(wide.columns)}): statistic={fried.statistic:.6g}, df={len(wide.columns)-1}, p={fried.pvalue:.6g}, Kendall W={w:.6g}. Missing pairs: {missing}."
        pd.DataFrame([{"method": "Friedman repeated-measures test", "complete_seeds": len(wide), "layers": len(wide.columns), "statistic": fried.statistic, "df": len(wide.columns) - 1, "p_value": fried.pvalue, "kendall_W": w, "missing_pairs": str(missing)}]).to_csv(out / "roberta_rq1_statistics.csv", index=False)
    else: rq1_text = f"Insufficient complete seeds for Friedman test. Missing pairs: {missing}."
    (out / "roberta_rq1_statistics.txt").write_text(rq1_text + "\n")
    # Summary and leave-one-seed-out diagnostics.
    lines = ["# RoBERTa analysis summary", "", "## Experimental validation", f"Valid locked-schema matrix rows: {len(raw)} / 36.", f"Missing pairs: {missing or 'none'}.", f"Extra non-matrix JSON artifacts preserved: {[p.name for p in extras] or 'none'}.", "", "## RQ1 result", rq1_text, f"Observed best mean-utility layer: L{oracle+1} ({predictors.mean_utility_delta.max():.6f}).", "", "## RQ2 result", "Predictor correlations and ranking diagnostics are in the CSV outputs; they are layer-level diagnostics (n=12), not independent seed-level predictive validation.", "", "## RQ3 result", "Budget analysis reports top-k recovery of the oracle-best single layer only. It does not measure joint adaptation of k layers.", "", "## Robustness observations", "Leave-one-seed-out summaries are reported below; predictors use the fixed 512-example probe with probe_seed=42, which is a limitation rather than a probe robustness test."]
    for seed in (0, 1, 2):
        subset = raw[raw.seed != seed].groupby("layer").utility_delta.mean().sort_values(ascending=False)
        lines.append(f"- Leave out seed {seed}: top layer(s) {', '.join('L'+str(i+1) for i in subset.index[:3])}.")
    leave_rows = []
    for seed in (0, 1, 2):
        subset = raw[raw.seed != seed].groupby("layer").utility_delta.mean().sort_values(ascending=False)
        for rank, (layer_id, value) in enumerate(subset.items(), 1):
            leave_rows.append({"held_out_seed": seed, "layer": layer_id, "reported_layer": f"L{layer_id+1}", "mean_utility_delta_remaining_seeds": value, "rank": rank})
    pd.DataFrame(leave_rows).to_csv(out / "roberta_leave_one_seed.csv", index=False)
    lines += ["", "## Safe paper claims", "Placement utility varies across layers, subject to the incomplete L4/seed2 matrix cell. Predictor associations should be described as exploratory layer-level evidence.", "", "## Claims not supported", "Do not claim a complete 36-run matrix, causal predictor validity, independent generalization, or measured joint multi-layer LoRA utility until L4/seed2 is completed."]
    (out / "roberta_analysis_summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "roberta")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "roberta")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows, missing, extras = discover(args.results_dir)
    validate(rows, missing, extras, args.allow_incomplete)
    analyze(rows, missing, extras, args.output_dir)
    print(f"Analyzed {len(rows)} valid matrix runs; missing={missing}; output={args.output_dir}")


if __name__ == "__main__":
    main()
