"""
harness/aggregate_results.py

Collects every results/layer{L}_seed{S}.json + results/importance_seed{S}.json,
builds the layer -> utility / layer -> {CKA, gradient, squared-gradient proxy} table, computes
Spearman correlation between each predictor and utility, prints a formatted
table, and saves plots to analysis/.

Usage:
    python -m harness.aggregate_results
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

PREDICTOR_COLS = ["cka", "grad_norm", "fisher"]


def load_train_runs():
    rows = []
    for f in sorted(RESULTS_DIR.glob("layer*_seed*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    return pd.DataFrame(rows)


def load_importance_runs():
    rows = []
    for f in sorted(RESULTS_DIR.glob("importance_seed*.json")):
        with open(f) as fh:
            rec = json.load(fh)
        for layer_str, cka_val in rec["cka"].items():
            rows.append({
                "seed": rec["seed"],
                "layer": int(layer_str),
                "cka": cka_val,
                "grad_norm": rec["grad_norm"][layer_str],
                "fisher": rec["fisher"][layer_str],
            })
    return pd.DataFrame(rows)


def build_table(train_df: pd.DataFrame, imp_df: pd.DataFrame) -> pd.DataFrame:
    utility = train_df.groupby("layer").agg(
        frozen_val_acc_mean=("frozen_val_acc", "mean"),
        frozen_val_acc_std=("frozen_val_acc", "std"),
        lora_val_acc_mean=("lora_val_acc", "mean"),
        lora_val_acc_std=("lora_val_acc", "std"),
        utility_delta_mean=("utility_delta", "mean"),
        utility_delta_std=("utility_delta", "std"),
    ).reset_index()

    predictors = imp_df.groupby("layer")[PREDICTOR_COLS].agg(["mean", "std"]).reset_index()
    predictors.columns = ["layer"] + [f"{col}_{stat}" for col in PREDICTOR_COLS for stat in ("mean", "std")]

    merged = pd.merge(utility, predictors, on="layer").sort_values("layer")
    return merged


def compute_correlations(table: pd.DataFrame) -> dict:
    corrs = {}
    for col in PREDICTOR_COLS:
        rho, pval = spearmanr(table[f"{col}_mean"], table["utility_delta_mean"])
        corrs[col] = {"spearman_rho": round(float(rho), 4), "p_value": round(float(pval), 4)}
    return corrs


def print_pretty_table(table: pd.DataFrame, corrs: dict):
    print()
    print("=" * 72)
    print(" LAYER -> UTILITY / IMPORTANCE-PREDICTOR TABLE (dummy-layer sweep)")
    print("=" * 72)
    display_df = table.copy()
    display_df["utility"] = display_df.apply(
        lambda r: f"{r.utility_delta_mean:.3f} +/- {r.utility_delta_std:.3f}", axis=1
    )
    display_df = display_df[["layer", "utility", "frozen_val_acc_mean", "lora_val_acc_mean"] +
                            [f"{col}_mean" for col in PREDICTOR_COLS]]
    display_df.columns = ["Layer", "Utility delta (mean +/- std)", "Frozen Val Acc", "LoRA Val Acc",
                          "CKA-input", "Grad Norm", "Squared-grad proxy"]
    print(tabulate(display_df, headers="keys", tablefmt="fancy_grid",
                    showindex=False, floatfmt=".4f"))

    print("\nSpearman correlation (predictor vs. utility), per the plan's checkpoint:")
    predictor_labels = {"cka": "CKA-input", "grad_norm": "Grad norm",
                        "fisher": "Squared-grad proxy"}
    corr_rows = [[predictor_labels[name], c["spearman_rho"], c["p_value"]]
                 for name, c in corrs.items()]
    print(tabulate(corr_rows, headers=["Predictor", "Spearman rho", "p-value"],
                    tablefmt="fancy_grid", floatfmt=".4f"))
    print()


def make_plots(table: pd.DataFrame, corrs: dict, train_df: pd.DataFrame):
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. layer -> utility curve (bar + error bars)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(table["layer"], table["utility_delta_mean"], yerr=table["utility_delta_std"],
           capsize=4, color="#4C72B0", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Layer index (LoRA placement)")
    ax.set_ylabel("Utility delta (LoRA - frozen)")
    ax.set_title("Layer -> adaptation utility (dummy-layer sweep)")
    ax.set_xticks(table["layer"])
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "layer_utility_curve.png", dpi=150)
    plt.close(fig)

    # 2. layer -> each predictor, normalized, overlaid with utility
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    norm_util = (table["utility_delta_mean"] - table["utility_delta_mean"].min()) / (
        table["utility_delta_mean"].max() - table["utility_delta_mean"].min() + 1e-9)
    ax.plot(table["layer"], norm_util, marker="o", linewidth=2.5, color="black",
            label="Utility delta (normalized)")
    colors = {"cka": "#DD8452", "grad_norm": "#55A868", "fisher": "#C44E52"}
    for col in PREDICTOR_COLS:
        norm_pred = (table[f"{col}_mean"] - table[f"{col}_mean"].min()) / (table[f"{col}_mean"].max() - table[f"{col}_mean"].min() + 1e-9)
        ax.plot(table["layer"], norm_pred, marker="s", linewidth=1.8, linestyle="--",
                color=colors[col], label=f"{col.upper()} (rho={corrs[col]['spearman_rho']})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Normalized value")
    ax.set_title("Predictors vs. utility, by layer")
    ax.set_xticks(table["layer"])
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "predictor_vs_utility.png", dpi=150)
    plt.close(fig)

    # 3. scatter: predictor value vs utility, one panel per predictor
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, PREDICTOR_COLS):
        ax.scatter(table[f"{col}_mean"], table["utility_delta_mean"], s=90, color=colors[col],
                   edgecolor="black", zorder=3)
        for _, row in table.iterrows():
            ax.annotate(f"L{int(row.layer)}", (row[f'{col}_mean'], row.utility_delta_mean),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel(col.upper())
        ax.set_ylabel("Utility delta")
        ax.set_title(f"{col.upper()} vs utility\nSpearman rho={corrs[col]['spearman_rho']}")
    fig.suptitle("Predictability check (RQ2): does each cheap signal track utility?")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "predictor_scatter.png", dpi=150)
    plt.close(fig)

    # 4. loss curves per layer (one seed each, first seed found)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for layer, grp in train_df.groupby("layer"):
        row = grp.iloc[0]
        ax.plot(row["loss_curve"], label=f"layer {layer}", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Training loss curves by LoRA layer placement")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "loss_curves.png", dpi=150)
    plt.close(fig)

    print(f"Saved 4 plots to {ANALYSIS_DIR}/")


def main():
    train_df = load_train_runs()
    imp_df = load_importance_runs()
    if train_df.empty or imp_df.empty:
        print("No results found yet -- run train_one_layer.py and importance.py first.")
        return

    table = build_table(train_df, imp_df)
    corrs = compute_correlations(table)

    print_pretty_table(table, corrs)
    make_plots(table, corrs, train_df)

    table.to_csv(ANALYSIS_DIR / "layer_utility_table.csv", index=False)
    with open(ANALYSIS_DIR / "correlations.json", "w") as f:
        json.dump(corrs, f, indent=2)

    best_layer = int(table.loc[table["utility_delta_mean"].idxmax(), "layer"])
    best_predictor = max(corrs, key=lambda k: corrs[k]["spearman_rho"])
    raw = pd.merge(train_df, imp_df, on=["layer", "seed"], how="inner")
    raw.to_csv(ANALYSIS_DIR / "raw_layer_seed_results.csv", index=False)
    print(f"Saved raw per-seed joined results to {ANALYSIS_DIR / 'raw_layer_seed_results.csv'}")
    print(f"Checkpoint verdict: best empirical layer by utility delta = {best_layer}; "
          f"best predictor by Spearman rho = {best_predictor.upper()} "
          f"(rho={corrs[best_predictor]['spearman_rho']})")


if __name__ == "__main__":
    main()
