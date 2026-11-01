#!/usr/bin/env python3
"""Descriptive Phase 4 allocation summary and pre-specified comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

CONFIGS = ["uniform_all_r2", "utility_top4_r6", "utility_top6_r4", "early3_r8",
           "utility_top3_r8", "spread_k3_r8", "strided_k6_r4"]
PAIRS = [("early3_r8", "utility_top3_r8"), ("utility_top6_r4", "strided_k6_r4"),
         ("early3_r8", "spread_k3_r8"), ("utility_top3_r8", "spread_k3_r8")]


def records(roots):
    output = []
    for root in roots:
        for path in Path(root).rglob("result.json"):
            try:
                value = json.loads(path.read_text())
                if value.get("configuration", {}).get("id") in CONFIGS and value.get("status") == "artifact_ready":
                    output.append(value)
            except (OSError, json.JSONDecodeError):
                continue
    return output


def mean(values):
    return statistics.mean(values) if values else None


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def make_summary(rows):
    summaries = []
    for config in CONFIGS:
        group = [r for r in rows if r["configuration"]["id"] == config]
        if group:
            accuracies = {r["seed"]: r["validation_metrics"]["accuracy"] for r in group}
            utilities = [r.get("utility") for r in group if r.get("utility") is not None]
            first = group[0]["configuration"]
            reported = first.get("reported_layers") or [f"L{layer + 1}" for layer in first.get("layers", [])]
            layers = ",".join(reported)
            rank, alpha = first["rank"], first["alpha"]
        else:
            accuracies, utilities, layers, rank, alpha = {}, [], "", "", ""
        summaries.append({"Configuration": config, "Layers": layers,
                          "Rank": rank, "Alpha": alpha, "Seeds": ",".join(map(str, sorted(accuracies))),
                          "Mean validation accuracy": mean(list(accuracies.values())),
                          "Standard deviation": sd(list(accuracies.values())),
                          "Mean LoRA utility": mean(utilities),
                          "Individual seed accuracies": json.dumps({str(k): accuracies[k] for k in sorted(accuracies)})})
    return summaries


def comparisons(rows):
    by = {}
    for row in rows:
        by.setdefault(row["configuration"]["id"], {})[row["seed"]] = row["validation_metrics"]["accuracy"]
    output = []
    for left, right in PAIRS:
        shared = sorted(set(by.get(left, {})) & set(by.get(right, {})))
        diffs = [by[right][seed] - by[left][seed] for seed in shared]
        output.append({"left": left, "right": right, "difference_definition": f"{right} - {left}",
                       "paired_seeds": shared, "paired_differences": diffs,
                       "mean_paired_difference": mean(diffs), "sd_paired_difference": sd(diffs),
                       "n_pairs": len(diffs)})
    return output


def interpretations(summary):
    means = {row["Configuration"]: row["Mean validation accuracy"] for row in summary}
    text = ["Scope: RoBERTa-base + SST-2 + current Phase 4 protocol."]
    if all(means.get(name) is not None for name in ("spread_k3_r8", "early3_r8", "utility_top3_r8")):
        spread, early, utility = means["spread_k3_r8"], means["early3_r8"], means["utility_top3_r8"]
        if spread > early and spread > utility:
            text.append("Preliminary evidence supports the depth-coverage hypothesis at k=3.")
        elif abs(spread - early) <= max(0.0, (0.002)) and abs(spread - utility) <= max(0.0, (0.002)):
            text.append("The k=3 experiment does not provide clear evidence for depth coverage.")
        elif spread < early and spread < utility:
            text.append("Simple depth coverage alone does not explain the existing allocation results.")
        else:
            text.append("The k=3 comparison is mixed; the raw seed values and uncertainty should guide interpretation.")
    if all(means.get(name) is not None for name in ("strided_k6_r4", "utility_top6_r4")):
        if means["strided_k6_r4"] > means["utility_top6_r4"]:
            text.append("The result supports broader depth coverage relative to concentrated mid-to-late placement.")
        else:
            text.append("Depth coverage alone is insufficient as an explanation.")
    text.append("These preliminary seed comparisons are descriptive and do not establish universal generalization.")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = records(args.roots)
    summary = make_summary(rows)
    pairs = comparisons(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "phase4_followup_summary.csv").open("w", newline="") as handle:
        fields = list(summary[0]) if summary else ["Configuration"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    (args.output_dir / "phase4_followup_paired_comparisons.json").write_text(json.dumps(pairs, indent=2) + "\n")
    (args.output_dir / "phase4_followup_interpretation.txt").write_text("\n".join(interpretations(summary)) + "\n")
    print(f"Wrote {len(summary)} configuration summaries and {len(pairs)} paired comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
