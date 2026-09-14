# RoBERTa analysis summary

## Experimental validation
Valid locked-schema matrix rows: 35 / 36.
Missing pairs: [(3, 2)].
Extra non-matrix JSON artifacts preserved: ['pilot_layer0_seed0.json'].

## RQ1 result
Friedman repeated-measures test on complete seeds (n=2, layers=12): statistic=17.7092, df=11, p=0.0885776, Kendall W=0.804965. Missing pairs: [(3, 2)].
Observed best mean-utility layer: L10 (0.092508).

## RQ2 result
Predictor correlations and ranking diagnostics are in the CSV outputs; they are layer-level diagnostics (n=12), not independent seed-level predictive validation.

## RQ3 result
Budget analysis reports top-k recovery of the oracle-best single layer only. It does not measure joint adaptation of k layers.

## Robustness observations
Leave-one-seed-out summaries are reported below; predictors use the fixed 512-example probe with probe_seed=42, which is a limitation rather than a probe robustness test.
- Leave out seed 0: top layer(s) L4, L10, L7.
- Leave out seed 1: top layer(s) L10, L8, L7.
- Leave out seed 2: top layer(s) L10, L8, L7.

## Safe paper claims
Placement utility varies across layers, subject to the incomplete L4/seed2 matrix cell. Predictor associations should be described as exploratory layer-level evidence.

## Claims not supported
Do not claim a complete 36-run matrix, causal predictor validity, independent generalization, or measured joint multi-layer LoRA utility until L4/seed2 is completed.
