# RoBERTa analysis summary

## Experimental validation
Valid locked-schema matrix rows: 36 / 36 (complete 36-run matrix).
Missing pairs: none.
Extra non-matrix JSON artifacts preserved: ['pilot_layer0_seed0.json'].

## RQ1 result
Friedman repeated-measures test on complete seeds (n=3, layers=12): statistic=24.5584, df=11, p=0.0105737, Kendall W=0.744195. Missing pairs: [].
Observed best mean-utility layer: L10 (0.092508).

## RQ2 result
Predictors are aggregated by layer across seeds before correlation and ranking analysis. These are layer-level diagnostics (n=12), not independent seed-level predictive validation.

## RQ3 result
Budget analysis reports top-k recovery of the oracle-best single layer only. It does not measure joint adaptation of k layers.

## Robustness observations
Leave-one-seed-out summaries are reported below; predictors use the fixed 512-example probe with probe_seed=42, which is a limitation rather than a probe robustness test.
- Leave out seed 0: top layer(s) L10, L7, L8.
- Leave out seed 1: top layer(s) L10, L8, L7.
- Leave out seed 2: top layer(s) L10, L8, L7.

## Safe paper claims
Placement utility varies across layers. Predictor associations should be described as exploratory layer-level evidence.

## Claims not supported
Do not claim causal predictor validity, independent generalization, or measured joint multi-layer LoRA utility.
