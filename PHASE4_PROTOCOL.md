# Phase 4 protocol

## Research question

Can a gradient-based layer-selection rule derived from single-layer LoRA experiments identify effective multi-layer adapter configurations under a matched parameter budget on SST-2?

This phase is limited to RoBERTa-base on SST-2. It is not a claim about other models, tasks, or universal layer-selection superiority.

## Selection rule and configurations

Phase 3's saved ranking artifact identifies zero-based layers `[9, 10, 11]`, reported as L10, L11, and L12, as the top three gradient-ranked layers. The Phase 4 manifest verifies this before any run.

The two deterministic random sets are generated with `random.Random(20261028).sample(range(12), 3)` called twice and sorted:

- Random A: zero-based `[8, 9, 11]` = L9, L10, L12
- Random B: zero-based `[0, 3, 9]` = L1, L4, L10

Training seeds are `[0, 1, 2]`. The layer-set seed is separate from training seeds.

The seven configurations and 21 planned runs are recorded in `phase4_manifest.json`.

## Model and training settings

- Model and tokenizer: `roberta-base`, both loaded from the exact resolved revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b` recorded in `phase4_config.json`. The runner fails if this field is absent and never silently falls back to `main`.
- Dataset: `nyu-mll/glue`, `sst2`.
- Layer placement: complete `roberta.encoder.layer[i]`, where zero-based 0–11 are L1–L12.
- Backbone and embeddings: frozen.
- Classifier head: trainable and included in the trainable count.
- LoRA targets: query and value.
- LoRA alpha: 16.
- Maximum sequence length: 128.
- Batch size: 32.
- Learning rate: `3e-4`.
- Optimizer: AdamW with explicit `weight_decay=0.01`, matching the existing Phase 3 AdamW default.
- Scheduler: none.
- Warmup: 0.
- Epochs: 5.
- Gradient accumulation: 1.
- Precision: fp32.
- Checkpoint rule: final epoch; holdout performance is not used for checkpoint selection or early stopping.

The all-layer rank-2 configuration is the matched-budget comparison set. All-layer rank-8 is a higher-budget reference.

## Evaluation split

`phase4_split.py` creates a fixed stratified holdout from the original SST-2 training split:

- source: original `train` split only
- split seed: `20261028`
- requested holdout fraction: 10%
- source size: 67,349
- training subset: 60,614
- holdout: 6,735
- stratification: label-wise allocation using largest-remainder rounding, followed by seeded within-label shuffling
- persisted fields: source indices, dataset `idx` values, labels, label counts, and dataset fingerprint

Every Phase 4 configuration uses the same holdout. Models train only on the remaining 90%.

The existing Phase 3 validation split is not used by the Phase 4 runner. Phase 3 did not persist a separate global example-ID manifest, and SST-2 `idx` values are split-local, so this protocol does not claim text-level identity verification against every Phase 3 validation example. The source split distinction is explicit and reproducible.

## Metrics and artifacts

Each run gets an exclusive directory under `results/phase4/{run_id}/`. It writes `status.json`, `result.json`, `predictions.json`, and a final `COMPLETE.json` marker. Aggregation accepts a run only when the final marker exists and all artifact run IDs agree. The result and prediction files are atomically written individually; the completion marker is the joint completion signal.

If a run is interrupted, its directory remains without `COMPLETE.json` and is excluded from aggregation. Do not delete it automatically. To explicitly preserve and rerun an incomplete run, use:

```bash
python run_phase4.py --configuration CONFIG_ID --seed SEED --archive-incomplete
```

Completed runs refuse to overwrite; rerun them in a new output directory.

Results include:

- accuracy
- cross-entropy loss
- positive-class precision, recall, and F1
- epoch training losses
- selected epoch and checkpoint rule
- raw predictions and labels
- trainable parameter count
- split fingerprint and sizes
- seed, configuration, runtime, and software versions

`phase4_aggregate.py` writes `analysis/phase4/phase4_master.csv`, `phase4_summary.csv`, and an aggregation status file. Incomplete runs are reported and excluded.

The holdout is evaluated using a fixed final-epoch rule. It is not used for model selection.

## Parameter budget

The measured classifier-head count is 592,130. LoRA parameters are 3,072 per rank unit per selected layer:

| Configuration | LoRA parameters | Classifier | Total trainable |
|---|---:|---:|---:|
| Three layers, rank 8 | 73,728 | 592,130 | 665,858 |
| Twelve layers, rank 2 | 73,728 | 592,130 | 665,858 |
| Twelve layers, rank 8 | 294,912 | 592,130 | 887,042 |

The preflight measures these counts from the actual PEFT model implementation.

## Limitations

- One model and one task.
- Three seeds per configuration.
- The Phase 3 gradient ranking was informed by validation performance; Phase 4 protects its primary evaluation with a new fixed holdout, but the selection rule itself is not an untouched independent discovery.
- Gradient norm and squared-gradient proxy are rank-equivalent under the current Phase 3 implementation and should be treated as one gradient-based predictor family.
- No joint multi-layer utility is inferred from Phase 3; Phase 4 directly measures the specified multi-layer configurations.
- The protocol does not include hyperparameter search.
