# Phase 4 expanded protocol implementation report

Date: 2026-10-30

## Scope and safety

The expanded protocol was implemented additively. The legacy `phase4/` code,
existing `phase4_v2_config.json`, original v2 manifest/selection semantics,
historical outputs, and the pre-existing deletion of `PHASE4_PROTOCOL.md` were
not changed. No model or dataset was downloaded; no training, split
generation, gradient artifact generation, or final-test evaluation occurred.

## Implemented

- Expanded six-configuration core with deterministic random A/B/C placement,
  explicit aliases/collisions, 18 run entries, and unique-count reporting.
- Exact 66-pair enumeration and seed-20261036 execution permutation. Each
  pair records its canonical ID, layers, rank 12, alpha 24, seed 0,
  permutation seed, execution position, protocol version, and split hash.
- Pre-result replication selection with seed-20261037, stable deduplication,
  and a maximum of 11 pairs / 22 additional-seed runs.
- Prespecified gradient, Phase 3 rank-baseline, and depth scores plus
  recovery/percentile/regret/top-k summaries and a sum-to-zero additive OLS
  descriptive model. Residuals are explicitly not labeled definitive
  interactions.
- Explicit screening/replication data boundaries exposing only train and
  validation pools. Final-test loading is isolated in `phase4_v2_final_eval.py`.
- Probe ID/label checks, source/split hash checks, and optional local-files-only
  instantiated PEFT parameter verification.
- Hash-verified expanded completion markers and runtime/hardware metadata
  support.

## Verification

```text
python3 -m py_compile [all new expanded modules and test]
PASS

python3 -m unittest tests.test_phase4_v2_expanded tests.test_phase4_v2_protocol -v
Ran 19 tests in 0.6s
OK (skipped=1)
```

The skipped test is the PyTorch gradient-equivalence test because this
interpreter has no `torch`. Parameter arithmetic is covered synthetically;
the exact instantiated PEFT check remains a Kaggle preflight action because
the required local model cache and ML dependencies are unavailable here.

`git diff --check` is required after review and is reported with the final
handoff. The complete repository test suite remains dependency-limited by the
pre-existing missing `torch` and `pandas` packages.

## Remaining Kaggle approval steps

1. Record environment and GPU type without training.
2. Run the complete tests, including PyTorch gradient equivalence.
3. Load the approved local dataset/model cache and validate source fingerprint,
   IDs, labels, quarantine, split hashes, and probe integrity.
4. Instantiate PEFT models locally and verify 665,858 trainable parameters for
   every core configuration and pairwise rank/alpha budget.
5. Generate and review the expanded-core and pair manifests, then freeze their
   hashes.
6. Measure quota/runtime on the available GPU; do not assume 106 runs or
   equivalent GPU runtimes.
7. Obtain separate approval before any training or final-test evaluation.
