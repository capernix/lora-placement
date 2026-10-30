# Phase 4 v2 implementation audit

Date: 2026-10-29

## Summary

Implemented the Phase 4 v2 protocol additively. No legacy code or artifact was
modified, no dataset/model was downloaded, no final-test evaluation was run,
and no training was launched.

Created:

- `phase4_v2_split.py`
- `phase4_v2_selection.py`
- `phase4_v2_config.json`
- `phase4_v2_manifest.py`
- `phase4_v2_preflight.py`
- `phase4_v2_io.py`
- `phase4_v2_aggregate.py`
- `run_phase4_v2.py`
- `phase4_v2_environment.py`
- `tests/test_phase4_v2_protocol.py`

The final split and selection artifacts were intentionally not generated. The
training manifest cannot be generated until the real v2 selection artifact
exists and passes validation.

## Safety and legacy state

Initial status contained the pre-existing deletion of `PHASE4_PROTOCOL.md`,
untracked legacy context/paper files, the untracked `phase4/` artifact bundle,
and the prior audit report. All requested v2 paths were initially absent. The
pre-existing deletion and artifacts remain untouched.

The v2 runner writes only under an explicitly supplied output directory, such
as `results/phase4_v2`, and uses a new `phase4_v2` protocol marker.

## Implemented safeguards

- Reconstructs the old probe using the exact procedure in `roberta/data.py`.
- Verifies the old source fingerprint, source size, complete ID mapping, and
  recorded holdout partition before quarantine.
- Confirms the old probe/holdout union size is 7,197 and the remaining pool is
  60,152 examples.
- Generates exact v2 pool sizes of 47,640/6,000/512/6,000.
- Enforces pairwise disjointness and quarantine exclusion.
- Records IDs, labels, class counts, allocation seeds, pool hashes, and a
  manifest hash.
- Accumulates weighted batch gradients by summation rather than concatenation.
- Freezes the classifier head during scoring while retaining gradients through
  the frozen head to query/value weights.
- Uses fixed-length padding for the production scorer.
- Requires a hashed selection artifact before manifest generation.
- Fails closed on collision between the selected gradient set and
  `random_a_r8 = [8, 9, 11]`.
- Generates exactly nine primary run entries.
- Adds actual-model parameter checks for 665,858 trainable parameters per
  primary configuration when dependencies and local model files are available.
- Makes v2 aggregation manifest-authoritative and rejects missing, duplicate,
  incomplete, unexpected, or mismatched runs.
- Makes status and completion-marker states consistent for v2 runs.
- Adds checkpoint hashing and environment recording.

## Test commands and outcomes

### Passed

```text
python3 -m py_compile phase4_v2_split.py phase4_v2_selection.py \
  phase4_v2_manifest.py phase4_v2_preflight.py phase4_v2_io.py \
  phase4_v2_aggregate.py run_phase4_v2.py \
  tests/test_phase4_v2_protocol.py
```

```text
python3 -m unittest tests.test_phase4_v2_protocol -v
Ran 10 tests in 0.657s
OK (skipped=1)
```

The ten passing tests cover legacy probe reconstruction, 50-example overlap,
7,197-example quarantine arithmetic, deterministic splitting, disjointness,
quarantine exclusion, mismatch failures, selection/ranking, collision logic,
manifest freezing, parameter arithmetic, lifecycle handling, and incomplete
run rejection.

### Skipped

The full-probe versus batched-gradient numerical test was skipped because the
current interpreter has no `torch` installation. The test is present and will
run automatically in the approved Kaggle environment.

### Legacy suite

```text
python3 -m unittest discover -s tests
```

Result: failed during collection for pre-existing dependency reasons:

- `test_harness_invariants`: `ModuleNotFoundError: No module named 'torch'`
- `test_roberta_infrastructure`: `ModuleNotFoundError: No module named 'torch'`
- `test_phase4_infrastructure`: `ModuleNotFoundError: No module named 'pandas'`

The legacy failures occurred before test execution and are unrelated to v2
changes. No legacy test or source file was modified.

### No-training checks

Passed:

```text
python3 phase4_v2_preflight.py --help
python3 -c 'import run_phase4_v2, phase4_v2_selection, phase4_v2_split'
```

No v2 preflight with real artifacts was run because the v2 split, selection,
and final manifest do not yet exist, and the current interpreter lacks the
required ML packages.

## Environment

Current interpreter:

```text
/usr/bin/python3
Python 3.12.3
```

Unavailable packages include `torch`, `numpy`, `pandas`, `scipy`,
`transformers`, `peft`, and `datasets`.

The additive `phase4_v2_environment.py` recorder is dependency-tolerant and
will record interpreter, package, git, CUDA, and GPU information once run in
Kaggle. The exact-model preflight intentionally uses local model files only;
it will not download a missing model.

## Remaining blockers

No methodological blocker was found in the implementation review.

Operational blockers remain:

1. The current environment lacks the ML dependencies.
2. The actual SST-2 dataset fingerprint and model revision must be available
   in the approved Kaggle environment.
3. The torch-dependent gradient-equivalence test has not yet executed.
4. The exact instantiated-model parameter check has not yet executed.
5. The v2 split, selection artifact, and nine-run manifest have intentionally
   not been generated locally.

## Exact next approved stage

In the Kaggle environment, after confirming package versions and local/cache
policy:

```bash
python3 phase4_v2_environment.py --output phase4_v2_environment.json
python3 -m unittest tests.test_phase4_v2_protocol -v
python3 phase4_v2_split.py --output phase4_v2_split.json
python3 phase4_v2_selection.py \
  --config phase4_v2_config.json \
  --split phase4_v2_split.json \
  --output phase4_v2_selection.json
python3 phase4_v2_manifest.py \
  --config phase4_v2_config.json \
  --split phase4_v2_split.json \
  --selection phase4_v2_selection.json \
  --output phase4_v2_manifest.json
python3 phase4_v2_preflight.py \
  --config phase4_v2_config.json \
  --split phase4_v2_split.json \
  --selection phase4_v2_selection.json \
  --manifest phase4_v2_manifest.json \
  --model-check
```

These commands generate no training runs. Training remains a separate step
requiring explicit approval after the selection hash, manifest, environment
record, tests, and preflight output have been reviewed.
