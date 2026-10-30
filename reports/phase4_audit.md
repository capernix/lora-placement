# Phase 4 audit: status before further experiments

Audit date: 2026-10-29. This report is based on the current worktree, the
untracked run artifacts under `phase4/` and `results/`, and the tracked Phase
3 artifacts at `HEAD` (`ed0c92b`). No training, model download, or code change
was performed. The existing `PHASE4_PROTOCOL.md` deletion in the worktree was
not repaired.

## 1. Executive verdict

The planned Phase 4 comparison is not yet clean enough for confirmatory claims.
The reported overlap is real: reproducing `random.Random(42).sample` over the
67,349-example SST-2 training split and intersecting it with
`phase4_split.json` gives exactly 50 shared source indices. The Phase 3 probe
is used in `compute_importance` to form task-loss gradients, and the Phase 4
manifest selects `[9, 10, 11]` from the resulting gradient ranking. Therefore
the labels of those 50 future holdout examples entered the adaptive selection
signal. The saved artifacts do not contain per-example gradients or a
leave-50-out ranking, so the marginal effect of those 50 examples on the
selected set is unverified; their leakage pathway is confirmed.

Only one Phase 4 run artifact is present: `gradient_top3_r8_seed0`, with
accuracy `0.9158129175946548`, 6,735 evaluation examples, and 886.985 seconds
of recorded training time. It is useful exploratory evidence only and must be
rerun for a clean primary comparison. The other 20 planned runs are absent or
unverifiable. Do not launch the remaining runs yet.

Minimum defensible primary correction: freeze a new protocol/version, create a
new final evaluation set disjoint from a new selection probe, recompute the
gradient selection using that probe, and run the complete seven-configuration
by-three-seed matrix under the new split. This is 21 clean protocol runs; the
existing run is not reusable as a confirmatory result. A cheaper alternative
can preserve fixed, non-gradient configurations on the existing holdout, but
would leave the holdout partially exposed and should be labeled exploratory,
not a clean final benchmark.

## 2. Evidence-based audit findings

| Finding | Evidence | Severity | Scientific consequence | Required correction | Rerun? |
|---|---|---:|---|---|---|
| Probe and Phase 4 holdout overlap | `roberta/data.py:16-18` samples 512 from `encoded["train"]` with seed 42. `phase4_split.json` records the 6,735 holdout indices. Direct reproduction gives 50 shared indices: `2552, 4165, 6057, ... , 66307`. | critical | Holdout labels influenced the gradient signal used to choose the top-three configuration. | Use a probe and final test partition that are disjoint; record example IDs and fingerprints for both. | Yes for the gradient configuration and for a clean final benchmark. |
| The 50 labels did enter selection, but their marginal effect is unknown | `roberta/importance.py:33-39` computes task-loss gradients using `batch["labels"]`; `phase4_manifest.py` verifies the selected gradient layers `[9,10,11]`. No per-example gradients or leave-out ranking artifact is saved. | critical | It is invalid to call the current holdout untouched for adaptive selection. It is not justified to claim that the 50 examples definitely changed the ranking. | Recompute the selection probe without final-test examples; optionally save per-example contributions or a leave-out sensitivity diagnostic. | Yes for a clean selection claim. |
| Phase 3 probe also belongs to the Phase 3 training source split | `roberta/train.py:156-168` loads the full train split for fitting and a subset of that same split for the probe. | medium | This is not Phase 4 test leakage by itself, but it means the signal is task-aware and computed from data also used for Phase 3 adaptation. | Describe it as a pre-fine-tuning, task-aware signal; use a dedicated selection pool in the clean protocol. | No for historical evidence; yes if claiming a clean new selection study. |
| Phase 4 has no validation split or adaptive checkpoint rule | `run_phase4.py:122-148` trains on 60,614 examples and evaluates the holdout only after epoch 5; `phase4_config.json` says `checkpoint_rule: final_epoch`. | medium | Fixed-epoch comparison is valid if prespecified, but there is no validation set for future tuning or model selection. | Reserve validation for any tuning; keep final test single-use and final-epoch only if that is the locked rule. | Not required if settings remain frozen. |
| Current completed artifact is outside the runner/aggregator layout and lacks the saved checkpoint | The artifact is in `phase4/`, while `run_phase4.py:167` and `phase4_aggregate.py:52` expect `results/phase4/`. `phase4/result.json` points to `results/phase4/gradient_top3_r8_seed0/checkpoint`, which is absent in the worktree. | high | Full rerun provenance and weight reproducibility cannot be established from the artifact alone. | Use a versioned output root, preserve checkpoint or explicitly document a weights-not-retained policy, and store an existence/hash field. | Yes for primary evidence. |
| Completion and status disagree | `phase4/status.json` says `running`; `phase4/COMPLETE.json` says `completed`. `phase4_aggregate.py:19-27` trusts the completion marker and does not inspect status. `phase4_io.py:90-111` never changes status to completed. | high | A stale status can conceal interrupted or copied artifacts; aggregation can accept a run whose lifecycle record disagrees. | Make one completion state authoritative and validate status/result/marker consistency. | No for the numeric run, but rerun or repair infrastructure before new runs. |
| Phase 4 parameter matching is correct in source and in the saved run | `phase4_preflight.py` measures trainable parameters from PEFT. The saved run reports 665,858. The intended counts are 665,858 for 3×rank-8 and 12×rank-2, and 887,042 for 12×rank-8. | low | The all-layer rank-8 reference is not a matched-budget comparison and must not be interpreted as one. | Report it as a higher-capacity reference; compare the two 665,858 groups for the primary budget claim. | No. |
| Phase 3 and Phase 4 classifier counts differ because the LoRA configuration differs | Phase 3 JSONs report 616,706 = 592,130 classifier + 24,576 single-layer rank-8 LoRA. Phase 4 reports 665,858 = 592,130 classifier + 73,728 3-layer rank-8 LoRA. | low | The counts are internally coherent; comparing absolute utility across phases without noting the capacity change is misleading. | Report counts per phase/configuration and avoid cross-phase accuracy comparisons as if capacity matched. | No. |
| Gradient norm and squared-gradient proxy are the same ranking | `roberta/importance.py:52-55` computes one concatenated gradient vector per layer, then its L2 norm and mean squared value. All candidate layers have equal parameter count and equal batch counts. Tracked ranking files show identical ranks and Spearman rho `0.5944055944055945`. | high | The proxy is not an independent predictor in this implementation. | Treat these as one gradient-derived signal family unless a normalized statistic with demonstrated non-equivalence is introduced. | No for historical analysis; yes only for a redesigned signal experiment. |
| CKA is not a defensible current predictor | `roberta/importance.py:41-48` compares each layer's pooled position-0 hidden state to `hidden_states[0]`, the embedding output. At position 0, the input token/position representation is effectively constant across examples. The denominator only guards exact zero (`:10`); saved values are approximately `1e-12`–`2e-10`, indicating numerical near-degeneracy rather than useful variation. | high | Existing CKA associations should not be used as evidence of predictive validity. | Replace with a prespecified non-degenerate reference and numerical diagnostics, or remove CKA from claims. | No for Phase 4; yes before any CKA claim. |
| Phase 3 selection rule is reproducible but adaptive | `analysis/roberta/roberta_ranking_analysis.csv` ranks L12, L11, L10 as the top three gradient layers; `phase4_manifest.py` verifies zero-based `[9,10,11]`. The ranking is computed from a fixed probe, not from an independent external dataset. | high | The set is a fixed historical decision, not an untouched discovery result. | For primary evidence, recompute on the clean selection probe and version the resulting selection artifact. | Yes for clean primary evidence. |
| Friedman analysis uses the correct blocking concept, but has only three blocks | `analysis/analyze_roberta.py` pivots utility by `seed` and `layer` and calls `friedmanchisquare`; each seed appears once under every layer. The reported result is statistic 24.5584, df 11, p 0.0105737, W 0.744195. | medium | It is a repeated-measures exploratory result with n=3 blocks, not 36 independent observations. Small-sample p-values and post-hoc claims are fragile. | Keep seed-blocked analysis; report paired effect sizes/intervals and prespecify any post-hoc procedure. | No for the historical test; rerun only if the protocol/split changes. |
| Stored rounding changes six Phase 3 utility differences by 1e-6 | Six raw JSON rows have `utility_delta != round(lora_val_acc - frozen_val_acc, 6)` when recomputed from stored rounded accuracies. | low | Small discrepancies can affect exact ties/ranks near boundaries, though not the headline result here. | Store integer correct counts and unrounded accuracy/utility, derive utility during analysis. | No; reanalysis from counts/raw precision is sufficient. |
| Revisions are mostly pinned, but metadata is inconsistent | Phase 3 JSONs resolve model commit `e2da8e2f...`; Phase 4 config uses the same resolved model commit. `phase4_config.json` says tokenizer revision `main`, while `run_phase4.py:62` passes the resolved model revision to tokenizer loading. Phase 3 runtime is torch 2.10/transformers 5.0/peft 0.19.1/datasets 5.0; Phase 4 is torch 2.11/transformers 5.16.1/peft 0.20/datasets 5.0.1. | medium | The model weights are comparable, but software changes can affect PEFT initialization, serialization, or numerics. | Pin and record model, tokenizer, code commit, dependency versions, and CUDA/runtime in every clean run. | Yes for a strict reproducibility rerun; not necessarily to interpret history. |
| Seed handling is explicit but GPU determinism is not guaranteed | Phase 4 `set_seed` seeds Python, NumPy, Torch, and CUDA (`run_phase4.py:29-35`); loaders use the run seed (`:128`). No deterministic-algorithm policy is set, and workers are used by `roberta.train._loader`. | medium | Same nominal seed may not be bitwise identical across hardware/software. | Record hardware and determinism settings; run duplicate smoke/repro checks before the matrix. | No for existing numbers; yes for protocol validation. |
| Aggregation checks the completion marker but not the manifest or full schema | `phase4_aggregate.py:19-27` validates marker/result/prediction run IDs, then aggregates. It does not verify all expected runs, config fields, split IDs, checkpoint existence, or result-vs-manifest equality. | medium | A malformed or misplaced run can silently enter a summary; missing runs are not a complete-comparison error. | Aggregate against a frozen manifest and fail/report on missing, duplicate, schema, split, and checkpoint mismatches. | No for audit; required before new evidence. |

## 3. Confirmed versus unverified

### Confirmed

- The 50-example probe/holdout intersection exists at the source-index level.
- The probe gradients use labels and the selected top-three layers are L10–L12.
- Phase 4 training uses the 90% complement of the current holdout and evaluates the same 6,735 examples for every configuration.
- The one available run used the intended top-three layers, rank 8, query/value targets, explicit weight decay 0.01, five epochs, and 665,858 trainable parameters.
- The two primary budget designs are parameter matched; the all-layer rank-8 reference is not.
- CKA is numerically near-degenerate under the current reference definition.
- Squared-gradient and gradient norm are rank-equivalent under the saved implementation.

### Unverified

- Whether removing the 50 overlapping examples changes any layer's gradient score or the selected top three.
- Whether the root `phase4/` artifact was copied from a completed `results/phase4/...` directory and whether its checkpoint still exists elsewhere.
- Whether the current holdout was inspected repeatedly to make later adaptive decisions; the repository does not log analyst decisions.
- Bitwise reproducibility across devices and the exact tokenizer revision used by historical Phase 3 jobs beyond the shared model revision evidence.

## 4. Recommended clean protocol

### Preferred protocol: Phase 4 v2

1. Freeze a new protocol and manifest before looking at any new final-test metric.
2. From the same dataset fingerprint, create disjoint pools: a selection probe `P`, a training pool `D`, a validation pool `V` for any predeclared tuning, and a final test `T`. `P ∩ T = ∅`, and no labels from `T` may enter selection, tuning, checkpoint choice, or configuration choice. Persist source indices, dataset IDs, labels, fingerprints, and hashes for all pools.
3. Recompute all gradient scores on `P` using one frozen, unadapted model and a predeclared aggregation rule. Save the selected top three and the complete score table. Do not select from final-test results.
4. Use identical `D`, `V`, preprocessing, model/tokenizer revision, optimizer, weight decay, epoch/checkpoint rule, and metric for every configuration. If no tuning is required, keep the final-epoch rule and treat `V` as unused or omit it.
5. Make the primary comparison only between `gradient_top3_r8`, `random_a_r8`, `random_b_r8`, `early_r8`, `middle_r8`, and `uniform_all_r2`, all at 665,858 trainable parameters. Keep `all_layer_reference_r8` as a separately labeled higher-capacity reference.
6. Define random A/B before final-test evaluation. They are two fixed placements, not an estimate of a random-placement distribution. A distributional random baseline would require many independently sampled triples, not two draws.
7. Treat seed 0/1/2 as training replicates. The random-set-generation seed is a configuration-sampling seed, not an additional training replicate.
8. Evaluate `T` once at the end. Report per-seed accuracy, loss, F1, integer counts, and—if utility is primary—a matched frozen-head baseline and utility delta. Never use `T` for early stopping or selection.
9. Keep the old artifacts under an explicit `legacy`/`exploratory` label. Do not delete or overwrite them.

### Minimal-cost alternative

Retain the current fixed holdout only for fixed configurations that have never
been selected using its labels, recompute the gradient probe from a new pool
disjoint from that holdout, and rerun all three gradient-top-3 seeds. This
avoids rerunning any already-completed non-adaptive configuration, but the
holdout has already been exposed through the old probe and one observed result.
It is therefore suitable only for an explicitly exploratory comparison, not a
clean final-test claim. The current `gradient_top3_r8_seed0` cannot be upgraded
to primary evidence merely by adding seeds because its selection path is
already contaminated.

## 5. Run-by-run validity matrix

The manifest plans 21 runs. Only the row below is present in the current
artifact tree. “Primary” means valid for the preferred clean protocol, not
whether the file is structurally complete.

| Configuration | Layers | Rank | Seed | Current artifact | Primary status | Action |
|---|---|---:|---:|---|---|---|
| gradient_top3_r8 | L10,L11,L12 | 8 | 0 | complete marker + result + predictions in `phase4/`; stale status; checkpoint absent | exploratory only | rerun under v2 |
| gradient_top3_r8 | L10,L11,L12 | 8 | 1 | missing | missing/unverifiable | run under v2 |
| gradient_top3_r8 | L10,L11,L12 | 8 | 2 | missing | missing/unverifiable | run under v2 |
| random_a_r8 | L9,L10,L12 | 8 | 0 | missing | missing/unverifiable | run under v2 |
| random_a_r8 | L9,L10,L12 | 8 | 1 | missing | missing/unverifiable | run under v2 |
| random_a_r8 | L9,L10,L12 | 8 | 2 | missing | missing/unverifiable | run under v2 |
| random_b_r8 | L1,L4,L10 | 8 | 0 | missing | missing/unverifiable | run under v2 |
| random_b_r8 | L1,L4,L10 | 8 | 1 | missing | missing/unverifiable | run under v2 |
| random_b_r8 | L1,L4,L10 | 8 | 2 | missing | missing/unverifiable | run under v2 |
| early_r8 | L1,L2,L3 | 8 | 0 | missing | missing/unverifiable | run under v2 |
| early_r8 | L1,L2,L3 | 8 | 1 | missing | missing/unverifiable | run under v2 |
| early_r8 | L1,L2,L3 | 8 | 2 | missing | missing/unverifiable | run under v2 |
| middle_r8 | L5,L6,L7 | 8 | 0 | missing | missing/unverifiable | run under v2 |
| middle_r8 | L5,L6,L7 | 8 | 1 | missing | missing/unverifiable | run under v2 |
| middle_r8 | L5,L6,L7 | 8 | 2 | missing | missing/unverifiable | run under v2 |
| uniform_all_r2 | L1–L12 | 2 | 0 | missing | missing/unverifiable | run under v2 |
| uniform_all_r2 | L1–L12 | 2 | 1 | missing | missing/unverifiable | run under v2 |
| uniform_all_r2 | L1–L12 | 2 | 2 | missing | missing/unverifiable | run under v2 |
| all_layer_reference_r8 | L1–L12 | 8 | 0 | missing | missing/unverifiable | run under v2 as higher-capacity reference |
| all_layer_reference_r8 | L1–L12 | 8 | 1 | missing | missing/unverifiable | run under v2 as higher-capacity reference |
| all_layer_reference_r8 | L1–L12 | 8 | 2 | missing | missing/unverifiable | run under v2 as higher-capacity reference |

## 6. Minimum rerun plan and compute estimate

### Preferred primary comparison

- 21 runs under the new versioned split and clean selection probe.
- The existing top-three seed-0 run is not counted as reusable primary work.
- The top-three layer set itself must be regenerated from the clean probe. If
  it changes, the new set replaces `[9,10,11]` in the v2 manifest; the old set
  remains a historical exploratory configuration.
- Do not rerun Phase 3's 36 single-layer trainings merely to repair this
  Phase 4 issue. A new gradient scoring pass is needed; a new single-layer
  training matrix is not, unless the research question changes.

The observed Phase 4 run took 886.985 seconds (14.8 minutes). A naive estimate
for 21 v2 runs at that same rate is 18,626.7 seconds, or 5.17 hours of serial
GPU time. For the 20 runs after excluding the existing exploratory artifact,
the analogous lower-bound estimate is 17,739.7 seconds, or 4.93 hours. These
are estimates, not guarantees: all-layer configurations may take longer than
the three-layer run because the detach point is earlier and more of the model
participates in backward computation. Phase 3 single-layer recorded runtimes
range from 810.144 to 1,540.964 seconds, mean 1,175.045 seconds; they are a
poor but relevant hardware/software reference, not a replacement for a
configuration-specific benchmark.

### Minimal alternative cost

If the current holdout is retained as an exploratory evaluation set and only
the contaminated adaptive configuration is repaired, the minimum new training
work is three gradient-top-3 seeds (or three seeds for whatever clean probe
selects), plus one cheap probe-scoring pass. This does not establish a clean
final benchmark and must not be reported as one.

## 7. Statistical and implementation corrections

- Keep Friedman as a seed-blocked omnibus diagnostic for Phase 3, but report
  the effective repeated-measures units as three seeds, not 36 independent
  observations. Add paired per-layer contrasts and multiplicity-controlled
  post-hoc comparisons only if prespecified.
- Analyze Phase 4 with a configuration-level model that respects seed
  replication. Do not treat the 21 rows as independent examples. Report
  per-seed values and uncertainty; three seeds are a small replication count.
- Compute utility from stored integer correct counts or unrounded accuracy,
  not rounded six-decimal fields. Phase 3 has six 1e-6 inconsistencies.
- Use the same primary metric and split across all configurations. Absolute
  accuracy is acceptable for the current fixed-final-epoch Phase 4 design;
  utility delta is preferable if a matched frozen-head baseline is added.
- Do not describe the gradient p-value (`p≈0.0415`) as confirmatory; it is a
  layer-level exploratory association with 12 effective units, and it does
  not survive the stated Bonferroni alpha 0.025.
- Do not use current CKA values as evidence. A replacement must compare
  non-degenerate representations, check centered covariance norms, and save
  denominator diagnostics.
- Do not present squared-gradient as independent from gradient norm under the
  current equal-size implementation.
- Pin the exact model and tokenizer commit, repository commit, dependency
  versions, Python/CUDA metadata, and determinism policy in each run.
- Make the manifest authoritative for aggregation and validate every expected
  run, split fingerprint/IDs, configuration, count, metric schema, and
  completion state. The current completion marker is authoritative in code,
  but its disagreement with `status.json` is a lifecycle defect.
- Preserve or hash checkpoints if full reproducibility is required. If
  checkpoints are intentionally omitted, say so and do not claim weight-level
  reproducibility from result JSONs alone.

## 8. Pairwise pilot assessment

Decision now: **stop/go = STOP**. The 24-pair pilot is not justified before a
clean Phase 4 comparison. Phase 3 establishes layer-level variation and an
exploratory gradient association; it does not establish additivity, interaction
structure, or prediction of joint multi-layer utility. Two random triples and
one incomplete Phase 4 run cannot support the pairwise question.

If Phase 4 later supports a pilot, prespecify it as follows:

- Hypothesis: a pre-fine-tuning signal predicts pair utility well enough to
  improve selection regret over random selection at the same parameter budget.
- Baselines: random pair selection; the best individual-layer utility sum or
  mean; an additive model fitted only on training pairs; gradient-score sum;
  and a depth/adjacency heuristic where scientifically relevant.
- Define the target. The least ambitious target is unseen pairs made from the
  same 12 known layers. Generalization to pairs containing unseen layers needs
  layer-blocked evaluation and is a different, harder claim.
- A random 16/8 split is weak because pairs sharing a layer are dependent.
  For known-layer composition, use repeated prespecified pair splits and
  report layer-aware uncertainty. For unseen-layer generalization, hold out
  complete layers, not merely pairs.
- Fit any predictor on training pairs only. Evaluate held-out pairs once and
  measure regret relative to the oracle pair, top-1/top-k recovery, and
  probability of selecting a configuration within a prespecified utility
  tolerance. Correlation alone is insufficient.
- Include replicated training for the best predicted pair, the oracle pair,
  and random controls. Record the complete ledger: probe/scoring cost, all
  pair-training costs, replications, failed/partial jobs, storage, and
  evaluation cost.
- Go only if the clean Phase 4 result shows a repeatable budget-matched
  placement effect and the prespecified single-layer baselines beat random
  selection by a meaningful regret margin. Stop if Phase 4 shows no stable
  effect, if pair predictions do not beat the additive/random baselines on
  held-out pairs, or if the result depends on the contaminated holdout.

## 9. Claims currently supported

The current evidence supports the cautious statement that single-layer LoRA
utility varies across RoBERTa layers under the Phase 3 setup, with an
exploratory positive association between gradient norm and layer-mean utility.
It supports the engineering fact that the proposed primary Phase 4 budget
matches three rank-8 layers to twelve rank-2 layers.

It does not support an untouched-holdout claim, independent predictive
validity of the gradient signal, CKA evidence, independent information from
the squared-gradient proxy, superiority of the gradient-selected
configuration, generalization beyond SST-2/roberta-base, or any pairwise
prediction claim.

## 10. Exact next commands after approval of protocol changes

These are intentionally not being run in this audit:

```bash
# After implementing and reviewing a versioned v2 split/manifest:
python3 phase4_preflight.py --config phase4_v2_config.json \
  --manifest phase4_v2_manifest.json --split phase4_v2_split.json

# Run only after the preflight output, clean probe-selection artifact, and
# output directory have been reviewed and approved:
python3 run_phase4.py --config phase4_v2_config.json \
  --split phase4_v2_split.json --output-dir results/phase4_v2 \
  --configuration CONFIG_ID --seed SEED

python3 phase4_aggregate.py --input-dir results/phase4_v2 \
  --output-dir analysis/phase4_v2
```

The commands above are a template, not authorization to train. Proposed code
changes should be reviewed and approved before implementation.
