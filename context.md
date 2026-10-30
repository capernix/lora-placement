# ```markdown

# \# Research Project Memory Context

# 

# \## Project

# 

# Repository: `/home/capernix/projects/lora-placement`

# 

# Research area: parameter-efficient fine-tuning, LoRA placement, layer importance, parameter-budget allocation, and multimodal extension.

# 

# The project investigates whether LoRA placement within a frozen transformer encoder affects downstream adaptation utility and whether cheap pre-fine-tuning signals can identify useful adapter locations.

# 

# Do not change the research direction without explicit approval. Do not launch expensive training automatically.

# 

# \---

# 

# \## Phase 1: Dummy harness

# 

# The synthetic/dummy LoRA harness was completed and validated.

# 

# Important invariants established:

# 

# \- Exactly one LoRA placement.

# \- Only the classifier/task head and selected LoRA A/B parameters are trainable.

# \- Frozen backbone parameters do not change.

# \- Repeated identical seeds/configurations are reproducible.

# \- Trainable parameter counts use `requires\_grad=True`.

# 

# Existing tests remain in:

# 

# \- `tests/test\_harness\_invariants.py`

# 

# Do not modify the dummy harness unless an independent integration bug is discovered.

# 

# \---

# 

# \## Phase 2: RoBERTa implementation

# 

# The real RoBERTa pipeline is under:

# 

# \- `roberta/config.py`

# \- `roberta/data.py`

# \- `roberta/model.py`

# \- `roberta/importance.py`

# \- `roberta/train.py`

# \- `run\_roberta\_pilot.py`

# \- `run\_roberta\_sweep.py`

# 

# Model:

# 

# \- `roberta-base`

# \- Requested revision: `main`

# \- Resolved revision:

# &#x20; `e2da8e2f811d1448a5b465c236feacd80ffbac7b`

# \- Tokenizer uses the same model revision.

# \- Architecture: 12 RoBERTa encoder layers.

# \- Zero-based internal layers: `0–11`

# \- Human labels: `L1–L12`

# \- Placement unit: complete `roberta.encoder.layer\[i]`

# \- Embeddings and classifier are not placement candidates.

# 

# Dataset:

# 

# \- Hugging Face dataset: `nyu-mll/glue`

# \- Config: `sst2`

# \- Train split: 67,349 examples

# \- Validation split: 872 examples

# \- Maximum length: 128

# 

# LoRA:

# 

# \- Targets: `query` and `value`

# \- Rank: 8

# \- Alpha: 16

# \- Only the selected encoder layer receives LoRA in Phase 3.

# \- Classifier head is trainable.

# \- Original RoBERTa backbone remains frozen.

# 

# Training:

# 

# \- Optimizer: AdamW

# \- Learning rate: `3e-4`

# \- Batch size: 32

# \- Epochs: 5

# \- No scheduler

# \- No warmup

# \- Effective AdamW default weight decay: `0.01`

# \- FP32

# \- Same classifier-head initialization for matched baseline/LoRA runs.

# 

# Baseline:

# 

# \- Frozen RoBERTa encoder

# \- Trainable classifier head

# \- Utility:

# &#x20; `utility\_delta = lora\_val\_acc - frozen\_val\_acc`

# 

# Predictors:

# 

# \- Fixed 512-example SST-2 train probe

# \- Probe seed: 42

# \- Predictors computed before LoRA adaptation

# \- CKA compares pooled `<s>` layer representations to the pooled input representation.

# \- Gradient predictor uses task-loss gradients.

# \- Squared-gradient predictor is an empirical squared-gradient / diagonal-Fisher-style proxy.

# \- Do not call it formal Fisher information.

# 

# \---

# 

# \## Phase 3: Completed single-layer RoBERTa matrix

# 

# The canonical Phase 3 matrix is complete:

# 

# \- 12 layers × 3 seeds

# \- 36 valid JSON result files

# \- Files:

# &#x20; `results/roberta/layer00\_seed00.json`

# &#x20; through

# &#x20; `results/roberta/layer11\_seed02.json`

# \- Separate pilot artifact:

# &#x20; `results/roberta/pilot\_layer0\_seed0.json`

# &#x20; This is preserved but excluded from the canonical matrix.

# 

# All 36 canonical runs passed validation:

# 

# \- No duplicate `(layer, seed)` pairs

# \- Consistent model/revision/dataset/settings

# \- Identical trainable parameter counts

# \- `backbone\_unchanged == true`

# \- Complete predictor fields

# \- No raw JSONs were modified during analysis.

# 

# Canonical Phase 3 analysis outputs:

# 

# \- `analysis/roberta/roberta\_master.csv`

# \- `analysis/roberta/roberta\_layer\_summary.csv`

# \- `analysis/roberta/roberta\_predictor\_summary.csv`

# \- `analysis/roberta/roberta\_ranking\_analysis.csv`

# \- `analysis/roberta/roberta\_ranking\_metrics.csv`

# \- `analysis/roberta/roberta\_budget\_selection.csv`

# \- `analysis/roberta/roberta\_rq1\_statistics.csv`

# \- `analysis/roberta/roberta\_rq1\_statistics.txt`

# \- `analysis/roberta/roberta\_leave\_one\_seed.csv`

# \- `analysis/roberta/roberta\_analysis\_summary.md`

# \- Figures in `analysis/roberta/figures/`

# 

# Phase 3 RQ1 findings:

# 

# Mean utility by layer:

# 

# \- L1: 0.072248

# \- L2: 0.080657

# \- L3: 0.077217

# \- L4: 0.085627

# \- L5: 0.086391

# \- L6: 0.087538

# \- L7: 0.090214

# \- L8: 0.090979

# \- L9: 0.086774

# \- L10: 0.092508

# \- L11: 0.087156

# \- L12: 0.082951

# 

# Best mean utility: L10.

# 

# Friedman repeated-measures test:

# 

# \- Statistic: 24.5584

# \- df: 11

# \- p-value: 0.0105737

# \- Kendall’s W: 0.744195

# 

# Interpretation:

# 

# \- Placement materially affects utility in this experiment.

# \- L10 is the best average layer.

# \- L7–L10 are generally strong.

# \- Results are specific to RoBERTa-base/SST-2 and should not be generalized universally.

# 

# Phase 3 RQ2 findings:

# 

# Layer-level predictor correlations with mean utility:

# 

# | Predictor | Pearson r | Spearman rho | p |

# |---|---:|---:|---:|

# | CKA-input | -0.0956 | -0.1608 | 0.6175 |

# | Gradient norm | 0.1586 | 0.5944 | 0.0415 |

# | Squared-gradient proxy | -0.0217 | 0.5944 | 0.0415 |

# 

# Predictor rankings:

# 

# \- CKA top layer: L4

# \- Gradient norm top layer: L12

# \- Squared-gradient proxy top layer: L12

# \- Actual utility-best layer: L10

# 

# Top-k recovery of actual oracle-best layer L10:

# 

# \- CKA: not recovered at k=1, 2, or 3

# \- Gradient norm: recovered at k=3

# \- Squared-gradient proxy: recovered at k=3

# 

# Important predictor caveat:

# 

# The gradient norm and squared-gradient proxy are rank-equivalent by construction.

# 

# In `roberta/importance.py`:

# 

# ```python

# values = torch.cat(\[grad.reshape(-1) for \_, grad in grads])

# grad\_norm\[layer] = values.norm().item()

# squared\_gradient\_proxy\[layer] = values.square().mean().item()

# ```

# 

# Since all candidate layers have equal query/value gradient parameter counts:

# 

# ```text

# mean(g^2) = ||g||\_2^2 / N

# ```

# 

# Therefore they induce the same layer ranking. Treat them as one gradient-based predictor family, not independent predictors.

# 

# \---

# 

# \## Phase 4 preparation

# 

# Phase 4 was prepared but the 21-run matrix has NOT been launched.

# 

# Original Phase 4 proposal:

# 

# \- Gradient-selected top 3: L10, L11, L12, rank 8

# \- Random set A: L9, L10, L12, rank 8

# \- Random set B: L1, L4, L10, rank 8

# \- Early: L1, L2, L3, rank 8

# \- Middle: L5, L6, L7, rank 8

# \- Uniform all-layer rank 2

# \- All-layer reference rank 8

# \- Seeds: 0, 1, 2

# \- Total: 21 runs

# 

# Random layer generation:

# 

# ```python

# random.Random(20261028).sample(range(12), 3)

# ```

# 

# called twice and sorted:

# 

# \- Random A zero-based: `\[8, 9, 11]`

# \- Random B zero-based: `\[0, 3, 9]`

# 

# Phase 4 files:

# 

# \- `phase4\_config.json`

# \- `phase4\_manifest.json`

# \- `phase4\_split.json`

# \- `phase4\_split.py`

# \- `phase4\_manifest.py`

# \- `phase4\_preflight.py`

# \- `phase4\_io.py`

# \- `run\_phase4.py`

# \- `phase4\_aggregate.py`

# \- `PHASE4\_PROTOCOL.md`

# \- `tests/test\_phase4\_infrastructure.py`

# 

# Phase 4 split:

# 

# \- Source: original SST-2 train split

# \- Source size: 67,349

# \- Training subset: 60,614

# \- Holdout: 6,735

# \- Stratified by label

# \- Split seed: 20261028

# \- Dataset fingerprint: `e0f6962d602757cb`

# \- No train/holdout overlap

# \- Exact indices, IDs, labels, counts, and fingerprint saved in `phase4\_split.json`

# 

# Phase 4 parameter counts:

# 

# \- Classifier head: 592,130

# \- Three rank-8 layers:

# &#x20; - LoRA params: 73,728

# &#x20; - Total trainable: 665,858

# \- Twelve rank-2 layers:

# &#x20; - LoRA params: 73,728

# &#x20; - Total trainable: 665,858

# \- Twelve rank-8 layers:

# &#x20; - LoRA params: 294,912

# &#x20; - Total trainable: 887,042

# 

# Phase 4 infrastructure fixes completed:

# 

# \- Exact resolved revision is propagated to model and tokenizer.

# \- Missing revision causes a hard failure.

# \- Dataset fingerprint and split indices are validated.

# \- Each run gets an exclusive directory.

# \- Atomic JSON writes are used.

# \- `COMPLETE.json` is written only after result and predictions are both validated.

# \- Incomplete runs are excluded from aggregation.

# \- Completed runs refuse overwrite.

# \- Incomplete runs require explicit archival.

# \- Aggregation outputs mean/std metrics across completed seeds.

# 

# Phase 4 tests:

# 

# ```text

# Ran 12 tests

# OK

# ```

# 

# Phase 4 smoke test passed:

# 

# ```text

# Phase 4 smoke test passed: gradient\_top3\_r8, seed=0, trainable=665858

# ```

# 

# No Phase 4 training runs have been launched.

# 

# Do not launch Phase 4 unless explicitly requested. The project may now move to a multimodal study instead.

# 

# \---

# 

# \## Phase 4 runner commands

# 

# Smoke test only:

# 

# ```bash

# .venv/bin/python run\_phase4.py \\

# &#x20; --smoke-test \\

# &#x20; --configuration gradient\_top3\_r8 \\

# &#x20; --seed 0

# ```

# 

# One real run, not currently approved unless explicitly requested:

# 

# ```bash

# .venv/bin/python run\_phase4.py \\

# &#x20; --configuration gradient\_top3\_r8 \\

# &#x20; --seed 0 \\

# &#x20; --split phase4\_split.json \\

# &#x20; --output-dir results/phase4

# ```

# 

# Aggregate completed Phase 4 runs:

# 

# ```bash

# .venv/bin/python phase4\_aggregate.py \\

# &#x20; --input-dir results/phase4 \\

# &#x20; --output-dir analysis/phase4

# ```

# 

# Recovery of an incomplete run is explicit:

# 

# ```bash

# .venv/bin/python run\_phase4.py \\

# &#x20; --configuration gradient\_top3\_r8 \\

# &#x20; --seed 0 \\

# &#x20; --archive-incomplete

# ```

# 

# Completed runs cannot be overwritten; use a new output directory for intentional reruns.

# 

# \---

# 

# \## Current recommended research direction

# 

# Do not simply claim:

# 

# > Gradient-based LoRA layer selection is novel.

# 

# That is no longer defensible by itself.

# 

# Relevant papers in `papers/` and recent literature include:

# 

# \- Flexora: validation-driven/bilevel layer selection.

# \- DPLoRA: squared-gradient-based heterogeneous rank allocation and pruning.

# \- EVA: activation-variance-based initialization and adaptive rank allocation.

# \- FILet: data-aware Fisher-subspace LoRA initialization.

# \- L2-LoRA: layer-specific regularization based on knowledge localization.

# \- A Layer-wise Analysis of SFT: middle layers stable, final layers highly sensitive.

# \- FoRA-UA: extreme parameter-budget reduction, with multimodal evaluation left open.

# 

# The key gap is:

# 

# > Whether cheap pre-adaptation, task-aware signals can select LoRA locations across modalities under a fixed parameter budget.

# 

# This is more defensible than introducing another generic LoRA variant.

# 

# \---

# 

# \## Recommended next project

# 

# A practical multimodal extension:

# 

# \### Model

# 

# `openai/clip-vit-base-patch32`

# 

# \### Task

# 

# Image-text retrieval.

# 

# \### Dataset candidate

# 

# Flickr8k or another small fixed image-caption retrieval dataset.

# 

# \### Proposed question

# 

# > Can pre-fine-tuning task-gradient and activation-based signals identify which vision and language layers should receive LoRA under a fixed parameter budget?

# 

# CLIP creates a two-axis layer definition:

# 

# ```text

# tower ∈ {vision, text}

# layer ∈ {0, ..., 11}

# ```

# 

# Candidate adapter locations:

# 

# \- vision encoder layers

# \- text encoder layers

# \- projection layers as a separate control

# 

# Potential configurations:

# 

# \- vision-only selected layers

# \- text-only selected layers

# \- balanced vision/text selection

# \- random matched-budget selection

# \- activation-variance selection

# \- gradient-based selection

# \- full/reference adaptation

# 

# Metrics:

# 

# \- image-to-text Recall@1/5/10

# \- text-to-image Recall@1/5/10

# \- mean retrieval recall

# \- trainable parameter count

# \- runtime

# 

# Critical methodological point:

# 

# The vision and text towers have different dimensions, so equal LoRA ranks do not necessarily mean equal parameter budgets. Budget matching must use actual trainable parameter counts.

# 

# Do not call gradient norm and mean squared gradient separate predictors. They are rank-equivalent when candidate parameter counts are equal.

# 

# \---

# 

# \## Recommended staged plan

# 

# \### Stage A: literature-grounded design

# 

# Define:

# 

# \- model

# \- dataset

# \- exact split

# \- tower/layer units

# \- LoRA target modules

# \- matched parameter budget

# \- gradient probe

# \- activation-variance baseline

# \- retrieval metrics

# 

# \### Stage B: one-seed multimodal pilot

# 

# Run only:

# 

# \- frozen/zero-shot baseline

# \- gradient-selected configuration

# \- random matched-budget configuration

# \- activation-selected configuration

# 

# Use one seed first.

# 

# \### Stage C: expand only if machinery is trustworthy

# 

# If the pilot is valid:

# 

# \- use three seeds

# \- add vision-only/text-only/balanced comparisons

# \- add retrieval robustness checks

# \- compare against a simple full-layer reference

# 

# Do not jump directly to a large LLaVA/Qwen-VL experiment. That would confound layer placement with connector tuning, language-model adaptation, and generative instruction tuning.

# 

# \---

# 

# \## Literature interpretation

# 

# Recent work suggests:

# 

# \- Layer selection matters.

# \- Layer importance is architecture- and task-dependent.

# \- Validation-driven selection is expensive and can introduce selection bias.

# \- Squared-gradient budget allocation is already being explored.

# \- Activation variance and Fisher-subspace methods are active alternatives.

# \- Multimodal modality imbalance is a major factor.

# \- A clean cross-tower, fixed-budget, pre-adaptation selection study remains a reasonable gap.

# 

# The strongest eventual paper framing would be:

# 

# > A controlled study of modality-aware LoRA placement under a fixed parameter budget, testing whether pre-adaptation task signals transfer from unimodal language models to multimodal vision-language models.

# 

# \---

# 

# \## Operating constraints for the next chat

# 

# \- Do not launch Phase 4’s 21 runs automatically.

# \- Do not modify raw Phase 3 JSONs.

# \- Do not overwrite existing analysis artifacts without explicit intent.

# \- Do not introduce formal Fisher claims without implementing a real estimator.

# \- Do not treat gradient norm and squared-gradient proxy as independent predictors.

# \- Do not assume late layers are universally best.

# \- Do not start with a large MLLM unless the small multimodal pilot succeeds.

# \- Prefer code inspection, small smoke tests, and reproducible manifests before training.

# \- If searching for current papers, internet access is available.

# ```

