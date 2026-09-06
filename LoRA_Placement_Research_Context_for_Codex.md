# Codex Context: LoRA Layer Placement Research Project

## 0. Purpose of this document

This document is the source of truth for understanding the current research direction and the immediate engineering work.

The project is **not** trying to prove that a particular LoRA placement, CKA metric, gradient metric, or Fisher metric is best. Existing experimental claims from an earlier project/PDF should **not** be trusted as established findings.

The goal is to build a rigorous experimental pipeline that lets the data answer the research questions.

The current uploaded dummy-layer harness is a prototype for validating that pipeline before moving to real models.

---

# 1. Research goal

The central research problem is:

> **How should limited LoRA adaptation capacity be allocated across layers, especially when moving from unimodal language models to multimodal models?**

A sharper version is:

> **Can inexpensive, pre-fine-tuning layer-importance signals predict where LoRA adaptation will be useful, and does that relationship transfer from unimodal language adaptation to multimodal adaptation?**

The project should study three related concepts:

### 1.1 Intrinsic / pre-fine-tuning importance

A signal that tells us something about a layer before LoRA fine-tuning.

Examples currently under consideration:
- CKA-based representation statistics
- downstream-task gradient norm
- empirical Fisher / squared-gradient statistic

These are not all "training-free" in the strict sense.

For this project, prefer the terminology:

> **pre-fine-tuning, low-cost importance signals**

Gradient/Fisher are task-aware because they use the downstream task loss and labels.

### 1.2 Adaptation utility

How much benefit is obtained by putting LoRA at a particular layer.

The preferred primary quantity should be:

`utility(layer) = LoRA performance(layer) - frozen-baseline performance`

Absolute validation performance should also be reported.

### 1.3 Allocation efficiency

Given a fixed LoRA parameter budget, how effectively can we choose which layers receive LoRA?

This becomes the practical component of the study.

---

# 2. Current research questions

Use these as the working RQs.

## RQ1

> **How does the utility of LoRA adaptation vary across layers in unimodal and multimodal transformer models?**

This establishes whether placement actually matters and how the layer-utility profile differs between settings.

## RQ2

> **To what extent can pre-fine-tuning, low-cost layer-importance signals predict LoRA adaptation utility, and does their predictive relationship transfer across the two settings?**

The important part is that the predictor must be computed **before LoRA fine-tuning**.

Do not select or redefine a metric after seeing which one correlates best.

## RQ3

> **Can importance-guided layer selection achieve competitive performance while reducing the number of LoRA configurations that must be trained?**

This turns prediction into a practical selection problem.

The project should compare importance-guided selection against appropriate baselines rather than claiming superiority without evidence.

---

# 3. Scope for the semester

Keep the project focused.

## Primary settings

### Setting A: Unimodal language

Use RoBERTa as the controlled baseline.

The initial experiment should sweep individual RoBERTa layers one at a time.

### Setting B: Multimodal vision-language

Use the existing CLIP + RoBERTa multimodal setup and its downstream task.

The multimodal experiment is the main extension of the controlled language experiment.

## Explicitly out of scope for now

Do NOT expand the semester project into:
- speech/audio
- code models
- many additional foundation models
- a huge combinatorial layer-placement search
- many LoRA variants such as DoRA/PiSSA/AdaLoRA before the core phenomenon is established

Those can be future work.

---

# 4. What the current dummy harness is for

The uploaded harness is a **sanity/unit-test environment**, not research evidence.

Its structure is:

- frozen toy sequential backbone
- six synthetic blocks
- one LoRA adapter inserted into exactly one layer
- trainable task head
- synthetic binary classification
- pre-fine-tuning importance calculation
- aggregation of layer utility and predictor values
- Spearman correlation and plots

Relevant files:

- `harness/model.py`
- `harness/train_one_layer.py`
- `harness/importance.py`
- `harness/aggregate_results.py`

The dummy model explicitly uses a synthetic task whose label generation is associated with a particular internal layer. Therefore, any strong correlation in this toy setup must NOT be presented as evidence that the corresponding metric predicts LoRA placement in real models.

The dummy harness only needs to demonstrate that:
1. a layer can be isolated for LoRA;
2. the rest of the backbone remains frozen;
3. training runs reproducibly;
4. pre-fine-tuning predictors can be computed;
5. results can be joined correctly;
6. utility/predictor analyses run end-to-end.

Do not keep adding sophisticated research features to the toy model unless necessary to validate the real pipeline.

---

# 5. Current harness implementation

## `model.py`

The toy model contains:
- `NUM_LAYERS = 6`
- hidden dimension 64
- rank-4 LoRA
- alpha 16
- frozen backbone
- trainable linear head

`inject_lora(layer_idx)` replaces the selected block's linear layer with a LoRA-wrapped linear layer.

The backbone is frozen and only:
- the task head
- LoRA A/B parameters

are trainable during the one-layer sweep.

This is appropriate for the unit-test setup.

## `train_one_layer.py`

For each run:
1. set seed;
2. generate synthetic dataset;
3. split train/validation;
4. create deterministic frozen backbone;
5. inject LoRA into exactly one layer;
6. train head + LoRA;
7. record validation accuracy, training accuracy, loss curve, parameter count and elapsed time.

This is the correct basic structure.

## `importance.py`

Currently computes:
- CKA
- gradient norm
- squared-gradient empirical Fisher proxy

before LoRA training.

The important scientific property is that these predictors are computed before adaptation.

However, the exact definitions need to be made explicit and, where appropriate, improved.

## `aggregate_results.py`

Currently:
- averages utility across seeds per layer;
- averages predictor values across seeds per layer;
- calculates Spearman correlation;
- creates plots.

This is fine as an initial diagnostic, but the real experiment needs richer per-seed analysis and should not rely primarily on p-values from correlations over only 6 or 12 layers.

---

# 6. Required changes to the dummy harness before RoBERTa

These are the immediate engineering tasks.

## 6.1 Add a frozen-baseline measurement

Currently utility is effectively treated as validation accuracy.

Add a frozen baseline evaluation.

Record at minimum:

```text
frozen_val_acc
lora_val_acc
utility_delta
```

where:

```text
utility_delta = lora_val_acc - frozen_val_acc
```

The primary layer-utility analysis should use `utility_delta`, while absolute accuracy remains a reported metric.

Reason:

A layer that produces 80% after LoRA is not necessarily more useful for adaptation than a layer that goes from 70% to 79%.

We care about the benefit attributable to adaptation.

---

# 7. Clarify CKA

The current CKA implementation compares each layer representation with the raw input representation.

Conceptually this measures representation similarity/retention relative to the input.

Do not silently redefine this.

Instead, make the definition explicit.

Potentially support two separately named signals:

### CKA-input

```text
CKA(h_l, h_0)
```

where `h_0` is the input representation.

Interpretation:
> How similar is the layer representation to the original input?

### CKA-local-change

```text
1 - CKA(h_{l-1}, h_l)
```

Interpretation:
> How much does the representation change at this layer?

If both are implemented, treat them as distinct predictors.

Do not choose whichever one gives the strongest result after seeing the real data.

Document the motivation for each.

---

# 8. Clarify gradient norm and Fisher

The current implementation computes gradients of the downstream BCE loss with respect to each block's pretrained weight matrix.

That is a legitimate signal.

Define it precisely.

### Gradient norm

For layer l:

```text
G_l = || dL / dW_l ||
```

where `L` is the downstream task loss evaluated before LoRA fine-tuning.

### Empirical Fisher proxy

Current implementation uses the element-wise squared gradient averaged over the weight matrix:

```text
F_l = mean((dL / dW_l)^2)
```

This should be described as an empirical diagonal-Fisher-style / squared-gradient proxy, not necessarily as a full Fisher information calculation.

Important:

These metrics are **task-aware**, because they use the downstream task loss and labels.

Do not call them simply "training-free."

Use:
> pre-fine-tuning, low-cost, task-aware signals

unless the methodology is changed.

---

# 9. Prevent information leakage

The predictor pipeline must obey this rule:

```text
pretrained/frozen model
        +
allowed downstream probe data
        ↓
importance signal
        ↓
layer ranking
        ↓
LoRA fine-tuning
```

Never:

```text
LoRA fine-tuning
        ↓
changed representation
        ↓
importance calculation
        ↓
claim of prediction
```

No post-fine-tuning information may be used to construct a predictor that is presented as pre-fine-tuning.

If any metric requires labels, say so explicitly.

---

# 10. Improve the aggregation/statistical analysis

Do not collapse everything immediately to one mean per layer.

Retain raw observations:

```text
layer × seed × utility
layer × seed × predictor
```

For the real experiments, report:
- mean and standard deviation across seeds;
- per-seed results where useful;
- Spearman correlation as a ranking diagnostic;
- Kendall tau where useful;
- confidence intervals/bootstrap where justified;
- top-k layer identification;
- ranking agreement;
- regret from predictor-selected layers;
- actual downstream performance after selecting layers using the predictor.

With only 12 RoBERTa layers, correlation p-values should not be treated as the primary evidence.

The scientifically important question is whether a predictor actually helps select useful layers.

---

# 11. The real RoBERTa experiment

Once the dummy harness passes the sanity checks, move to RoBERTa.

## Initial sweep

Perform a one-layer-at-a-time sweep:

```text
L1
L2
L3
...
L12
```

Keep everything constant except LoRA placement.

Control:
- dataset split;
- model checkpoint;
- tokenizer;
- LoRA rank;
- LoRA alpha;
- learning rate;
- optimizer;
- training epochs/steps;
- batch size;
- evaluation protocol;
- random seeds.

Start with approximately 3 seeds for development.

Increase to 5 if compute and variance justify it.

Do not prematurely run an enormous sweep.

---

# 12. RoBERTa measurement pipeline

For each layer:

## Before LoRA training

Compute:

```text
CKA-input
optional CKA-local-change
gradient norm
Fisher proxy
```

using a fixed probe protocol.

Save the raw predictor values.

## Baseline

Evaluate the frozen model.

Save:

```text
frozen_val_acc
```

## LoRA run

Train LoRA at exactly one layer.

Save:

```text
lora_val_acc
utility_delta
trainable_parameters
training time
loss curve
checkpoint/configuration metadata
```

The resulting dataset should allow us to build:

```text
layer → actual adaptation utility
layer → predictor values
```

and compare their rankings.

---

# 13. Core RoBERTa analysis

The first major figure/table should show:

```text
Layer
↓
Actual LoRA utility
↓
CKA
↓
Gradient norm
↓
Fisher
```

Then answer:

### RQ1
Is utility actually layer-dependent?

### RQ2
Do pre-fine-tuning signals rank useful layers?

Do NOT assume:
- positive correlation;
- negative correlation;
- CKA superiority;
- gradient superiority;
- Fisher superiority.

Any result is acceptable if the methodology is sound.

---

# 14. Multimodal experiment

After the unimodal experiment is stable, apply the methodology to the existing CLIP + RoBERTa multimodal model.

The key question is not simply:

> Which layer is best?

The deeper question is:

> Does the relationship between pre-fine-tuning importance and LoRA utility transfer from unimodal language adaptation to multimodal adaptation?

Consider meaningful components separately:
- language encoder layers;
- vision encoder layers;
- fusion/connector components if the architecture contains meaningful trainable placement points.

Do not force a fake one-to-one correspondence between CLIP and RoBERTa layers.

The multimodal architecture should determine what counts as a placement unit.

---

# 15. Fixed-budget layer selection

After establishing single-layer utility profiles, conduct a modest fixed-budget experiment for RQ3.

Do NOT brute-force all possible subsets.

Instead compare methods such as:

1. random layer selection;
2. simple baseline heuristic;
3. predictor-guided selection;
4. small empirical oracle based on actual validation utility.

All comparisons should use comparable LoRA parameter budgets.

The question is:

> Can a cheap pre-fine-tuning signal choose a small set of useful layers without training every possible placement configuration?

Measure:
- final downstream performance;
- number of configurations trained;
- parameter count;
- regret relative to the empirical oracle;
- consistency across seeds.

Only propose a new selection method if the evidence supports one.

---

# 16. What NOT to do

Do not:
- optimize metrics to obtain a desired correlation;
- claim a toy-model correlation as a research finding;
- call label-dependent gradients/Fisher "task-free";
- use post-fine-tuning representations as predictors;
- report only the best seed;
- compare methods with different LoRA parameter budgets;
- run huge combinatorial searches before establishing the basic layer-utility phenomenon;
- add speech/code merely to make the project sound broader;
- add DoRA/PiSSA/AdaLoRA/etc. before the core LoRA placement question is understood;
- make universal claims from two domains;
- assume the old PDF's conclusions are correct.

---

# 17. Reproducibility requirements

Every real experiment should save enough metadata to reproduce the run.

At minimum:

```text
model/checkpoint
dataset
split
seed
layer/component
LoRA rank
LoRA alpha
learning rate
optimizer
batch size
epochs/steps
probe configuration
evaluation metric
trainable parameter count
runtime
```

Save checkpoints for important real runs.

The result files should be machine-readable.

Prefer a stable schema that can later be aggregated across:
- model;
- domain;
- layer;
- seed;
- predictor;
- LoRA configuration.

---

# 18. Suggested project architecture

Aim toward a reusable experiment harness rather than scripts that only work for the toy model.

Conceptually:

```text
configs/
  roberta.yaml
  multimodal.yaml

models/
  roberta_adapter.py
  multimodal_adapter.py
  toy.py

datasets/
  ...

importance/
  cka.py
  gradients.py
  fisher.py

experiments/
  sweep.py
  fixed_budget.py

analysis/
  utility.py
  prediction.py
  selection.py
  plots.py

results/
  raw/
  processed/
```

The exact directory structure can change.

The important design goal is:

> The experimental logic should be model/config driven, not hard-coded around the six-layer toy model.

---

# 19. Development order

## Step 1 — Finish dummy harness

Implement:
- frozen baseline;
- utility delta;
- explicit CKA definitions;
- precise gradient/Fisher definitions;
- raw per-seed result preservation.

Then run the toy sweep only to verify correctness.

## Step 2 — Build RoBERTa adapter

Make it possible to specify:

```text
model
dataset
layer
rank
alpha
seed
```

from configuration/CLI.

## Step 3 — Validate one RoBERTa layer manually

Before launching all 12 layers:
- verify LoRA is actually inserted where intended;
- verify only intended parameters are trainable;
- verify frozen parameters do not change;
- verify evaluation works;
- verify checkpoint loading works.

## Step 4 — Run the full single-layer sweep

Start with 3 seeds.

## Step 5 — Analyze RQ1/RQ2

Do not proceed to complicated selection experiments until the basic results are trustworthy.

## Step 6 — Multimodal adaptation

Port the same experimental logic to CLIP + RoBERTa.

## Step 7 — Fixed-budget selection

Test whether importance-guided selection has practical value.

## Step 8 — Paper analysis

Only after the results are available should the final claims, hypotheses, and proposed method be finalized.

---

# 20. Literature / novelty context

The project should be positioned carefully.

Existing work already explores:
- layer-wise PEFT;
- layer/block importance;
- adaptive layer/rank allocation;
- training-free or low-cost layer selection;
- multimodal PEFT module importance;
- vision-language layer filtering.

Therefore:

> "We sweep LoRA over layers and compute CKA/gradient/Fisher correlations"

is not sufficient by itself as the novelty claim.

The potentially stronger contribution is the **empirical study of the relationship between pre-fine-tuning importance and actual LoRA adaptation utility, especially whether that relationship transfers between unimodal language and multimodal vision-language settings, plus the practical question of budgeted layer selection.**

Do not claim this as established novelty until a literature audit verifies it.

Literature review should be timeboxed and focused on the closest work rather than becoming an open-ended task.

---

# 21. Current conceptual paper structure

A likely structure is:

## Introduction
Limited PEFT budgets make placement a decision problem.

## Related Work
- LoRA/PEFT
- layer selection
- adaptive PEFT
- multimodal PEFT
- importance-based selection

## Experimental Framework
Define:
- layer utility;
- pre-fine-tuning predictors;
- selection/benchmarking methodology.

## Unimodal Study
RoBERTa single-layer sweep.

## Multimodal Study
CLIP + RoBERTa layer/component sweep.

## Predictor Analysis
How well do importance signals predict actual adaptation utility?

## Budgeted Selection
Can predictor-guided selection reduce the number of configurations that must be trained?

## Discussion
What transfers and what does not between unimodal and multimodal settings?

## Limitations / Future Work
Potential extension to speech/audio, code, larger models, other PEFT methods, etc.

---

# 22. Scientific philosophy

The project should be hypothesis-driven but **result-agnostic**.

We are not trying to prove:

> "late layers are best."

We are not trying to prove:

> "CKA predicts LoRA placement."

We are not trying to prove:

> "gradient norm is better than Fisher."

Instead, we are asking questions where the answer is empirically determined.

A null result can be valuable.

For example:

> "Pre-fine-tuning importance signals do not reliably predict LoRA utility."

could be a meaningful finding if demonstrated carefully.

Likewise:

> "The predictor works in RoBERTa but fails in multimodal adaptation."

could be more interesting than a universal positive result.

The goal is to produce a study that is difficult to dismiss because:
- definitions are precise;
- predictors do not leak information;
- baselines are controlled;
- seeds are handled correctly;
- parameter budgets are matched;
- conclusions follow the data.

---

# 23. Immediate next action for Codex

Before changing large parts of the project:

1. Inspect the current repository structure and existing README/configuration.
2. Inspect the four dummy harness files in context.
3. Implement the smallest clean set of changes needed to:
   - add frozen baseline;
   - calculate utility delta;
   - make CKA definitions explicit;
   - preserve per-seed predictor/utility data;
   - improve aggregation without destroying raw results.
4. Add tests/checks for:
   - exactly one LoRA layer being trainable;
   - backbone weights remaining frozen;
   - predictor calculation occurring before LoRA training;
   - reproducibility under the same seed.
5. Run the dummy sweep.
6. Verify the output schema and analysis.
7. Do NOT begin a large RoBERTa sweep until these checks pass.
8. Then implement the RoBERTa adapter and validate one layer manually.
9. Only then launch the complete RoBERTa sweep.

When making code changes, prioritize correctness and scientific validity over adding features.

---

# 24. Final success criterion

The project is successful if we can go from:

```text
pretrained model
      ↓
cheap pre-fine-tuning layer signals
      ↓
predicted layer ranking
      ↓
limited LoRA training budget
      ↓
competitive downstream performance
```

and rigorously quantify where and when this works.

The final paper should explain not just **which layer wins**, but:

> **why layer placement matters, whether inexpensive pre-fine-tuning signals tell us where adaptation will help, and whether those relationships remain valid when moving from unimodal language to multimodal vision-language adaptation.**
