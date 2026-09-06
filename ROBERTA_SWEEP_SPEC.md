# RoBERTa Sweep — Locked Decisions

Answers to every open item in the repo audit. Treat this as the spec Codex
implements against — if something here turns out to be wrong once you're
running real experiments, change it here first, then propagate, don't let
the code and this doc drift apart.

## Model

- Checkpoint: `roberta-base`, revision pinned to `main` (record the actual commit hash once you first download it, so re-runs six months from now use the identical weights).
- Tokenizer: `AutoTokenizer.from_pretrained("roberta-base")`, same revision.
- Max sequence length: 128. SST-2 sentences are short single sentences; 128 is generous headroom without wasting compute on padding.

## Dataset

- Source: `nyu-mll/glue`, config `sst2` (Hugging Face `datasets`).
- Split: standard `train` / `validation` — GLUE's SST-2 test split has no public labels, so validation is your only real eval set. Don't touch it until final numbers; if you need a tuning set, carve one out of train.

## Layer definition

- One placement unit = one full `roberta.encoder.layer[i]` (attention + intermediate + output), matching your existing proposal. Confirmed correct.
- Embeddings and the classification head are **excluded** from placement — they're either always-frozen (embeddings) or always-trainable (classifier head), identically across all 12 runs. Only the LoRA injection point varies.
- Indexing: 0-based internally (`layer[0]`...`layer[11]`, matches PyTorch/HF convention — don't fight the framework). Report as **1-based L1–L12** in results tables and the paper. Put this mapping in one place (a constant or a comment in `model.py`), not re-derived per script.

## LoRA config

- Target modules: **query + value only**, applied together as one adapter per block. This is the standard lightweight LoRA setup from the original paper and keeps trainable-parameter count identical across all 12 runs — important, since a varying param count across layers would confound "utility" with "adaptation capacity," which is exactly the variable RQ1 is supposed to isolate.
- Rank: 8. Alpha: 16 (alpha = 2×rank is the common default ratio). **Your toy harness currently uses rank 4 / alpha 16** — that's a 4x scaling factor, unusually aggressive. Fix this before the real sweep; a mismatched rank/alpha ratio changes effective LoRA magnitude in a way that's easy to mistake for a layer-placement effect.

## Baseline / utility definition

- Baseline = frozen RoBERTa encoder + trainable classifier head only (matches your existing `frozen_val_acc`). This is **not** a fully fine-tuned baseline — say so explicitly in the paper, since it changes what `utility_delta` means: "benefit of adapting layer i over doing nothing to the encoder," not "benefit over full fine-tuning."
- Classifier head: trained from scratch, from a fixed init seed shared across all runs at a given training seed (so head initialization isn't an uncontrolled variable layer-to-layer).

## Training

- Optimizer: AdamW.
- Learning rate: 3e-4 (LoRA's small parameter count tolerates a higher LR than full fine-tuning would).
- Batch size: 32.
- Epochs: 5 — SST-2 converges fast; watch val loss and consider cutting to 3 if you see overfitting past epoch 3 in the pilot run.
- Seeds: 3 (0, 1, 2), applied to both model init (classifier head) and data shuffling.

## Probe protocol (importance metrics)

- Fixed 512-example subset of `train`, drawn with `probe_seed=42`, reused **identically** for CKA, gradient norm, and Fisher — this is what makes the three predictors comparable to each other, not just each internally consistent.
- CKA representation: **pooled** **`<s>`** **(CLS-equivalent) token**, not token-level. This matches the granularity of what the classifier head actually consumes — using token-level CKA while the downstream task is sequence-level would measure a different thing than what predicts classification utility.

## Checkpointing

- Do **not** save full model weights per run — 36 runs × RoBERTa-sized checkpoints is wasted storage for no analysis benefit. Save only the result JSON (metrics + importance scores).
- Exception: if a later budget-selection run (RQ3) becomes a candidate for the "small oracle," save that specific checkpoint — you'll want it if a reviewer asks for a qualitative example.

## What's still genuinely open (not a Codex question, an you-and-advisor question)

- Whether 5 epochs is enough or SST-2 needs more/fewer — this is an empirical call from the pilot run, not something to lock in advance.
- Whether to add MNLI as a confirmatory second unimodal task once SST-2 results are in — deferred per the semester-scope cut; revisit only if time allows after the multimodal sweep, never before it.