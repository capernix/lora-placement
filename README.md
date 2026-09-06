# LoRA Placement Transfer — Dummy-Layer Harness

Engineering scaffold for the LoRA placement transfer project (Track B,
Week 1). This is **not** the real RoBERTa/CLIP sweep — it's a small
synthetic stand-in used to validate the harness end-to-end before the
real sweep starts in Week 2, per the project plan.

## What's here

- `harness/model.py` — frozen toy backbone (6 sequential blocks) + `LoRALinear` adapter (rank 4, alpha 16)
- `harness/data.py` — synthetic binary classification task; label depends on a specific layer's representation, so there's a real (if synthetic) pattern to recover
- `harness/train_one_layer.py` — trains a LoRA adapter on exactly one layer, logs metrics to `results/<run_id>.json`
- `harness/importance.py` — computes CKA-input / gradient norm / empirical squared-gradient (diagonal-Fisher-style) proxy per layer, pre-fine-tuning, on a frozen probe batch
- `harness/aggregate_results.py` — builds the layer→utility-delta / layer→predictor table, preserves a joined per-seed CSV, computes Spearman correlation, and saves plots to `analysis/`
- `run_sweep.py` — orchestrates the full sweep end-to-end

## Usage

```bash
python3 -m pip install torch numpy pandas scipy matplotlib tabulate

# full sweep (6 layers x 3 seeds)
python3 run_sweep.py

# or individual pieces
python3 -m harness.train_one_layer --layer 3 --seed 0
python3 -m harness.importance --seed 0 --probe-size 256
python3 -m harness.aggregate_results
```

## Known limitations / next steps for whoever picks this up

- Runs log frozen baseline accuracy, LoRA accuracy, utility delta, trainable parameter count, and loss curve. Model checkpoints are still not saved.
- This uses a synthetic backbone/task, not RoBERTa-base or CLIP+RoBERTa+ScienceQA. Swapping in the real models means replacing `model.py`'s `ToyBackbone` with the real HF model wrapper, and `data.py` with the real SST-2 / ScienceQA loaders — the training loop, importance computation, and aggregation logic should carry over largely unchanged.
- Existing generated results and analysis artifacts must be regenerated after harness changes; toy correlations are sanity checks, not findings.

## Validation

Run the invariant checks with:

```bash
python3 -m unittest discover -s tests
```

The environment must provide Python 3 and the dependencies above, including PyTorch.
