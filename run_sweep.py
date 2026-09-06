"""
run_sweep.py

Drives the full dummy-layer harness test end-to-end, exactly as the
Week-1 plan specifies: "Get this running end-to-end on 1-2 dummy layers
before Week 2 -- you want to find harness bugs now, not during the real
sweep." Here we run all 6 dummy layers x 3 seeds (=18 runs) since it's
cheap, to get a fuller demo signal than the plan's minimum.

Usage:
    python3 run_sweep.py
"""
import time
from harness.model import NUM_LAYERS
from harness.train_one_layer import run as train_run
from harness.importance import compute_importance
from harness import aggregate_results

SEEDS = [0, 1, 2]


def main():
    t0 = time.time()
    print(f"Sweep: {NUM_LAYERS} layers x {len(SEEDS)} seeds = {NUM_LAYERS * len(SEEDS)} LoRA runs\n")

    print("--- Step 1/3: importance signals (CKA-input / grad-norm / squared-gradient proxy) per seed ---")
    for seed in SEEDS:
        rec = compute_importance(seed, probe_size=256)
        print(f"  seed={seed}: computed for {len(rec['cka'])} layers")

    print("\n--- Step 2/3: training one LoRA layer at a time ---")
    for layer in range(NUM_LAYERS):
        for seed in SEEDS:
            record = train_run(layer, seed, epochs=60, verbose=False)
            print(f"  layer={layer} seed={seed} -> lora_val_acc={record['lora_val_acc']:.4f} "
                  f"(params={record['n_trainable_params']}, {record['elapsed_sec']}s)")

    print("\n--- Step 3/3: aggregating results, correlating, plotting ---")
    aggregate_results.main()

    print(f"\nTotal sweep time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
