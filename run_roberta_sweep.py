#!/usr/bin/env python3
import argparse
import tempfile
from pathlib import Path

import torch
from transformers import DataCollatorWithPadding

from roberta.config import RobertaSweepConfig
from roberta.data import load_sst2
from roberta.io import is_complete_result, result_path, write_json_atomic
from roberta.model import inject_single_layer_lora, load_model, load_tokenizer, trainable_parameters
from roberta.sweep import parse_csv_ints, parse_jobs, pending_jobs, print_status, requested_jobs


def smoke_test(output_dir):
    config = RobertaSweepConfig()
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires CUDA; no CUDA device is available")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    tokenizer = load_tokenizer(config)
    train, validation, probe = load_sst2(tokenizer, config.max_length, config.probe_size, config.probe_seed)
    model = inject_single_layer_lora(load_model(config, 0), config, 0)
    names = [name for name, parameter in trainable_parameters(model)]
    assert any("encoder.layer.0" in name and "lora_A" in name for name in names)
    assert any("encoder.layer.0" in name and "lora_B" in name for name in names)
    assert all("encoder.layer.0" in name or "classifier" in name for name in names)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    batch = collator([train[i] for i in range(2)])
    device = next(model.parameters()).device
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    model.train()
    model(**batch).loss.backward()
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        marker = Path(temp_dir) / "smoke_marker.json"
        write_json_atomic(marker, {"status": "smoke_only", "train_size": len(train), "validation_size": len(validation), "probe_size": len(probe)})
        assert not is_complete_result(marker)
    print("Smoke test passed (no research result was written).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--jobs", help="Explicit layer:seed pairs, e.g. 0:0,1:2")
    parser.add_argument("--output-dir", default="results/roberta")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.smoke_test:
        output_dir.mkdir(parents=True, exist_ok=True)
        smoke_test(output_dir)
        return
    jobs = parse_jobs(args.jobs) if args.jobs else requested_jobs(parse_csv_ints(args.layers), parse_csv_ints(args.seeds))
    if args.status:
        print_status(output_dir, jobs)
        return
    jobs_to_run = pending_jobs(output_dir, jobs) if args.skip_existing else jobs
    for layer, seed in jobs_to_run:
        print(f"Running layer={layer} seed={seed}", flush=True)
        from roberta.train import run_experiment
        run_experiment(layer=layer, seed=seed, output_dir=output_dir)


if __name__ == "__main__":
    main()
