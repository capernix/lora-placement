#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from roberta.sweep import parse_csv_ints, partition_jobs, pending_jobs, requested_jobs


def main():
    parser = argparse.ArgumentParser(description="Run independent sweep jobs on two GPUs.")
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-dir", default="results/roberta")
    parser.add_argument("--logs-dir", default="logs/roberta")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise SystemExit("At least two CUDA GPUs are required")
    jobs = requested_jobs(parse_csv_ints(args.layers), parse_csv_ints(args.seeds))
    if args.skip_existing:
        jobs = pending_jobs(args.output_dir, jobs)
    buckets = partition_jobs(jobs, 2)
    logs = Path(args.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    processes, handles = [], []
    for gpu, bucket in enumerate(buckets):
        if not bucket:
            continue
        job_arg = ",".join(f"{layer}:{seed}" for layer, seed in bucket)
        command = [sys.executable, "run_roberta_sweep.py", "--jobs", job_arg, "--output-dir", args.output_dir]
        if args.skip_existing:
            command.append("--skip-existing")
        handle = (logs / f"worker_gpu{gpu}.log").open("w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        processes.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
        handles.append(handle)
    codes = [process.wait() for process in processes]
    for handle in handles:
        handle.close()
    if any(code != 0 for code in codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
