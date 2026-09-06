from itertools import product

from .config import RobertaSweepConfig
from .io import is_complete_result, result_path


def parse_csv_ints(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_jobs(value):
    return [tuple(map(int, item.strip().split(":"))) for item in value.split(",")]


def requested_jobs(layers, seeds, config=None):
    config = config or RobertaSweepConfig()
    if any(layer < 0 or layer >= config.num_layers for layer in layers):
        raise ValueError(f"layers must be in [0, {config.num_layers - 1}]")
    return list(product(layers, seeds))


def pending_jobs(output_dir, jobs):
    return [job for job in jobs if not is_complete_result(result_path(output_dir, *job))]


def print_status(output_dir, jobs):
    pending = pending_jobs(output_dir, jobs)
    print(f"Total: {len(jobs)}")
    print(f"Completed: {len(jobs) - len(pending)}")
    print(f"Remaining: {len(pending)}")
    if pending:
        print("Missing: " + ", ".join(f"(layer={l}, seed={s})" for l, s in pending))


def partition_jobs(jobs, workers=2):
    buckets = [[] for _ in range(workers)]
    loads = [0] * workers
    for job in sorted(jobs, key=lambda pair: pair[0], reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        buckets[index].append(job)
        loads[index] += 1
    return buckets
