import json
import os
import tempfile
from pathlib import Path

REQUIRED_RESULT_KEYS = {
    "status", "model", "requested_revision", "resolved_model_revision", "dataset",
    "layer", "reported_layer", "seed", "max_length", "lora_rank", "lora_alpha",
    "target_modules", "learning_rate", "batch_size", "epochs", "probe_size",
    "probe_seed", "frozen_val_acc", "lora_val_acc", "utility_delta",
    "n_trainable_params", "predictors", "training_time_sec",
    "gpu_peak_memory_allocated_mb",
}


def result_path(output_dir, layer, seed):
    return Path(output_dir) / f"layer{layer:02d}_seed{seed:02d}.json"


def is_complete_result(path):
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("status") != "completed" or not REQUIRED_RESULT_KEYS <= record.keys():
        return False
    try:
        expected_delta = round(record["lora_val_acc"] - record["frozen_val_acc"], 6)
    except (KeyError, TypeError):
        return False
    return record.get("utility_delta") == expected_delta


def write_json_atomic(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(record, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
