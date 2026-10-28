"""Completion-safe Phase 4 run artifacts and exclusive run reservations."""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path


class RunExistsError(RuntimeError):
    pass


class IncompleteRunError(RunExistsError):
    pass


class CompletedRunError(RunExistsError):
    pass


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def run_paths(output_dir, run_id):
    directory = Path(output_dir) / run_id
    return {
        "directory": directory,
        "lock": directory / ".run.lock",
        "status": directory / "status.json",
        "result": directory / "result.json",
        "predictions": directory / "predictions.json",
        "complete": directory / "COMPLETE.json",
    }


class RunReservation:
    def __init__(self, paths):
        self.paths = paths
        self.completed = False

    def __enter__(self):
        self.paths["directory"].mkdir(parents=True, exist_ok=False)
        try:
            self.paths["lock"].mkdir()
        except Exception:
            shutil.rmtree(self.paths["directory"], ignore_errors=True)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.completed:
            self.paths["lock"].rmdir()
        return False


def reserve_run(output_dir, run_id, running_metadata):
    paths = run_paths(output_dir, run_id)
    if paths["directory"].exists():
        if paths["complete"].exists():
            raise CompletedRunError(f"completed run already exists: {paths['directory']}")
        files = sorted(str(path.relative_to(paths["directory"])) for path in paths["directory"].iterdir())
        raise IncompleteRunError(
            f"incomplete run exists: {paths['directory']}; artifacts={files}. "
            "Choose a new output directory or explicitly archive/recover this run."
        )
    reservation = RunReservation(paths)
    reservation.__enter__()
    atomic_json(paths["status"], {"status": "running", "run_id": run_id, "started_at": time.time(), **running_metadata})
    return reservation


def write_completed_artifacts(reservation, result, predictions):
    paths = reservation.paths
    if result.get("run_id") != predictions.get("run_id"):
        raise ValueError("result and prediction run IDs differ")
    if result.get("status") != "artifact_ready":
        raise ValueError("result must be artifact_ready before completion")
    atomic_json(paths["result"], result)
    atomic_json(paths["predictions"], predictions)
    # Validate both complete JSON artifacts before publishing completion.
    loaded_result = read_json(paths["result"])
    loaded_predictions = read_json(paths["predictions"])
    if loaded_result.get("run_id") != loaded_predictions.get("run_id"):
        raise ValueError("written artifacts failed run-ID validation")
    atomic_json(paths["complete"], {
        "status": "completed",
        "run_id": result["run_id"],
        "result": paths["result"].name,
        "predictions": paths["predictions"].name,
        "completed_at": time.time(),
    })
    reservation.completed = True
    paths["lock"].rmdir()


def archive_incomplete(output_dir, run_id):
    """Explicit recovery action: preserve and rename an incomplete run directory."""
    paths = run_paths(output_dir, run_id)
    if not paths["directory"].exists():
        raise FileNotFoundError(paths["directory"])
    if paths["complete"].exists():
        raise CompletedRunError(f"refusing to archive completed run: {paths['directory']}")
    target = paths["directory"].with_name(f"{run_id}.incomplete.{int(time.time())}")
    shutil.move(paths["directory"], target)
    return target
