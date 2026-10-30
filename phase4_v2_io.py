"""Additive, completion-safe I/O for Phase 4 v2 artifacts."""
from __future__ import annotations

import hashlib
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


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_payload(payload):
    return hashlib.sha256(canonical_json(payload)).hexdigest()


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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_paths(output_dir, run_id):
    directory = Path(output_dir) / run_id
    return {"directory": directory, "lock": directory / ".run.lock",
            "status": directory / "status.json", "result": directory / "result.json",
            "predictions": directory / "predictions.json", "complete": directory / "COMPLETE.json",
            "checkpoint": directory / "checkpoint"}


class RunReservation:
    def __init__(self, paths):
        self.paths = paths
        self.completed = False

    def __exit__(self, exc_type, exc, tb):
        if not self.completed and self.paths["lock"].exists():
            self.paths["lock"].rmdir()
        return False


def reserve_run(output_dir, run_id, running_metadata):
    paths = run_paths(output_dir, run_id)
    if paths["directory"].exists():
        if paths["complete"].exists():
            raise CompletedRunError(f"completed run already exists: {paths['directory']}")
        raise IncompleteRunError(f"incomplete run exists: {paths['directory']}")
    paths["directory"].mkdir(parents=True, exist_ok=False)
    try:
        paths["lock"].mkdir()
    except Exception:
        shutil.rmtree(paths["directory"], ignore_errors=True)
        raise
    atomic_json(paths["status"], {"protocol_version": "phase4_v2", "status": "running",
                                   "run_id": run_id, "started_at": time.time(), **running_metadata})
    return RunReservation(paths)


def write_completed_artifacts(reservation, result, predictions, checkpoint_hash=None):
    paths = reservation.paths
    if result.get("run_id") != predictions.get("run_id"):
        raise ValueError("result and prediction run IDs differ")
    if result.get("status") != "artifact_ready":
        raise ValueError("result must be artifact_ready before completion")
    if checkpoint_hash is not None:
        result["checkpoint_sha256"] = checkpoint_hash
    atomic_json(paths["result"], result)
    atomic_json(paths["predictions"], predictions)
    loaded_result = read_json(paths["result"])
    loaded_predictions = read_json(paths["predictions"])
    if loaded_result.get("run_id") != loaded_predictions.get("run_id"):
        raise ValueError("written artifacts failed run-ID validation")
    atomic_json(paths["status"], {"protocol_version": "phase4_v2", "status": "completed",
                                   "run_id": result["run_id"], "completed_at": time.time()})
    atomic_json(paths["complete"], {"protocol_version": "phase4_v2", "status": "completed",
                                     "run_id": result["run_id"], "result": paths["result"].name,
                                     "predictions": paths["predictions"].name,
                                     "completed_at": time.time()})
    reservation.completed = True
    paths["lock"].rmdir()


def mark_failed(reservation, error):
    atomic_json(reservation.paths["status"], {"protocol_version": "phase4_v2",
                                               "status": "failed", "run_id": reservation.paths["directory"].name,
                                               "error": repr(error), "failed_at": time.time()})


def archive_incomplete(output_dir, run_id):
    paths = run_paths(output_dir, run_id)
    if not paths["directory"].exists():
        raise FileNotFoundError(paths["directory"])
    if paths["complete"].exists():
        raise CompletedRunError(f"refusing to archive completed run: {paths['directory']}")
    target = paths["directory"].with_name(f"{run_id}.incomplete.{int(time.time())}")
    shutil.move(paths["directory"], target)
    return target
