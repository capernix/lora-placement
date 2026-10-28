import json
import tempfile
import unittest
from pathlib import Path

from phase4_io import (CompletedRunError, IncompleteRunError, archive_incomplete,
                       reserve_run, run_paths, write_completed_artifacts)
from phase4_aggregate import load_completed
from phase4_split import validate_split
from run_phase4 import make_sweep_config


class Phase4InfrastructureTests(unittest.TestCase):
    def test_completion_marker_is_required_and_published_last(self):
        with tempfile.TemporaryDirectory() as directory:
            reservation = reserve_run(directory, "example_seed0", {"configuration_id": "example", "seed": 0})
            paths = reservation.paths
            self.assertFalse(paths["complete"].exists())
            result = {"status": "artifact_ready", "run_id": "example_seed0"}
            predictions = {"run_id": "example_seed0", "predictions": [0], "labels": [0]}
            write_completed_artifacts(reservation, result, predictions)
            self.assertTrue(paths["complete"].exists())
            self.assertEqual(json.loads(paths["complete"].read_text())["status"], "completed")
            with self.assertRaises(CompletedRunError):
                reserve_run(directory, "example_seed0", {})

    def test_incomplete_runs_are_not_overwritten_and_can_be_archived_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            reservation = reserve_run(directory, "example_seed0", {})
            reservation.__exit__(RuntimeError, RuntimeError("interrupted"), None)
            with self.assertRaises(IncompleteRunError):
                reserve_run(directory, "example_seed0", {})
            archived = archive_incomplete(directory, "example_seed0")
            self.assertTrue(archived.exists())
            replacement = reserve_run(directory, "example_seed0", {})
            replacement.__exit__(None, None, None)

    def test_split_validation_rejects_overlap(self):
        payload = {"source_size": 4, "train_indices": [0, 1, 2], "holdout_indices": [2, 3], "holdout_ids": [2, 3], "holdout_labels": [0, 1]}
        with self.assertRaises(ValueError):
            validate_split(payload)

    def test_phase4_revision_is_required_and_propagated(self):
        config = {
            "model": {"name": "roberta-base", "resolved_revision": "revision-test"},
            "dataset": {"name": "nyu-mll/glue", "config": "sst2", "max_length": 128},
            "lora": {"alpha": 16, "target_modules": ["query", "value"]},
            "training": {"learning_rate": 3e-4, "batch_size": 32, "epochs": 5},
        }
        self.assertEqual(make_sweep_config(config, 8).model_revision, "revision-test")
        config["model"].pop("resolved_revision")
        with self.assertRaises(ValueError):
            make_sweep_config(config, 8)

    def test_aggregation_ignores_uncompleted_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            reservation = reserve_run(directory, "complete_seed0", {})
            result = {"status": "artifact_ready", "run_id": "complete_seed0", "configuration": {"id": "cfg", "layers": [0], "rank": 8}, "seed": 0, "n_trainable_params": 1, "holdout_metrics": {"accuracy": 1.0, "loss": 0.0, "precision_positive": 1.0, "recall_positive": 1.0, "f1_positive": 1.0}, "runtime_sec": 1, "selected_epoch": 5, "dataset_fingerprint": "x", "model": {"resolved_revision": "r"}}
            predictions = {"run_id": "complete_seed0", "predictions": [1], "labels": [1]}
            write_completed_artifacts(reservation, result, predictions)
            incomplete = reserve_run(directory, "incomplete_seed0", {})
            incomplete.__exit__(RuntimeError, RuntimeError("interrupted"), None)
            master, ignored = load_completed(directory)
            self.assertEqual(len(master), 1)
            self.assertEqual(ignored, ["incomplete_seed0"])


if __name__ == "__main__":
    unittest.main()
