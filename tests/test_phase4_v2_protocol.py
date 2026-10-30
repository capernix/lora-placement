import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from phase4_v2_aggregate import load_completed
from phase4_v2_io import (CompletedRunError, IncompleteRunError, mark_failed,
                           reserve_run, write_completed_artifacts)
from phase4_v2_manifest import build_manifest
from phase4_v2_preflight import validate_split_manifest
from phase4_v2_selection import (accumulate_probe_gradients, aggregate_scorer_scores,
                                 build_selection_artifact, collision_with_random)
from phase4_v2_split import (EXPECTED_FINGERPRINT, create_v2_split,
                             load_legacy_source_mapping, reconstruct_legacy_probe_indices,
                             verify_legacy_probe_reconstruction)


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "phase4_v2_config.json").read_text())


def legacy_artifacts():
    payload, index_to_id, _ = load_legacy_source_mapping(ROOT / "phase4_split.json")
    probe = verify_legacy_probe_reconstruction((ROOT / "roberta/data.py").read_text(), payload, index_to_id)
    return payload, probe


class Phase4V2ProtocolTests(unittest.TestCase):
    def test_legacy_probe_reconstruction_and_quarantine_arithmetic(self):
        payload, probe = legacy_artifacts()
        overlap = set(probe["indices"]) & set(payload["holdout_indices"])
        union = set(probe["indices"]) | set(payload["holdout_indices"])
        self.assertEqual(len(probe["indices"]), 512)
        self.assertEqual(len(overlap), 50)
        self.assertEqual(len(union), 7197)
        self.assertEqual(len(set(probe["ids"])), 512)
        self.assertEqual(len(set(range(payload["source_size"])) - union), 60152)
        self.assertEqual(probe["indices"], reconstruct_legacy_probe_indices())

    def test_split_is_deterministic_disjoint_and_quarantined(self):
        legacy, probe = legacy_artifacts()
        labels = [index % 2 for index in range(67349)]
        ids = list(range(67349))
        first = create_v2_split(labels, ids, EXPECTED_FINGERPRINT, legacy, probe)
        second = create_v2_split(labels, ids, EXPECTED_FINGERPRINT, legacy, probe)
        self.assertEqual(first, second)
        pools = first["pools"]
        self.assertEqual({name: data["size"] for name, data in pools.items()},
                         {"train": 47640, "validation": 6000, "selection_probe": 512, "final_test": 6000})
        all_indices = []
        for data in pools.values():
            all_indices.extend(data["indices"])
        self.assertEqual(len(all_indices), len(set(all_indices)))
        quarantine = set(legacy["holdout_indices"]) | set(probe["indices"])
        self.assertFalse(quarantine & set(all_indices))
        self.assertEqual(len(all_indices), 60152)
        self.assertEqual(sum(pools[name]["class_counts"]["0"] for name in pools),
                         sum(label == 0 for label in labels) - sum(labels[i] == 0 for i in quarantine))

    def test_split_fails_on_fingerprint_and_size(self):
        legacy, probe = legacy_artifacts()
        labels = [index % 2 for index in range(67349)]
        ids = list(range(67349))
        with self.assertRaises(ValueError):
            create_v2_split(labels, ids, "wrong", legacy, probe)
        with self.assertRaises(ValueError):
            create_v2_split(labels[:-1], ids[:-1], EXPECTED_FINGERPRINT, legacy, probe)

    def test_selection_scores_and_collision_rule(self):
        scores = [{str(layer): float(layer) for layer in range(12)},
                  {str(layer): float(layer) for layer in range(12)}]
        result = aggregate_scorer_scores(scores)
        self.assertEqual(result["selected_layers"], [9, 10, 11])
        self.assertTrue(collision_with_random([11, 8, 9]))
        self.assertFalse(collision_with_random([0, 1, 2]))

    def test_manifest_requires_selection_and_freezes_nine_runs(self):
        cfg = config()
        split = {"manifest_sha256": "split-hash", "dataset_fingerprint": EXPECTED_FINGERPRINT}
        selection = build_selection_artifact(
            cfg, split,
            [{str(layer): float(layer) for layer in range(12)} for _ in range(3)],
            "probe-hash",
        )
        manifest = build_manifest(cfg, split, selection)
        self.assertEqual(manifest["run_count"], 9)
        self.assertEqual(len({run["run_id"] for run in manifest["runs"]}), 9)
        self.assertEqual({entry["expected_trainable_parameters"] for entry in manifest["configurations"]}, {665858})
        by_id = {entry["id"]: entry for entry in manifest["configurations"]}
        self.assertEqual((by_id["gradient_top3_r8"]["rank"], by_id["gradient_top3_r8"]["alpha"]), (8, 16))
        self.assertEqual((by_id["random_a_r8"]["rank"], by_id["random_a_r8"]["alpha"]), (8, 16))
        self.assertEqual((by_id["uniform_all_r2"]["rank"], by_id["uniform_all_r2"]["alpha"]), (2, 4))
        self.assertTrue(all("alpha" in run for run in manifest["runs"]))

    def test_preflight_validates_generated_split(self):
        legacy, probe = legacy_artifacts()
        labels = [index % 2 for index in range(67349)]
        ids = list(range(67349))
        split = create_v2_split(labels, ids, EXPECTED_FINGERPRINT, legacy, probe)
        validate_split_manifest(split, config())

    def test_trainable_budget_formula(self):
        classifier = 592130
        per_rank_per_layer = 3072
        self.assertEqual(classifier + 3 * 8 * per_rank_per_layer, 665858)
        self.assertEqual(classifier + 12 * 2 * per_rank_per_layer, 665858)

    def test_completion_lifecycle_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            reservation = reserve_run(directory, "example_seed0", {"seed": 0})
            paths = reservation.paths
            result = {"status": "artifact_ready", "run_id": "example_seed0",
                      "configuration": {"id": "x", "layers": [0], "rank": 8},
                      "seed": 0, "n_trainable_params": 665858,
                      "final_test_metrics": {"accuracy": 0.5, "loss": 0.7}}
            predictions = {"run_id": "example_seed0", "predictions": [0], "labels": [1]}
            write_completed_artifacts(reservation, result, predictions)
            self.assertEqual(json.loads(paths["status"].read_text())["status"], "completed")
            self.assertTrue(paths["complete"].exists())
            with self.assertRaises(CompletedRunError):
                reserve_run(directory, "example_seed0", {})
            incomplete = reserve_run(directory, "incomplete_seed0", {})
            incomplete.__exit__(RuntimeError, RuntimeError("interrupt"), None)
            with self.assertRaises(IncompleteRunError):
                reserve_run(directory, "incomplete_seed0", {})

    def test_aggregator_rejects_missing_run(self):
        manifest = {"runs": [{"run_id": "missing_seed0", "configuration_id": "x", "seed": 0,
                               "expected_trainable_parameters": 665858}]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_completed(directory, manifest)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class GradientEquivalenceTests(unittest.TestCase):
    def test_batched_gradient_equals_full_probe_mean_gradient(self):
        import torch
        import torch.nn.functional as F

        class Output:
            def __init__(self, loss):
                self.loss = loss

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query = torch.nn.Linear(3, 3, bias=False)
                self.value = torch.nn.Linear(3, 3, bias=False)
                self.classifier = torch.nn.Linear(3, 2)

            def forward(self, input_ids, labels):
                hidden = self.value(self.query(input_ids))
                return Output(F.cross_entropy(self.classifier(hidden), labels))

        torch.manual_seed(17)
        model = Tiny().eval()
        targets = [model.query.weight, model.value.weight]
        inputs = torch.randn(5, 3)
        labels = torch.tensor([0, 1, 0, 1, 1])
        full = {"input_ids": inputs, "labels": labels}
        reference = [gradient.detach().clone() for gradient in torch.autograd.grad(model(**full).loss, targets)]
        batches = [
            {"input_ids": inputs[:2], "labels": labels[:2]},
            {"input_ids": inputs[2:4], "labels": labels[2:4]},
            {"input_ids": inputs[4:], "labels": labels[4:]},
        ]
        actual = accumulate_probe_gradients(model, batches, targets, total_examples=5)
        for expected, observed in zip(reference, actual):
            torch.testing.assert_close(expected, observed, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
