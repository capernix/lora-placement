import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification

from roberta.io import is_complete_result, result_path, write_json_atomic
from roberta.model import inject_single_layer_lora, register_frozen_prefix_detach, trainable_parameters
from roberta.sweep import partition_jobs, pending_jobs, requested_jobs
from roberta.config import RobertaSweepConfig


def tiny_config():
    return RobertaConfig(
        vocab_size=101, max_position_embeddings=32, num_hidden_layers=12,
        hidden_size=32, intermediate_size=64, num_attention_heads=4,
        num_labels=2, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )


class RobertaInfrastructureTests(unittest.TestCase):
    def test_frozen_prefix_detach_matches_reference_for_edge_and_middle_layers(self):
        torch.manual_seed(7)
        inputs = {
            "input_ids": torch.tensor([[1, 4, 5, 2, 0], [1, 8, 3, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]),
            "labels": torch.tensor([0, 1]),
        }
        for layer in (0, 5, 11):
            reference = RobertaForSequenceClassification(tiny_config())
            optimized = RobertaForSequenceClassification(tiny_config())
            optimized.load_state_dict(copy.deepcopy(reference.state_dict()))
            torch.manual_seed(99)
            reference = inject_single_layer_lora(reference, RobertaSweepConfig(), layer)
            torch.manual_seed(99)
            optimized = inject_single_layer_lora(optimized, RobertaSweepConfig(), layer)
            hook = register_frozen_prefix_detach(optimized, layer)
            reference.train()
            optimized.train()
            reference_out = reference(**inputs)
            optimized_out = optimized(**inputs)
            reference_out.loss.backward()
            optimized_out.loss.backward()
            torch.testing.assert_close(optimized_out.loss, reference_out.loss, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(optimized_out.logits, reference_out.logits, rtol=1e-5, atol=1e-6)
            reference_grads = dict(reference.named_parameters())
            optimized_grads = dict(optimized.named_parameters())
            for name, parameter in reference_grads.items():
                if parameter.requires_grad:
                    self.assertIsNotNone(optimized_grads[name].grad, name)
                    torch.testing.assert_close(optimized_grads[name].grad, parameter.grad, rtol=1e-4, atol=1e-5)
            hook.remove()

    def test_lora_scope_and_trainable_count(self):
        model = inject_single_layer_lora(RobertaForSequenceClassification(tiny_config()), RobertaSweepConfig(), 5)
        names = [name for name, parameter in trainable_parameters(model)]
        self.assertTrue(any("encoder.layer.5" in name and "lora_A" in name for name in names))
        self.assertTrue(any("encoder.layer.5" in name and "lora_B" in name for name in names))
        self.assertTrue(any(name.startswith("base_model.model.classifier") for name in names))
        self.assertTrue(all("encoder.layer.5" in name or "classifier" in name for name in names))
        self.assertTrue(all(not parameter.requires_grad for name, parameter in model.named_parameters()
                            if "roberta.embeddings" in name or ("roberta.encoder.layer" in name and "encoder.layer.5" not in name)))

    def test_result_completion_and_pending_detection(self):
        fields = {
            "status": "completed", "model": "roberta-base", "requested_revision": "main",
            "resolved_model_revision": "abc", "dataset": "nyu-mll/glue/sst2", "layer": 0,
            "reported_layer": "L1", "seed": 0, "max_length": 128, "lora_rank": 8,
            "lora_alpha": 16, "target_modules": ["query", "value"], "learning_rate": 3e-4,
            "batch_size": 32, "epochs": 5, "probe_size": 512, "probe_seed": 42,
            "frozen_val_acc": 0.8, "lora_val_acc": 0.85, "utility_delta": 0.05,
            "n_trainable_params": 1, "predictors": {}, "training_time_sec": 1,
            "gpu_peak_memory_allocated_mb": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = result_path(directory, 0, 0)
            write_json_atomic(path, fields)
            self.assertTrue(is_complete_result(path))
            self.assertEqual(pending_jobs(directory, [(0, 0), (0, 1)]), [(0, 1)])
            path.write_text("{")
            self.assertFalse(is_complete_result(path))
            fields.pop("status")
            write_json_atomic(path, fields)
            self.assertFalse(is_complete_result(path))

    def test_partitioning_is_complete_and_disjoint(self):
        jobs = requested_jobs(range(12), (0, 1, 2))
        buckets = partition_jobs(jobs, 2)
        flattened = [job for bucket in buckets for job in bucket]
        self.assertEqual(sorted(flattened), sorted(jobs))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertLessEqual(abs(len(buckets[0]) - len(buckets[1])), 1)


if __name__ == "__main__":
    unittest.main()
