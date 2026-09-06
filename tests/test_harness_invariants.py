import unittest

import torch

from harness.model import ToyBackbone
from harness.train_one_layer import run


class HarnessInvariantTests(unittest.TestCase):
    def test_exactly_one_lora_placement_and_only_expected_parameters_train(self):
        model = ToyBackbone()
        model.inject_lora(3)

        adapters = [
            (i, block.linear)
            for i, block in enumerate(model.blocks)
            if hasattr(block.linear, "lora_A")
        ]
        self.assertEqual([i for i, _ in adapters], [3])

        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(
            trainable_names,
            {"head.weight", "head.bias", "blocks.3.linear.lora_A", "blocks.3.linear.lora_B"},
        )

    def test_frozen_backbone_does_not_change(self):
        model = ToyBackbone()
        model.inject_lora(2)
        before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.startswith("blocks.") and ".base." in name
        }
        optimizer = torch.optim.SGD(model.trainable_parameters(), lr=0.1)
        x = torch.randn(16, 64)
        y = torch.randint(0, 2, (16,)).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
        for name, old in before.items():
            self.assertTrue(torch.equal(old, dict(model.named_parameters())[name]))

    def test_same_seed_and_config_is_reproducible(self):
        kwargs = {"layer": 1, "seed": 31415, "epochs": 3, "n_samples": 96}
        first = run(**kwargs)
        second = run(**kwargs)
        try:
            for key in ("frozen_val_acc", "lora_val_acc", "utility_delta", "train_acc", "loss_curve"):
                self.assertEqual(first[key], second[key])
        finally:
            from harness.train_one_layer import RESULTS_DIR
            (RESULTS_DIR / "layer1_seed31415.json").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
