"""
Shared model definitions for the LoRA-placement dummy-layer harness.

This is NOT RoBERTa/CLIP -- it's a small synthetic stand-in with the same
shape of problem (N sequential "layers", LoRA injected into exactly one
layer at a time, frozen backbone otherwise). Its job is to shake out
harness bugs before Week 2's real sweep, per the project plan.
"""
import torch
import torch.nn as nn

INPUT_DIM = 64
HIDDEN_DIM = 64
NUM_LAYERS = 6
LORA_RANK = 4
LORA_ALPHA = 16
LABEL_SOURCE_LAYER = 4  # which block's representation the synthetic label depends on


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank (LoRA) update.

    y = W0 x + (alpha / r) * B A x
    W0 stays frozen; only A and B are trained.
    """

    def __init__(self, base: nn.Linear, rank: int = LORA_RANK, alpha: int = LORA_ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def lora_parameters(self):
        return [self.lora_A, self.lora_B]


class Block(nn.Module):
    """One backbone 'layer': Linear -> Tanh. Stands in for a transformer block."""

    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.linear(x))


class ToyBackbone(nn.Module):
    """Frozen random backbone of NUM_LAYERS blocks, with a trainable task head.

    A LoRA adapter can be injected into exactly one block via `inject_lora`.
    """

    def __init__(self, num_layers=NUM_LAYERS, dim=HIDDEN_DIM, backbone_seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(backbone_seed)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(num_layers)])
        # deterministic backbone init, independent of the training seed
        with torch.no_grad():
            for blk in self.blocks:
                blk.linear.weight.copy_(torch.randn(blk.linear.weight.shape, generator=g) * 0.7)
                blk.linear.bias.zero_()
        for p in self.parameters():
            p.requires_grad = False

        self.head = nn.Linear(dim, 1)
        # The backbone is frozen, but the task head is trained for every run.
        for p in self.head.parameters():
            p.requires_grad = True
        self.num_layers = num_layers
        self.lora_layer_idx = None

    def inject_lora(self, layer_idx: int, rank: int = LORA_RANK, alpha: int = LORA_ALPHA):
        block = self.blocks[layer_idx]
        block.linear = LoRALinear(block.linear, rank=rank, alpha=alpha)
        self.lora_layer_idx = layer_idx
        return block.linear.lora_parameters()

    def trainable_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        # Keep the helper defensive: only the selected adapter may be trainable.
        if self.lora_layer_idx is not None:
            selected = set(self.blocks[self.lora_layer_idx].linear.lora_parameters())
            params = [p for p in params if p in selected or p in set(self.head.parameters())]
        return params

    def forward(self, x, return_activations=False):
        acts = [x]
        h = x
        for blk in self.blocks:
            h = blk(h)
            acts.append(h)
        logits = self.head(h).squeeze(-1)
        if return_activations:
            return logits, acts
        return logits
