import torch
from torch.utils.data import DataLoader


def linear_cka(x, y):
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    numerator = (x.T @ y).norm(p="fro") ** 2
    denominator = (x.T @ x).norm(p="fro") * (y.T @ y).norm(p="fro")
    return 0.0 if denominator.item() == 0 else (numerator / denominator).item()


def compute_importance(model, probe, collate_fn, layer_count=12, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    hidden_states = []
    layer_grads = {i: [] for i in range(layer_count)}
    query_value_params = []
    for i in range(layer_count):
        block = model.roberta.encoder.layer[i]
        query_value_params.append((i, "query", block.attention.self.query.weight))
        query_value_params.append((i, "value", block.attention.self.value.weight))
        block.attention.self.query.weight.requires_grad = True
        block.attention.self.value.weight.requires_grad = True

    total = len(probe)
    loader = DataLoader(probe, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            hidden_states.append(outputs.hidden_states)
        # Re-run with autograd for the task-aware pre-fine-tuning predictors.
        outputs = model(**batch, output_hidden_states=False)
        loss = outputs.loss * (len(batch["labels"]) / total)
        grads = torch.autograd.grad(loss, [p for _, _, p in query_value_params],
                                    retain_graph=False, allow_unused=False)
        for (layer, kind, _), grad in zip(query_value_params, grads):
            layer_grads[layer].append((kind, grad.detach().cpu()))

    # hidden_states is a list of batches, each containing embeddings + 12 blocks.
    cka = {}
    reference = torch.cat([batch_states[0][:, 0, :].cpu() for batch_states in hidden_states])
    for layer in range(layer_count):
        representation = torch.cat(
            [batch_states[layer + 1][:, 0, :].cpu() for batch_states in hidden_states]
        )
        cka[layer] = linear_cka(representation, reference)

    grad_norm = {}
    squared_gradient_proxy = {}
    for layer, grads in layer_grads.items():
        values = torch.cat([grad.reshape(-1) for _, grad in grads])
        grad_norm[layer] = values.norm().item()
        squared_gradient_proxy[layer] = values.square().mean().item()

    return {
        "cka_input": cka,
        "grad_norm": grad_norm,
        "squared_gradient_proxy": squared_gradient_proxy,
        "probe_size": total,
    }
