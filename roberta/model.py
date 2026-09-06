from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from .config import RobertaSweepConfig


def load_tokenizer(config: RobertaSweepConfig):
    return AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)


def load_model(config: RobertaSweepConfig, seed: int):
    # The seed controls the newly initialized classifier head.
    import torch
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=2,
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def inject_single_layer_lora(model, config: RobertaSweepConfig, layer: int):
    if not 0 <= layer < config.num_layers:
        raise ValueError(f"layer must be in [0, {config.num_layers - 1}], got {layer}")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        layers_to_transform=[layer],
        layers_pattern=["layer"],
        modules_to_save=["classifier"],
    )
    adapted = get_peft_model(model, lora_config)
    return adapted


def trainable_parameters(model):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad]


def register_frozen_prefix_detach(model, layer: int):
    """Detach the input to a selected block whose prefix is frozen.

    The hook removes only unnecessary backward traversal through blocks before
    ``layer``; it leaves the selected block, suffix, and classifier gradients
    unchanged.
    """
    block = model.base_model.model.roberta.encoder.layer[layer]

    def detach_prefix(_module, inputs):
        return (inputs[0].detach(), *inputs[1:])

    return block.register_forward_pre_hook(detach_prefix)


def resolved_model_revision(model):
    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        raise RuntimeError("Hugging Face did not expose a resolved model commit hash")
    return revision
