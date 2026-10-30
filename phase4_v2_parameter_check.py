"""Optional exact PEFT parameter verification; local-files-only by design."""
from __future__ import annotations

def actual_parameter_count(model_name, revision, layers, rank, alpha, target_modules=("query", "value")):
    from transformers import AutoModelForSequenceClassification
    from peft import LoraConfig, TaskType, get_peft_model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision, num_labels=2, local_files_only=True)
    for parameter in model.parameters(): parameter.requires_grad = False
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha, target_modules=list(target_modules), layers_to_transform=list(layers), layers_pattern=["layer"], modules_to_save=["classifier"])
    adapted = get_peft_model(model, lora)
    return sum(p.numel() for p in adapted.parameters() if p.requires_grad)

def verify_expected_count(*args, expected=665858, **kwargs):
    count = actual_parameter_count(*args, **kwargs)
    if count != expected: raise ValueError(f"actual instantiated PEFT count {count}, expected {expected}")
    return count
