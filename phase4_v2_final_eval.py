"""Final-test evaluation entry point. Screening and replication do not import this module."""
from __future__ import annotations
import json
from pathlib import Path

def load_final_test_only(split_manifest, dataset):
    pool = split_manifest["pools"]["final_test"]
    return dataset.select(pool["indices"])

def evaluate_completed_checkpoint(model, checkpoint_dir, final_dataset, collator, batch_size):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    # This function accepts only an already-selected final_test dataset.
    import torch
    model.eval(); device = next(model.parameters()).device; predictions, labels = [], []
    with torch.no_grad():
        for start in range(0, len(final_dataset), batch_size):
            batch = collator([final_dataset[i] for i in range(start, min(start + batch_size, len(final_dataset)))])
            batch = {k: v.to(device) for k, v in batch.items()}; y = batch.pop("labels")
            predictions.extend(model(**batch).logits.argmax(-1).cpu().tolist()); labels.extend(y.cpu().tolist())
    return {"predictions": predictions, "labels": labels, "n": len(labels)}
