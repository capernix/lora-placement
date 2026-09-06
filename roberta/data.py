import random

from datasets import load_dataset


def load_sst2(tokenizer, max_length=128, probe_size=512, probe_seed=42):
    raw = load_dataset("nyu-mll/glue", "sst2")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    encoded = raw.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format("torch")

    rng = random.Random(probe_seed)
    indices = sorted(rng.sample(range(len(encoded["train"])), probe_size))
    probe = encoded["train"].select(indices)
    return encoded["train"], encoded["validation"], probe
