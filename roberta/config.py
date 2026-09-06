from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RobertaSweepConfig:
    model_name: str = "roberta-base"
    model_revision: str = "main"
    dataset_name: str = "nyu-mll/glue"
    dataset_config: str = "sst2"
    max_length: int = 128
    num_layers: int = 12
    lora_rank: int = 8
    lora_alpha: int = 16
    target_modules: tuple[str, ...] = ("query", "value")
    learning_rate: float = 3e-4
    batch_size: int = 32
    epochs: int = 5
    probe_size: int = 512
    probe_seed: int = 42

    def as_dict(self):
        return asdict(self)
