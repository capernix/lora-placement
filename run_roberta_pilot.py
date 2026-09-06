import argparse
import json

from roberta.train import run_pilot


def main():
    parser = argparse.ArgumentParser(description="Run one locked RoBERTa pilot configuration")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    record = run_pilot(layer=args.layer, seed=args.seed)
    print(json.dumps({
        "reported_layer": record["reported_layer"],
        "seed": record["seed"],
        "resolved_model_revision": record["resolved_model_revision"],
        "n_trainable_params": record["n_trainable_params"],
        "frozen_val_acc": record["frozen_val_acc"],
        "lora_val_acc": record["lora_val_acc"],
        "utility_delta": record["utility_delta"],
        "backbone_unchanged": record["backbone_unchanged"],
        "predictors": record["predictors"],
    }, indent=2))


if __name__ == "__main__":
    main()
