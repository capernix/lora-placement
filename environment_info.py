#!/usr/bin/env python3
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {installed('torch')}")
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name}; memory={props.total_memory / 2**30:.2f} GiB")
    for package in ("transformers", "peft", "datasets", "tokenizers", "accelerate"):
        print(f"{package}: {installed(package)}")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    print(f"Git commit: {commit}")


if __name__ == "__main__":
    main()
