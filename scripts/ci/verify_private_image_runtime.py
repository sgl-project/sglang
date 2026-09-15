"""Verify packages required by the private-delivery image."""

import importlib.metadata

for distribution in ("sglang", "sgl-kernel", "onion-ai-data", "deep-ep"):
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed-as-distribution"
    print(f"{distribution}={version}")

import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda}")

import sglang

print(f"sglang-module={sglang.__file__}")

import eic

print(f"eic-module={eic.__file__}")
