"""Verify private-delivery packages without initializing the EIC client.

The EIC SDK aborts during import on a build runner without its runtime
environment. Real client initialization and read/write checks belong in a
configured task Pod; image CI only proves that the SDK module is installed.
"""

import importlib.metadata
import importlib.util

for distribution in ("sglang", "sgl-kernel", "eic", "onion-ai-data", "deep-ep"):
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed-as-distribution"
    print(f"{distribution}={version}")

eic_spec = importlib.util.find_spec("eic")
if eic_spec is None:
    raise SystemExit("EIC SDK module is not installed")
print(f"eic-module={eic_spec.origin}")

import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda}")

import sglang

print(f"sglang-module={sglang.__file__}")
