"""Backward-compatible import shim.

Use `petit_nvfp4.py` for implementation. Keep this module for existing imports.
"""

from sglang.srt.layers.quantization.petit_nvfp4 import (  # noqa: F401
    PetitNvFp4Config,
    PetitNvFp4LinearMethod,
)

__all__ = ["PetitNvFp4Config", "PetitNvFp4LinearMethod"]
