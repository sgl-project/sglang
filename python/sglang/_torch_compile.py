"""SGLang-local wrapper for import-time ``torch.compile`` decorators."""

from __future__ import annotations

import sys
from typing import Any

import torch


def _using_triton_stub() -> bool:
    triton = sys.modules.get("triton")
    return bool(getattr(triton, "__sglang_stub__", False))


def sglang_compile(*args: Any, **kwargs: Any) -> Any:
    """Compile an SGLang callable, or keep it eager with the Triton stub.

    The macOS import stub is sufficient for defining Triton kernels, but it is
    not a compiler implementation and cannot be inspected by TorchInductor.
    Limit the eager fallback to SGLang call sites instead of replacing the
    process-wide ``torch.compile`` function.
    """
    if _using_triton_stub():
        kwargs["disable"] = True
    return torch.compile(*args, **kwargs)
