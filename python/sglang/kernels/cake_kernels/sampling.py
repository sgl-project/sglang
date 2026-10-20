"""Blackwell Cake softmax through FlashInfer's public API."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@lru_cache(maxsize=None)
def _device_supported(device_index: int) -> bool:
    import torch

    return (
        torch.cuda.get_device_capability(device_index) == (10, 3)
        and find_spec("flashinfer.jit.cake_blackwell_softmax") is not None
    )


def supports_softmax(logits: torch.Tensor) -> bool:
    """The large-vocabulary domain qualified against the eager Torch caller."""
    import torch

    return (
        logits.is_cuda
        and torch.version.cuda is not None
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and 1 <= logits.shape[0] <= 64
        and 128256 <= logits.shape[1] <= 262144
        and logits.shape[1] % 8 == 0
        and logits.is_contiguous()
        and _device_supported(logits.device.index)
    )


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """Return softmax probabilities through FlashInfer's public API.

    FlashInfer owns the architecture-specific Cake dispatch and its fallback
    for rows outside the optimized shape domain. No sampling distribution or
    random-number-generator behavior is changed by this operation.
    """
    from flashinfer.sampling import softmax as flashinfer_softmax

    return flashinfer_softmax(logits)
