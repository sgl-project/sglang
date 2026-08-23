"""Shared helpers for fail-closed Kimi-K3 AITER batch-1 fusions."""

from __future__ import annotations

import torch

_FP8_MAX = 448.0


def quantize_fp8_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP32 [out, in] weight with one FP32 scale per row."""
    if weight.ndim != 2 or not weight.is_cuda or not weight.is_contiguous():
        raise ValueError("expected a contiguous CUDA [out, in] weight")
    row_amax = weight.float().abs().amax(dim=1)
    scale = (row_amax / _FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (
        (weight.float() / scale[:, None])
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
    return quantized.contiguous(), scale.contiguous()
