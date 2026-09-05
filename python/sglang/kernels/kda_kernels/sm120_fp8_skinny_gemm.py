"""Lazy dispatch gate for the KDA SM120 FP8 skinny GEMM."""

from __future__ import annotations

import functools
import importlib.util
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


_SUPPORTED_M = (1, 2, 4, 8, 9)

# Exact (K, N) projection shapes are enabled only at M values that passed
# BF16 numerical comparison, cold-L2 benchmarks against the available SGLang fast
# paths, and model-level E2E validation on RTX PRO 6000 Blackwell.
_QUALIFIED_M_BY_PROJECTION = {
    (5120, 8192): _SUPPORTED_M,
    (5120, 16384): (2, 4, 8),
    (6144, 5120): _SUPPORTED_M,
    (4096, 4096): _SUPPORTED_M,
    (5120, 7168): _SUPPORTED_M,
    (5120, 5120): (2, 4, 8, 9),
    (5120, 34816): _SUPPORTED_M,
    (4096, 5376): (1,),
}
_PRODUCTION_SHAPES = frozenset(
    (m, k, n)
    for (k, n), supported_m in _QUALIFIED_M_BY_PROJECTION.items()
    for m in supported_m
)


@functools.cache
def _has_runtime(device: torch.device) -> bool:
    import torch

    try:
        return importlib.util.find_spec(
            "flashinfer"
        ) is not None and torch.cuda.get_device_capability(device) == (12, 0)
    except ModuleNotFoundError:
        return False


def _supports(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    output_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> bool:
    import torch

    if (
        input.device.type != "cuda"
        or input.ndim != 2
        or input.dtype != torch.bfloat16
        or not input.is_contiguous()
        or input_scale is None
        or output_scale is None
        or bias is not None
    ):
        return False

    m, k = input.shape
    if weight.ndim != 2:
        return False
    n = weight.shape[1]
    if (m, k, n) not in _PRODUCTION_SHAPES:
        return False
    if (
        weight.dtype != torch.float8_e4m3fn
        or weight.shape[0] != k
        or weight.stride() != (1, k)
    ):
        return False
    if (
        input_scale.dtype != torch.float32
        or input_scale.numel() != 1
        or not input_scale.is_contiguous()
        or output_scale.dtype != torch.float32
        or output_scale.numel() != 1
        or not output_scale.is_contiguous()
    ):
        return False
    if any(t.device != input.device for t in (weight, input_scale, output_scale)):
        return False
    return _has_runtime(input.device)


def try_sm120_fp8_skinny_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    output_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor | None:
    """Run the E2E-qualified KDA path, or return ``None`` for fallback."""
    if not _supports(
        input,
        weight,
        input_scale,
        output_scale,
        bias,
    ):
        return None

    assert input_scale is not None
    assert output_scale is not None
    from sglang.kernels.kda_kernels.sm120_fp8_skinny_gemm_sm120 import (
        _run_sm120_fp8_skinny_gemm,
    )

    return _run_sm120_fp8_skinny_gemm(input, weight, input_scale, output_scale)


__all__ = ["try_sm120_fp8_skinny_gemm"]
