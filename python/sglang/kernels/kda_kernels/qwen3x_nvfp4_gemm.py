"""Lazy dispatch gate for the KDA Qwen3.x SM120 NVFP4 GEMM."""

from __future__ import annotations

import functools
import importlib.util
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


_SUPPORTED_SHAPES = frozenset(
    (m, k, n)
    for m in (1, 2, 4, 8)
    for k, n in (
        (2560, 18432),
        (9216, 2560),
        (4096, 24576),
        (12288, 4096),
    )
).union({(9, 17408, 5120)})


@functools.cache
def _has_runtime(device: torch.device) -> bool:
    import torch

    try:
        return (
            importlib.util.find_spec("cutlass") is not None
            and importlib.util.find_spec("cuda.bindings.driver") is not None
            and torch.cuda.get_device_capability(device) == (12, 0)
        )
    except ModuleNotFoundError:
        return False


def _supports(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: Optional[torch.Tensor],
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> bool:
    import torch

    if (
        input.device.type != "cuda"
        or input.ndim != 2
        or input.dtype != torch.uint8
        or input_sf is None
        or out_dtype != torch.bfloat16
    ):
        return False

    m, packed_k = input.shape
    k = packed_k * 2
    n = int(out_features)
    if (m, k, n) not in _SUPPORTED_SHAPES:
        return False
    if input.stride() != (packed_k, 1):
        return False
    if (
        weight.dtype != torch.uint8
        or weight.shape != (packed_k, n)
        or weight.stride() != (1, packed_k)
    ):
        return False

    scale_k = k // 16
    padded_m = ((m + 127) // 128) * 128
    if (
        input_sf.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or input_sf.shape != (padded_m, scale_k)
        or input_sf.stride() != (scale_k, 1)
    ):
        return False
    if (
        weight_sf.dtype != torch.float8_e4m3fn
        or weight_sf.shape != (scale_k, n)
        or weight_sf.stride() != (1, scale_k)
    ):
        return False
    if alpha.dtype != torch.float32 or alpha.numel() != 1 or not alpha.is_contiguous():
        return False
    if any(t.device != input.device for t in (weight, input_sf, weight_sf, alpha)):
        return False
    return _has_runtime(input.device)


def try_qwen3x_nvfp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: Optional[torch.Tensor],
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor | None:
    """Run the E2E-qualified KDA GEMM, or return ``None`` for fallback."""
    if not _supports(
        input, weight, input_sf, weight_sf, alpha, out_dtype, out_features
    ):
        return None

    from sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm_sm120 import (
        _run_qwen3x_nvfp4_gemm,
    )

    assert input_sf is not None
    return _run_qwen3x_nvfp4_gemm(input, weight, input_sf, weight_sf, alpha)


__all__ = ["try_qwen3x_nvfp4_gemm"]
