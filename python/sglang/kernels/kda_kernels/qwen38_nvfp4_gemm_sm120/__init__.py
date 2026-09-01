# SPDX-License-Identifier: Apache-2.0

# KDA provenance: this kernel was automatically optimized by the Humanize2
# workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
# (https://github.com/mit-han-lab/kernel-design-agents).
# Source: https://github.com/BBuf/KDA-Pilot/pull/195 @
# 516c976cee824a236679adf6eb525275a0a9a120.

from __future__ import annotations

import torch

_QWEN35_DECODE_KN = frozenset(
    {
        # Qwen3.5-4B
        (2560, 18432),
        (9216, 2560),
        # Qwen3.5-9B
        (4096, 24576),
        (12288, 4096),
    }
)
_QWEN38_DECODE_KN = frozenset(
    {
        (5120, 34816),
        (17408, 5120),
        (5120, 248320),
    }
)
_QWEN38_PREFILL_KN = frozenset(
    {
        (5120, 34816),
        (17408, 5120),
        (5120, 248320),
    }
)
_QWEN35_DECODE_M = frozenset({1, 2, 4, 8, 9})
# This is the only role that improved the full Qwen3.8-27B DSpark serving
# benchmark without changing acceptance. Qwen3.5-4B/9B ordinary-decode shapes
# are enabled only after their own SM120 multi-weight and E2E validation.
_E2E_VALIDATED_MKN = frozenset(
    (m, k, n) for m in (1, 2, 4, 8) for k, n in _QWEN35_DECODE_KN
).union(
    {
        # Qwen3.8-27B DSpark verification
        (9, 17408, 5120),
    }
)


def can_use_kda_nvfp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> bool:
    """Return whether this call matches the captured SM120 production contract."""
    if not input.is_cuda or input.ndim != 2 or input.dtype != torch.uint8:
        return False
    if out_dtype != torch.bfloat16:
        return False

    m, packed_k = input.shape
    k = packed_k * 2
    n = int(out_features)
    if not (
        (m in _QWEN35_DECODE_M and (k, n) in _QWEN35_DECODE_KN)
        or (m in (1, 9) and (k, n) in _QWEN38_DECODE_KN)
        or (m == 4369 and (k, n) in _QWEN38_PREFILL_KN)
    ):
        return False
    if input.stride() != (packed_k, 1):
        return False
    if weight.dtype != torch.uint8 or weight.shape != (packed_k, n):
        return False
    if weight.stride() != (1, packed_k):
        return False

    scale_k = k // 16
    padded_m = ((m + 127) // 128) * 128
    # FlashInfer 0.6.17 exposes the same E4M3 scale bytes as uint8, while
    # newer builds may preserve the float8_e4m3fn dtype on the view.
    if input_sf.dtype not in (torch.uint8, torch.float8_e4m3fn):
        return False
    if input_sf.shape != (padded_m, scale_k) or input_sf.stride() != (scale_k, 1):
        return False
    if weight_sf.dtype != torch.float8_e4m3fn:
        return False
    if weight_sf.shape != (scale_k, n) or weight_sf.stride() != (1, scale_k):
        return False
    if alpha.dtype != torch.float32 or alpha.numel() != 1 or not alpha.is_contiguous():
        return False

    tensors = (weight, input_sf, weight_sf, alpha)
    if any(t.device != input.device for t in tensors):
        return False
    props = torch.cuda.get_device_properties(input.device)
    return (props.major, props.minor) == (12, 0)


def can_dispatch_kda_nvfp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> bool:
    """Return whether this call belongs to the E2E-validated serving fast path."""
    if input.ndim != 2:
        return False
    m, packed_k = input.shape
    if (m, packed_k * 2, int(out_features)) not in _E2E_VALIDATED_MKN:
        return False
    return can_use_kda_nvfp4_gemm(
        input, weight, input_sf, weight_sf, alpha, out_dtype, out_features
    )


def kda_nvfp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    """Run the KDA-generated Qwen3.x ModelOpt NVFP4 GEMM."""
    if not can_use_kda_nvfp4_gemm(
        input, weight, input_sf, weight_sf, alpha, out_dtype, out_features
    ):
        raise ValueError("unsupported call for the KDA Qwen3.x NVFP4 GEMM")

    from .gemm import decode_fp4_gemm, large_fp4_gemm

    if input.shape[0] <= 9:
        return decode_fp4_gemm(input, weight, input_sf, weight_sf, alpha)
    return large_fp4_gemm(input, weight, input_sf, weight_sf, alpha, out_dtype)


__all__ = [
    "can_dispatch_kda_nvfp4_gemm",
    "can_use_kda_nvfp4_gemm",
    "kda_nvfp4_gemm",
]
