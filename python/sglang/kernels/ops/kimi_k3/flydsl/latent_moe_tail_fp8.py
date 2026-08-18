# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Typed row-scaled-FP8 Kimi-K3 latent-tail entry point for gfx950."""

from __future__ import annotations

import functools
import math

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.utils import is_flydsl_available

_LATENT_DIM = 3584
_HIDDEN_DIM = 7168
_FP8_MAX = 448.0


def quantize_latent_moe_tail_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one contiguous BF16 up-projection to row-scaled OCP E4M3."""

    if (
        not weight.is_cuda
        or weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (_HIDDEN_DIM, _LATENT_DIM)
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "latent-tail source weight must be contiguous CUDA BF16 [7168,3584]"
        )
    weight_f32 = weight.float()
    amax = weight_f32.abs().amax(dim=1)
    scale = torch.where(
        amax > 0,
        amax / _FP8_MAX,
        torch.ones_like(amax),
    )
    packed = (
        (weight_f32 / scale[:, None])
        .clamp(min=-_FP8_MAX, max=_FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return packed, scale.contiguous()


def supports_latent_moe_tail_fp8(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    epsilon: float,
) -> bool:
    """Fail closed unless the exact MI355X TP8 B1 contract is present."""

    tensors = (routed, shared, rms_weight, up_weight, up_scale)
    return (
        all(tensor.is_cuda for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.is_contiguous() for tensor in tensors)
        and routed.dtype == torch.bfloat16
        and shared.dtype == torch.bfloat16
        and rms_weight.dtype == torch.bfloat16
        and up_weight.dtype == torch.float8_e4m3fn
        and up_scale.dtype == torch.float32
        and tuple(routed.shape) == (1, _LATENT_DIM)
        and tuple(shared.shape) == (1, _HIDDEN_DIM)
        and tuple(rms_weight.shape) == (_LATENT_DIM,)
        and tuple(up_weight.shape) == (_HIDDEN_DIM, _LATENT_DIM)
        and tuple(up_scale.shape) == (_HIDDEN_DIM,)
        and math.isfinite(epsilon)
        and epsilon > 0.0
        and is_flydsl_available()
        and get_gfx_runtime() == "gfx950"
    )


@functools.cache
def _compiled_latent_moe_tail_fp8():
    from .kernels.latent_moe_tail_fp8_gfx950 import (
        build_b1_latent_moe_tail_fp8_persistent_module,
    )

    return build_b1_latent_moe_tail_fp8_persistent_module(
        rows_per_wave=1,
        cu_count=240,
        waves_per_eu=2,
        weight_cache_modifier=2,
    )


def latent_moe_tail_fp8(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    epsilon: float,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse RMSNorm, FP8-weight GEMV, BF16 materialization, and shared add."""

    if not supports_latent_moe_tail_fp8(
        routed,
        shared,
        rms_weight,
        up_weight,
        up_scale,
        epsilon,
    ):
        raise NotImplementedError("unsupported Kimi-K3 FP8 latent-tail contract")
    if out is None:
        out = torch.empty_like(shared)
    elif (
        out.device != routed.device
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or tuple(out.shape) != (1, _HIDDEN_DIM)
    ):
        raise ValueError(
            "out must be contiguous BF16 shape (1,7168) on the input device"
        )

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    _compiled_latent_moe_tail_fp8()(
        ptr_arg(routed),
        ptr_arg(shared),
        ptr_arg(rms_weight),
        ptr_arg(up_weight),
        ptr_arg(up_scale),
        ptr_arg(out),
        float(epsilon),
        stream=torch.cuda.current_stream(routed.device),
    )
    return out


__all__ = [
    "latent_moe_tail_fp8",
    "quantize_latent_moe_tail_weight",
    "supports_latent_moe_tail_fp8",
]
