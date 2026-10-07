# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Typed row-scaled-FP8 Kimi-K3 latent-tail entry point for gfx950."""

from __future__ import annotations

import functools
import importlib.util
import math

import torch
from aiter.jit.utils.chip_info import get_gfx_runtime

_LATENT_DIM = 3584
_HIDDEN_DIM = 7168
_FP8_MAX = 448.0
_TOKEN_BUCKETS = (1, 2, 4)


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
    prefix: torch.Tensor | None = None,
) -> bool:
    """Fail closed unless an exact MI355X TP8 token bucket is present."""

    tensors = (routed, shared, rms_weight, up_weight, up_scale)
    if prefix is not None:
        tensors = (*tensors, prefix)
    num_tokens = routed.shape[0] if routed.ndim == 2 else -1
    return (
        all(tensor.is_cuda for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.is_contiguous() for tensor in tensors)
        and routed.dtype == torch.bfloat16
        and shared.dtype == torch.bfloat16
        and rms_weight.dtype == torch.bfloat16
        and up_weight.dtype == torch.float8_e4m3fn
        and up_scale.dtype == torch.float32
        and num_tokens in _TOKEN_BUCKETS
        and tuple(routed.shape) == (num_tokens, _LATENT_DIM)
        and tuple(shared.shape) == (num_tokens, _HIDDEN_DIM)
        and tuple(rms_weight.shape) == (_LATENT_DIM,)
        and tuple(up_weight.shape) == (_HIDDEN_DIM, _LATENT_DIM)
        and tuple(up_scale.shape) == (_HIDDEN_DIM,)
        and (
            prefix is None
            or (
                prefix.dtype == torch.bfloat16
                and tuple(prefix.shape) == (num_tokens, _HIDDEN_DIM)
            )
        )
        and math.isfinite(epsilon)
        and epsilon > 0.0
        and importlib.util.find_spec("flydsl") is not None
        and get_gfx_runtime() == "gfx950"
    )


@functools.cache
def _compiled_latent_moe_tail_fp8(
    num_tokens: int, add_prefix: bool, skip_rms: bool = False
):
    from .kernels.latent_moe_tail_fp8_gfx950 import (
        build_latent_moe_tail_fp8_persistent_module,
    )

    return build_latent_moe_tail_fp8_persistent_module(
        num_tokens=num_tokens,
        add_prefix=add_prefix,
        skip_rms=skip_rms,
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
    prefix: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    skip_rms: bool = False,
) -> torch.Tensor:
    """Fuse RMSNorm, FP8-weight GEMV, BF16 materialization, and shared add."""

    if not supports_latent_moe_tail_fp8(
        routed,
        shared,
        rms_weight,
        up_weight,
        up_scale,
        epsilon,
        prefix,
    ):
        raise NotImplementedError("unsupported Kimi-K3 FP8 latent-tail contract")
    if out is None:
        out = torch.empty_like(shared)
    elif (
        out.device != routed.device
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or tuple(out.shape) != tuple(shared.shape)
    ):
        raise ValueError(
            "out must be contiguous BF16 with shared's shape on the input device"
        )

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    prefix_arg = prefix if prefix is not None else shared
    _compiled_latent_moe_tail_fp8(int(routed.shape[0]), prefix is not None, skip_rms)(
        ptr_arg(routed),
        ptr_arg(shared),
        ptr_arg(rms_weight),
        ptr_arg(up_weight),
        ptr_arg(up_scale),
        ptr_arg(prefix_arg),
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
