# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 B1 pre-route projections with row-scaled FP8 weights."""

from __future__ import annotations

import functools
import math

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.utils import is_flydsl_available

_BATCH_SIZE = 1
_HIDDEN_SIZE = 7168
_ROUTED_SIZE = 3584
_SHARED_GATE_UP_SIZE = 1536
_SHARED_INTERMEDIATE_SIZE = _SHARED_GATE_UP_SIZE // 2
_ROUTER_SIZE = 896


def is_kimi_k3_moe_preroute_fp8_available() -> bool:
    """Return whether the fixed-shape gfx950 FlyDSL kernels can be built."""

    return is_flydsl_available() and get_gfx_runtime() == "gfx950"


def _same_device(*tensors: torch.Tensor) -> bool:
    return len({tensor.device for tensor in tensors}) == 1


def supports_kimi_k3_moe_dual_projection_fp8(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
) -> bool:
    """Return whether the fixed Kimi-K3 dual-projection path is supported."""

    tensors = (
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
    )
    return (
        all(tensor.is_cuda for tensor in tensors)
        and hidden.dtype == torch.bfloat16
        and routed_weight.dtype == torch.float8_e4m3fn
        and routed_scale.dtype == torch.float32
        and shared_weight.dtype == torch.float8_e4m3fn
        and shared_scale.dtype == torch.float32
        and tuple(hidden.shape) == (_BATCH_SIZE, _HIDDEN_SIZE)
        and tuple(routed_weight.shape) == (_ROUTED_SIZE, _HIDDEN_SIZE)
        and tuple(routed_scale.shape) == (_ROUTED_SIZE,)
        and tuple(shared_weight.shape) == (_SHARED_GATE_UP_SIZE, _HIDDEN_SIZE)
        and tuple(shared_scale.shape) == (_SHARED_GATE_UP_SIZE,)
        and all(tensor.is_contiguous() for tensor in tensors)
        and _same_device(*tensors)
        and is_kimi_k3_moe_preroute_fp8_available()
    )


@functools.cache
def _dual_projection_launcher():
    from .kernels.kimi_k3_dual_projection_fp8_gfx950 import (
        build_kimi_k3_b1_dual_projection_fp8_module,
    )

    return build_kimi_k3_b1_dual_projection_fp8_module(
        rows_per_wave=2,
        cu_count=248,
        waves_per_eu=0,
        weight_cache_modifier=2,
        hidden_to_lds=True,
    )


def kimi_k3_moe_dual_projection_fp8(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one BF16 token with two row-scaled FP8 weight matrices."""

    if not supports_kimi_k3_moe_dual_projection_fp8(
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
    ):
        raise ValueError("unsupported Kimi-K3 dual-projection inputs")

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    routed_output = hidden.new_empty((_BATCH_SIZE, _ROUTED_SIZE))
    shared_output = hidden.new_empty((_BATCH_SIZE, _SHARED_GATE_UP_SIZE))
    _dual_projection_launcher()(
        ptr_arg(hidden),
        ptr_arg(routed_weight),
        ptr_arg(routed_scale),
        ptr_arg(shared_weight),
        ptr_arg(shared_scale),
        ptr_arg(routed_output),
        ptr_arg(shared_output),
        stream=torch.cuda.current_stream(hidden.device),
    )
    return routed_output, shared_output


def supports_kimi_k3_moe_tri_projection_fp8(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    """Return whether the mixed-precision tri-projection is supported."""

    tensors = (
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
        router_weight,
    )
    return (
        all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
        and hidden.dtype == torch.bfloat16
        and hidden.ndim == 2
        and hidden.shape[0] in (1, 2)
        and hidden.shape[1] == _HIDDEN_SIZE
        and routed_weight.dtype == torch.float8_e4m3fn
        and tuple(routed_weight.shape) == (_ROUTED_SIZE, _HIDDEN_SIZE)
        and routed_scale.dtype == torch.float32
        and tuple(routed_scale.shape) == (_ROUTED_SIZE,)
        and shared_weight.dtype == torch.float8_e4m3fn
        and tuple(shared_weight.shape) == (_SHARED_GATE_UP_SIZE, _HIDDEN_SIZE)
        and shared_scale.dtype == torch.float32
        and tuple(shared_scale.shape) == (_SHARED_GATE_UP_SIZE,)
        and router_weight.is_cuda
        and router_weight.device == hidden.device
        and router_weight.dtype == torch.bfloat16
        and tuple(router_weight.shape) == (_ROUTER_SIZE, _HIDDEN_SIZE)
        and router_weight.is_contiguous()
        and _same_device(*tensors)
        and is_kimi_k3_moe_preroute_fp8_available()
    )


@functools.cache
def _tri_projection_launcher(num_tokens: int):
    from .kernels.kimi_k3_tri_projection_fp8_gfx950 import (
        build_kimi_k3_b1_tri_projection_fp8_module,
    )

    return build_kimi_k3_b1_tri_projection_fp8_module(
        num_tokens=num_tokens,
        rows_per_wave=1,
        cu_count=248,
        waves_per_eu=0,
        weight_cache_modifier=2,
        hidden_to_lds=True,
    )


def kimi_k3_moe_tri_projection_fp8(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project routed, shared, and FP32 router outputs in one wide grid."""

    if not supports_kimi_k3_moe_tri_projection_fp8(
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
        router_weight,
    ):
        raise ValueError("unsupported Kimi-K3 tri-projection inputs")

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    num_tokens = int(hidden.shape[0])
    routed_output = hidden.new_empty((num_tokens, _ROUTED_SIZE))
    shared_output = hidden.new_empty((num_tokens, _SHARED_GATE_UP_SIZE))
    router_output = hidden.new_empty(
        (num_tokens, _ROUTER_SIZE),
        dtype=torch.float32,
    )
    _tri_projection_launcher(num_tokens)(
        ptr_arg(hidden),
        ptr_arg(routed_weight),
        ptr_arg(routed_scale),
        ptr_arg(shared_weight),
        ptr_arg(shared_scale),
        ptr_arg(router_weight),
        ptr_arg(routed_output),
        ptr_arg(shared_output),
        ptr_arg(router_output),
        stream=torch.cuda.current_stream(hidden.device),
    )
    return routed_output, shared_output, router_output


def supports_kimi_k3_shared_down_fp8(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> bool:
    """Return whether the fused Kimi-K3 SiTU/shared-down path is supported."""

    tensors = (gate_up, weight, weight_scale)
    return (
        gate_up.is_cuda
        and gate_up.dtype == torch.bfloat16
        and gate_up.ndim == 2
        and gate_up.shape[0] in (1, 2)
        and gate_up.shape[1] == _SHARED_GATE_UP_SIZE
        and gate_up.is_contiguous()
        and supports_kimi_k3_shared_down_fp8_weight(
            weight,
            weight_scale,
            device=gate_up.device,
        )
        and _same_device(*tensors)
    )


def supports_kimi_k3_shared_down_fp8_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> bool:
    """Return whether a shared-down weight pair has the kernel's contract."""

    tensors = (weight, weight_scale)
    return (
        all(tensor.is_cuda for tensor in tensors)
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale.dtype == torch.float32
        and tuple(weight.shape) == (_HIDDEN_SIZE, _SHARED_INTERMEDIATE_SIZE)
        and tuple(weight_scale.shape) == (_HIDDEN_SIZE,)
        and all(tensor.is_contiguous() for tensor in tensors)
        and _same_device(*tensors)
        and (device is None or weight.device == device)
        and is_kimi_k3_moe_preroute_fp8_available()
    )


@functools.cache
def _shared_down_launcher(
    num_tokens: int,
    situ_beta: float,
    situ_linear_beta: float,
):
    from .kernels.kimi_k3_shared_down_fp8_gfx950 import (
        build_kimi_k3_b1_shared_down_fp8_module,
    )

    return build_kimi_k3_b1_shared_down_fp8_module(
        num_tokens=num_tokens,
        rows_per_wave=1,
        cu_count=248,
        waves_per_eu=0,
        weight_cache_modifier=2,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )


def kimi_k3_shared_down_fp8(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SiTU and project with a row-scaled FP8 weight matrix."""

    if (
        not math.isfinite(situ_beta)
        or not math.isfinite(situ_linear_beta)
        or situ_beta <= 0.0
        or situ_linear_beta <= 0.0
    ):
        raise ValueError("SiTU beta values must be finite and positive")
    if not supports_kimi_k3_shared_down_fp8(
        gate_up,
        weight,
        weight_scale,
    ):
        raise ValueError("unsupported Kimi-K3 shared-down inputs")

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    if out is None:
        output = gate_up.new_empty((gate_up.shape[0], _HIDDEN_SIZE))
    elif (
        out.device != gate_up.device
        or out.dtype != torch.bfloat16
        or tuple(out.shape) != (gate_up.shape[0], _HIDDEN_SIZE)
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous BF16 [M,7168] on the same device"
        )
    else:
        output = out
    _shared_down_launcher(
        int(gate_up.shape[0]),
        float(situ_beta),
        float(situ_linear_beta),
    )(
        ptr_arg(gate_up),
        ptr_arg(weight),
        ptr_arg(weight_scale),
        ptr_arg(output),
        stream=torch.cuda.current_stream(gate_up.device),
    )
    return output
