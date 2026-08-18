# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Typed AITER entry point for Kimi-K3 KDA group-64 E4M3 projection."""

from __future__ import annotations

import functools

import torch

from .kernels.kimi_k3_kda_input_group64_gfx950 import (
    build_kimi_k3_kda_input_group64_module,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

_HIDDEN = 7168
_PADDED_OUTPUT = 6288
_LOGICAL_OUTPUT = 6284
_GROUP = 64
_GROUPS_PER_ROW = _HIDDEN // _GROUP
_FP8_MAX = 448.0
_ROWS_PER_WAVE = 2
_CU_COUNT = 256
_WEIGHT_CACHE_MODIFIER = 2


def _is_gfx950(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.version.hip:
        return False
    try:
        properties = torch.cuda.get_device_properties(device)
    except (AssertionError, RuntimeError):
        return False
    arch = getattr(properties, "gcnArchName", "")
    return str(arch).split(":", 1)[0] == "gfx950"


def supports_kimi_k3_kda_input_group64(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> bool:
    """Fail closed unless every fixed gfx950 contract is satisfied."""

    tensors = (hidden, weight, scale)
    return (
        _is_gfx950(hidden.device)
        and hidden.dtype == torch.bfloat16
        and weight.dtype == torch.float8_e4m3fn
        and scale.dtype == torch.float32
        and hidden.ndim == 2
        and hidden.shape[0] in (1, 2)
        and hidden.shape[1] == _HIDDEN
        and tuple(weight.shape) == (_LOGICAL_OUTPUT, _HIDDEN)
        and tuple(scale.shape) == (_LOGICAL_OUTPUT, _GROUPS_PER_ROW)
        and all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
    )


def quantize_kimi_k3_kda_input_group64(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepack the 6,284 checkpoint-owned rows once, outside decode."""

    if (
        not weight.is_cuda
        or weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (_PADDED_OUTPUT, _HIDDEN)
        or not weight.is_contiguous()
    ):
        raise ValueError("KDA source weight must be contiguous CUDA BF16 [6288,7168]")
    padding = weight[_LOGICAL_OUTPUT:]
    if bool(torch.count_nonzero(padding).item()):
        raise ValueError("KDA projection padding rows must be exactly zero")
    source = (
        weight[:_LOGICAL_OUTPUT]
        .float()
        .reshape(_LOGICAL_OUTPUT, _GROUPS_PER_ROW, _GROUP)
    )
    amax = source.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / _FP8_MAX, torch.ones_like(amax))
    packed = (
        (source / scale[..., None])
        .clamp(min=-_FP8_MAX, max=_FP8_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(_LOGICAL_OUTPUT, _HIDDEN)
        .contiguous()
    )
    return packed, scale.contiguous()


@functools.lru_cache(maxsize=2)
def _launcher(num_tokens: int):
    return build_kimi_k3_kda_input_group64_module(
        num_tokens=num_tokens,
        rows_per_wave=_ROWS_PER_WAVE,
        cu_count=_CU_COUNT,
        waves_per_eu=0,
        weight_cache_modifier=_WEIGHT_CACHE_MODIFIER,
        hidden_to_lds=True,
    )


def kimi_k3_kda_input_group64(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch only after the caller has passed the typed support predicate."""

    if not supports_kimi_k3_kda_input_group64(hidden, weight, scale):
        raise ValueError("unsupported Kimi-K3 KDA group64 projection contract")
    if output is None:
        output = hidden.new_empty((hidden.shape[0], _PADDED_OUTPUT))
    elif (
        output.dtype != torch.bfloat16
        or tuple(output.shape) != (hidden.shape[0], _PADDED_OUTPUT)
        or output.device != hidden.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous BF16 [M,6288] on the same device")
    launcher = _launcher(int(hidden.shape[0]))
    launcher(
        ptr_arg(hidden),
        ptr_arg(weight),
        ptr_arg(scale),
        ptr_arg(output),
        stream=torch.cuda.current_stream(hidden.device),
    )
    return output
