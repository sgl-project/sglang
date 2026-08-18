# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 MLA output-gate dispatch."""

import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.utils import is_flydsl_available

_HIDDEN = 7168
_OUTPUT = 1536
_ROWS_PER_BLOCK = 4
_WAVES_PER_EU = 2
_WEIGHT_CACHE_MODIFIER = 0


def _is_gfx950_flydsl_available() -> bool:
    if not is_flydsl_available():
        return False
    try:
        return get_gfx_runtime() == "gfx950"
    except (AssertionError, KeyError, RuntimeError):
        return False


def supports_kimi_k3_mla_gate(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    attention_output: torch.Tensor,
) -> bool:
    """Return whether the fixed gfx950 BF16 gate specialization owns the call."""

    tensors = (hidden, gate_weight, attention_output)
    return (
        all(tensor.is_cuda for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.dtype == torch.bfloat16 for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
        and tuple(hidden.shape) == (1, _HIDDEN)
        and tuple(gate_weight.shape) == (_OUTPUT, _HIDDEN)
        and tuple(attention_output.shape) == (1, _OUTPUT)
        and _is_gfx950_flydsl_available()
    )


@functools.cache
def _compiled_kimi_k3_mla_gate(
    rows_per_block: int,
    waves_per_eu: int,
    weight_cache_modifier: int,
):
    from .kernels.kimi_k3_mla_gate_epilogue_gfx950 import (
        build_kimi_k3_mla_gate_module,
    )

    return build_kimi_k3_mla_gate_module(
        rows_per_block,
        waves_per_eu,
        weight_cache_modifier,
    )


def _launch_kimi_k3_mla_gate(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    attention_output: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    _compiled_kimi_k3_mla_gate(
        _ROWS_PER_BLOCK,
        _WAVES_PER_EU,
        _WEIGHT_CACHE_MODIFIER,
    )(
        ptr_arg(hidden),
        ptr_arg(gate_weight),
        ptr_arg(attention_output),
        ptr_arg(output),
        stream=torch.cuda.current_stream(hidden.device),
    )
    return output


def kimi_k3_mla_gate(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    attention_output: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project and apply the BF16 Kimi-K3 full-MLA output gate."""

    if not supports_kimi_k3_mla_gate(hidden, gate_weight, attention_output):
        raise NotImplementedError(
            "kimi_k3_mla_gate requires contiguous gfx950 BF16 tensors with "
            "shapes (1, 7168), (1536, 7168), and (1, 1536)"
        )
    if out is None:
        out = torch.empty_like(attention_output)
    elif (
        out.device != hidden.device
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or tuple(out.shape) != (1, _OUTPUT)
    ):
        raise ValueError(
            "out must be contiguous BF16 shape (1, 1536) on the input device"
        )
    return _launch_kimi_k3_mla_gate(
        hidden,
        gate_weight,
        attention_output,
        out,
    )
