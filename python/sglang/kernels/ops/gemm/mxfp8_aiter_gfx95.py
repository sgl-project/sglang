# SPDX-License-Identifier: Apache-2.0
"""gfx950 aiter dense route for 32x32-block ue8m0 fp8 checkpoints: aiter's MXFP8 A8W8 GEMM on the
plain [N, K] fp8 weight and its compact [N/32, K/32] ue8m0 block scales, with the activation as fp8 +
ue8m0 [M, K/32]; aiter picks the kernel config from its tuned MXFP8 tables."""

from __future__ import annotations

import importlib.util
from functools import cache
from typing import Optional

import torch
from aiter import gemm_a8w8_blockscale

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_operand
from sglang.srt.environ import envs


@cache
def _aiter_has_preshuffled_mxfp8() -> bool:
    # gemm_a8w8_blockscale forwards preshuffled group32 operands to FlyDSL since ROCm/aiter#5896
    return importlib.util.find_spec("aiter.ops.flydsl.batched_gemm_a8w8") is not None


def aiter_preshuffle_supported(n: int, k: int) -> bool:
    """Whether an [N, K] weight of the aiter route is kept (16, 16)-preshuffled."""
    return (
        envs.SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE.get()
        and n % 16 == 0
        and k % 32 == 0
        and _aiter_has_preshuffled_mxfp8()
    )


def aiter_mxfp8_blockscaled_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_ue8m0: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    weight_preshuffled: bool = False,
) -> torch.Tensor:
    """Dense linear of the aiter route. input is bf16 (plain or on the fp8 grid, where the per-32
    requant is exact), or fp8 e4m3 with input_scale ue8m0 [M, K/32]. weight_preshuffled: weight is
    aiter's (16, 16) shuffle of the [N, K] weight (aiter_preshuffle_supported)."""
    input_2d = input.view(-1, input.shape[-1])
    n = weight.shape[0]
    if input_2d.shape[0] == 0:
        out = input_2d.new_empty((0, n), dtype=torch.bfloat16)
    else:
        xq, xs = mxfp8_operand(input_2d, input_scale)
        out = gemm_a8w8_blockscale(
            xq,
            weight,
            xs,
            weight_scale_ue8m0,
            dtype=torch.bfloat16,
            isBpreshuffled=weight_preshuffled,
        )
    if bias is not None:
        out = out + bias
    return out.view(*input.shape[:-1], n)
