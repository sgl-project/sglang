# SPDX-License-Identifier: Apache-2.0
"""gfx950 aiter dense route for 32x32-block ue8m0 fp8 checkpoints: aiter's MXFP8 A8W8 GEMM on the
plain [N, K] fp8 weight and its compact [N/32, K/32] ue8m0 block scales, with the activation as fp8 +
ue8m0 [M, K/32]; aiter picks the kernel config from its tuned MXFP8 tables."""

from __future__ import annotations

from typing import Optional

import torch
from aiter import gemm_a8w8_blockscale

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize


def aiter_mxfp8_blockscaled_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_ue8m0: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense linear of the aiter route. input is bf16 (plain or on the fp8 grid, where the per-32
    requant is exact), or fp8 e4m3 with input_scale ue8m0 [M, K/32]."""
    input_2d = input.view(-1, input.shape[-1])
    n = weight.shape[0]
    if input_2d.shape[0] == 0:
        out = input_2d.new_empty((0, n), dtype=torch.bfloat16)
    else:
        if input_scale is None:
            xq, xs = mxfp8_e4m3_quantize(input_2d.to(torch.bfloat16))
        else:
            assert input_2d.dtype == torch.float8_e4m3fn, input_2d.dtype
            xq, xs = input_2d.contiguous(), input_scale.contiguous()
        out = gemm_a8w8_blockscale(
            xq, weight, xs, weight_scale_ue8m0, dtype=torch.bfloat16
        )
    if bias is not None:
        out = out + bias
    return out.view(*input.shape[:-1], n)
