# SPDX-License-Identifier: Apache-2.0
"""FP8 activations with one scale shared by a 128x128 token/channel block."""

import torch
import triton
import triton.language as tl


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _block_quant_activation(
    x,
    quant,
    scales,
    rows,
    width: tl.constexpr,
    stride_m,
    stride_k,
    scale_min: tl.constexpr,
):
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    col = tl.program_id(1) * 128 + tl.arange(0, 128)
    row = row.to(tl.int64)
    col = col.to(tl.int64)
    offsets = row[:, None] * stride_m + col[None, :] * stride_k
    values = tl.load(x + offsets, mask=row[:, None] < rows, other=0).to(tl.float32)
    peak = _maximum(tl.reduce(tl.abs(values), None, _maximum), scale_min)
    # ATen divides by a scalar via its FP32 reciprocal and by a tensor via div.rn.
    scale = peak * (1.0 / 448.0)
    fp8 = tl.div_rn(values, scale).to(tl.float8e4nv)
    output_offsets = row[:, None] * width + col[None, :]
    tl.store(quant + output_offsets, fp8, mask=row[:, None] < rows)
    scale_offsets = row * (width // 128) + tl.program_id(1)
    tl.store(scales + scale_offsets, scale, mask=row < rows)


def quantize_block_fp8_activation(x: torch.Tensor, scale_min: float):
    rows, width = x.shape
    quant = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, width // 128), device=x.device, dtype=torch.float32)
    _block_quant_activation[(triton.cdiv(rows, 128), width // 128)](
        x, quant, scales, rows, width, *x.stride(), scale_min, num_warps=8
    )
    return quant, scales
