# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Adapted from vLLM layers/activation.py::_swiglustep_and_mul_kernel.

The public BaseFusedOp owns validation and allocation; this leaf only launches
the FP32 activation with the source's arithmetic and final BF16 cast order.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_clamp_and_mul_kernel(
    input,
    out,
    input_stride,
    out_stride,
    width: tl.constexpr,
    limit: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < width
    gate = tl.load(input + row * input_stride + cols, mask=mask, other=0).to(tl.float32)
    up = tl.load(input + row * input_stride + cols + width, mask=mask, other=0).to(
        tl.float32
    )
    gate = tl.minimum(tl.sigmoid(gate) * gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    product = gate * up
    tl.store(out + row * out_stride + cols, product.to(out.dtype.element_ty), mask=mask)


def _silu_clamp_and_mul(
    input: torch.Tensor, out: torch.Tensor, *, limit: float
) -> None:
    rows, twice_width = input.shape
    width = twice_width // 2
    _silu_clamp_and_mul_kernel[(rows, triton.cdiv(width, 1024))](
        input,
        out,
        input.stride(0),
        out.stride(0),
        width=width,
        limit=limit,
        BLOCK_SIZE=1024,
    )
