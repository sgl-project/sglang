# SPDX-License-Identifier: Apache-2.0
"""SwiGLU (silu(gate) * up, bf16 rounding after the SiLU and after the product,
as the eager formula) fused with the per-tensor fp8 absmax, for the fp8
per-tensor GEMM path: the fc1 output is read once, the bf16 product and its
scale come out together, and a static per-tensor quant produces the fc2 fp8
input. Replaces silu, mul and the dynamic absmax pass (three extra passes over
the [rows, hidden] activation)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0


@triton.jit
def _silu_mul_absmax_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    rows,
    hidden,
    stride_row,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r < rows)[:, None] & (c < hidden)[None, :]
    base = x_ptr + r[:, None].to(tl.int64) * stride_row
    gate = tl.load(base + c[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + hidden + c[None, :], mask=mask, other=0.0).to(tl.float32)
    act = gate * tl.sigmoid(gate)
    act = act.to(tl.bfloat16).to(tl.float32)
    prod = (act * up).to(tl.bfloat16)
    tl.store(out_ptr + r[:, None].to(tl.int64) * hidden + c[None, :], prod, mask=mask)
    amax = tl.max(tl.abs(prod.to(tl.float32)))
    tl.atomic_max(scale_ptr, tl.math.div_rn(amax, 448.0))


def silu_mul_per_tensor_fp8(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden [rows, 2 * n] bf16 (row-major, gate | up) -> (fp8 [rows, n],
    scale [1] fp32 = absmax / 448), i.e. the ``(qinput, scale)`` the per-tensor
    fp8 linear accepts as a prequantized input."""
    from sglang.kernels.ops.quantization.fp8_kernel import (
        fp8_dtype,
        sgl_per_tensor_quant_fp8,
    )

    if hidden.ndim != 2 or hidden.dtype != torch.bfloat16 or hidden.stride(-1) != 1:
        raise ValueError("expected a row-major bf16 [rows, 2 * hidden] tensor")
    rows, twice = hidden.shape
    n = twice // 2
    out = torch.empty(rows, n, dtype=torch.bfloat16, device=hidden.device)
    scale = torch.full((1,), 1e-12, dtype=torch.float32, device=hidden.device)
    block_r, block_c = 16, 256
    grid = (triton.cdiv(rows, block_r), triton.cdiv(n, block_c))
    with torch.get_device_module().device(hidden.device):
        _silu_mul_absmax_kernel[grid](
            hidden,
            out,
            scale,
            rows,
            n,
            hidden.stride(0),
            BLOCK_R=block_r,
            BLOCK_C=block_c,
            num_warps=4,
        )
    q = torch.empty(rows, n, dtype=fp8_dtype, device=hidden.device)
    sgl_per_tensor_quant_fp8(out, q, scale, is_static=True)
    return q, scale


__all__ = ["silu_mul_per_tensor_fp8"]
