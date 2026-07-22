# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_quantize_kernel(
    x_ptr,
    out_ptr,
    scale_inv,
    M,
    x_row_stride,
    out_row_stride,
    N: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    m_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_idx < M
    n_idx = tl.arange(0, N)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    x_off = m_idx[:, None] * x_row_stride + n_idx[None, :]
    x = tl.load(x_ptr + x_off, mask=m_mask[:, None])

    x_fp8 = (x.to(tl.float32) * scale_inv).to(FP8_DTYPE)

    out_off = m_idx[:, None] * out_row_stride + n_idx[None, :]
    tl.store(out_ptr + out_off, x_fp8, mask=m_mask[:, None])

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _flatten_to_2d(x: torch.Tensor):
    """Flatten leading dims onto the row stride; returns (M, N, row_stride).

    Accepts contiguous tensors and last-dim slice views (e.g.
    ``kv[..., qk_nope:]``) where leading dims still pack onto a uniform row
    stride.
    """
    assert x.stride(-1) == 1, f"expected stride-1 inner dim, got stride={x.stride(-1)}"
    N = x.shape[-1]
    if x.ndim == 1:
        return 1, N, N
    M = x.numel() // N
    row_stride = x.stride(-2)
    for d in range(x.ndim - 2):
        expected = x.shape[d + 1] * x.stride(d + 1)
        if x.stride(d) != expected:
            raise ValueError(
                f"cannot flatten dim {d}: stride={x.stride(d)} but expected "
                f"shape[{d+1}]*stride[{d+1}]={expected}. Tensor shape={tuple(x.shape)}, "
                f"stride={tuple(x.stride())}."
            )
    return M, N, row_stride


def fp8_quantize(
    x: torch.Tensor,
    scale_inv: float = 1.0,
    out: Optional[torch.Tensor] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Cast a BF16/FP16 tensor to FP8 with an optional per-tensor scale.

    Computes ``out = saturate((x * scale_inv) -> fp8)`` element-wise. When
    ``scale_inv == 1.0`` the multiply is dropped at compile time (pure cast).

    Args:
        x: BF16 or FP16 tensor. Must have stride(-1) == 1; leading dims must
           pack uniformly onto the row stride (true for contiguous tensors and
           for last-dim slice views like ``kv[..., qk_nope:]``).
        scale_inv: scalar multiplier applied before the cast (i.e. ``1/scale``).
        out: optional pre-allocated FP8 output. Same shape as ``x``.
        fp8_dtype: ``torch.float8_e4m3fn`` (default) or ``torch.float8_e5m2``.
        enable_pdl: opt into Programmatic Dependent Launch (Hopper+).

    Returns:
        FP8 tensor with the same shape as ``x``.
    """
    assert x.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"fp8_quantize input must be bf16/fp16, got {x.dtype}"
    assert fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    M, N, x_row_stride = _flatten_to_2d(x)

    if out is None:
        out = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
    else:
        assert out.shape == x.shape and out.dtype == fp8_dtype
    out_M, _, out_row_stride = _flatten_to_2d(out)
    assert out_M == M

    fp8_dtype_const = tl.float8e4nv if fp8_dtype is torch.float8_e4m3fn else tl.float8e5

    if M <= 2048:
        block_m = 4
    elif M <= 16384:
        block_m = 16
    else:
        block_m = 32
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(M, block_m),)

    # launch_pdl is NVIDIA-only; the HIP backend rejects unknown kwargs.
    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}

    _fp8_quantize_kernel[grid](
        x,
        out,
        scale_inv,
        M,
        x_row_stride,
        out_row_stride,
        N=N,
        FP8_DTYPE=fp8_dtype_const,
        BLOCK_M=block_m,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kwargs,
    )
    return out
