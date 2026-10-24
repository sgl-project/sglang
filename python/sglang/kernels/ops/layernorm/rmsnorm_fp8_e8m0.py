"""Fused RMSNorm + per-token 1x128 fp8 quant with power-of-two (ue8m0) scales.

The e8m0 counterpart of aiter's ``fused_rms_fp8_group_quant(...,
output_unquantized_inp1=True, transpose_scale=True)``: one launch emits the bf16
normed row (for the readers that want it unquantized) and the fp8 operand of an
e8m0 mxscale GEMM. The scale is ``ceil_pow2(amax / 448)`` (aiter's default
RoundUp / DeepSeek-V4 reference ``act_quant(scale_fmt="ue8m0")``), stored as one
e8m0 byte per group in column-major order: a contiguous ``[K/128, M]`` buffer
viewed as ``[M, K/128]``, the layout ``per_group_quant_hip(transpose_scale=True,
scale_type=fp8_e8m0)`` emits and aiter's mxscale bpreshuffle GEMM reads.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0


@triton.jit
def _rmsnorm_fp8_e8m0_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    M,
    N,
    x_stride,
    y_stride,
    q_stride,
    eps,
    BLOCK_N: tl.constexpr,
    GROUP: tl.constexpr,
    NUM_G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd * w).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * y_stride + offs, y, mask=mask)

    # Quantize the rounded bf16 row, as a separate quant of the norm output would.
    yg = tl.reshape(y.to(tl.float32), (NUM_G, GROUP))
    amax = tl.max(tl.abs(yg), axis=1)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    e = (bits >> 23) & 0xFF
    e = tl.where(((bits & 0x7FFFFF) != 0) & (e < 0xFF), e + 1, e)
    inv = tl.where(e > 0, 1.0 / (e << 23).to(tl.float32, bitcast=True), 0.0)
    q = tl.clamp(yg * inv[:, None], -448.0, 448.0).to(q_ptr.dtype.element_ty)
    q_offs = tl.reshape(offs, (NUM_G, GROUP))
    tl.store(
        q_ptr + row * q_stride + q_offs,
        q,
        mask=tl.reshape(mask, (NUM_G, GROUP)),
    )
    g = tl.arange(0, NUM_G)
    tl.store(s_ptr + g * M + row, e.to(tl.uint8), mask=g * GROUP < N)


def rmsnorm_fp8_e8m0_quant(
    x: torch.Tensor, weight: torch.Tensor, eps: float, group_size: int = 128
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Returns ``((q_fp8 [M, N], scale_e8m0 [M, N/128] column-major bytes), y [M, N])``."""
    assert x.dim() == 2 and x.stride(-1) == 1
    m, n = x.shape
    assert n % group_size == 0
    num_g = n // group_size
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((num_g, m), dtype=torch.float8_e8m0fnu, device=x.device)
    if m > 0:
        num_g_pow2 = triton.next_power_of_2(num_g)
        block_n = num_g_pow2 * group_size
        num_warps = 4 if block_n <= 2048 else (8 if block_n <= 4096 else 16)
        _rmsnorm_fp8_e8m0_kernel[(m,)](
            x,
            weight,
            y,
            q,
            s.view(torch.uint8),
            m,
            n,
            x.stride(0),
            y.stride(0),
            q.stride(0),
            eps,
            BLOCK_N=block_n,
            GROUP=group_size,
            NUM_G=num_g_pow2,
            num_warps=num_warps,
        )
    return (q, s.view(m, num_g)), y
