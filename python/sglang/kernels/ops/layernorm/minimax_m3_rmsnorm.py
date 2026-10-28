# SPDX-License-Identifier: Apache-2.0
"""Fused Gemma RMSNorm Triton kernels for MiniMax-M3 on AMD ROCm.

Gemma RMSNorm = ``normalize(x) * (1 + weight)``, computed in a single fp32 pass.
On ROCm with AITER, ``GemmaRMSNorm.forward_hip`` otherwise falls back to a
~8-op PyTorch sequence: ``sgl_kernel``'s Gemma kernels are CUDA-only, and
AITER's ``rmsnorm2d_fwd`` requires weight.dtype == activation.dtype (fp32
weight + bf16 activation silently corrupts on gfx950). These kernels read
strided inputs, so they serve both the full-hidden norms and the per-head
q/k/index norms (non-contiguous ``qkv.split`` views).
"""

import torch
import triton
import triton.language as tl


# Gemma (1 + weight) RMSNorm over strided rows; no other Triton RMSNorm covers that.
@triton.jit
def _gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_cols,
    stride_row,
    stride_col,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_row + cols * stride_col, mask=mask, other=0.0).to(
        tl.float32
    )
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = x * rstd * (1.0 + w)
    tl.store(
        out_ptr + row * n_cols + cols,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _gemma_fused_add_rmsnorm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    n_cols,
    stride_xrow,
    stride_xcol,
    stride_rrow,
    stride_rcol,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(
        x_ptr + row * stride_xrow + cols * stride_xcol, mask=mask, other=0.0
    ).to(tl.float32)
    r = tl.load(
        res_ptr + row * stride_rrow + cols * stride_rcol, mask=mask, other=0.0
    ).to(tl.float32)
    s = x + r
    # residual_out is the pre-norm sum (consumed by the next layer's add).
    tl.store(
        res_out_ptr + row * n_cols + cols,
        s.to(res_out_ptr.dtype.element_ty),
        mask=mask,
    )
    var = tl.sum(s * s, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = s * rstd * (1.0 + w)
    tl.store(
        out_ptr + row * n_cols + cols,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _gemma_fused_add_rmsnorm_mxfp4_quant_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    q_ptr,
    scale_ptr,
    res_out_ptr,
    n_cols,
    stride_xrow,
    stride_xcol,
    stride_rrow,
    stride_rcol,
    scale_n_pad,
    eps,
    BLOCK_N: tl.constexpr,
):
    # The add + norm below is _gemma_fused_add_rmsnorm_kernel's, statement for
    # statement (same layout, same reduction), so the bf16 output it quantizes
    # is bit-identical to that kernel's. Keep the two in sync.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(
        x_ptr + row * stride_xrow + cols * stride_xcol, mask=mask, other=0.0
    ).to(tl.float32)
    r = tl.load(
        res_ptr + row * stride_rrow + cols * stride_rcol, mask=mask, other=0.0
    ).to(tl.float32)
    s = x + r
    tl.store(
        res_out_ptr + row * n_cols + cols,
        s.to(res_out_ptr.dtype.element_ty),
        mask=mask,
    )
    var = tl.sum(s * s, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = s * rstd * (1.0 + w)
    # MXFP4 of the norm output as written in the activation dtype, the way
    # AITER per_1x32_f4_quant quantizes it: E8M0 scale ceil_pow2(amax / 6)
    # (RoundUp) per 32 values with amax floored at 1e-10, E2M1 round to
    # nearest even.
    y = out.to(res_out_ptr.dtype.element_ty).to(tl.float32)
    NB: tl.constexpr = BLOCK_N // 32
    yb = tl.reshape(y, (NB, 32))
    amax = tl.maximum(tl.max(tl.abs(yb), axis=1, keep_dims=True), 1e-10)
    u = (amax * (1.0 / 6.0)).to(tl.int32, bitcast=True)
    e = (u >> 23) & 0xFF
    e = tl.where((e < 0xFF) & ((u & 0x7FFFFF) != 0), e + 1, e)
    e = tl.minimum(e, 254)
    qx = yb * tl.exp2((127 - e).to(tl.float32))
    # gfx950 conversion, two values per instruction; four bytes (one thread's
    # eight values) per 32-bit store.
    lo, hi = tl.split(tl.reshape(qx, (NB, 16, 2)))
    byts = (
        tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            args=[lo, hi, 1.0],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        & 0xFF
    )
    shifts = tl.arange(0, 4) * 8
    words = tl.sum(tl.reshape(byts, (NB, 4, 4)) << shifts[None, None, :], axis=2)
    words = tl.reshape(words, (NB * 4,))
    wcols = tl.arange(0, NB * 4)
    tl.store(q_ptr + row * (n_cols // 8) + wcols, words, mask=wcols < n_cols // 8)
    # E8M0 scales in the tile order of the AITER ASM FP4 GEMM (as
    # per_1x32_f4_quant(shuffle=True) writes them).
    c = tl.arange(0, NB)
    sidx = (
        (row % 32) // 16
        + ((c % 8) // 4) * 2
        + (row % 16) * 4
        + (c % 4) * 64
        + (c // 8) * 256
        + (row // 32) * 32 * scale_n_pad
    )
    tl.store(scale_ptr + sidx, tl.reshape(e, (NB,)).to(tl.uint8), mask=c < n_cols // 32)


def _num_warps(block_n: int) -> int:
    if block_n >= 4096:
        return 16
    if block_n >= 1024:
        return 8
    return 4


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma RMSNorm = normalize(x) * (1 + weight), fp32 math, single pass."""
    orig_shape = x.shape
    n = orig_shape[-1]
    x2 = x.reshape(-1, n)
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(n)
    _gemma_rmsnorm_kernel[(m,)](
        x2,
        weight,
        out,
        n,
        x2.stride(0),
        x2.stride(1),
        eps,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
    )
    return out.reshape(orig_shape)


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    """Fused (x + residual) then Gemma RMSNorm; returns (normed, pre-norm sum)."""
    orig_shape = x.shape
    n = orig_shape[-1]
    x2 = x.reshape(-1, n)
    r2 = residual.reshape(-1, n)
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    res_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(n)
    _gemma_fused_add_rmsnorm_kernel[(m,)](
        x2,
        r2,
        weight,
        out,
        res_out,
        n,
        x2.stride(0),
        x2.stride(1),
        r2.stride(0),
        r2.stride(1),
        eps,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
    )
    return out.reshape(orig_shape), res_out.reshape(orig_shape)


def gemma_fused_add_rmsnorm_mxfp4_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    """gemma_fused_add_rmsnorm with the normed output MXFP4-quantized (gfx950).

    Returns ``(x_fp4, x_scales, residual_out)``: ``x_fp4`` uint8 ``(m, n // 2)``
    (two E2M1 codes per byte), ``x_scales`` uint8 E8M0 ``(round_up(m, 256),
    n // 32)`` in the AITER ASM FP4 GEMM tile order (rows >= m are not
    written), and the pre-norm sum. Bit-identical to gemma_fused_add_rmsnorm
    followed by AITER ``per_1x32_f4_quant(..., shuffle=True)``.
    """
    m, n = x.shape
    assert residual.shape == x.shape and n % 256 == 0, (x.shape, residual.shape)
    x_fp4 = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    x_scales = torch.empty(
        ((m + 255) // 256 * 256, n // 32), dtype=torch.uint8, device=x.device
    )
    res_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(n)
    _gemma_fused_add_rmsnorm_mxfp4_quant_kernel[(m,)](
        x,
        residual,
        weight,
        x_fp4.view(torch.int32),
        x_scales,
        res_out,
        n,
        x.stride(0),
        x.stride(1),
        residual.stride(0),
        residual.stride(1),
        n // 32,
        eps,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
    )
    return x_fp4, x_scales, res_out
