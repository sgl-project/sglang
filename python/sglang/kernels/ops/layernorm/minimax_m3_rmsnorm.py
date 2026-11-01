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

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.aiter_fp8_quant_triton import (
    fp8_group_quant,
    fp8_per_token_quant,
)


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
    ROUND_SUM: tl.constexpr = False,
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
    if ROUND_SUM:
        # normalize the sum as stored (the eager path adds in the activation dtype)
        s = s.to(res_out_ptr.dtype.element_ty).to(tl.float32)
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
def _gemma_add_rmsnorm_fp8_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    q_ptr,
    scale_ptr,
    n_rows,
    n_cols,
    stride_xrow,
    stride_xcol,
    stride_rrow,
    stride_rcol,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    STORE_OUT: tl.constexpr,
    GROUP: tl.constexpr,
    TRANSPOSE_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # The norm is the code of the two kernels above, launched the same way, so the
    # quantized values are those of the separate norm and aiter quant kernels.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    s = tl.load(
        x_ptr + row * stride_xrow + cols * stride_xcol, mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_RESIDUAL:
        r = tl.load(
            res_ptr + row * stride_rrow + cols * stride_rcol, mask=mask, other=0.0
        ).to(tl.float32)
        s = s + r
        tl.store(
            res_out_ptr + row * n_cols + cols,
            s.to(res_out_ptr.dtype.element_ty),
            mask=mask,
        )
    var = tl.sum(s * s, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (s * rstd * (1.0 + w)).to(x_ptr.dtype.element_ty)
    if STORE_OUT:
        tl.store(out_ptr + row * n_cols + cols, out, mask=mask)
    if GROUP == 0:
        q, scale = fp8_per_token_quant(out.to(tl.float32), 0)
        tl.store(scale_ptr + row, scale)
    else:
        q, scale = fp8_group_quant(out.to(tl.float32), GROUP)
        g = tl.arange(0, BLOCK_N // GROUP)
        if TRANSPOSE_SCALE:
            scale_offs = g * n_rows + row
        else:
            scale_offs = row * (n_cols // GROUP) + g
        tl.store(scale_ptr + scale_offs, scale, mask=g < n_cols // GROUP)
    tl.store(q_ptr + row * n_cols + cols, q, mask=mask)


@triton.jit
def _gemma_add_rmsnorm_fp8_4rows_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    q_ptr,
    scale_ptr,
    n_rows,
    n_cols,
    stride_xrow,
    stride_xcol,
    stride_rrow,
    stride_rcol,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    STORE_OUT: tl.constexpr,
    GROUP: tl.constexpr,
    TRANSPOSE_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # The row code of _gemma_add_rmsnorm_fp8_kernel (same layout, same bytes) over four
    # rows: one weight load, and one layout conversion for the four rows' group scales.
    cols = tl.arange(0, BLOCK_N)
    w = tl.load(w_ptr + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
    base = tl.program_id(0) * 4
    for i in tl.static_range(4):
        row = base + i
        in_rows = row < n_rows
        mask = (cols < n_cols) & in_rows
        s = tl.load(
            x_ptr + row * stride_xrow + cols * stride_xcol, mask=mask, other=0.0
        ).to(tl.float32)
        if HAS_RESIDUAL:
            r = tl.load(
                res_ptr + row * stride_rrow + cols * stride_rcol, mask=mask, other=0.0
            ).to(tl.float32)
            s = s + r
            tl.store(
                res_out_ptr + row * n_cols + cols,
                s.to(res_out_ptr.dtype.element_ty),
                mask=mask,
            )
        var = tl.sum(s * s, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        out = (s * rstd * (1.0 + w)).to(x_ptr.dtype.element_ty)
        if STORE_OUT:
            tl.store(out_ptr + row * n_cols + cols, out, mask=mask)
        if GROUP == 0:
            q, scale = fp8_per_token_quant(out.to(tl.float32), 0)
            tl.store(scale_ptr + row, scale, mask=in_rows)
        else:
            q, scale = fp8_group_quant(out.to(tl.float32), GROUP)
            if i == 0:
                scale0 = scale
            elif i == 1:
                scale1 = scale
            elif i == 2:
                scale2 = scale
            else:
                scale3 = scale
        tl.store(q_ptr + row * n_cols + cols, q, mask=mask)
    if GROUP != 0:
        # Column k holds the scales of row base + k.
        scales = tl.reshape(
            tl.join(tl.join(scale0, scale2), tl.join(scale1, scale3)),
            [BLOCK_N // GROUP, 4],
        )
        g = tl.arange(0, BLOCK_N // GROUP)
        rows = base + tl.arange(0, 4)
        if TRANSPOSE_SCALE:
            scale_offs = g[:, None] * n_rows + rows[None, :]
        else:
            scale_offs = rows[None, :] * (n_cols // GROUP) + g[:, None]
        tl.store(
            scale_ptr + scale_offs,
            scales,
            mask=(g < n_cols // GROUP)[:, None] & (rows < n_rows)[None, :],
        )


# Measured on MI350X at 6144 columns: four rows per program win from 2048 rows.
_FOUR_ROWS_MIN_ROWS = 2048


@triton.jit
def _gemma_fused_add_rmsnorm_mxfp4_quant_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    q_ptr,
    scale_ptr,
    res_out_ptr,
    n_rows,
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
    # A program handles rows pid, pid + num_programs, ... and loads the next
    # row's inputs while it reduces and quantizes the current one, so a row's
    # reduction does not leave the memory system idle.
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x_next = tl.load(
        x_ptr + pid * stride_xrow + cols * stride_xcol, mask=mask, other=0.0
    )
    r_next = tl.load(
        res_ptr + pid * stride_rrow + cols * stride_rcol, mask=mask, other=0.0
    )
    for row in range(pid, n_rows, step):
        x = x_next.to(tl.float32)
        r = r_next.to(tl.float32)
        nxt = row + step
        nmask = mask & (nxt < n_rows)
        x_next = tl.load(
            x_ptr + nxt * stride_xrow + cols * stride_xcol, mask=nmask, other=0.0
        )
        r_next = tl.load(
            res_ptr + nxt * stride_rrow + cols * stride_rcol, mask=nmask, other=0.0
        )
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
        tl.store(
            scale_ptr + sidx, tl.reshape(e, (NB,)).to(tl.uint8), mask=c < n_cols // 32
        )


@functools.lru_cache(maxsize=None)
def _num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _num_row_programs(m: int, device: torch.device) -> int:
    # Two programs per CU (16-warp programs); each walks rows with stride grid.
    return max(1, min(m, 2 * _num_cus(device)))


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
    round_sum: bool = False,
):
    """Fused (x + residual) then Gemma RMSNorm; returns (normed, pre-norm sum).
    ``round_sum`` normalizes the sum rounded to the activation dtype, as the eager path does."""
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
        ROUND_SUM=round_sum,
        num_warps=_num_warps(block_n),
    )
    return out.reshape(orig_shape), res_out.reshape(orig_shape)


def gemma_add_rmsnorm_fp8(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    group_size: Optional[int] = None,
    transpose_scale: bool = False,
    keep_bf16: bool = False,
):
    """Gemma RMSNorm of ``x`` (plus ``residual`` when given) quantized to FP8 as aiter does.

    Per token (``group_size`` None, scale ``[m, 1]``) like ``per_token_quant_hip``, or per
    ``1 x group_size`` (scale ``[m, n / group_size]``, column-major when
    ``transpose_scale``) like ``get_hip_quant(QuantType.per_1x128)``. Returns
    ``((fp8 [m, n], fp32 scale), residual_out)``, with the bf16 norm as a third element of
    the tuple when ``keep_bf16``; ``residual_out`` is the pre-norm sum, or ``x`` itself
    without a residual.
    """
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    m = x2.shape[0]
    q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    group = group_size or 0
    if group:
        assert n % group == 0
        scale_shape = (n // group, m) if transpose_scale else (m, n // group)
    else:
        scale_shape = (m, 1)
    scale = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device) if keep_bf16 else q
    if residual is None:
        r2, res_out = x2, x2
    else:
        r2 = residual.reshape(-1, n)
        res_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(n)
    kernel, grid = _gemma_add_rmsnorm_fp8_kernel, m
    if m >= _FOUR_ROWS_MIN_ROWS:
        kernel, grid = _gemma_add_rmsnorm_fp8_4rows_kernel, (m + 3) // 4
    kernel[(grid,)](
        x2,
        r2,
        weight,
        out,
        res_out,
        q,
        scale,
        m,
        n,
        x2.stride(0),
        x2.stride(1),
        r2.stride(0),
        r2.stride(1),
        eps,
        HAS_RESIDUAL=residual is not None,
        STORE_OUT=keep_bf16,
        GROUP=group,
        TRANSPOSE_SCALE=transpose_scale,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
    )
    if group and transpose_scale:
        scale = scale.t()
    quantized = (q, scale, out) if keep_bf16 else (q, scale)
    return quantized, x if residual is None else res_out


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
    _gemma_fused_add_rmsnorm_mxfp4_quant_kernel[(_num_row_programs(m, x.device),)](
        x,
        residual,
        weight,
        x_fp4.view(torch.int32),
        x_scales,
        res_out,
        m,
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
