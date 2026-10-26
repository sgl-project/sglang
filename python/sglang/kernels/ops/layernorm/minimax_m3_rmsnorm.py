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
def _gemma_add_rmsnorm_fp8_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    q_ptr,
    scale_ptr,
    zeros_ptr,
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
    ZERO_COLS: tl.constexpr,
    BLOCK_ZERO: tl.constexpr,
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
    if ZERO_COLS > 0:
        zcols = tl.arange(0, BLOCK_ZERO)
        tl.store(
            zeros_ptr + row * ZERO_COLS + zcols,
            tl.zeros((BLOCK_ZERO,), zeros_ptr.dtype.element_ty),
            mask=zcols < ZERO_COLS,
        )


@triton.jit
def _gemma_add_rmsnorm_fp8_4rows_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    q_ptr,
    scale_ptr,
    zeros_ptr,
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
    ZERO_COLS: tl.constexpr,
    BLOCK_ZERO: tl.constexpr,
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
        if ZERO_COLS > 0:
            zcols = tl.arange(0, BLOCK_ZERO)
            tl.store(
                zeros_ptr + row * ZERO_COLS + zcols,
                tl.zeros((BLOCK_ZERO,), zeros_ptr.dtype.element_ty),
                mask=(zcols < ZERO_COLS) & in_rows,
            )
    if GROUP != 0:
        # column k: the scales of row base + k
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


# measured on MI350X at 6144 columns: four rows per program win from 2048 rows
_FOUR_ROWS_MIN_ROWS = 2048


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


def gemma_add_rmsnorm_fp8(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    group_size: Optional[int] = None,
    transpose_scale: bool = False,
    keep_bf16: bool = False,
    zeros: Optional[torch.Tensor] = None,
):
    """Gemma RMSNorm of ``x`` (plus ``residual`` when given) quantized to FP8 as aiter
    does: per token (``group_size`` None, scale ``[m, 1]``) like ``per_token_quant_hip``,
    or per ``1 x group_size`` (scale ``[m, n / group_size]``, column-major when
    ``transpose_scale``) like ``get_hip_quant(QuantType.per_1x128)``.

    Returns ``((fp8 [m, n], fp32 scale), residual_out)``, with the bf16 norm as a third
    element of the tuple when ``keep_bf16``; ``residual_out`` is the pre-norm sum, or
    ``x`` itself without a residual. ``zeros``, a contiguous ``[m, *]`` tensor, is
    zero-filled by the same launch.
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
    zero_cols = 0
    if zeros is not None:
        assert zeros.is_contiguous() and zeros.shape[0] == m
        zero_cols = zeros.numel() // m
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
        q if zeros is None else zeros,
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
        ZERO_COLS=zero_cols,
        BLOCK_ZERO=triton.next_power_of_2(max(zero_cols, 1)),
        num_warps=_num_warps(block_n),
    )
    if group and transpose_scale:
        scale = scale.t()
    quantized = (q, scale, out) if keep_bf16 else (q, scale)
    return quantized, x if residual is None else res_out
