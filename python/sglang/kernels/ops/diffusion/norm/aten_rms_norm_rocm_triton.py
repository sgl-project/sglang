# SPDX-License-Identifier: Apache-2.0
"""Bit-exact replica of ROCm aten's affine-free RMSNorm on short half rows.

For a row of N = 64, 128 or 256 fp16/bf16 values, torch's ROCm
``vectorized_layer_norm_kernel<T, float, rms_norm=true>`` lets lane t of the
first wavefront add the squares of elements 4t..4t+3 in order (half squares
are exact in fp32, so FMA contraction cannot change the sum). ``shfl_down``
then adds lane i + h into lane i for h = N/8 .. 1 (higher offsets only meet
empty lanes); the mean square is that sum / N and the output ``rstd * x`` with
``rstd = rsqrt(mean + eps)``. One 256-thread block per row leaves that kernel
far from bandwidth, behind a ``.contiguous()`` copy of strided rows; this one
normalizes many rows per program and reads the strided rows directly.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_ROWS_PER_PROG = 64
_WIDTHS = (64, 128, 256)


@triton.jit
def _aten_rstd(x, eps, width: tl.constexpr, rows: tl.constexpr):
    """``rsqrt(mean(x * x) + eps)`` per row of fp32 ``x`` [rows, width], in aten's order."""
    # lane t owns elements 4t..4t+3 and adds their squares in order
    lanes: tl.constexpr = width // 4
    even, odd = tl.split(tl.reshape(x, (rows, lanes, 2, 2), can_reorder=False))
    x0, x2 = tl.split(even)
    x1, x3 = tl.split(odd)
    acc = x0 * x0
    acc = acc + x1 * x1
    acc = acc + x2 * x2
    acc = acc + x3 * x3
    # shfl_down fold: lane i += lane i + half
    if lanes >= 64:
        acc = tl.sum(tl.reshape(acc, (rows, 2, 32), can_reorder=False), axis=1)
    if lanes >= 32:
        acc = tl.sum(tl.reshape(acc, (rows, 2, 16), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows, 2, 8), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows, 2, 4), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows, 2, 2), can_reorder=False), axis=1)
    return tl.rsqrt(tl.sum(acc, axis=1) / width + eps)


@triton.jit
def _aten_rms_norm_rocm_kernel(
    y_ptr,
    x_ptr,
    n_rows,
    rows_per_outer,
    outer_stride,
    row_stride,
    eps,
    width: tl.constexpr,
    rows: tl.constexpr,
):
    row = tl.program_id(0) * rows + tl.arange(0, rows)
    live = row < n_rows
    row64 = row.to(tl.int64)
    src = (row64 // rows_per_outer) * outer_stride + (
        row64 % rows_per_outer
    ) * row_stride
    offs = tl.arange(0, width)
    x = tl.load(x_ptr + src[:, None] + offs[None, :], mask=live[:, None], other=0.0)
    x = x.to(tl.float32)
    y = _aten_rstd(x, eps, width, rows)[:, None] * x
    tl.store(
        y_ptr + row64[:, None] * width + offs[None, :],
        y.to(y_ptr.dtype.element_ty),
        mask=live[:, None],
    )


@triton.jit
def _aten_rms_norm_neox_rope_qk_kernel(
    q_out,
    k_out,
    q_ptr,
    k_ptr,
    cache_ptr,
    pos_ptr,
    n_rows,
    heads,
    token_stride,
    head_stride,
    eps,
    width: tl.constexpr,
    rot: tl.constexpr,
    rows: tl.constexpr,
):
    # row = (token, head); the NeoX pair of column c < rot / 2 is c + rot / 2
    row = tl.program_id(0) * rows + tl.arange(0, rows)
    live = row < n_rows
    row64 = row.to(tl.int64)
    token = row64 // heads
    src = token * token_stride + (row64 % heads) * head_stride
    offs = tl.arange(0, width)
    half: tl.constexpr = rot // 2
    lo = offs < half
    hi = (offs >= half) & (offs < rot)
    partner = tl.where(lo, offs + half, tl.where(hi, offs - half, offs))
    angle = tl.where(lo, offs, tl.where(hi, offs - half, 0))
    rotated = live[:, None] & (lo | hi)[None, :]
    cache_row = tl.load(pos_ptr + token, mask=live, other=0) * rot
    cos = tl.load(
        cache_ptr + cache_row[:, None] + angle[None, :], mask=rotated, other=0.0
    )
    sin = tl.load(
        cache_ptr + cache_row[:, None] + half + angle[None, :], mask=rotated, other=0.0
    )
    cos = cos.to(tl.float32)
    sin = sin.to(tl.float32)
    dtype = q_out.dtype.element_ty
    for which in tl.static_range(2):
        x_ptr = q_ptr if which == 0 else k_ptr
        y_ptr = q_out if which == 0 else k_out
        x = tl.load(x_ptr + src[:, None] + offs[None, :], mask=live[:, None], other=0.0)
        x = x.to(tl.float32)
        xp = tl.load(
            x_ptr + src[:, None] + partner[None, :], mask=live[:, None], other=0.0
        ).to(tl.float32)
        rstd = _aten_rstd(x, eps, width, rows)[:, None]
        # the RMSNorm output is stored in BF16, then the rope kernel's BF16
        # operators round each product and the sum
        y = (rstd * x).to(dtype).to(tl.float32)
        yp = (rstd * xp).to(dtype).to(tl.float32)
        a = (y * cos).to(dtype).to(tl.float32)
        b = (yp * sin).to(dtype).to(tl.float32)
        out = tl.where(lo[None, :], a - b, tl.where(hi[None, :], a + b, y))
        tl.store(
            y_ptr + row64[:, None] * width + offs[None, :],
            out.to(dtype),
            mask=live[:, None],
        )


def can_use_aten_rms_norm_rocm(x: torch.Tensor) -> bool:
    """Affine-free half rows whose last dim is contiguous and one of _WIDTHS."""
    return (
        torch.version.hip is not None
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.dim() >= 2
        and x.shape[-1] in _WIDTHS
        and x.stride(-1) == 1
        and not torch.compiler.is_compiling()
    )


def aten_rms_norm_rocm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``F.rms_norm(x, x.shape[-1:], eps=eps)`` (no weight), contiguous out.

    ``x`` may be a strided view whose rows are evenly spaced within its
    second-to-last dim and whose leading dims collapse into one stride.
    """
    width = x.shape[-1]
    lead = x.reshape(-1, x.shape[-2], width) if x.dim() > 2 else x[None]
    if lead.stride(-1) != 1:
        raise ValueError("rows must be contiguous along the normalized dim")
    outer, inner = lead.shape[0], lead.shape[1]
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    n_rows = outer * inner
    grid = (triton.cdiv(n_rows, _ROWS_PER_PROG),)
    _aten_rms_norm_rocm_kernel[grid](
        out,
        lead,
        n_rows,
        inner,
        lead.stride(0),
        lead.stride(1),
        eps,
        width=width,
        rows=_ROWS_PER_PROG,
    )
    return out


def can_use_aten_rms_norm_neox_rope_qk_rocm(
    query: torch.Tensor,
    key: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
) -> bool:
    """[B, S, H, D] BF16 q/k views sharing strides, whose batch and token dims
    collapse into one stride, with a ``[pos, cos | sin]`` cache of the q dtype.
    The rope kernel's FP16 arithmetic rounds differently on HIP, so FP16 stays
    on the two-kernel path."""
    if not (
        can_use_aten_rms_norm_rocm(query)
        and query.dtype == torch.bfloat16
        and query.dim() == 4
        and key.shape == query.shape
        and key.stride() == query.stride()
        and key.dtype == query.dtype
        and key.device == query.device
        and query.stride(0) == query.shape[1] * query.stride(1)
    ):
        return False
    rot = cache.shape[-1] if cache.dim() == 2 else 0
    return (
        cache.dtype == query.dtype
        and cache.device == query.device
        and cache.is_contiguous()
        and 2 <= rot <= query.shape[-1]
        and rot % 2 == 0
        and positions.dtype == torch.int64
        and positions.device == query.device
        and positions.is_contiguous()
        and positions.numel() == query.shape[0] * query.shape[1]
    )


def aten_rms_norm_neox_rope_qk_rocm(
    query: torch.Tensor,
    key: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``aten_rms_norm_rocm`` of ``query`` and ``key``, then sgl_kernel's NeoX
    ``rotary_embedding`` of both (``positions`` index ``cache`` per batch-major
    token), in one launch: contiguous ``(query, key)`` with the same bytes."""
    batch, tokens, heads, width = query.shape
    q_out = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    k_out = torch.empty_like(q_out)
    n_rows = batch * tokens * heads
    _aten_rms_norm_neox_rope_qk_kernel[(triton.cdiv(n_rows, _ROWS_PER_PROG),)](
        q_out,
        k_out,
        query,
        key,
        cache,
        positions,
        n_rows,
        heads,
        query.stride(1),
        query.stride(2),
        eps,
        width=width,
        rot=cache.shape[-1],
        rows=_ROWS_PER_PROG,
        enable_fp_fusion=False,
    )
    return q_out, k_out
