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
    rstd = tl.rsqrt(tl.sum(acc, axis=1) / width + eps)
    y = rstd[:, None] * x
    tl.store(
        y_ptr + row64[:, None] * width + offs[None, :],
        y.to(y_ptr.dtype.element_ty),
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
