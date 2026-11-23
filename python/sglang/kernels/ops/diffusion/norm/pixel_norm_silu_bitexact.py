# SPDX-License-Identifier: Apache-2.0
"""Bit-exact per-pixel RMS norm + SiLU, written in the next conv's layout.

The LTX video VAE runs ``conv(silu(x / sqrt(mean(x**2, dim=1) + eps)))`` with
no affine parameters. Under autocast ``x`` is half precision, ``pow`` promotes
to fp32, and everything up to the conv stays fp32, so eager spends seven
full-size passes per norm: the cast to fp32, ``pow``, ``mean``, a broadcast
``div`` that aten does not vectorise on channels_last, ``silu``, the layout /
temporal-pad copy and the cast back to the conv dtype.

``pixel_norm_silu(x, eps, out)`` keeps aten's ``torch.mean`` (the reduction
order is the reference's) and replays the rest with the same IEEE operations:

- squares: ``fp32(x) * fp32(x)`` in ``x``'s own strides (aten ``pow(x, 2)``),
  so ``torch.mean(..., dim=1, keepdim=True)`` sees the tensor eager gives it;
- finish: ``q = fp32(x) / sqrt(mean_sq + fp32(eps))`` and aten's SiLU
  ``q / (1 + expf(-q))`` (``tl.exp`` is an approximation; ``silu_f32`` replays
  CUDA's ``expf``), rounded once to ``out.dtype`` -- the single rounding the
  conv's autocast cast applies to the fp32 activation.

``out`` is the destination: a ``[B, C, T, H, W]`` view whose batch elements are
dense NDHWC blocks, e.g. the body of a temporally padded channels_last_3d
buffer. ``x`` may be dense channels_last_3d or contiguous NCDHW.

Reference: ``F.silu(x.float() / torch.sqrt(torch.mean(x.float() ** 2, dim=1,
keepdim=True) + eps)).to(out.dtype)``. Callers verify the first call with
``BitExactFusionGate`` and keep the eager chain as the fallback.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.numerics import silu_f32

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_MAX_CHANNELS = 4096
_SQUARE_BLOCK = 4096


@triton.jit
def _square_kernel(x_ptr, out_ptr, total, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # aten pow(x, 2) is one fp32 multiply.
    tl.store(out_ptr + offs, x * x, mask=mask)


@triton.jit
def _finish_kernel(
    x_ptr,
    mean_sq_ptr,
    out_ptr,
    n_rows,
    eps,
    CHANNELS: tl.constexpr,
    X_CHANNELS_LAST: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One program normalises a [ROWS, BLOCK_C] tile: consecutive (t, h, w)
    # pixels by consecutive channels.
    row_ids = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    col_ids = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = row_ids < n_rows
    mask = row_mask[:, None] & (col_ids < CHANNELS)[None, :]
    out_offs = row_ids[:, None] * CHANNELS + col_ids[None, :]
    if X_CHANNELS_LAST:
        x_offs = out_offs
    else:
        x_offs = col_ids[None, :].to(tl.int64) * n_rows + row_ids[:, None]

    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.load(mean_sq_ptr + row_ids, mask=row_mask, other=1.0)
    rms = tl.sqrt_rn(mean_sq + eps)
    q = tl.div_rn(x, tl.broadcast_to(rms[:, None], ROWS, BLOCK_C))
    y = silu_f32(q)
    tl.store(out_ptr + out_offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _is_ndhwc_block(t: torch.Tensor) -> bool:
    """Every batch element of ``t`` is a dense NDHWC block (``C > 1``)."""
    _, c, _, h, w = t.shape
    expected = (None, 1, h * w * c, w * c, c)
    return all(
        size == 1 or stride == want
        for size, stride, want in zip(t.shape[1:], t.stride()[1:], expected[1:])
    )


def can_use_pixel_norm_silu(x: torch.Tensor, eps: float, out: torch.Tensor) -> bool:
    """True when :func:`pixel_norm_silu` can fill ``out`` from ``x``. Never raises."""
    return (
        isinstance(x, torch.Tensor)
        and isinstance(out, torch.Tensor)
        and x.is_cuda
        and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and not (torch.is_grad_enabled() and x.requires_grad)
        and x.dim() == 5
        and x.numel() > 0
        and 1 < x.shape[1] <= _MAX_CHANNELS
        and x.dtype in _SUPPORTED_DTYPES
        and (x.is_contiguous(memory_format=torch.channels_last_3d) or x.is_contiguous())
        and isinstance(eps, float)
        and out.device == x.device
        and out.dtype in _SUPPORTED_DTYPES
        and out.shape == x.shape
        and _is_ndhwc_block(out)
    )


def pixel_norm_silu(x: torch.Tensor, eps: float, out: torch.Tensor) -> torch.Tensor:
    """``out[...] = silu(x / sqrt(mean(x**2, dim=1, keepdim=True) + eps))`` in fp32,
    rounded once to ``out.dtype``. Guard with :func:`can_use_pixel_norm_silu`."""
    if not can_use_pixel_norm_silu(x, eps, out):
        raise ValueError(
            "unsupported input for pixel_norm_silu: got x "
            f"{tuple(x.shape)}/{x.dtype} strides {tuple(x.stride())}, out "
            f"{tuple(out.shape)}/{out.dtype} strides {tuple(out.stride())}"
        )
    batch, channels = x.shape[:2]
    n_rows = x.numel() // (batch * channels)
    channels_last = x.is_contiguous(memory_format=torch.channels_last_3d)
    if channels_last:
        # Whole channel rows, ~4K elements per program: 32 pixels at C <= 128
        # down to 4 at C = 1024. Loads and stores are both contiguous.
        block_c = triton.next_power_of_2(channels)
        rows = max(1, min(32, 4096 // block_c))
    else:
        # NCDHW -> NDHWC transposes the tile, so keep both sides in runs of
        # 64 elements: 64 pixels of 64 channels.
        block_c = min(64, triton.next_power_of_2(channels))
        rows = 64
    num_warps = 8 if rows * block_c >= 2048 else 4
    grid = (triton.cdiv(n_rows, rows), triton.cdiv(channels, block_c))

    # Same shape and strides as eager's ``x ** 2``, so aten reduces it identically.
    squares = torch.empty_like(x, dtype=torch.float32)
    if squares.stride() != x.stride():
        raise ValueError("pixel_norm_silu needs a dense, non-overlapping input")
    with torch.cuda.device(x.device):
        _square_kernel[(triton.cdiv(x.numel(), _SQUARE_BLOCK),)](
            x, squares, x.numel(), BLOCK=_SQUARE_BLOCK
        )
        mean_sq = torch.mean(squares, dim=1, keepdim=True)
        del squares
        for n in range(batch):
            _finish_kernel[grid](
                x[n],
                mean_sq[n],
                out[n],
                n_rows,
                eps,
                CHANNELS=channels,
                X_CHANNELS_LAST=channels_last,
                ROWS=rows,
                BLOCK_C=block_c,
                num_warps=num_warps,
            )
    return out


__all__ = ["can_use_pixel_norm_silu", "pixel_norm_silu"]
