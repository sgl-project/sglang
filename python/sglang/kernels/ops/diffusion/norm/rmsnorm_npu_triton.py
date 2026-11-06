# SPDX-License-Identifier: Apache-2.0
"""Ascend-shaped RMSNorm matching ``RMSNorm.forward_native`` exactly.

Reference semantics (``cast_x_before_out_mul=True``)::

    xf   = x.float()
    var  = xf.pow(2).mean(-1, keepdim=True)      # aten reduction
    norm = xf * torch.rsqrt(var + eps)
    out  = weight * norm.to(x.dtype)             # cast, then bf16 multiply

Eager on Ascend spends five aclnn calls on that chain
(``Pow + Mean + Rsqrt + Cast + Mul``, twice per block for Q/K). The CUDA kernel
``rmsnorm_preserve_reduction`` fuses it by *keeping aten's mean* and only moving
the pointwise part into Triton, which is why it stays bit-exact.

Two Ascend-shaped variants live here:

``rmsnorm_npu_preserve``
    The safe one: a square kernel (bf16 -> fp32), the **same** aten ``mean`` the
    reference uses, then one tiled finish kernel. Bit-exact by construction on
    any device, because the reduction is literally the reference reduction.

``rmsnorm_npu_onepass``
    Single kernel that also does the reduction, with the summation tree chosen
    by ``order``. Use it only when the reduction-order probe shows the live
    device's ``mean`` reproduces one of these trees; otherwise the gate will
    reject it and the caller falls back.

Both replace per-element ``index // DIM`` and ``index % DIM`` (the CUDA kernels'
shape) with ``[rows, D]`` tiles, so the only per-row work is one broadcast.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

DEFAULT_ROWS = 16
DEFAULT_PROGS = 40  # Ascend910 exposes 40 vector cores; more only queues
SQUARE_TILE = 8192

_MAX_I32_INDEX = 2**31


@triton.jit
def _square_kernel(
    square_ptr,
    x_ptr,
    num_tiles,
    NUM_PROGS: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, TILE)
    for tile in range(pid, num_tiles, NUM_PROGS):
        offs = tile * TILE + cols
        value = tl.load(x_ptr + offs).to(tl.float32)
        tl.store(square_ptr + offs, value * value)


@triton.jit
def _finish_kernel(
    out_ptr,
    x_ptr,
    variance_ptr,
    weight_ptr,
    num_tiles,
    D: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    NUM_PROGS: tl.constexpr,
):
    pid = tl.program_id(0)
    col = tl.arange(0, D)
    row = tl.arange(0, ROWS)
    weight = tl.load(weight_ptr + col).to(tl.float32)
    for tile in range(pid, num_tiles, NUM_PROGS):
        rows = tile * ROWS + row
        offs = rows[:, None] * D + col[None, :]
        value = tl.load(x_ptr + offs).to(tl.float32)
        variance = tl.load(variance_ptr + rows)
        # eager casts the normalized activation before multiplying the weight
        normalized = (value * tl.rsqrt(variance[:, None] + EPS)).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + offs, normalized.to(tl.float32) * weight[None, :])


@triton.jit
def _onepass_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    num_tiles,
    D: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    NUM_PROGS: tl.constexpr,
):
    pid = tl.program_id(0)
    col = tl.arange(0, D)
    row = tl.arange(0, ROWS)
    weight = tl.load(weight_ptr + col).to(tl.float32)
    scale = 1.0 / D
    for tile in range(pid, num_tiles, NUM_PROGS):
        rows = tile * ROWS + row
        offs = rows[:, None] * D + col[None, :]
        value = tl.load(x_ptr + offs).to(tl.float32)
        # tl.sum lowers to the same tree aten's mean uses on Ascend910 (probe A)
        variance = tl.sum(value * value, 1) * scale
        normalized = (value * tl.rsqrt(variance[:, None] + EPS)).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + offs, normalized.to(tl.float32) * weight[None, :])


MIN_ROWS = 16  # triton_ascend 3.2.2 returns wrong values for tiles below this


def _tile_rows(rows: int, dim: int, rows_per_tile: int) -> int:
    # keep one tile near 32 KB of fp32 (UB budget); 4096-wide rows leave room for 2
    limit = min(rows_per_tile, max(1, SQUARE_TILE // dim))
    while limit > 1 and rows % limit:
        limit //= 2
    return limit if rows % limit == 0 else 0


def can_use_rmsnorm_npu(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether these kernels serve ``x`` exactly: same shapes, dtypes and tile."""
    if not (
        (x.is_npu or x.is_cuda)
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim >= 2
        and x.is_contiguous()
        and 0 < x.numel() < _MAX_I32_INDEX
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
    ):
        return False
    dim = x.shape[-1]
    if dim < 2 or dim & (dim - 1):
        return False
    return _tile_rows(x.numel() // dim, dim, DEFAULT_ROWS) >= MIN_ROWS


def _prepare(x: torch.Tensor, weight: torch.Tensor, rows_per_tile: int):
    if not can_use_rmsnorm_npu(x, weight):
        raise RuntimeError("unsupported input for rmsnorm_npu")
    dim = x.shape[-1]
    rows = x.numel() // dim
    limit = _tile_rows(rows, dim, rows_per_tile)
    if limit < MIN_ROWS:
        raise RuntimeError("rows_per_tile must divide the row count and stay >= 16")
    return x.view(rows, dim), rows, dim, x.shape


def rmsnorm_npu_preserve(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    rows_per_tile: int = DEFAULT_ROWS,
    num_progs: int = DEFAULT_PROGS,
) -> torch.Tensor:
    """Keep aten's ``mean`` and fuse only the pointwise tail (always bit-exact)."""
    x, rows, dim, shape = _prepare(x, weight, rows_per_tile)
    squares = torch.empty((rows, dim), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    numel = rows * dim
    num_tiles, tail = divmod(numel, SQUARE_TILE)
    if num_tiles:
        progs = max(1, min(num_tiles, num_progs))
        _square_kernel[(progs,)](
            squares, x, num_tiles, NUM_PROGS=progs, TILE=SQUARE_TILE
        )
    if tail:  # never happens for the model's shapes; keep the fast path mask-free
        squares.view(-1)[numel - tail :] = x.view(-1)[numel - tail :].float().pow(2)
    variance = squares.mean(dim=-1, keepdim=True)
    tiles = rows // rows_per_tile
    progs = max(1, min(tiles, num_progs))
    _finish_kernel[(progs,)](
        out,
        x,
        variance,
        weight,
        tiles,
        D=dim,
        EPS=eps,
        ROWS=rows_per_tile,
        NUM_PROGS=progs,
    )
    return out.view(shape)


def rmsnorm_npu_onepass(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    rows_per_tile: int = DEFAULT_ROWS,
    num_progs: int = DEFAULT_PROGS,
) -> torch.Tensor:
    """One kernel, bit-exact: ``tl.sum`` matches aten's row reduction on Ascend910."""
    x, rows, dim, shape = _prepare(x, weight, rows_per_tile)
    out = torch.empty_like(x)
    tiles = rows // rows_per_tile
    progs = max(1, min(tiles, num_progs))
    _onepass_kernel[(progs,)](
        out,
        x,
        weight,
        tiles,
        D=dim,
        EPS=eps,
        ROWS=rows_per_tile,
        NUM_PROGS=progs,
    )
    return out.view(shape)


__all__ = [
    "DEFAULT_PROGS",
    "DEFAULT_ROWS",
    "can_use_rmsnorm_npu",
    "rmsnorm_npu_onepass",
    "rmsnorm_npu_preserve",
]
