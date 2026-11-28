"""Peer-memory pushes for the DSA prefill shard-Q exchange.

Every rank writes straight into its peers' symmetric buffers in the layout the
receiver reads, so the exchange needs no all-to-all and no re-layout copy.
Rank ``s`` owns rows ``[bounds[s], bounds[s + 1])`` of the batch:
``push_rows_to_peers`` sends each owner its rows of a local head-sharded
``[rows, W]`` tensor at column ``dst_col``; ``push_cols_to_peers`` sends column
block ``j`` of this rank's ``[count, world * W]`` rows back to rank ``j`` at row
``row0``. Widths are in 4-byte words.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _push_rows_kernel(
    src,
    peer_ptrs,
    bounds,
    width,
    dst_stride,
    dst_col,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    peer = tl.program_id(1)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(bounds + peer).to(tl.int64)
    hi = tl.load(bounds + peer + 1).to(tl.int64)
    row = lo + i
    mask = (row < hi) & (col < width)
    dst = tl.load(peer_ptrs + peer).to(tl.pointer_type(tl.int32))
    vals = tl.load(src + row * width + col, mask=mask)
    tl.store(dst + i * dst_stride + dst_col + col, vals, mask=mask)


@triton.jit
def _push_cols_kernel(
    src,
    peer_ptrs,
    row0,
    width,
    world,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = (pid // world).to(tl.int64)
    peer = pid % world
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < width
    dst = tl.load(peer_ptrs + peer).to(tl.pointer_type(tl.int32))
    vals = tl.load(src + i * world * width + peer * width + col, mask=mask)
    tl.store(dst + (row0 + i) * width + col, vals, mask=mask)


def _words(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous() and (x.numel() * x.element_size()) % 4 == 0
    return x.view(torch.uint8).view(torch.int32)


def push_rows_to_peers(
    src: torch.Tensor,
    peer_ptrs: torch.Tensor,
    bounds: torch.Tensor,
    *,
    max_count: int,
    dst_stride_bytes: int,
    dst_col_bytes: int,
) -> None:
    """Rank ``s`` gets ``src[bounds[s]:bounds[s + 1]]`` (``[rows, ...]``,
    contiguous) at rows ``[0, count)`` of its buffer, ``dst_col_bytes`` into
    each row of ``dst_stride_bytes``. ``peer_ptrs`` int64 and ``bounds`` int32
    are device tensors of ``world`` and ``world + 1`` entries."""
    width = src[0].numel() * src.element_size() // 4
    world = peer_ptrs.shape[0]
    grid = (max_count, world, triton.cdiv(width, _BLOCK))
    _push_rows_kernel[grid](
        _words(src),
        peer_ptrs,
        bounds,
        width,
        dst_stride_bytes // 4,
        dst_col_bytes // 4,
        BLOCK=_BLOCK,
    )


def push_cols_to_peers(
    src: torch.Tensor, peer_ptrs: torch.Tensor, *, row0: int
) -> None:
    """Rank ``j`` gets column block ``j`` of ``src`` (``[count, world, ...]``,
    contiguous) at rows ``[row0, row0 + count)`` of its buffer."""
    count, world = src.shape[0], src.shape[1]
    if count == 0:
        return
    width = src[0, 0].numel() * src.element_size() // 4
    grid = (count * world, triton.cdiv(width, _BLOCK))
    _push_cols_kernel[grid](_words(src), peer_ptrs, row0, width, world, BLOCK=_BLOCK)
