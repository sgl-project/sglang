from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_slot(src, dst, n, row, at, cols, SCATTER: tl.constexpr):
    # gather: dst[row] = src[at]; scatter: dst[at] = src[row] (cast to dst's dtype)
    m = cols < n
    if SCATTER:
        v = tl.load(src + row * n + cols, mask=m)
        tl.store(dst + at * n + cols, v.to(dst.dtype.element_ty), mask=m)
    else:
        v = tl.load(src + at * n + cols, mask=m)
        tl.store(dst + row * n + cols, v.to(dst.dtype.element_ty), mask=m)


@triton.jit
def _rows_kernel(
    idx_ptr,
    s0,
    d0,
    n0,
    s1,
    d1,
    n1,
    s2,
    d2,
    n2,
    s3,
    d3,
    n3,
    s4,
    d4,
    n4,
    s5,
    d5,
    n5,
    HAS0: tl.constexpr,
    HAS1: tl.constexpr,
    HAS2: tl.constexpr,
    HAS3: tl.constexpr,
    HAS4: tl.constexpr,
    HAS5: tl.constexpr,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program == one (row, column block). Every buffer shares the row index, so
    # a single launch moves every buffer's row instead of one indexing op each.
    # int64 offsets: row * n can pass 2**31 for a wide buffer
    row = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1)
    at = tl.load(idx_ptr + row).to(tl.int64)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    if HAS0:
        _copy_slot(s0, d0, n0, row, at, cols, SCATTER)
    if HAS1:
        _copy_slot(s1, d1, n1, row, at, cols, SCATTER)
    if HAS2:
        _copy_slot(s2, d2, n2, row, at, cols, SCATTER)
    if HAS3:
        _copy_slot(s3, d3, n3, row, at, cols, SCATTER)
    if HAS4:
        _copy_slot(s4, d4, n4, row, at, cols, SCATTER)
    if HAS5:
        _copy_slot(s5, d5, n5, row, at, cols, SCATTER)


_SLOTS = 6


def _row_width(buf: torch.Tensor) -> int:
    """Flattened per-row element count (trailing dims), 1 for a 1-D buffer."""
    return math.prod(buf.shape[1:])


def _empty_like_rows(buf: torch.Tensor, m: int) -> torch.Tensor:
    """Output buffer for `m` gathered rows of `buf` (same trailing dims/dtype/device)."""
    return torch.empty((m, *buf.shape[1:]), dtype=buf.dtype, device=buf.device)


def _launch_rows(
    indices: torch.Tensor,
    pairs: Sequence[Optional[tuple[torch.Tensor, torch.Tensor]]],
    scatter: bool,
) -> None:
    """One launch moving row r of every (src, dst) pair: dst[r] = src[indices[r]], or
    with ``scatter`` dst[indices[r]] = src[r]. A pair may be None (slot unused)."""
    m = indices.shape[0]
    pairs = list(pairs) + [None] * (_SLOTS - len(pairs))
    widths = [_row_width(p[0]) for p in pairs if p is not None]
    if m == 0 or not widths:
        return
    args = []
    for p in pairs:
        if p is None:
            # Dummy operands for a disabled slot: the pointers must be valid even
            # though the kernel never dereferences them (gated off by HAS<k>).
            args += [indices, indices, 1]
        else:
            src, dst = p
            n = _row_width(src)
            if n != _row_width(dst):
                raise ValueError(
                    f"row widths differ: {tuple(src.shape)} {tuple(dst.shape)}"
                )
            # the operand indexed by row (the scatter's source, the gather's output)
            rows = src if scatter else dst
            if rows.shape[0] < m:
                raise ValueError(f"{rows.shape[0]} rows for {m} indices")
            args += [src, dst, n]
    max_n = max(widths)
    block = min(1024, triton.next_power_of_2(max_n))
    _rows_kernel[(m, triton.cdiv(max_n, block))](
        indices,
        *args,
        *(p is not None for p in pairs),
        SCATTER=scatter,
        BLOCK=block,
    )


def gather_spec_extras(
    indices: torch.Tensor,
    topk_p_buf: torch.Tensor,
    topk_index_buf: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    hidden_states_buf: Optional[torch.Tensor],
    dsa_topk_indices_buf: Optional[torch.Tensor] = None,
):
    """Gather spec extras (topk_p / topk_index / bonus_tokens / optional hidden
    states / optional DSA top-k seed) by a shared row index in a single fused Triton
    launch (one kernel for all buffers) instead of one advanced-index gather per
    buffer. A buffer that is None gives None."""
    # Source buffers are allocated once (torch.empty/full) and only ever mutated
    # in place, so they are guaranteed row-contiguous. `indices` flows from
    # several producers (req_pool_indices, filtered/merged future_indices); the
    # kernel addresses it linearly, so normalize layout here (no-op when already
    # contiguous) to avoid a silent wrong-result on a strided index tensor.
    indices = indices.contiguous()
    m = indices.shape[0]
    bufs = (
        topk_p_buf,
        topk_index_buf,
        output_tokens_buf,
        hidden_states_buf,
        dsa_topk_indices_buf,
    )
    outs = tuple(None if b is None else _empty_like_rows(b, m) for b in bufs)
    _launch_rows(
        indices,
        [None if b is None else (b, o) for b, o in zip(bufs, outs)],
        scatter=False,
    )
    return outs


def scatter_spec_extras(
    indices: torch.Tensor,
    pairs: Sequence[Optional[tuple[torch.Tensor, torch.Tensor]]],
) -> None:
    """``buf[indices] = rows.to(buf.dtype)`` for every (rows, buf) pair (at most 6,
    None skipped) in one launch: the relay's stash of a step's spec extras. ``buf``
    rows are contiguous; ``rows`` is made so (a no-op when it already is). Indices
    must be distinct, as an index_put of the same rows would need them to be."""
    if not indices.is_cuda:
        # no Triton launch off the GPU: the same writes, one index_put each
        for p in pairs:
            if p is not None:
                p[1][indices] = p[0].to(p[1].dtype)
        return
    indices = indices.contiguous()
    _launch_rows(
        indices,
        [None if p is None else (p[0].contiguous(), p[1]) for p in pairs],
        scatter=True,
    )
