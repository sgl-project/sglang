"""Fused schedule builder for the DeepSeek-V4 FP4 indexer on HIP.

AITER's ``compute_prefill_schedule`` derives the persistent-grid schedule with
~27 small torch ops before it launches ``_prefill_cta_info_kernel``, and the
sglang side adds a page-table pad, a ``row_to_batch`` arange and a
``local_starts`` zero-fill on top. Every one of those is a few thousand
elements, so the whole preamble is pure launch latency. On a 128k/1k conc-64
DP8TP8 MTP trace it measured 33 dispatches and 146us per C4 layer per step --
11.5% of all GPU kernel time -- against an 8.6us logits kernel.

Everything the preamble computes is a reduction or a scan over the row count,
so it collapses into a single kernel; the only real dependency is that
``_prefill_cta_info_kernel`` needs the finished prefix sums. That leaves two
dispatches for the whole schedule.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_KV_BLOCK_SIZE = 64

# Rows are query tokens: tens to a few hundred for MTP target-verify, up to the
# chunked-prefill size for EXTEND. The prep kernel holds one row per lane, so
# past this it hands back to AITER's torch preamble -- a prefill that wide is
# compute-bound and does not care about ~30 extra launches.
MAX_FUSED_ROWS = 4096

# Columns per page-table tile. The padded table is the row's page count rounded
# up to a multiple of 4, plus 4 more for the scheduler's one-chunk lookahead.
_PT_BLOCK = 256


@triton.jit
def _pad_page_table_tile(
    src_ptr,
    dst_ptr,
    src_stride,
    dst_stride,
    row,
    tile,
    w_src,
    w_dst,
    PT_BLOCK: tl.constexpr,
):
    """Copy one tile of a page-table row, zero-filling the scheduling pad."""
    cols = tile * PT_BLOCK + tl.arange(0, PT_BLOCK)
    inside = cols < w_dst
    vals = tl.load(
        src_ptr + row * src_stride + cols,
        mask=inside & (cols < w_src),
        other=0,
    )
    tl.store(dst_ptr + row * dst_stride + cols, vals, mask=inside)


@triton.jit
def _prefill_schedule_prep_kernel(
    le_ptr,  # [T] int32  local_ends (== c4_seq_lens)
    chunks_ptr,  # [T] int32  out
    incl_ptr,  # [T] int32  out, inclusive prefix sum of per-row CTA counts
    excl_ptr,  # [T] int32  out, exclusive prefix sum
    rb_ptr,  # [T] int32  out, row_to_batch
    ls_ptr,  # [T] int32  out, local_starts
    scalars_ptr,  # [2] int32  out, [safe, total_splits]
    pt_src_ptr,  # [T, w_src] int32
    pt_dst_ptr,  # [T, w_dst] int32
    pt_src_stride,
    pt_dst_stride,
    T,
    P,
    s_max,
    w_src,
    w_dst,
    pt_tiles_per_row,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    PT_BLOCK: tl.constexpr,
):
    """Whole FP4 prefill schedule preamble, plus the page-table pad.

    Program 0 owns the schedule (a handful of reductions and one scan over the
    rows); the rest pad the page table. The two halves write disjoint buffers
    and neither reads the other, so they ride in one dispatch.
    """
    pid = tl.program_id(0)

    if pid == 0:
        off = tl.arange(0, BLOCK_T)
        live = off < T
        le = tl.load(le_ptr + off, mask=live, other=0)
        # ceil(le / block_k); a non-positive length contributes no chunks, which
        # matches the reference's clamp(floor_div(le + block_k - 1), min=0).
        chunks = tl.where(live, (tl.maximum(le, 0) + (BLOCK_K - 1)) // BLOCK_K, 0)

        # A split factor s fits when the persistent grid can host every
        # (row, split) pair: sum_i ceil(chunks_i / s) <= P. ceil(c/s) is
        # non-increasing in s, so the sum is too and the smallest fitting s is
        # a binary search instead of the reference's [s_max, T] materialization.
        any_fits = tl.sum((chunks + (s_max - 1)) // s_max) <= P
        lo = 1
        hi = s_max
        for _ in tl.static_range(32):
            searching = lo < hi
            mid = tl.where(searching, (lo + hi) // 2, lo)
            fits = tl.sum((chunks + (mid - 1)) // mid) <= P
            lo = tl.where(searching & (fits == 0), mid + 1, lo)
            hi = tl.where(searching & fits, mid, hi)
        # No s fits: give every row its own CTA and let the grid clip, matching
        # the reference's max_chunks fallback.
        safe = tl.where(any_fits, lo, tl.maximum(tl.max(chunks), 1)).to(tl.int32)

        ctas_r = (chunks + (safe - 1)) // safe
        incl = tl.cumsum(ctas_r, axis=0)

        tl.store(chunks_ptr + off, chunks, mask=live)
        tl.store(incl_ptr + off, incl, mask=live)
        tl.store(excl_ptr + off, incl - ctas_r, mask=live)
        # sglang schedules one row per query token over the row's whole window,
        # so row_to_batch is the identity and every local start is 0.
        tl.store(rb_ptr + off, off.to(tl.int32), mask=live)
        tl.store(ls_ptr + off, tl.zeros([BLOCK_T], tl.int32), mask=live)
        # Rows past T contribute 0, so the scan is flat there and its max is
        # incl[T - 1].
        tl.store(scalars_ptr, safe)
        tl.store(scalars_ptr + 1, tl.max(incl).to(tl.int32))
    else:
        job = pid - 1
        _pad_page_table_tile(
            pt_src_ptr,
            pt_dst_ptr,
            pt_src_stride,
            pt_dst_stride,
            job // pt_tiles_per_row,
            job % pt_tiles_per_row,
            w_src,
            w_dst,
            PT_BLOCK,
        )


@triton.jit
def _pad_page_table_kernel(
    src_ptr,
    dst_ptr,
    src_stride,
    dst_stride,
    w_src,
    w_dst,
    PT_BLOCK: tl.constexpr,
):
    """Standalone page-table pad for callers with no schedule to build."""
    _pad_page_table_tile(
        src_ptr,
        dst_ptr,
        src_stride,
        dst_stride,
        tl.program_id(0),
        tl.program_id(1),
        w_src,
        w_dst,
        PT_BLOCK,
    )


def padded_page_table_shape(page_table: torch.Tensor) -> Tuple[int, int, int]:
    """Rows, logical width, and the 256-token-scheduling padded width.

    The padding rule is shared with the logits sizing in ``fp4_indexer_hip``;
    imported lazily because that module reaches back into this one.
    """
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import _guarded_pages

    rows, logical_width = page_table.shape
    return rows, logical_width, _guarded_pages(logical_width)


def _as_int32_2d(page_table: torch.Tensor) -> torch.Tensor:
    """Make the page table int32 with a unit column stride, without a copy."""
    if page_table.dtype is not torch.int32:
        page_table = page_table.to(torch.int32)
    if page_table.stride(1) != 1:
        page_table = page_table.contiguous()
    return page_table


def pad_page_table(
    page_table: torch.Tensor, out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, int]:
    """Pad a page table for 256-token scheduling in a single dispatch.

    Replaces the ``new_zeros`` + masked ``copy_`` pair: the kernel writes every
    output element, so the destination never needs pre-zeroing.
    """
    page_table = _as_int32_2d(page_table)
    rows, logical_width, padded_width = padded_page_table_shape(page_table)
    if out is None:
        out = page_table.new_empty((rows, padded_width + 4))
    else:
        assert out.shape == (rows, padded_width + 4), f"{out.shape=} {rows=}"
    w_dst = padded_width + 4
    if rows:
        _pad_page_table_kernel[(rows, triton.cdiv(w_dst, _PT_BLOCK))](
            page_table,
            out,
            page_table.stride(0),
            out.stride(0),
            logical_width,
            w_dst,
            PT_BLOCK=_PT_BLOCK,
        )
    return out, padded_width * _KV_BLOCK_SIZE


class PrefillScheduleBuffers:
    """Row-indexed scratch the prep kernel fills and the cta_info kernel reads.

    One allocation, sliced into views, so refreshing the schedule costs no
    dispatches beyond the kernels themselves. Past ``MAX_FUSED_ROWS`` only the
    row metadata is needed, because AITER's preamble owns the prefix sums.
    """

    __slots__ = (
        "fused",
        "storage",
        "chunks",
        "incl",
        "excl",
        "row_to_batch",
        "local_starts",
        "safe",
        "total_splits",
    )

    def __init__(self, rows: int, device: torch.device):
        self.fused = rows <= MAX_FUSED_ROWS
        if not self.fused:
            self.storage = torch.empty(2 * rows, dtype=torch.int32, device=device)
            # Identity rows over their whole window; constant across refreshes,
            # so these two dispatches happen once per buffer, not per build.
            self.row_to_batch = self.storage[0:rows]
            self.local_starts = self.storage[rows : 2 * rows]
            torch.arange(rows, dtype=torch.int32, device=device, out=self.row_to_batch)
            self.local_starts.zero_()
            self.chunks = self.incl = self.excl = None
            self.safe = self.total_splits = None
            return
        self.storage = torch.empty(5 * rows + 2, dtype=torch.int32, device=device)
        self.chunks = self.storage[0:rows]
        self.incl = self.storage[rows : 2 * rows]
        self.excl = self.storage[2 * rows : 3 * rows]
        self.row_to_batch = self.storage[3 * rows : 4 * rows]
        self.local_starts = self.storage[4 * rows : 5 * rows]
        self.safe = self.storage[5 * rows : 5 * rows + 1]
        self.total_splits = self.storage[5 * rows + 1 : 5 * rows + 2]


def build_prefill_schedule(
    *,
    page_table: torch.Tensor,
    local_ends: torch.Tensor,
    cta_info_out: torch.Tensor,
    parallel_unit_num: int,
    max_seq_len: int,
    block_k: int = 256,
    guarded_out: Optional[torch.Tensor] = None,
    buffers: Optional[PrefillScheduleBuffers] = None,
) -> Tuple[torch.Tensor, PrefillScheduleBuffers]:
    """Pad the page table and build the FP4 prefill schedule in two dispatches.

    Equivalent to ``_guard_page_table`` + an identity ``row_to_batch`` + a zero
    ``local_starts`` + AITER's ``compute_prefill_schedule``, writing the same
    ``cta_info`` rows through AITER's own ``_prefill_cta_info_kernel``.
    """
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        _prefill_cta_info_kernel,
        compute_prefill_schedule,
    )

    page_table = _as_int32_2d(page_table)
    rows, logical_width, padded_width = padded_page_table_shape(page_table)
    total_rows = local_ends.shape[0]
    assert total_rows <= rows, (
        f"local_ends rows {total_rows} exceed the page table's {rows}; the "
        "schedule kernel indexes both by row"
    )
    assert parallel_unit_num >= total_rows, (
        f"parallel_unit_num={parallel_unit_num} < rows={total_rows} would "
        "silently drop rows past the last slot"
    )

    if guarded_out is None:
        guarded_out = page_table.new_empty((rows, padded_width + 4))
    if buffers is None:
        buffers = PrefillScheduleBuffers(total_rows, page_table.device)

    w_dst = padded_width + 4
    pt_tiles_per_row = triton.cdiv(w_dst, _PT_BLOCK)

    if not buffers.fused:
        # Too many rows to hold one per lane; pad the table and let AITER's
        # torch preamble build the schedule.
        _pad_page_table_kernel[(rows, pt_tiles_per_row)](
            page_table,
            guarded_out,
            page_table.stride(0),
            guarded_out.stride(0),
            logical_width,
            w_dst,
            PT_BLOCK=_PT_BLOCK,
        )
        compute_prefill_schedule(
            buffers.row_to_batch,
            buffers.local_starts,
            local_ends,
            block_k=block_k,
            parallel_unit_num=parallel_unit_num,
            max_seq_len=max_seq_len,
            cta_info_out=cta_info_out,
        )
        return guarded_out, buffers

    _prefill_schedule_prep_kernel[(1 + rows * pt_tiles_per_row,)](
        local_ends,
        buffers.chunks,
        buffers.incl,
        buffers.excl,
        buffers.row_to_batch,
        buffers.local_starts,
        buffers.safe,
        page_table,
        guarded_out,
        page_table.stride(0),
        guarded_out.stride(0),
        total_rows,
        parallel_unit_num,
        max(1, (max_seq_len + block_k - 1) // block_k),
        logical_width,
        w_dst,
        pt_tiles_per_row,
        BLOCK_K=block_k,
        BLOCK_T=max(16, triton.next_power_of_2(total_rows)),
        PT_BLOCK=_PT_BLOCK,
    )

    BLOCK_P = 256
    _prefill_cta_info_kernel[(triton.cdiv(parallel_unit_num, BLOCK_P),)](
        buffers.incl,
        buffers.excl,
        buffers.chunks,
        buffers.row_to_batch,
        buffers.local_starts,
        local_ends,
        buffers.safe,
        buffers.total_splits,
        cta_info_out,
        total_rows,
        parallel_unit_num,
        BLOCK_P=BLOCK_P,
    )
    return guarded_out, buffers
