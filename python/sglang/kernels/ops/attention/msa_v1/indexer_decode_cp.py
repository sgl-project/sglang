"""Context-parallel MSA-v1 decode indexer: exchange kernels around the stock scorer.

Rank r owns index heads [r * HL, (r + 1) * HL) and scores every head over its
page share: global page p belongs to rank (p // G) % CP, at compact column
(p // (CP * G)) * G + p % G. Scores are scattered back to the head owner's
full-width row, so the owner runs the stock top-k on bit-identical inputs.

Symmetric buffers, per rank: a two-slot query staging area [2, rows, 4 * 128] e4m3
and the compact score receive buffer [CP - 1, HL, rows, W] fp32. In both the
payload is the arrival flag (0xFF bytes, 0xFFFFFFFF words mark a slot not yet
written) and the consumer re-arms what it read, so no fence or host sync is needed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_PAGE = tl.constexpr(128)
_HEAD_DIM = 128
NUM_INDEX_HEADS = 4
# Pages per contiguous rank chunk; 8 fp32 scores fill one 32-byte sector.
PAGE_GRANULE = 8
# Receive-buffer word not yet written by the peer; scores never carry it.
SENTINEL = 0xFFFFFFFF
_SENTINEL = tl.constexpr(SENTINEL)
_CANONICAL_NAN = tl.constexpr(0x7FFFFFFF)
_BLOCK = 1024


@triton.jit
def _peer_ptr(ptrs, rank, dtype: tl.constexpr):
    return tl.load(ptrs + rank).to(tl.pointer_type(dtype))


@triton.jit
def _rank_page_count(n, CP: tl.constexpr, rank, G: tl.constexpr):
    # Pages of `rank` among global pages [0, n).
    rem = n % (CP * G)
    return (n // (CP * G)) * G + tl.minimum(tl.maximum(rem - rank * G, 0), G)


@triton.jit
def _global_page(j, CP: tl.constexpr, rank, G: tl.constexpr):
    return (j // G) * (CP * G) + rank * G + j % G


@triton.jit
def _compact_page_table_kernel(
    page_table_ptr,  # [B, P] int32
    final_seq_lens_ptr,  # [B] int32, including the query rows
    compact_table_ptr,  # [B, W] int32 out
    compact_seq_lens_ptr,  # [B] int32 out
    P,
    W,
    QLEN: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    seq = tl.load(final_seq_lens_ptr + b)
    # The stock scheduler's page count: pages before the last query's own page.
    hist = tl.minimum(tl.maximum((seq - 1) // _PAGE, 0), P)
    n_local = tl.minimum(_rank_page_count(hist, CP, RANK, G), W)
    for start in range(0, n_local, BLOCK):
        j = start + tl.arange(0, BLOCK)
        mask = j < n_local
        v = tl.load(
            page_table_ptr + b.to(tl.int64) * P + _global_page(j, CP, RANK, G),
            mask=mask,
        )
        tl.store(compact_table_ptr + b.to(tl.int64) * W + j, v, mask=mask)
    # Every query slot sits one page past the compact history, so the scorer
    # stores all n_local pages for every query; the causal cut is per row later.
    tl.store(compact_seq_lens_ptr + b, n_local * _PAGE + QLEN)


@triton.jit
def _q_exchange_kernel(
    q_ptr,  # [rows, HL, 128] e4m3 as uint8, this rank's heads
    q_row_stride,
    stage_ptrs,  # [CP] int64: [2, rows_max, H * 128] uint8 staging of every rank
    tile_ptr,  # [rows, H * 128] uint8, the scorer's query tile on this rank
    epoch_ptr,  # [1] int32, exchange calls so far
    stage_slot_bytes,
    H: tl.constexpr,
    HL: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
):
    t = tl.program_id(0)
    # A slot is rewritten two calls later, which needs this rank's next query,
    # sent only after this call re-armed it.
    slot = (tl.load(epoch_ptr) & 1).to(tl.int64) * stage_slot_bytes
    own = tl.arange(0, HL * 128)
    row = tl.load(q_ptr + t.to(tl.int64) * q_row_stride + own)
    # 0xFF (-NaN) is the not-yet-written flag; +NaN scores the same.
    row = tl.where(row == 0xFF, 0x7F, row).to(tl.uint8)
    offset = t.to(tl.int64) * (H * 128) + RANK * (HL * 128) + own
    for r in tl.static_range(CP):
        if r != RANK:
            tl.store(_peer_ptr(stage_ptrs, r, tl.uint8) + slot + offset, row)
    tl.store(tile_ptr + offset, row)
    full = tl.arange(0, H * 128)
    peer = full // (HL * 128) != RANK
    stage = _peer_ptr(stage_ptrs, RANK, tl.uint8) + slot + t.to(tl.int64) * (H * 128)
    v = tl.load(stage + full, volatile=True)
    pending = tl.max(((v == 0xFF) & peer).to(tl.int32), axis=0) > 0
    while pending:
        v = tl.load(stage + full, volatile=True)
        pending = tl.max(((v == 0xFF) & peer).to(tl.int32), axis=0) > 0
    tl.store(tile_ptr + t.to(tl.int64) * (H * 128) + full, v, mask=peer)
    tl.store(stage + full, tl.full(full.shape, 0xFF, tl.uint8), mask=peer)


@triton.jit
def _score_scatter_kernel(
    scores_ptr,  # [H, rows, W] fp32 compact scores of every head
    num_valid_pages_ptr,  # [rows] int32, the stock plan's (own page + 1)
    recv_ptrs,  # [CP] int64: [CP - 1, HL, rows, W] fp32 receive buffer per rank
    full_ptr,  # [rows, HL, P] fp32 full-width rows of this rank's heads
    epoch_ptr,  # [1] int32, bumped once every exchange of this call has read it
    rows,
    P,
    W,
    HL: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.load(num_valid_pages_ptr + t)
    # Rows of at most 16 pages never read scores (forced-tail top-k contract).
    if n > 16:
        count = _rank_page_count(n - 1, CP, RANK, G)
        src = scores_ptr + (h * rows + t).to(tl.int64) * W
        owner = h // HL
        for start in range(0, count, BLOCK):
            j = start + tl.arange(0, BLOCK)
            mask = j < count
            v = tl.load(src + j, mask=mask)
            if owner == RANK:
                dst = full_ptr + (t * HL + h % HL).to(tl.int64) * P
                tl.store(dst + _global_page(j, CP, RANK, G), v, mask=mask)
            else:
                # The payload is the arrival flag: keep real scores off the
                # sentinel bits (every NaN ranks alike in the top-k).
                bits = v.to(tl.uint32, bitcast=True)
                nan = tl.full(bits.shape, _CANONICAL_NAN, tl.uint32)
                v = tl.where(bits == _SENTINEL, nan, bits).to(tl.float32, bitcast=True)
                slot = RANK - (RANK > owner).to(tl.int32)
                dst = (
                    _peer_ptr(recv_ptrs, owner, tl.float32)
                    + ((slot * HL + h % HL) * rows + t).to(tl.int64) * W
                )
                tl.store(dst + j, v, mask=mask)
    if t == 0 and h == 0:
        tl.store(epoch_ptr, tl.load(epoch_ptr) + 1)


@triton.jit
def _gather_scores_kernel(
    recv_ptr,  # [CP - 1, HL, rows, W] fp32, this rank's receive buffer
    full_ptr,  # [rows, HL, P] fp32
    num_valid_pages_ptr,  # [rows] int32
    rows,
    P,
    W,
    HL: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Wait for each peer's compact scores, place them in the full rows, re-arm."""
    hl = tl.program_id(0) // rows
    t = tl.program_id(0) % rows
    n = tl.load(num_valid_pages_ptr + t)
    if n > 16:
        dst = full_ptr + (t * HL + hl).to(tl.int64) * P
        for r in tl.static_range(CP):
            if r != RANK:
                count = _rank_page_count(n - 1, CP, r, G)
                slot = r - (r > RANK)
                src = recv_ptr + ((slot * HL + hl) * rows + t).to(tl.int64) * W
                for start in range(0, count, BLOCK):
                    j = start + tl.arange(0, BLOCK)
                    mask = j < count
                    bits = tl.load(src + j, mask=mask, other=0, volatile=True)
                    bits = bits.to(tl.uint32, bitcast=True)
                    pending = tl.max((bits == _SENTINEL).to(tl.int32), axis=0) > 0
                    while pending:
                        bits = tl.load(src + j, mask=mask, other=0, volatile=True)
                        bits = bits.to(tl.uint32, bitcast=True)
                        pending = tl.max((bits == _SENTINEL).to(tl.int32), axis=0) > 0
                    tl.store(
                        dst + _global_page(j, CP, r, G),
                        bits.to(tl.float32, bitcast=True),
                        mask=mask,
                    )
                    sentinel = tl.full(j.shape, _SENTINEL, tl.uint32)
                    tl.store(src + j, sentinel.to(tl.float32, bitcast=True), mask=mask)


def compact_page_table(
    *,
    page_table: torch.Tensor,
    final_seq_lens: torch.Tensor,
    compact_table: torch.Tensor,
    compact_seq_lens: torch.Tensor,
    query_length: int,
    cp_size: int,
    cp_rank: int,
) -> None:
    batch, max_pages = page_table.shape
    _compact_page_table_kernel[(batch,)](
        page_table,
        final_seq_lens,
        compact_table,
        compact_seq_lens,
        max_pages,
        compact_table.shape[1],
        QLEN=query_length,
        CP=cp_size,
        RANK=cp_rank,
        G=PAGE_GRANULE,
        BLOCK=_BLOCK,
    )


def exchange_index_q(
    *,
    q: torch.Tensor,
    stage_ptrs: torch.Tensor,
    stage_slot_bytes: int,
    tile: torch.Tensor,
    epoch: torch.Tensor,
    cp_size: int,
    cp_rank: int,
) -> None:
    rows, heads_local, dim = q.shape
    assert dim == _HEAD_DIM and q.stride(2) == 1 and q.stride(1) == _HEAD_DIM
    _q_exchange_kernel[(rows,)](
        q.view(torch.uint8),
        q.stride(0),
        stage_ptrs,
        tile,
        epoch,
        stage_slot_bytes,
        H=NUM_INDEX_HEADS,
        HL=heads_local,
        CP=cp_size,
        RANK=cp_rank,
        num_warps=1,
    )


def scatter_scores(
    *,
    scores: torch.Tensor,
    num_valid_pages: torch.Tensor,
    recv_ptrs: torch.Tensor,
    full_scores: torch.Tensor,
    epoch: torch.Tensor,
    cp_size: int,
    cp_rank: int,
) -> None:
    heads, rows, width = scores.shape
    _score_scatter_kernel[(rows, heads)](
        scores,
        num_valid_pages,
        recv_ptrs,
        full_scores,
        epoch,
        rows,
        full_scores.shape[-1],
        width,
        HL=heads // cp_size,
        CP=cp_size,
        RANK=cp_rank,
        G=PAGE_GRANULE,
        BLOCK=_BLOCK,
        num_warps=4,
    )


def gather_scores(
    *,
    recv: torch.Tensor,
    full_scores: torch.Tensor,
    num_valid_pages: torch.Tensor,
    width: int,
    cp_size: int,
    cp_rank: int,
) -> None:
    rows, heads_local, max_pages = full_scores.shape
    _gather_scores_kernel[(heads_local * rows,)](
        recv,
        full_scores,
        num_valid_pages,
        rows,
        max_pages,
        width,
        HL=heads_local,
        CP=cp_size,
        RANK=cp_rank,
        G=PAGE_GRANULE,
        BLOCK=_BLOCK,
        num_warps=4,
    )
