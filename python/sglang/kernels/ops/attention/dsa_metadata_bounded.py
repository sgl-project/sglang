"""DSA replay metadata with a live-length-bounded, data-independent grid.

Drop-in variants of ``dsa.metadata.fused_dsa_{decode,target_verify,
draft_extend}_metadata``. Those launch ``1 + rows * cdiv(width, 128)``
programs, where ``width`` is the page-table width: ``--context-length``
(1M tokens for GLM-5.x) rounded up. Almost every program only loads its
row's kv length and exits, yet launching ~8K programs per row costs
5-6 us per decode row and 100 us for 24 target-verify rows on GB300.

Here each request row gets ``num_splits`` programs (bounded by a program
budget shared by the rows, a per-request cap and the width), and every
program strides over that row's *live* column blocks, read from ``seq_lens``
on device. The grid depends only on (rows, width, budget, cap), so the launch
stays CUDA-graph safe; the traversal depends on the replay-time kv lengths.

The set of written elements is identical to the unbounded kernels': column
block ``cb`` of a row is written iff ``cb * 128 < kv_len`` and
``cb < cdiv(width, 128)``, with the same masks and values. Target verify
additionally loads each request's ``req_to_token`` block once and stores it
to the request's ``next_n`` expanded rows.

``fused_dsa_decode_metadata_multi`` writes the decode metadata of up to four
CUDA-graph metadata sets (the per-step backends of a multi-step MTP draft)
in one launch, instead of a precompute into a scratch table plus full-width
copies.
"""

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.metadata import fused_dsa_draft_extend_metadata

_BLOCK_N = 128
# One wave of 4-warp programs on a 148-160 SM Blackwell part (16 resident per
# SM). Rows beyond 16 per pass loop; idle programs exit after two loads.
DEFAULT_PROGRAM_BUDGET = 2048
# Programs per request at most (0: no cap). 256 programs cover 32K tokens per
# pass; at 1k-8k contexts a request has 8-72 live blocks, so a small batch
# would otherwise launch up to ~2K programs that only load seq_lens and exit.
DEFAULT_MAX_SPLITS_PER_ROW = 256
# Destination metadata sets per multi-destination launch.
MAX_MULTI_DSTS = 4


def bounded_num_splits(
    row_units: int,
    width: int,
    program_budget: int,
    max_splits_per_row: int = DEFAULT_MAX_SPLITS_PER_ROW,
) -> int:
    """Programs per row: ``budget // rows``, at most ``max_splits_per_row``
    (when > 0) and the width blocks, at least 1."""
    splits = max(1, program_budget) // max(1, row_units)
    if max_splits_per_row > 0:
        splits = min(splits, max_splits_per_row)
    return max(1, min(triton.cdiv(width, _BLOCK_N), splits))


@triton.jit
def _store_seqlens(
    cache_seqlens,
    cu_seqlens_k,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    offs,
    mask,
    seq_i32,
    cu,
    dsa_seq,
    dsa_cu,
):
    tl.store(cache_seqlens + offs, seq_i32, mask=mask)
    tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
    tl.store(cu_seqlens_k + 1 + offs, cu, mask=mask)
    tl.store(dsa_cache_seqlens + offs, dsa_seq, mask=mask)
    tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
    tl.store(dsa_cu_seqlens_k + 1 + offs, dsa_cu, mask=mask)


@triton.jit
def _store_page_block(
    page_table_1,
    real_page_table,
    rows_i64,
    offs_n,
    vals,
    mask,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    physical_page_size: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    HAS_PAGE_TABLE_1: tl.constexpr,
):
    # rows_i64 / offs_n / vals / mask broadcast against each other: 1D for one
    # output row, [rows, 1] x [1, cols] for a request's expanded rows.
    if HAS_PAGE_TABLE_1:
        tl.store(
            page_table_1
            + rows_i64 * page_table_stride_0
            + offs_n * page_table_stride_1,
            vals,
            mask=mask,
        )
    if HAS_REAL_PAGE_TABLE:
        real_mask = mask & ((offs_n % physical_page_size) == 0)
        real_cols = offs_n // physical_page_size
        tl.store(
            real_page_table
            + rows_i64 * real_page_table_stride_0
            + real_cols * real_page_table_stride_1,
            vals // physical_page_size,
            mask=real_mask,
        )


# ---------------------------------------------------------------------------
# Decode (one output row per request), 1..4 destination metadata sets
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "max_len",
        "num_splits",
    ]
)
def _fused_dsa_decode_metadata_bounded_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens_0,
    cu_seqlens_k_0,
    page_table_1_0,
    dsa_cache_seqlens_0,
    dsa_cu_seqlens_k_0,
    real_page_table_0,
    cache_seqlens_1,
    cu_seqlens_k_1,
    page_table_1_1,
    dsa_cache_seqlens_1,
    dsa_cu_seqlens_k_1,
    real_page_table_1,
    cache_seqlens_2,
    cu_seqlens_k_2,
    page_table_1_2,
    dsa_cache_seqlens_2,
    dsa_cu_seqlens_k_2,
    real_page_table_2,
    cache_seqlens_3,
    cu_seqlens_k_3,
    page_table_1_3,
    dsa_cache_seqlens_3,
    dsa_cu_seqlens_k_3,
    real_page_table_3,
    seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    bs: tl.constexpr,
    max_len,
    num_splits,
    dsa_index_topk: tl.constexpr,
    physical_page_size: tl.constexpr,
    NUM_DST: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    HAS_PAGE_TABLE_1: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid == 0:
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        seq_i32 = seq.to(tl.int32)
        dsa_seq = tl.minimum(seq_i32, dsa_index_topk)
        cu = tl.cumsum(seq_i32, 0)
        dsa_cu = tl.cumsum(dsa_seq, 0)
        _store_seqlens(
            cache_seqlens_0,
            cu_seqlens_k_0,
            dsa_cache_seqlens_0,
            dsa_cu_seqlens_k_0,
            offs_b,
            mask_b,
            seq_i32,
            cu,
            dsa_seq,
            dsa_cu,
        )
        if NUM_DST > 1:
            _store_seqlens(
                cache_seqlens_1,
                cu_seqlens_k_1,
                dsa_cache_seqlens_1,
                dsa_cu_seqlens_k_1,
                offs_b,
                mask_b,
                seq_i32,
                cu,
                dsa_seq,
                dsa_cu,
            )
        if NUM_DST > 2:
            _store_seqlens(
                cache_seqlens_2,
                cu_seqlens_k_2,
                dsa_cache_seqlens_2,
                dsa_cu_seqlens_k_2,
                offs_b,
                mask_b,
                seq_i32,
                cu,
                dsa_seq,
                dsa_cu,
            )
        if NUM_DST > 3:
            _store_seqlens(
                cache_seqlens_3,
                cu_seqlens_k_3,
                dsa_cache_seqlens_3,
                dsa_cu_seqlens_k_3,
                offs_b,
                mask_b,
                seq_i32,
                cu,
                dsa_seq,
                dsa_cu,
            )
        return

    page_pid = pid - 1
    row = page_pid // num_splits
    split_id = page_pid - row * num_splits
    row_ok = row < bs

    req_idx = tl.load(
        req_pool_indices + row * req_pool_indices_stride, mask=row_ok, other=0
    )
    kv_len = tl.load(seq_lens + row * seq_lens_stride, mask=row_ok, other=0).to(
        tl.int32
    )
    # Same block set as the unbounded kernel: cb * BLOCK_N < kv_len, inside the width.
    num_live_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_N), tl.cdiv(max_len, BLOCK_N))
    row_i64 = row.to(tl.int64)
    for col_block in tl.range(
        split_id, num_live_blocks, num_splits, num_stages=NUM_STAGES
    ):
        offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = row_ok & (offs_n < max_len)
        vals = tl.load(
            req_to_token
            + req_idx * req_to_token_stride_0
            + offs_n * req_to_token_stride_1,
            mask=mask,
            other=0,
        ).to(tl.int32)
        _store_page_block(
            page_table_1_0,
            real_page_table_0,
            row_i64,
            offs_n,
            vals,
            mask,
            page_table_stride_0,
            page_table_stride_1,
            real_page_table_stride_0,
            real_page_table_stride_1,
            physical_page_size,
            HAS_REAL_PAGE_TABLE,
            HAS_PAGE_TABLE_1,
        )
        if NUM_DST > 1:
            _store_page_block(
                page_table_1_1,
                real_page_table_1,
                row_i64,
                offs_n,
                vals,
                mask,
                page_table_stride_0,
                page_table_stride_1,
                real_page_table_stride_0,
                real_page_table_stride_1,
                physical_page_size,
                HAS_REAL_PAGE_TABLE,
                HAS_PAGE_TABLE_1,
            )
        if NUM_DST > 2:
            _store_page_block(
                page_table_1_2,
                real_page_table_2,
                row_i64,
                offs_n,
                vals,
                mask,
                page_table_stride_0,
                page_table_stride_1,
                real_page_table_stride_0,
                real_page_table_stride_1,
                physical_page_size,
                HAS_REAL_PAGE_TABLE,
                HAS_PAGE_TABLE_1,
            )
        if NUM_DST > 3:
            _store_page_block(
                page_table_1_3,
                real_page_table_3,
                row_i64,
                offs_n,
                vals,
                mask,
                page_table_stride_0,
                page_table_stride_1,
                real_page_table_stride_0,
                real_page_table_stride_1,
                physical_page_size,
                HAS_REAL_PAGE_TABLE,
                HAS_PAGE_TABLE_1,
            )


def _resolve_tables(
    page_table_1: Optional[torch.Tensor],
    real_page_table: Optional[torch.Tensor],
    physical_page_size: int,
):
    """Same table contract as dsa_metadata: page_size == 1 writes page_table_1
    only (real IS page_table_1); page_table_1 may be dropped when page_size > 1."""
    has_real_page_table = physical_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
    else:
        assert page_table_1 is not None
        real_page_table = page_table_1
    has_page_table_1 = page_table_1 is not None
    if not has_page_table_1:
        assert has_real_page_table
        page_table_1 = real_page_table  # dummy pointer for stride args
    return page_table_1, real_page_table, has_real_page_table, has_page_table_1


def fused_dsa_decode_metadata_multi(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    dsts: Sequence[tuple],
    bs: int,
    max_len: int,
    dsa_index_topk: int,
    physical_page_size: int,
    program_budget: int = DEFAULT_PROGRAM_BUDGET,
    max_splits_per_row: int = DEFAULT_MAX_SPLITS_PER_ROW,
    num_stages: int = 3,
) -> None:
    """Decode metadata of ``fused_dsa_decode_metadata`` into every destination.

    ``dsts`` holds 1..4 tuples ``(cache_seqlens, cu_seqlens_k, page_table_1,
    dsa_cache_seqlens, dsa_cu_seqlens_k, real_page_table)``; all page tables
    share strides. Each page-table row is written over its live prefix only
    (column blocks of 128 below ``seq_lens[row]``); the tail keeps stale values,
    as with the unbounded kernel, so consumers must bound reads by cache_seqlens.
    """
    assert 1 <= len(dsts) <= MAX_MULTI_DSTS
    if bs == 0:
        for cache_seqlens, cu_seqlens_k, _, _, dsa_cu_seqlens_k, _ in dsts:
            cu_seqlens_k[:1].zero_()
            dsa_cu_seqlens_k[:1].zero_()
        return

    resolved = []
    table_flags = set()
    for (
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
    ) in dsts:
        pt1, rpt, has_real, has_pt1 = _resolve_tables(
            page_table_1, real_page_table, physical_page_size
        )
        table_flags.add((has_real, has_pt1))
        resolved.append(
            (cache_seqlens, cu_seqlens_k, pt1, dsa_cache_seqlens, dsa_cu_seqlens_k, rpt)
        )
    assert len(table_flags) == 1, "destinations disagree on which tables exist"
    first_pt1, first_rpt = resolved[0][2], resolved[0][5]
    for d in resolved[1:]:
        assert d[2].stride() == first_pt1.stride(), "page_table_1 strides differ"
        assert d[5].stride() == first_rpt.stride(), "real_page_table strides differ"
    while len(resolved) < MAX_MULTI_DSTS:
        resolved.append(resolved[0])  # unused slots: valid dummy pointers

    num_splits = bounded_num_splits(bs, max_len, program_budget, max_splits_per_row)
    grid = (1 + bs * num_splits,)
    flat = [t for d in resolved for t in d]
    _fused_dsa_decode_metadata_bounded_kernel[grid](
        seq_lens,
        req_pool_indices,
        req_to_token,
        *flat,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        first_pt1.stride(0),
        first_pt1.stride(1),
        first_rpt.stride(0) if has_real else 0,
        first_rpt.stride(1) if has_real else 0,
        bs,
        max_len,
        num_splits,
        dsa_index_topk,
        physical_page_size,
        len(dsts),
        has_real,
        has_pt1,
        BLOCK_BS=triton.next_power_of_2(bs),
        BLOCK_N=_BLOCK_N,
        NUM_STAGES=num_stages,
    )


def fused_dsa_decode_metadata_bounded(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    max_len: int,
    dsa_index_topk: int,
    physical_page_size: int,
    program_budget: int = DEFAULT_PROGRAM_BUDGET,
    max_splits_per_row: int = DEFAULT_MAX_SPLITS_PER_ROW,
    num_stages: int = 3,
) -> None:
    """``fused_dsa_decode_metadata`` with a live-length-bounded grid."""
    fused_dsa_decode_metadata_multi(
        seq_lens,
        req_pool_indices,
        req_to_token,
        [
            (
                cache_seqlens,
                cu_seqlens_k,
                page_table_1,
                dsa_cache_seqlens,
                dsa_cu_seqlens_k,
                real_page_table,
            )
        ],
        bs=bs,
        max_len=max_len,
        dsa_index_topk=dsa_index_topk,
        physical_page_size=physical_page_size,
        program_budget=program_budget,
        max_splits_per_row=max_splits_per_row,
        num_stages=num_stages,
    )


# ---------------------------------------------------------------------------
# Target verify (next_n expanded rows per request)
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "max_seqlen_k",
        "num_splits",
    ]
)
def _fused_dsa_target_verify_metadata_bounded_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table_1,
    seqlens_expanded,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
    paged_mqa_ctx_lens_2d,
    seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    paged_mqa_ctx_lens_stride_0: tl.constexpr,
    paged_mqa_ctx_lens_stride_1: tl.constexpr,
    bs: tl.constexpr,
    max_seqlen_k,
    num_splits,
    dsa_index_topk: tl.constexpr,
    physical_page_size: tl.constexpr,
    next_n: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    HAS_PAGED_MQA_CTX_LENS: tl.constexpr,
    HAS_PAGE_TABLE_1: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_EXPANDED: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    expanded_size: tl.constexpr = bs * next_n

    if pid == 0:
        # Identical to _fused_dsa_target_verify_metadata_kernel's program 0.
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        cache_seq = seq.to(tl.int32) + next_n
        cu = tl.cumsum(cache_seq, 0)

        tl.store(cache_seqlens + offs_b, cache_seq, mask=mask_b)
        tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)

        offs_e = tl.arange(0, BLOCK_EXPANDED)
        mask_e = offs_e < expanded_size
        req_row = offs_e // next_n
        draft_off = offs_e - req_row * next_n
        base_seq = tl.load(
            seq_lens + req_row * seq_lens_stride,
            mask=mask_e,
            other=0,
        ).to(tl.int32)
        expanded_seq = base_seq + draft_off + 1
        expanded_seq = tl.where(mask_e, expanded_seq, 0)
        dsa_seq = tl.minimum(expanded_seq, dsa_index_topk)
        dsa_cu = tl.cumsum(dsa_seq, 0)

        tl.store(seqlens_expanded + offs_e, expanded_seq, mask=mask_e)
        tl.store(dsa_cache_seqlens + offs_e, dsa_seq, mask=mask_e)
        tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(dsa_cu_seqlens_k + 1 + offs_e, dsa_cu, mask=mask_e)

        if HAS_PAGED_MQA_CTX_LENS:
            tl.store(
                paged_mqa_ctx_lens_2d
                + req_row * paged_mqa_ctx_lens_stride_0
                + draft_off * paged_mqa_ctx_lens_stride_1,
                base_seq + next_n,
                mask=mask_e,
            )
        return

    page_pid = pid - 1
    req_row = page_pid // num_splits
    split_id = page_pid - req_row * num_splits
    req_ok = req_row < bs

    req_idx = tl.load(
        req_pool_indices + req_row * req_pool_indices_stride, mask=req_ok, other=0
    )
    kv_len = (
        tl.load(seq_lens + req_row * seq_lens_stride, mask=req_ok, other=0).to(tl.int32)
        + next_n
    )
    num_live_blocks = tl.minimum(
        tl.cdiv(kv_len, BLOCK_N), tl.cdiv(max_seqlen_k, BLOCK_N)
    )
    # The request's next_n expanded rows all carry its page row.
    offs_r = tl.arange(0, BLOCK_R)
    out_rows = (req_row * next_n + offs_r).to(tl.int64)
    row_mask = req_ok & (offs_r < next_n)
    for col_block in tl.range(
        split_id, num_live_blocks, num_splits, num_stages=NUM_STAGES
    ):
        offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = offs_n < max_seqlen_k
        vals = tl.load(
            req_to_token
            + req_idx * req_to_token_stride_0
            + offs_n * req_to_token_stride_1,
            mask=req_ok & col_mask,
            other=0,
        ).to(tl.int32)
        _store_page_block(
            page_table_1,
            real_page_table,
            out_rows[:, None],
            offs_n[None, :],
            vals[None, :],
            row_mask[:, None] & col_mask[None, :],
            page_table_stride_0,
            page_table_stride_1,
            real_page_table_stride_0,
            real_page_table_stride_1,
            physical_page_size,
            HAS_REAL_PAGE_TABLE,
            HAS_PAGE_TABLE_1,
        )


def fused_dsa_target_verify_metadata_bounded(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    max_seqlen_k: int,
    dsa_index_topk: int,
    physical_page_size: int,
    next_n: int,
    paged_mqa_ctx_lens_2d: torch.Tensor = None,
    program_budget: int = DEFAULT_PROGRAM_BUDGET,
    max_splits_per_row: int = DEFAULT_MAX_SPLITS_PER_ROW,
    num_stages: int = 3,
) -> None:
    """``fused_dsa_target_verify_metadata`` with a live-length-bounded grid."""
    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return
    assert next_n > 0

    page_table_1, real_page_table, has_real_page_table, has_page_table_1 = (
        _resolve_tables(page_table_1, real_page_table, physical_page_size)
    )

    has_paged_mqa_ctx_lens = paged_mqa_ctx_lens_2d is not None
    if has_paged_mqa_ctx_lens:
        assert paged_mqa_ctx_lens_2d.dtype == torch.int32
        assert paged_mqa_ctx_lens_2d.dim() == 2
        assert paged_mqa_ctx_lens_2d.size(0) == bs
        assert paged_mqa_ctx_lens_2d.size(1) == next_n
    else:
        paged_mqa_ctx_lens_2d = page_table_1

    expanded_size = bs * next_n
    num_splits = bounded_num_splits(
        bs, max_seqlen_k, program_budget, max_splits_per_row
    )
    grid = (1 + bs * num_splits,)

    _fused_dsa_target_verify_metadata_bounded_kernel[grid](
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        seqlens_expanded,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        paged_mqa_ctx_lens_2d,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        paged_mqa_ctx_lens_2d.stride(0) if has_paged_mqa_ctx_lens else 0,
        paged_mqa_ctx_lens_2d.stride(1) if has_paged_mqa_ctx_lens else 0,
        bs,
        max_seqlen_k,
        num_splits,
        dsa_index_topk,
        physical_page_size,
        next_n,
        has_real_page_table,
        has_paged_mqa_ctx_lens,
        has_page_table_1,
        BLOCK_BS=triton.next_power_of_2(bs),
        BLOCK_EXPANDED=triton.next_power_of_2(expanded_size),
        BLOCK_R=triton.next_power_of_2(next_n),
        BLOCK_N=_BLOCK_N,
        NUM_STAGES=num_stages,
    )


# ---------------------------------------------------------------------------
# Draft extend, static per-request extend length (spec v2 replay)
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "total_len",
        "max_seqlen_k",
        "num_splits",
        "static_qo_len",
    ]
)
def _fused_dsa_draft_extend_metadata_bounded_kernel(
    seq_lens,
    extend_seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table_1,
    seqlens_expanded,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
    seq_lens_stride: tl.constexpr,
    extend_seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    bs: tl.constexpr,
    total_len,
    max_seqlen_k,
    num_splits,
    static_qo_len,
    dsa_index_topk: tl.constexpr,
    physical_page_size: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    HAS_PAGE_TABLE_1: tl.constexpr,
    HAS_EXTEND_SEQ_LENS: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_EXPANDED: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Static extend length only: every request extends by the same qo_len,
    # read from extend_seq_lens[0] / [row] (as the unbounded kernel does) or,
    # without the tensor, taken from static_qo_len.
    pid = tl.program_id(0)

    if pid == 0:
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        cache_seq = seq.to(tl.int32)
        cu = tl.cumsum(cache_seq, 0)

        tl.store(cache_seqlens + offs_b, cache_seq, mask=mask_b)
        tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)

        offs_e = tl.arange(0, BLOCK_EXPANDED)
        mask_e = offs_e < total_len
        if HAS_EXTEND_SEQ_LENS:
            qo0 = tl.load(extend_seq_lens).to(tl.int32)
        else:
            qo0 = tl.full((), 0, tl.int32) + static_qo_len
        req_row = offs_e // qo0
        local_off = offs_e - req_row * qo0
        qo_len_for_row = tl.zeros((BLOCK_EXPANDED,), tl.int32) + qo0

        base_seq = tl.load(
            seq_lens + req_row * seq_lens_stride,
            mask=mask_e,
            other=0,
        ).to(tl.int32)
        # Clamp to >= 0 for padded rows (seq_len fill value < qo_len); see
        # _fused_dsa_draft_extend_metadata_kernel.
        expanded_seq = base_seq - qo_len_for_row + local_off + 1
        expanded_seq = tl.maximum(expanded_seq, 0)
        expanded_seq = tl.where(mask_e, expanded_seq, 0)
        dsa_seq = tl.minimum(expanded_seq, dsa_index_topk)
        dsa_cu = tl.cumsum(dsa_seq, 0)

        tl.store(seqlens_expanded + offs_e, expanded_seq, mask=mask_e)
        tl.store(dsa_cache_seqlens + offs_e, dsa_seq, mask=mask_e)
        tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(dsa_cu_seqlens_k + 1 + offs_e, dsa_cu, mask=mask_e)
        return

    page_pid = pid - 1
    req_row = page_pid // num_splits
    split_id = page_pid - req_row * num_splits
    req_ok = req_row < bs

    if HAS_EXTEND_SEQ_LENS:
        qo_len = tl.load(
            extend_seq_lens + req_row * extend_seq_lens_stride,
            mask=req_ok,
            other=0,
        ).to(tl.int32)
    else:
        qo_len = tl.where(req_ok, static_qo_len, 0)
    kv_len = tl.load(seq_lens + req_row * seq_lens_stride, mask=req_ok, other=0).to(
        tl.int32
    )
    num_live_blocks = tl.minimum(
        tl.cdiv(kv_len, BLOCK_N), tl.cdiv(max_seqlen_k, BLOCK_N)
    )
    prefix = req_row * qo_len
    offs_r = tl.arange(0, BLOCK_ROWS)
    out_rows = prefix + offs_r
    row_mask = req_ok & (offs_r < qo_len) & (out_rows < total_len)
    has_rows = req_ok & (qo_len > 0)
    req_idx = tl.load(
        req_pool_indices + req_row * req_pool_indices_stride,
        mask=has_rows,
        other=0,
    )
    out_rows_i64 = out_rows.to(tl.int64)
    for col_block in tl.range(
        split_id, num_live_blocks, num_splits, num_stages=NUM_STAGES
    ):
        offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = offs_n < max_seqlen_k
        vals = tl.load(
            req_to_token
            + req_idx * req_to_token_stride_0
            + offs_n * req_to_token_stride_1,
            mask=col_mask & has_rows,
            other=0,
        ).to(tl.int32)
        _store_page_block(
            page_table_1,
            real_page_table,
            out_rows_i64[:, None],
            offs_n[None, :],
            vals[None, :],
            row_mask[:, None] & col_mask[None, :],
            page_table_stride_0,
            page_table_stride_1,
            real_page_table_stride_0,
            real_page_table_stride_1,
            physical_page_size,
            HAS_REAL_PAGE_TABLE,
            HAS_PAGE_TABLE_1,
        )


def fused_dsa_draft_extend_metadata_bounded(
    seq_lens: torch.Tensor,
    extend_seq_lens: Optional[torch.Tensor],
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    total_len: int,
    max_seqlen_k: int,
    dsa_index_topk: int,
    physical_page_size: int,
    max_extend_len: int,
    max_total_len: int,
    static_extend_len: bool = False,
    program_budget: int = DEFAULT_PROGRAM_BUDGET,
    max_splits_per_row: int = DEFAULT_MAX_SPLITS_PER_ROW,
    num_stages: int = 3,
) -> None:
    """``fused_dsa_draft_extend_metadata`` with a live-length-bounded grid.

    Bounded for ``static_extend_len`` (every request extends by
    ``extend_seq_lens[0]``, or by ``max_extend_len`` when ``extend_seq_lens``
    is None); ragged extend lengths take the unbounded kernel.
    """
    if not static_extend_len:
        assert extend_seq_lens is not None
        return fused_dsa_draft_extend_metadata(
            seq_lens,
            extend_seq_lens,
            req_pool_indices,
            req_to_token,
            cache_seqlens,
            cu_seqlens_k,
            page_table_1,
            seqlens_expanded,
            dsa_cache_seqlens,
            dsa_cu_seqlens_k,
            real_page_table,
            bs=bs,
            total_len=total_len,
            max_seqlen_k=max_seqlen_k,
            dsa_index_topk=dsa_index_topk,
            physical_page_size=physical_page_size,
            max_extend_len=max_extend_len,
            max_total_len=max_total_len,
            static_extend_len=False,
        )

    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return
    if total_len == 0:
        cache = seq_lens.to(torch.int32)
        cache_seqlens.copy_(cache)
        cu_seqlens_k[:1].zero_()
        cu_seqlens_k[1 : bs + 1].copy_(torch.cumsum(cache, dim=0, dtype=torch.int32))
        dsa_cu_seqlens_k[:1].zero_()
        return
    assert total_len <= max_total_len
    assert max_extend_len > 0
    assert total_len <= bs * max_extend_len

    page_table_1, real_page_table, has_real_page_table, has_page_table_1 = (
        _resolve_tables(page_table_1, real_page_table, physical_page_size)
    )
    has_extend_seq_lens = extend_seq_lens is not None
    extend_arg = extend_seq_lens if has_extend_seq_lens else seq_lens  # dummy

    num_splits = bounded_num_splits(
        bs, max_seqlen_k, program_budget, max_splits_per_row
    )
    grid = (1 + bs * num_splits,)
    _fused_dsa_draft_extend_metadata_bounded_kernel[grid](
        seq_lens,
        extend_arg,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        seqlens_expanded,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        seq_lens.stride(0),
        extend_arg.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        bs,
        total_len,
        max_seqlen_k,
        num_splits,
        max_extend_len,
        dsa_index_topk,
        physical_page_size,
        has_real_page_table,
        has_page_table_1,
        has_extend_seq_lens,
        BLOCK_BS=triton.next_power_of_2(bs),
        BLOCK_EXPANDED=triton.next_power_of_2(max_total_len),
        BLOCK_ROWS=triton.next_power_of_2(max_extend_len),
        BLOCK_N=_BLOCK_N,
        NUM_STAGES=num_stages,
    )
