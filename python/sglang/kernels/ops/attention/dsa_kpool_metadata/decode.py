"""Pool-aware fused DSA decode metadata."""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa_kpool_metadata.scan import bounded_scan_num_splits


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "max_len",
        "num_splits",
    ]
)
def _fused_dsa_decode_metadata_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table_1,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
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
    index_kpool: tl.constexpr,
    real_page_size: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    HAS_PAGE_TABLE_1: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid == 0:
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        seq_i32 = seq.to(tl.int32)
        if index_kpool <= 1:
            dsa_seq = tl.minimum(seq_i32, dsa_index_topk)
        else:
            # Preserve the live partial pool after selecting pool-aligned history.
            full_pool_tokens = (seq_i32 // index_kpool) * index_kpool
            selected_history_tokens = tl.minimum(full_pool_tokens, dsa_index_topk)
            tail_tokens = seq_i32 - full_pool_tokens
            dsa_seq = selected_history_tokens + tail_tokens

        cu = tl.cumsum(seq_i32, 0)
        dsa_cu = tl.cumsum(dsa_seq, 0)

        tl.store(cache_seqlens + offs_b, seq_i32, mask=mask_b)
        tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)
        tl.store(dsa_cache_seqlens + offs_b, dsa_seq, mask=mask_b)
        tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(dsa_cu_seqlens_k + 1 + offs_b, dsa_cu, mask=mask_b)
        return

    page_pid = pid - 1
    row = page_pid // num_splits
    split_id = page_pid - row * num_splits

    req_idx = tl.load(
        req_pool_indices + row * req_pool_indices_stride,
        mask=row < bs,
        other=0,
    )
    kv_len = tl.load(
        seq_lens + row * seq_lens_stride,
        mask=row < bs,
        other=0,
    ).to(tl.int32)
    # Page-table row offsets can overflow int32 at 1M context.
    row_i64 = row.to(tl.int64)
    num_live_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_N), tl.cdiv(max_len, BLOCK_N))
    # Three stages hide latency across strided copy iterations.
    for col_block in tl.range(split_id, num_live_blocks, num_splits, num_stages=3):
        offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (row < bs) & (offs_n < max_len)
        vals = tl.load(
            req_to_token
            + req_idx * req_to_token_stride_0
            + offs_n * req_to_token_stride_1,
            mask=mask,
            other=0,
        ).to(tl.int32)
        if HAS_PAGE_TABLE_1:
            tl.store(
                page_table_1
                + row_i64 * page_table_stride_0
                + offs_n * page_table_stride_1,
                vals,
                mask=mask,
            )

        if HAS_REAL_PAGE_TABLE:
            real_mask = mask & ((offs_n % real_page_size) == 0)
            real_cols = offs_n // real_page_size
            tl.store(
                real_page_table
                + row_i64 * real_page_table_stride_0
                + real_cols * real_page_table_stride_1,
                vals // real_page_size,
                mask=real_mask,
            )


def fused_dsa_decode_metadata(
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
    real_page_size: int,
    index_kpool: int = 1,
) -> None:
    """Fill decode-graph DSA metadata (seqlens + page tables) from req_to_token.

    ``page_table_1`` (the wide page_size=1 table) is optional: pass ``None`` to
    skip materializing it and write only the compact ``real_page_table``
    (page_size=``real_page_size``). This is used by the fused decode CUDA graph,
    where the wide table is never read (attention uses topk_indices, the indexer
    uses real_page_table); ``real_page_size`` must be >1 in that case. When a
    tensor is passed, behavior is unchanged (both tables are written).

    Contract: each page-table row is written only over its live prefix
    ([:cache_seqlens]); the tail keeps stale values across CUDA-graph replays, so
    consumers must bound reads by cache_seqlens.

    The column scan is bounded inside the kernel by each row's own kv length
    (read at run time), so the cost scales with the live sequence lengths and
    not with ``max_len`` (the table width); the grid itself stays
    data-independent. See :func:`bounded_scan_num_splits`.
    """
    assert seq_lens.is_cuda
    assert req_pool_indices.is_cuda
    assert req_to_token.is_cuda
    assert cache_seqlens.is_cuda
    assert cu_seqlens_k.is_cuda
    assert dsa_cache_seqlens.is_cuda
    assert dsa_cu_seqlens_k.is_cuda

    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return
    assert index_kpool > 0

    has_real_page_table = real_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
        assert real_page_table.is_cuda
    else:
        # page_size==1: real IS page_table_1, so page_table_1 must be present.
        assert page_table_1 is not None
        real_page_table = page_table_1

    # page_table_1 (the wide page_size=1 table) may be dropped for the fused
    # decode CUDA graph; the kernel then writes only real_page_table.
    has_page_table_1 = page_table_1 is not None
    if not has_page_table_1:
        assert has_real_page_table
        page_table_1 = real_page_table  # dummy pointer for stride args
    else:
        assert page_table_1.is_cuda

    block_bs = triton.next_power_of_2(bs)
    block_n = 128
    num_col_blocks = triton.cdiv(max_len, block_n)
    num_splits = bounded_scan_num_splits(bs, num_col_blocks)
    grid = (1 + bs * num_splits,)

    _fused_dsa_decode_metadata_kernel[grid](
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        bs,
        max_len,
        num_splits,
        dsa_index_topk,
        index_kpool,
        real_page_size,
        has_real_page_table,
        has_page_table_1,
        BLOCK_BS=block_bs,
        BLOCK_N=block_n,
    )
