"""Pool-aware fused DSA verify metadata."""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa_kpool_metadata.scan import bounded_scan_num_splits


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "max_seqlen_k",
        "num_splits",
    ]
)
def _fused_dsa_target_verify_metadata_kernel(
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
    index_kpool: tl.constexpr,
    real_page_size: tl.constexpr,
    next_n: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    HAS_PAGED_MQA_CTX_LENS: tl.constexpr,
    HAS_PAGE_TABLE_1: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_EXPANDED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    expanded_size: tl.constexpr = bs * next_n

    if pid == 0:
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
        if index_kpool <= 1:
            dsa_seq = tl.minimum(expanded_seq, dsa_index_topk)
        else:
            # Preserve the live partial pool after selecting pool-aligned history.
            full_pool_tokens = (expanded_seq // index_kpool) * index_kpool
            selected_history_tokens = tl.minimum(full_pool_tokens, dsa_index_topk)
            tail_tokens = expanded_seq - full_pool_tokens
            dsa_seq = selected_history_tokens + tail_tokens
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
    out_row = page_pid // num_splits
    split_id = page_pid - out_row * num_splits

    req_row = out_row // next_n
    req_idx = tl.load(
        req_pool_indices + req_row * req_pool_indices_stride,
        mask=out_row < expanded_size,
        other=0,
    )
    kv_len = (
        tl.load(
            seq_lens + req_row * seq_lens_stride,
            mask=out_row < expanded_size,
            other=0,
        ).to(tl.int32)
        + next_n
    )
    # Output-row offsets can overflow int32 at 1M context.
    out_row_i64 = out_row.to(tl.int64)
    num_live_blocks = tl.minimum(
        tl.cdiv(kv_len, BLOCK_N), tl.cdiv(max_seqlen_k, BLOCK_N)
    )
    for col_block in tl.range(split_id, num_live_blocks, num_splits, num_stages=3):
        offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (out_row < expanded_size) & (offs_n < max_seqlen_k)
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
                + out_row_i64 * page_table_stride_0
                + offs_n * page_table_stride_1,
                vals,
                mask=mask,
            )

        if HAS_REAL_PAGE_TABLE:
            real_mask = mask & ((offs_n % real_page_size) == 0)
            real_cols = offs_n // real_page_size
            tl.store(
                real_page_table
                + out_row_i64 * real_page_table_stride_0
                + real_cols * real_page_table_stride_1,
                vals // real_page_size,
                mask=real_mask,
            )


def _prep_fused_dsa_target_verify_metadata_launch(
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
    real_page_size: int,
    next_n: int,
    paged_mqa_ctx_lens_2d: torch.Tensor = None,
    index_kpool: int = 1,
):
    assert seq_lens.is_cuda
    assert req_pool_indices.is_cuda
    assert req_to_token.is_cuda
    assert cache_seqlens.is_cuda
    assert cu_seqlens_k.is_cuda
    assert seqlens_expanded.is_cuda
    assert dsa_cache_seqlens.is_cuda
    assert dsa_cu_seqlens_k.is_cuda

    assert bs > 0
    assert next_n > 0
    assert index_kpool > 0

    has_real_page_table = real_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
        assert real_page_table.is_cuda
    else:
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

    has_paged_mqa_ctx_lens = paged_mqa_ctx_lens_2d is not None
    if has_paged_mqa_ctx_lens:
        assert paged_mqa_ctx_lens_2d.is_cuda
        assert paged_mqa_ctx_lens_2d.dtype == torch.int32
        assert paged_mqa_ctx_lens_2d.dim() == 2
        assert paged_mqa_ctx_lens_2d.size(0) == bs
        assert paged_mqa_ctx_lens_2d.size(1) == next_n
    else:
        paged_mqa_ctx_lens_2d = page_table_1

    expanded_size = bs * next_n
    block_bs = triton.next_power_of_2(bs)
    block_expanded = triton.next_power_of_2(expanded_size)
    block_n = 128
    num_col_blocks = triton.cdiv(max_seqlen_k, block_n)
    num_splits = bounded_scan_num_splits(expanded_size, num_col_blocks)
    grid = (1 + expanded_size * num_splits,)

    args = (
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
        index_kpool,
        real_page_size,
        next_n,
        has_real_page_table,
        has_paged_mqa_ctx_lens,
        has_page_table_1,
    )
    constexprs = dict(
        BLOCK_BS=block_bs,
        BLOCK_EXPANDED=block_expanded,
        BLOCK_N=block_n,
    )
    return grid, args, constexprs


def fused_dsa_target_verify_metadata(
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
    real_page_size: int,
    next_n: int,
    paged_mqa_ctx_lens_2d: torch.Tensor = None,
    index_kpool: int = 1,
) -> None:
    if bs == 0:
        assert cu_seqlens_k.is_cuda
        assert dsa_cu_seqlens_k.is_cuda
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return

    grid, args, constexprs = _prep_fused_dsa_target_verify_metadata_launch(
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
        bs,
        max_seqlen_k,
        dsa_index_topk,
        real_page_size,
        next_n,
        paged_mqa_ctx_lens_2d,
        index_kpool,
    )
    _fused_dsa_target_verify_metadata_kernel[grid](*args, **constexprs)
