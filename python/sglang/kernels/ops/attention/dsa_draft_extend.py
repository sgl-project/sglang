from typing import Optional

import torch
import triton

from sglang.kernels.ops.attention.dsa_metadata import (
    _fused_dsa_draft_extend_metadata_kernel,
)


def fused_dsa_uniform_draft_extend_metadata(
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
    qo_len: int,
    max_seqlen_k: int,
    dsa_index_topk: int,
    real_page_size: int,
) -> None:
    """Launch the shared DSA metadata kernel with a compile-time query width."""
    assert seq_lens.is_cuda
    assert req_pool_indices.is_cuda
    assert req_to_token.is_cuda
    assert cache_seqlens.is_cuda
    assert cu_seqlens_k.is_cuda
    assert seqlens_expanded.is_cuda
    assert dsa_cache_seqlens.is_cuda
    assert dsa_cu_seqlens_k.is_cuda
    assert qo_len > 0

    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return
    total_len = bs * qo_len

    has_real_page_table = real_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
        assert real_page_table.is_cuda
    else:
        assert page_table_1 is not None
        real_page_table = page_table_1

    # The wide page_size=1 table may be absent when only the compact real page
    # table is consumed. Dummy pointers below are compile-time-dead operands.
    has_page_table_1 = page_table_1 is not None
    if not has_page_table_1:
        assert has_real_page_table
        page_table_1 = real_page_table
    else:
        assert page_table_1.is_cuda

    block_bs = triton.next_power_of_2(bs)
    block_expanded = triton.next_power_of_2(total_len)
    block_rows = triton.next_power_of_2(qo_len)
    block_n = 128
    num_col_blocks = triton.cdiv(max_seqlen_k, block_n)
    grid = (1 + bs * num_col_blocks,)

    _fused_dsa_draft_extend_metadata_kernel[grid](
        seq_lens,
        seq_lens,  # Dummy extend widths; UNIFORM_QO_LEN removes all loads.
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        seqlens_expanded,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        seq_lens,  # Dummy qo_indptr; HAS_QO_INDPTR is false.
        seq_lens.stride(0),
        seq_lens.stride(0),
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
        dsa_index_topk,
        real_page_size,
        has_real_page_table,
        has_page_table_1,
        True,
        False,
        BLOCK_BS=block_bs,
        BLOCK_EXPANDED=block_expanded,
        BLOCK_ROWS=block_rows,
        BLOCK_N=block_n,
        UNIFORM_QO_LEN=qo_len,
    )
