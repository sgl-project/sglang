"""Fused CUDA-graph metadata update for the TRTLLM MLA backend."""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.kvcache.kv_indices import (
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)


@triton.jit
def update_trtllm_mla_graph_metadata_kernel(
    # inputs
    req_pool_indices_ptr,  # [bs] int
    seq_lens_ptr,  # [bs] int
    req_to_token_ptr,  # [pool_size, req_to_token_stride] int32
    # outputs
    seq_lens_k_ptr,  # [bs] int32, or None
    block_kv_indices_ptr,  # [bs, block_kv_indices_stride] int32
    # scalars
    seqlen_offset,  # added to seq_lens for seq_lens_k / KV coverage
    req_to_token_stride,
    block_kv_indices_stride,
    # constexpr
    PAGE_SIZE: tl.constexpr,
    HAS_SEQ_LENS_K: tl.constexpr,
    NUM_PAGE_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    blk = tl.program_id(axis=1)

    seqlen = (tl.load(seq_lens_ptr + pid) + seqlen_offset).to(tl.int32)
    if HAS_SEQ_LENS_K and blk == 0:
        tl.store(seq_lens_k_ptr + pid, seqlen)

    num_pages = tl.cdiv(seqlen, PAGE_SIZE)
    if blk * NUM_PAGE_PER_BLOCK < num_pages:
        req_pool_index = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
        page_idx = blk * NUM_PAGE_PER_BLOCK + tl.arange(0, NUM_PAGE_PER_BLOCK)
        mask = page_idx < num_pages
        token = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_stride
            + page_idx.to(tl.int64) * PAGE_SIZE,
            mask=mask,
            other=0,
        )
        tl.store(
            block_kv_indices_ptr
            + pid.to(tl.int64) * block_kv_indices_stride
            + page_idx,
            token // PAGE_SIZE,
            mask=mask,
        )


def update_trtllm_mla_graph_metadata(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    block_kv_indices: torch.Tensor,
    bs: int,
    seqlen_offset: int,
    page_size: int,
    seq_lens_k: Optional[torch.Tensor] = None,
):
    if bs == 0:
        return

    grid = (
        bs,
        get_num_kv_index_blocks_flashmla(block_kv_indices.shape[1], page_size),
    )
    update_trtllm_mla_graph_metadata_kernel[grid](
        req_pool_indices,
        seq_lens,
        req_to_token,
        seq_lens_k,
        block_kv_indices,
        seqlen_offset,
        req_to_token.stride(0),
        block_kv_indices.stride(0),
        PAGE_SIZE=page_size,
        HAS_SEQ_LENS_K=seq_lens_k is not None,
        NUM_PAGE_PER_BLOCK=get_num_page_per_block_flashmla(page_size),
    )
