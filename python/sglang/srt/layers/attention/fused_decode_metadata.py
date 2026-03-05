"""Fused Triton kernel for normal_decode_set_metadata.

Fuses the add+copy, gather, and integer-division+copy operations into a
single kernel launch.  The cumulative-sum (cu_seqlens_k) is kept as a
separate torch.cumsum call because it is a reduction that is already
efficient on small vectors (batch-size elements).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


@triton.jit
def _fused_decode_metadata_kernel(
    cache_seqlens_ptr,
    seq_lens_ptr,
    seq_len_delta,
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_table_ptr,
    req_to_token_stride: tl.constexpr,
    page_table_stride: tl.constexpr,
    page_size: tl.constexpr,
    max_seq_pages,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # --- Op 1: cache_seqlens[pid] = seq_lens[pid] + seq_len_delta -----------
    seq_len = tl.load(seq_lens_ptr + pid)
    cache_seqlen = (seq_len + seq_len_delta).to(tl.int32)
    tl.store(cache_seqlens_ptr + pid, cache_seqlen)

    # --- Op 3+4: gather from req_to_token, divide, store to page_table ------
    req_pool_idx = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    base_addr = req_pool_idx * req_to_token_stride

    num_loops = tl.cdiv(max_seq_pages, BLOCK_SIZE)
    for i in range(num_loops):
        offsets = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offsets < max_seq_pages

        # strided_indices[j] == j * page_size  (precomputed arange with step)
        token_offsets = offsets.to(tl.int64) * page_size

        vals = tl.load(
            req_to_token_ptr + base_addr + token_offsets,
            mask=mask,
        )

        page_vals = (vals // page_size).to(tl.int32)
        tl.store(
            page_table_ptr + pid.to(tl.int64) * page_table_stride + offsets,
            page_vals,
            mask=mask,
        )


def fused_normal_decode_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    strided_indices: torch.Tensor,
    max_seq_pages: int,
    seq_lens: torch.Tensor,
    seq_len_delta: int,
    page_size: int,
    swa_page_table: Optional[torch.Tensor] = None,
    token_to_kv_pool: Optional[SWAKVPool] = None,
):
    """Drop-in replacement for normal_decode_set_metadata using a fused Triton kernel.

    Reduces ~5 CUDA kernel launches to 2 (one Triton kernel + one cumsum).
    The SWA (Sliding Window Attention) path is handled with standard PyTorch
    ops as a fallback since it requires a Python-level lookup table.
    """
    bs = seq_lens.shape[0]
    if bs == 0:
        return

    _fused_decode_metadata_kernel[(bs,)](
        cache_seqlens_int32,
        seq_lens,
        seq_len_delta,
        req_to_token,
        req_pool_indices,
        page_table,
        req_to_token.stride(0),
        page_table.stride(0),
        page_size,
        max_seq_pages,
        BLOCK_SIZE=512,
    )

    cu_seqlens_k[1:].copy_(
        torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32)
    )

    if swa_page_table is not None and token_to_kv_pool is not None:
        assert isinstance(token_to_kv_pool, SWAKVPool)
        page_indices = req_to_token[
            req_pool_indices[:, None],
            strided_indices[:max_seq_pages][None, :],
        ]
        swa_page_indices = token_to_kv_pool.translate_loc_from_full_to_swa(
            page_indices
        )
        swa_page_table[:, :max_seq_pages].copy_(swa_page_indices // page_size)
