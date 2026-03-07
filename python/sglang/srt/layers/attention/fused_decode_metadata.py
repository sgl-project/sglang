"""
Fused Triton kernel for normal_decode_set_metadata (add+copy, gather, div+copy). cumsum kept separate.

Note: `strided_indices` is only used in the SWA path. The non-SWA kernel
computes equivalent offsets internally via `offsets * page_size`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
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

        token_offsets = offsets.to(tl.int64) * page_size

        vals = tl.load(
            req_to_token_ptr + base_addr + token_offsets,
            mask=mask,
            other=0,  # avoid undefined values in masked lanes
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
    """Fused Triton kernel replacing ~5 CUDA launches with 2 (kernel + cumsum). SWA path falls back to PyTorch."""

    bs = seq_lens.shape[0]
    if bs == 0:
        return

    # Validate that max_seq_pages fits within the allocated page_table.
    # The Triton kernel has no bounds checking, so this prevents silent
    # out-of-bounds memory writes.
    if max_seq_pages > page_table.shape[1]:
        raise ValueError(
            f"max_seq_pages ({max_seq_pages}) exceeds page_table capacity "
            f"({page_table.shape[1]}). This would cause out-of-bounds memory "
            f"access in the Triton kernel."
        )

    # swa_page_table and token_to_kv_pool are co-dependent; reject partial input.
    if (swa_page_table is None) != (token_to_kv_pool is None):
        raise ValueError(
            "swa_page_table and token_to_kv_pool must both be provided or both be None."
        )

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

    cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32))

    if swa_page_table is not None and token_to_kv_pool is not None:
        page_indices = req_to_token[
            req_pool_indices[:, None],
            strided_indices[:max_seq_pages][None, :],
        ]
        swa_page_indices = token_to_kv_pool.translate_loc_from_full_to_swa(page_indices)
        swa_page_table[:, :max_seq_pages].copy_(swa_page_indices // page_size)
