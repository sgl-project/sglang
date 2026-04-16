"""
Fused Triton kernel for ``normal_decode_set_metadata``.

Fuses three of the four ops performed by ``normal_decode_set_metadata`` into
a single Triton kernel launch:

    1. ``cache_seqlens = seq_lens + seq_len_delta``
    2. ``page_indices  = req_to_token[req_pool_indices, 0::page_size]``
    3. ``page_table    = page_indices // page_size``

The ``cu_seqlens_k`` cumsum is kept as a separate launch; fusing it would
require a block-level scan kernel (left for a follow-up PR). The SWA path
also stays in PyTorch because ``translate_loc_from_full_to_swa`` is not
trivially expressible in Triton.

Notes
-----
* ``strided_indices`` is only used by the SWA fallback; the non-SWA kernel
  computes equivalent offsets internally via ``offsets * page_size``.
* ``page_size`` is the only autotune key. It is effectively fixed per server,
  so autotune re-benchmarks at most once per page_size. Callers that rely on
  CUDA graph capture should warmup at least once per page_size before
  starting capture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps)
    for block_size in (128, 256, 512)
    for num_warps in (1, 2, 4, 8)
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["page_size"])
@triton.jit
def _fused_decode_metadata_kernel(
    cache_seqlens_ptr,
    seq_lens_ptr,
    seq_len_delta,
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_table_ptr,
    req_to_token_stride,
    page_table_stride,
    page_size: tl.constexpr,
    max_seq_pages,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Op 1: cache_seqlens[pid] = seq_lens[pid] + seq_len_delta
    seq_len = tl.load(seq_lens_ptr + pid)
    cache_seqlen = (seq_len + seq_len_delta).to(tl.int32)
    tl.store(cache_seqlens_ptr + pid, cache_seqlen)

    # Ops 2+3: page_table[pid, :max_seq_pages] =
    #     req_to_token[req_pool_indices[pid], 0::page_size] // page_size
    # Promote to int64 to avoid overflow when req_to_token has > 2^31 elements.
    req_pool_idx = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    base_addr = req_pool_idx * req_to_token_stride
    row_base = pid.to(tl.int64) * page_table_stride

    num_blocks = tl.cdiv(max_seq_pages, BLOCK_SIZE)
    for i in range(num_blocks):
        offsets = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offsets < max_seq_pages

        token_offsets = offsets.to(tl.int64) * page_size
        vals = tl.load(
            req_to_token_ptr + base_addr + token_offsets,
            mask=mask,
            other=0,
        )
        page_vals = (vals // page_size).to(tl.int32)

        tl.store(page_table_ptr + row_base + offsets, page_vals, mask=mask)


def _update_swa_page_table(
    swa_page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    strided_indices: torch.Tensor,
    max_seq_pages: int,
    page_size: int,
    token_to_kv_pool: "SWAKVPool",
) -> None:
    """PyTorch fallback for the SWA-only portion of ``set_metadata``."""
    page_indices = req_to_token[
        req_pool_indices[:, None],
        strided_indices[:max_seq_pages][None, :],
    ]
    swa_page_indices = token_to_kv_pool.translate_loc_from_full_to_swa(page_indices)
    swa_page_table[:, :max_seq_pages].copy_(swa_page_indices // page_size)


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
    token_to_kv_pool: Optional["SWAKVPool"] = None,
) -> None:
    """Fused drop-in replacement for ``normal_decode_set_metadata``.

    Replaces ~4 CUDA launches with 2 (fused kernel + cumsum). The SWA path
    still runs the original PyTorch fallback; see the module docstring.
    """
    bs = seq_lens.shape[0]
    if bs == 0:
        return

    if max_seq_pages > page_table.shape[1]:
        raise ValueError(
            f"max_seq_pages ({max_seq_pages}) exceeds page_table capacity "
            f"({page_table.shape[1]}). This would cause out-of-bounds writes "
            "inside the Triton kernel."
        )

    # swa_page_table and token_to_kv_pool are co-dependent; reject partial input.
    if (swa_page_table is None) != (token_to_kv_pool is None):
        raise ValueError(
            "swa_page_table and token_to_kv_pool must both be provided or "
            "both be None."
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
    )

    # Separate launch; a block-level scan could fuse it in a follow-up PR.
    cu_seqlens_k[1:].copy_(
        torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32)
    )

    if swa_page_table is not None:
        _update_swa_page_table(
            swa_page_table=swa_page_table,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            strided_indices=strided_indices,
            max_seq_pages=max_seq_pages,
            page_size=page_size,
            token_to_kv_pool=token_to_kv_pool,
        )
