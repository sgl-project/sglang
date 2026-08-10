# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Canonical kernel-facing page-table builder for the unified memory pool.

One fused gather+translate producing the per-batch canonical page table the
read-path choke point (`sglang.srt.mem_cache.kv_index_source`) hands to every
attention backend:

    out[b, c] = clamp(v2p[req_to_token[req[b], c * ps] // ps] * mult, 0)
                for c < ceil(seq_lens[b] / ps)          (the row's LIVE prefix)

- ``entry = v2p[virtual_page] * multiplier`` is the kernel-facing page id:
  physical at multiplier 1 (strided views), dense at the sub-pool's block
  count (dense views). Token-level consumers reconstruct
  ``token = entry * ps + offset`` — exact, because every kernel-facing id
  space is page-affine over virtual ids with in-page offsets preserved.
- PREFIX-ONLY per row: columns past the live prefix are never written. A
  capture-stable ``out=`` keeps stale values there (the fa3 stale-tail
  contract: consumers bound reads by cache_seqlens); a fresh zeros ``out=``
  reads as entry 0 = the page-0 sink, safe to dereference.
- Padding and tombstones land on the sink: ``-1`` req_to_token entries and
  tombstoned v2p rows (-1) both clamp to 0 (same contract as
  ``translate_v2p``).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_COLS = 256


@triton.jit
def build_kernel_page_table_kernel(
    req_to_token_ptr,  # in: [max_reqs, max_context] — VIRTUAL token ids
    req_pool_indices_ptr,  # in: [bs] — row per batch lane
    seq_lens_ptr,  # in: [bs]
    v2p_ptr,  # in: [num_pages + 1] int64 — virtual->physical page table
    out_ptr,  # out: [>=bs, >=max_pages] int32 — canonical entries
    req_stride,  # runtime: req_to_token row stride (elements)
    out_stride,  # runtime: out row stride (elements)
    mult,  # runtime: kernel_page_multiplier of the target sub-pool
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    blk = tl.program_id(1)
    req = tl.load(req_pool_indices_ptr + bid).to(tl.int64)
    seqlen = tl.load(seq_lens_ptr + bid)
    n_pages = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE

    cols = blk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_pages
    tok = tl.load(
        req_to_token_ptr + req * req_stride + cols.to(tl.int64) * PAGE_SIZE,
        mask=mask,
        other=0,
    ).to(tl.int64)
    # Truncating int division routes negative (padding) tokens to page 0, whose
    # v2p entry is the identity anchor; the final clamp keeps every padding or
    # tombstone lane on entry 0 = the sink.
    phys = tl.load(v2p_ptr + tok // PAGE_SIZE, mask=mask, other=0)
    entry = tl.maximum(phys * mult, 0).to(tl.int32)
    tl.store(out_ptr + bid.to(tl.int64) * out_stride + cols, entry, mask=mask)


def build_kernel_page_table(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    v2p: torch.Tensor,
    multiplier: int,
    page_size: int,
    max_pages: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Fill ``out``'s live prefix with canonical entries; see the module doc.

    ``out`` is caller-owned (fresh zeros for the eager path, the module's
    capture-stable buffer for replay) and only its ``[:bs, :max_pages]``
    region's live prefix is written — never rebound, never tail-cleared.
    """
    bs = int(req_pool_indices.numel())
    assert out.dtype == torch.int32, (
        f"build_kernel_page_table: out must be int32, got {out.dtype}"
    )
    assert out.dim() == 2 and out.shape[0] >= bs and out.shape[1] >= max_pages, (
        f"build_kernel_page_table: out {tuple(out.shape)} cannot hold "
        f"(bs={bs}, max_pages={max_pages})"
    )
    assert out.stride(1) == 1, "build_kernel_page_table: out rows must be packed"
    assert max_pages * page_size <= req_to_token.shape[1], (
        f"build_kernel_page_table: max_pages={max_pages} x ps={page_size} "
        f"exceeds req_to_token width {req_to_token.shape[1]}"
    )
    if bs == 0 or max_pages == 0:
        return out

    if not req_to_token.is_cuda:
        # Pure-torch CPU reference for the CUDA-only kernel (mirrors
        # translate_v2p's fallback; unit tests pin the shared semantics).
        cols = torch.arange(max_pages, device=req_to_token.device)
        live = cols[None, :] < (
            (seq_lens[:bs, None].to(torch.int64) + page_size - 1) // page_size
        )
        tok = req_to_token[
            req_pool_indices[:bs, None].to(torch.int64), (cols * page_size)[None, :]
        ].to(torch.int64)
        pages = torch.where(tok < 0, 0, tok // page_size)
        entry = (v2p[pages] * multiplier).clamp(min=0).to(torch.int32)
        dst = out[:bs, :max_pages]
        dst.copy_(torch.where(live, entry, dst))
        return out

    grid = (bs, triton.cdiv(max_pages, _BLOCK_COLS))
    build_kernel_page_table_kernel[grid](
        req_to_token,
        req_pool_indices,
        seq_lens,
        v2p,
        out,
        req_to_token.stride(0),
        out.stride(0),
        multiplier,
        PAGE_SIZE=page_size,
        BLOCK=_BLOCK_COLS,
    )
    return out
