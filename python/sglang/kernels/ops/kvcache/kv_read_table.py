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
"""Builds the unified memory pool's read indices.

One gather-and-translate: for each request row, read the virtual ids out of
`req_to_token` and convert each to the id the kernels can use.

    page(b, c)  = req_to_token[req[b], c * ps] // ps        -- the VIRTUAL page
    entry(b, c) = clamp(v2p[page(b, c)] * multiplier, 0)    -- kernel-facing

Two delivery forms over that one formula:

  PAGE TABLE    `out[b, c] = entry(b, c)`, rows a uniform stride apart, for a
                consumer whose kernel reads a page table directly.
  TOKEN STREAM  `out[row_starts[b] + p] = entry(b, p // ps) * ps + p % ps`, the
                indptr-addressed form a paged wrapper plans over. Converting an
                id keeps its offset inside the page, so the token id is exact.
                Its length is `sum(seq_lens)` -- one id per resident token,
                which the pool bounds, where a page table's width is bounded
                only by `max_context_len`.

PREFIX-ONLY per row: nothing past the row's live prefix is written, so a
caller-owned buffer keeps what it had there -- which is what lets a captured
cuda-graph buffer be refreshed in place. Readers bound themselves by
`cache_seqlens` and never look past the prefix.

A `-1` in `req_to_token` and a freed (`-1`) v2p row both clamp to entry 0, the
reserved padding slot, so a kernel dereferences padding, not a wild address.

The grid is sized from `bs` alone and each program strides over the items it
owns, bounded by the device-side lengths. A cuda-graph capture bakes the grid,
so a grid spanning the full width would replay `max_context_len`/BLOCK blocks
every step no matter how short the sequences actually are.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_ITEMS = 512
_NUM_WARPS = 8
# Enough blocks to fill the device without oversubscribing the item loop;
# measured on H100 over bs 1..256 x seq 1k..128k, flat within ~10% either side.
_TARGET_BLOCKS = 1024


@triton.jit
def build_kv_read_indices_kernel(
    req_to_token_ptr,  # in: [max_reqs, max_context] -- VIRTUAL token ids
    req_pool_indices_ptr,  # in: [bs] -- req_to_token row per batch lane
    seq_lens_ptr,  # in: [bs] -- live TOKENS per row
    v2p_ptr,  # in: [num_pages + 1] int64 -- virtual->physical page table
    row_starts_ptr,  # in: [bs + 1] or null -- CSR row starts; null = uniform
    kv_start_idx_ptr,  # in: [bs] or null -- first token of the row's window
    out_ptr,  # out: int32
    req_stride,  # runtime: req_to_token row stride (elements)
    out_stride,  # runtime: uniform row stride, used when row_starts is null
    mult,  # runtime: kernel_page_multiplier of the target sub-pool
    item_stride,  # runtime: items one program advances per loop trip
    PAGE_SIZE: tl.constexpr,
    EMIT_PER_TOKEN: tl.constexpr,
    OUT_INT64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    req = tl.load(req_pool_indices_ptr + bid).to(tl.int64)
    seqlen = tl.load(seq_lens_ptr + bid)
    # Derived here, not on the host: one elementwise op there costs a whole
    # launch, which a captured graph then replays every step.
    if EMIT_PER_TOKEN:
        n_items = seqlen
    else:
        n_items = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    kv_start = 0
    if kv_start_idx_ptr:
        kv_start = tl.load(kv_start_idx_ptr + bid).to(tl.int32)
    row_in = req_to_token_ptr + req * req_stride
    if row_starts_ptr:
        row_out = out_ptr + tl.load(row_starts_ptr + bid).to(tl.int64)
    else:
        row_out = out_ptr + bid.to(tl.int64) * out_stride

    for start in range(tl.program_id(1) * BLOCK, n_items, item_stride):
        item = start + tl.arange(0, BLOCK)
        mask = item < n_items
        pos = kv_start + item
        if EMIT_PER_TOKEN:
            page = pos // PAGE_SIZE
        else:
            page = pos
        tok = tl.load(row_in + page.to(tl.int64) * PAGE_SIZE, mask=mask, other=0).to(
            tl.int64
        )
        # Triton's `//` truncates toward zero, so `-1 // ps` is 0 for ps > 1 but
        # -1 at ps == 1, which would read one element BEFORE `v2p`.
        vpage = tl.where(tok < 0, 0, tok // PAGE_SIZE)
        entry = tl.maximum(tl.load(v2p_ptr + vpage, mask=mask, other=0) * mult, 0)
        if EMIT_PER_TOKEN:
            value = entry * PAGE_SIZE + pos % PAGE_SIZE
        else:
            value = entry
        if OUT_INT64:
            tl.store(row_out + item, value, mask=mask)
        else:
            tl.store(row_out + item, value.to(tl.int32), mask=mask)


def _launch(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    v2p: torch.Tensor,
    multiplier: int,
    page_size: int,
    max_items: int,
    out: torch.Tensor,
    out_stride: int,
    row_starts: Optional[torch.Tensor],
    kv_start_idx: Optional[torch.Tensor],
    emit_per_token: bool,
) -> None:
    bs = int(req_pool_indices.numel())
    item_programs = min(
        triton.cdiv(_TARGET_BLOCKS, bs), triton.cdiv(max_items, _BLOCK_ITEMS)
    )
    build_kv_read_indices_kernel[(bs, item_programs)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        v2p,
        row_starts,
        kv_start_idx,
        out,
        req_to_token.stride(0),
        out_stride,
        multiplier,
        item_programs * _BLOCK_ITEMS,
        PAGE_SIZE=page_size,
        EMIT_PER_TOKEN=emit_per_token,
        OUT_INT64=out.dtype == torch.int64,
        BLOCK=_BLOCK_ITEMS,
        num_warps=_NUM_WARPS,
    )


def _entries(
    *,
    req_to_token: torch.Tensor,
    req: int,
    page_cols: torch.Tensor,
    v2p: torch.Tensor,
    multiplier: int,
    page_size: int,
) -> torch.Tensor:
    """The formula above, in torch. The allocator's unit tests run on CPU, so
    without this the Triton kernel would have no coverage there."""
    tok = req_to_token[req, page_cols * page_size].to(torch.int64)
    return (v2p[torch.where(tok < 0, 0, tok // page_size)] * multiplier).clamp(min=0)


def build_kv_read_table(
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
    """Fill ``out``'s live prefix with PAGE TABLE entries.

    ``out`` is caller-owned (fresh zeros for the eager path, the module's
    capture-stable buffer for replay) and only its ``[:bs, :max_pages]``
    region's live prefix is written -- never rebound, never tail-cleared.
    """
    bs = int(req_pool_indices.numel())
    assert out.dtype == torch.int32, (
        f"build_kv_read_table: out must be int32, got {out.dtype}"
    )
    assert out.dim() == 2 and out.shape[0] >= bs and out.shape[1] >= max_pages, (
        f"build_kv_read_table: out {tuple(out.shape)} cannot hold "
        f"(bs={bs}, max_pages={max_pages})"
    )
    assert out.stride(1) == 1, "build_kv_read_table: out rows must be packed"
    assert (max_pages - 1) * page_size < req_to_token.shape[1], (
        f"build_kv_read_table: max_pages={max_pages} x ps={page_size} "
        f"exceeds req_to_token width {req_to_token.shape[1]}"
    )
    if bs == 0 or max_pages == 0:
        return out

    if not req_to_token.is_cuda:
        cols = torch.arange(max_pages, device=req_to_token.device)
        for b in range(bs):
            n_pages = (int(seq_lens[b]) + page_size - 1) // page_size
            live = min(n_pages, max_pages)
            out[b, :live] = _entries(
                req_to_token=req_to_token,
                req=int(req_pool_indices[b]),
                page_cols=cols[:live],
                v2p=v2p,
                multiplier=multiplier,
                page_size=page_size,
            ).to(torch.int32)
        return out

    _launch(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        v2p=v2p,
        multiplier=multiplier,
        page_size=page_size,
        max_items=max_pages,
        out=out,
        out_stride=out.stride(0),
        row_starts=None,
        kv_start_idx=None,
        emit_per_token=False,
    )
    return out


def build_kv_read_table_packed(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    v2p: torch.Tensor,
    indptr: torch.Tensor,
    multiplier: int,
    page_size: int,
    max_tokens: int,
    out: torch.Tensor,
    kv_start_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fill ``out``'s CSR rows with TOKEN STREAM ids.

    ``seq_lens`` counts tokens per row and ``indptr`` gives each row's start, so
    the live stream is ``sum(seq_lens)`` long; ``max_tokens`` is the capacity
    ``out`` must have for that, and callers holding a capture-stable buffer pass
    its size. ``kv_start_idx`` shifts a row's window start without moving where
    it lands.
    """
    bs = int(req_pool_indices.numel())
    assert out.dtype in (torch.int32, torch.int64), (
        f"build_kv_read_table_packed: out must be int32 or int64, got {out.dtype}"
    )
    assert out.dim() == 1 and out.numel() >= max_tokens, (
        f"build_kv_read_table_packed: out {tuple(out.shape)} cannot hold "
        f"max_tokens={max_tokens}"
    )
    assert indptr.numel() > bs, (
        f"build_kv_read_table_packed: indptr holds {indptr.numel()} entries, "
        f"need {bs + 1}"
    )
    if bs == 0 or max_tokens == 0:
        return out

    if not req_to_token.is_cuda:
        for b in range(bs):
            n = int(seq_lens[b])
            pos = torch.arange(n, device=req_to_token.device) + (
                0 if kv_start_idx is None else int(kv_start_idx[b])
            )
            entry = _entries(
                req_to_token=req_to_token,
                req=int(req_pool_indices[b]),
                page_cols=pos // page_size,
                v2p=v2p,
                multiplier=multiplier,
                page_size=page_size,
            )
            start = int(indptr[b])
            out[start : start + n] = (entry * page_size + pos % page_size).to(out.dtype)
        return out

    _launch(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        v2p=v2p,
        multiplier=multiplier,
        page_size=page_size,
        max_items=max_tokens,
        out=out,
        out_stride=0,
        row_starts=indptr,
        kv_start_idx=kv_start_idx,
        emit_per_token=True,
    )
    return out
