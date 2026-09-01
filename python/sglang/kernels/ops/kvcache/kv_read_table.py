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
"""Builds the per-batch read table for the unified memory pool.

One fused gather-and-translate. For each request row it reads the virtual ids
out of `req_to_token`, converts each to the id the kernels can use, and writes
the result into `out`:

    out[b, c] = clamp(v2p[req_to_token[req[b], c * ps] // ps] * multiplier, 0)
                for c < ceil(seq_lens[b] / ps)          -- the row's LIVE prefix

`v2p` is the pool's virtual->physical page table and `multiplier` scales a
physical page into the id space the per-layer views use (1 when one page maps
to one row-block). Since only the page number is rewritten, a token-level consumer can
rebuild flat ids as `entry * ps + offset`.

PREFIX-ONLY per row: columns past the live prefix are never written, so a
caller-owned buffer keeps what it had there -- which is what lets a captured
cuda-graph buffer be refreshed in place. Readers bound themselves by
`cache_seqlens` and never look past the prefix.

A `-1` in `req_to_token` and a freed (`-1`) v2p row both clamp to entry 0, the
reserved padding slot, so a kernel dereferences padding, not a wild address.

The grid is sized from `bs` alone and each program strides over the columns it
owns, bounded by the device-side `seq_lens`. A cuda-graph capture bakes the
grid, so a grid spanning `max_pages` would replay `max_context_len`/BLOCK column
blocks every step no matter how short the sequences actually are.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_COLS = 512
_NUM_WARPS = 8
# Enough blocks to fill the device without oversubscribing the column loop;
# measured on H100 over bs 1..256 x seq 1k..128k, flat within ~10% either side.
_TARGET_BLOCKS = 1024


@triton.jit
def build_kv_read_table_kernel(
    req_to_token_ptr,  # in: [max_reqs, max_context] -- VIRTUAL token ids
    req_pool_indices_ptr,  # in: [bs] -- row per batch lane
    seq_lens_ptr,  # in: [bs]
    v2p_ptr,  # in: [num_pages + 1] int64 -- virtual->physical page table
    out_ptr,  # out: [>=bs, >=max_pages] int32 -- the read table
    req_stride,  # runtime: req_to_token row stride (elements)
    out_stride,  # runtime: out row stride (elements)
    mult,  # runtime: kernel_page_multiplier of the target sub-pool
    col_stride,  # runtime: columns one program advances per loop trip
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    req = tl.load(req_pool_indices_ptr + bid).to(tl.int64)
    seqlen = tl.load(seq_lens_ptr + bid)
    n_pages = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    row_in = req_to_token_ptr + req * req_stride
    row_out = out_ptr + bid.to(tl.int64) * out_stride

    for start in range(tl.program_id(1) * BLOCK, n_pages, col_stride):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_pages
        tok = tl.load(row_in + cols.to(tl.int64) * PAGE_SIZE, mask=mask, other=0).to(
            tl.int64
        )
        # Triton's `//` truncates toward zero, so `-1 // ps` is 0 for ps > 1 but
        # -1 at ps == 1, which would read one element BEFORE `v2p`.
        page = tl.where(tok < 0, 0, tok // PAGE_SIZE)
        phys = tl.load(v2p_ptr + page, mask=mask, other=0)
        entry = tl.maximum(phys * mult, 0).to(tl.int32)
        tl.store(row_out + cols, entry, mask=mask)


@triton.jit
def build_kv_read_table_packed_kernel(
    req_to_token_ptr,  # in: [max_reqs, max_context] -- VIRTUAL token ids
    req_pool_indices_ptr,  # in: [bs] -- row per batch lane
    seq_lens_ptr,  # in: [bs]
    v2p_ptr,  # in: [num_pages + 1] int64 -- virtual->physical page table
    indptr_ptr,  # in: [bs + 1] -- CSR row starts into `out`
    kv_start_idx_ptr,  # in: [bs] or null -- first token of each row's window
    out_ptr,  # out: [>= indptr[bs]] int32 -- the packed token stream
    req_stride,  # runtime: req_to_token row stride (elements)
    mult,  # runtime: kernel_page_multiplier of the target sub-pool
    tok_stride,  # runtime: tokens one program advances per loop trip
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    req = tl.load(req_pool_indices_ptr + bid).to(tl.int64)
    n_tokens = tl.load(seq_lens_ptr + bid)
    kv_start = 0
    if kv_start_idx_ptr:
        kv_start = tl.load(kv_start_idx_ptr + bid).to(tl.int32)
    row_in = req_to_token_ptr + req * req_stride
    row_out = out_ptr + tl.load(indptr_ptr + bid).to(tl.int64)

    for start in range(tl.program_id(1) * BLOCK, n_tokens, tok_stride):
        col = start + tl.arange(0, BLOCK)
        mask = col < n_tokens
        pos = kv_start + col
        tok = tl.load(
            row_in + (pos // PAGE_SIZE).to(tl.int64) * PAGE_SIZE, mask=mask, other=0
        ).to(tl.int64)
        # Triton's `//` truncates toward zero, so `-1 // ps` is 0 for ps > 1 but
        # -1 at ps == 1, which would read one element BEFORE `v2p`.
        vpage = tl.where(tok < 0, 0, tok // PAGE_SIZE)
        phys = tl.load(v2p_ptr + vpage, mask=mask, other=0)
        entry = tl.maximum(phys * mult, 0)
        # Converting an id keeps its offset inside the page, so the token id is
        # exact: the same rebuild `create_flashinfer_kv_indices_triton` does.
        token = entry * PAGE_SIZE + pos % PAGE_SIZE
        tl.store(row_out + col, token.to(tl.int32), mask=mask)


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
    """Fill ``out``'s CSR rows with kernel-facing TOKEN ids.

    The rectangle form above feeds consumers that read a page table directly;
    this one writes the indptr-addressed stream the flashinfer wrappers plan
    over, skipping the rectangle they would otherwise repack. ``out`` holds
    ``sum(seq_lens)`` entries, which the pool bounds -- one id per resident
    token -- rather than the ``bs x max_context_len`` a rectangle needs.
    """
    bs = int(req_pool_indices.numel())
    assert (
        out.dtype == torch.int32
    ), f"build_kv_read_table_packed: out must be int32, got {out.dtype}"
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
        starts = indptr[: bs + 1].tolist()
        cols = torch.arange(max_tokens, device=req_to_token.device)
        for b in range(bs):
            n = int(seq_lens[b])
            pos = cols[:n] + (0 if kv_start_idx is None else int(kv_start_idx[b]))
            tok = req_to_token[int(req_pool_indices[b]), (pos // page_size) * page_size]
            vpage = torch.where(tok < 0, 0, tok.to(torch.int64) // page_size)
            entry = (v2p[vpage] * multiplier).clamp(min=0)
            out[starts[b] : starts[b] + n] = (entry * page_size + pos % page_size).to(
                torch.int32
            )
        return out

    tok_programs = min(
        triton.cdiv(_TARGET_BLOCKS, bs), triton.cdiv(max_tokens, _BLOCK_COLS)
    )
    build_kv_read_table_packed_kernel[(bs, tok_programs)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        v2p,
        indptr,
        kv_start_idx,
        out,
        req_to_token.stride(0),
        multiplier,
        tok_programs * _BLOCK_COLS,
        PAGE_SIZE=page_size,
        BLOCK=_BLOCK_COLS,
        num_warps=_NUM_WARPS,
    )
    return out


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
    """Fill ``out``'s live prefix with read-table entries.

    ``out`` is caller-owned (fresh zeros for the eager path, the module's
    capture-stable buffer for replay) and only its ``[:bs, :max_pages]``
    region's live prefix is written -- never rebound, never tail-cleared.
    """
    bs = int(req_pool_indices.numel())
    assert (
        out.dtype == torch.int32
    ), f"build_kv_read_table: out must be int32, got {out.dtype}"
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

    col_programs = min(
        triton.cdiv(_TARGET_BLOCKS, bs), triton.cdiv(max_pages, _BLOCK_COLS)
    )
    build_kv_read_table_kernel[(bs, col_programs)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        v2p,
        out,
        req_to_token.stride(0),
        out.stride(0),
        multiplier,
        col_programs * _BLOCK_COLS,
        PAGE_SIZE=page_size,
        BLOCK=_BLOCK_COLS,
        num_warps=_NUM_WARPS,
    )
    return out
