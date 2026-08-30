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
"""Turns the KV ids stored in `req_to_token` into ids attention kernels can use.

WHY THIS EXISTS. A KV slot can be named in more than one **id space**:

  * **virtual** - what `req_to_token` stores. Stable: it keeps naming the same
    logical slot even after the pool moves data around.
  * **physical** - where that slot sits in the pool right now.
  * **kernel-facing** - what a kernel can index the per-layer K/V tensors with.
    Same as physical for a plain pool; under the unified pool's per-layer views it
    is the physical page scaled by the per-page block count.

On a plain pool all three coincide and nothing here does any work. Under the
unified memory pool they differ, so somebody must convert - and if every
backend converts for itself, each one has to know the pool's internals and
each is a place to get it wrong. This module is the one place that converts.

WHAT BACKENDS GET. A `KVIndexTable`, which answers one question: *what do I
gather from, and which row is mine?*

    ids[row_ids[b], pos]        <- the gather every backend already does

    plain pool : ids = req_to_token, row_ids = req_pool_indices
                 (literally those objects - no copy, no kernel, no change)
    unified    : ids = a freshly built array of kernel-facing ids,
                 row_ids = arange(batch_size)

Backends call their own copy a *page table* (fa3) or a *block table*
(trtllm); this module calls what it hands them the **index table**.

WHY ONE TABLE IS ENOUGH FOR EVERYONE. Converting only ever rewrites the page
number and keeps the offset inside the page. So a page-granular table serves
both kinds of consumer: a block-table backend uses its rows as-is, and a
backend that wants a flat per-token list rebuilds one with

    token_id = entry * entry_page_size + pos % entry_page_size
"""

from __future__ import annotations

import weakref
from typing import Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.kvcache.kv_read_table import build_kv_read_table
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)


class KVIndexTable(msgspec.Struct, frozen=True):
    """Collection of what one batch gathers from."""

    ids: torch.Tensor  # 2-D array of KV ids to gather from
    row_ids: torch.Tensor  # which row belongs to batch lane b
    row_stride: int  # stride between rows of `ids`, in elements
    entry_page_size: int  # what one entry covers: 1 = a token, N = a page of N
    is_translated: bool  # entries are already kernel-facing ids
    sliding_window_ids: Optional[torch.Tensor]  # SWA models: the parallel swa array

    def sliding_window_read_ids(self) -> torch.Tensor:
        """Which array a sliding-window gather reads: the parallel swa array
        when translated, else the full-attention array, which the caller maps
        through the pool's own full->swa map."""
        return self.sliding_window_ids if self.is_translated else self.ids


class KVIndexTranslator:
    """Built once per ModelRunner."""

    def __init__(
        self,
        *,
        req_to_token: torch.Tensor,
        token_to_kv_pool_allocator,
        token_to_kv_pool,
        page_size: int,
        device: str,
    ):
        self.req_to_token = req_to_token
        self.page_size = page_size
        self.device = device

        self.is_translating = (
            isinstance(
                token_to_kv_pool_allocator,
                (UnifiedMambaTokenToKVPoolAllocator, UnifiedSWATokenToKVPoolAllocator),
            )
            and token_to_kv_pool_allocator.get_kvcache() is token_to_kv_pool
        )
        if self.is_translating:
            alloc = token_to_kv_pool_allocator
            self._full_v2p_table = alloc.full_v2p_page_table
            self._full_page_multiplier = alloc.kernel_page_multiplier
            self._translate_full = alloc.translate_kv_loc_for_kernel
            if isinstance(alloc, UnifiedSWATokenToKVPoolAllocator):
                self._swa_v2p_table = alloc.swa_v2p_page_table
                self._swa_page_multiplier = alloc.swa_kernel_page_multiplier
            else:
                self._swa_v2p_table = None
                self._swa_page_multiplier = 1
        else:
            self._full_v2p_table = None
            self._full_page_multiplier = 1
            self._translate_full = None
            self._swa_v2p_table = None
            self._swa_page_multiplier = 1

        self._rows: Optional[torch.Tensor] = (
            torch.arange(req_to_token.shape[0], dtype=torch.int64, device=device)
            if self.is_translating
            else None
        )
        self._capture_full_ids: Optional[torch.Tensor] = None
        self._capture_swa_ids: Optional[torch.Tensor] = None
        self._index_table_memo: Optional[Tuple[weakref.ref, KVIndexTable]] = None

    # -- capture-stable buffers ------------------------------------------------

    def ensure_capture_buffers(self, *, max_bs: int, max_context_len: int) -> None:
        """Idempotent. Zero-filled ``(max_bs, ceil(ctx/ps))`` int32 per kind:
        entry 0 is the reserved padding slot in every id space, so a captured
        graph replaying before any refresh reads padding, not garbage."""
        if not self.is_translating or self._capture_full_ids is not None:
            return
        max_pages = -(-max_context_len // self.page_size)
        self._capture_full_ids = torch.zeros(
            (max_bs, max_pages), dtype=torch.int32, device=self.device
        )
        if self._swa_v2p_table is not None:
            self._capture_swa_ids = torch.zeros(
                (max_bs, max_pages), dtype=torch.int32, device=self.device
            )

    # -- per-batch view --------------------------------------------------------

    def build_index_table(
        self,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        max_pages: Optional[int] = None,
        captured: bool = False,
    ) -> KVIndexTable:
        """The one per-batch entry point.

        Non-unified: the raw ``(req_to_token, req_pool_indices)`` passthrough,
        no tensor ops and no copies. Unified: a fresh table of width
        ``max_pages`` (eager), or a live-prefix refresh of the capture buffers
        (``captured=True``), returned WHOLE so its pointer is capture-bakeable.
        """
        if not self.is_translating:
            return KVIndexTable(
                ids=self.req_to_token,
                row_ids=req_pool_indices,
                row_stride=self.req_to_token.stride(0),
                entry_page_size=1,
                is_translated=False,
                sliding_window_ids=None,
            )

        bs = int(req_pool_indices.numel())
        if captured:
            assert self._capture_full_ids is not None, (
                "KVIndexTranslator.build_index_table(captured=True) before "
                "ensure_capture_buffers()"
            )
            out_full = self._capture_full_ids
            out_swa = self._capture_swa_ids
            width = out_full.shape[1] if max_pages is None else max_pages
        else:
            assert max_pages is not None, (
                "KVIndexTranslator.build_index_table: eager path needs max_pages "
                "(from the batch's seq_lens_cpu max)"
            )
            width = max_pages
            out_full = torch.zeros((bs, width), dtype=torch.int32, device=self.device)
            out_swa = (
                torch.zeros((bs, width), dtype=torch.int32, device=self.device)
                if self._swa_v2p_table is not None
                else None
            )

        build_kv_read_table(
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            v2p=self._full_v2p_table,
            multiplier=self._full_page_multiplier,
            page_size=self.page_size,
            max_pages=width,
            out=out_full,
        )
        if out_swa is not None:
            build_kv_read_table(
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                v2p=self._swa_v2p_table,
                multiplier=self._swa_page_multiplier,
                page_size=self.page_size,
                max_pages=width,
                out=out_swa,
            )
        return KVIndexTable(
            ids=out_full,
            row_ids=self._rows[:bs],
            row_stride=out_full.stride(0),
            entry_page_size=self.page_size,
            is_translated=True,
            sliding_window_ids=out_swa,
        )

    def fill_read_table(
        self,
        *,
        out: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Fill a backend-owned padded 2-D block table's live prefix with
        full-attention entries (trtllm_mla / flashmla consume such tables
        directly — their rows ARE the index table's rows).

        Prefix-only: columns past each row's live pages keep the backend's own
        fill (-1 sentinel or stale-but-unread values).

        Unified-only: callers dispatch on ``self.is_translating`` and keep
        their static builder otherwise.
        """
        assert (
            self.is_translating
        ), "KVIndexTranslator.fill_read_table on a pool that needs no translation"
        max_pages = min(
            out.shape[1],
            -(-self.req_to_token.shape[1] // self.page_size),
        )
        return build_kv_read_table(
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            v2p=self._full_v2p_table,
            multiplier=self._full_page_multiplier,
            page_size=self.page_size,
            max_pages=max_pages,
            out=out,
        )

    def index_table_for_batch(self, forward_batch) -> KVIndexTable:
        """Eager per-batch view, memoized in one slot keyed by batch identity
        so multi-consumer metadata builds share a build. The next batch
        replaces the slot; consumers only read during their own build. The
        captured path does not memoize — it refreshes its buffers per
        replay."""
        memo = self._index_table_memo
        if memo is not None and memo[0]() is forward_batch:
            return memo[1]
        max_pages = None
        if self.is_translating:
            # `seq_lens_sum` is the reliable "CPU mirror present" signal, not
            # `seq_lens_cpu`: the latter is a non-None but STALE slice on a
            # gpu_only batch, and a stale max under-sizes the table, leaving
            # the columns past it reading as the sink for tokens the kernel
            # wants. Fall back to the table's own width, which is always safe.
            slc = forward_batch.seq_lens_cpu
            if (
                forward_batch.seq_lens_sum is not None
                and slc is not None
                and slc.numel() > 0
            ):
                max_seq = int(slc.max())
            else:
                max_seq = self.req_to_token.shape[1]
            max_pages = max(-(-max_seq // self.page_size), 1)
        view = self.build_index_table(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            max_pages=max_pages,
        )
        self._index_table_memo = (weakref.ref(forward_batch), view)
        return view

    # -- token-level translate surface (the mixin / local-attn consumers) ------

    def translate_full_attn_ids(
        self, kv_indices: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Virtual token ids -> kernel-facing full-attention ids (the identity
        when no translation is needed, so callers never branch)."""
        if not self.is_translating:
            assert out is None, "passthrough translate takes no out="
            return kv_indices
        return self._translate_full(kv_indices, out=out)
