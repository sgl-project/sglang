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

A KV slot can be named in three id spaces:

  * **virtual** - what `req_to_token` stores. Keeps naming the same logical
    slot even after the pool moves data around.
  * **physical** - where that slot sits in the pool right now.
  * **kernel-facing** - what a kernel can index the per-layer K/V tensors
    with. Same as physical on a plain pool; under the unified pool it is the
    physical page scaled by the per-page block count.

All three coincide on a plain pool, so nothing here does any work there.

Backends get a `KVIndexTable`, which answers "what do I gather from, and
which row is mine?":

    ids[row_ids[b], pos]

    plain pool : ids = req_to_token, row_ids = req_pool_indices (those very
                 objects - no copy, no kernel)
    unified    : ids = a built array of kernel-facing ids,
                 row_ids = arange(batch_size)

Backends call their own copy a *page table* (fa3) or a *block table*
(trtllm); here it is the **index table**.

Converting only ever rewrites the page number and keeps the in-page offset, so
one page-granular table serves both kinds of consumer: a block-table backend
uses its rows as-is, and one that wants flat per-token ids rebuilds them as

    token_id = entry * entry_page_size + pos % entry_page_size

WRITES, IN TWO PHASES. The full-side write loc is rebound to kernel-facing
ids at ForwardBatch construction - the earliest consumer can snapshot it
right after. The sliding-window write loc is derived at the same moment as
read table, into the same index table.
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
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


class KVReadTables(msgspec.Struct, frozen=True):
    """One capture-stable destination, however many id spaces the pool has.

    A backend holds this and hands it back to `build_index_table(into=...)`;
    it never has to know whether there is a sliding-window space behind it.
    """

    full: torch.Tensor
    sliding_window: Optional[torch.Tensor]


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
            self._full_p2v_table = alloc.full_p2v_page_table
            self._full_page_multiplier = alloc.kernel_page_multiplier
            self._translate_full = alloc.translate_kv_loc_for_kernel
            if isinstance(alloc, UnifiedSWATokenToKVPoolAllocator):
                self._swa_v2p_table = alloc.swa_v2p_page_table
                self._swa_page_multiplier = alloc.swa_kernel_page_multiplier
                self._swa_write_loc_from_full = self._swa_write_loc_unified
            else:
                self._swa_v2p_table = None
                self._swa_page_multiplier = 1
                self._swa_write_loc_from_full = None
        else:
            self._full_v2p_table = None
            self._full_p2v_table = None
            self._full_page_multiplier = 1
            self._translate_full = None
            self._swa_v2p_table = None
            self._swa_page_multiplier = 1
            self._swa_write_loc_from_full = (
                token_to_kv_pool.translate_loc_from_full_to_swa
                if isinstance(token_to_kv_pool, SWAKVPool)
                else None
            )

        self._rows: Optional[torch.Tensor] = (
            torch.arange(req_to_token.shape[0], dtype=torch.int64, device=device)
            if self.is_translating
            else None
        )
        self._index_table_memo: Optional[Tuple[weakref.ref, KVIndexTable]] = None

    def make_capture_tables(
        self, *, max_bs: int, max_context_len: int
    ) -> Optional[KVReadTables]:
        """Capture-stable destinations for a backend to own, or None when this
        pool needs no translation and the backend will never fill any.

        Zero-filled: entry 0 is the reserved padding slot in every id space, so
        a captured graph replaying before its first refresh reads padding, not
        garbage.
        """
        if not self.is_translating:
            return None
        max_pages = -(-max_context_len // self.page_size)

        def _zeros():
            return torch.zeros(
                (max_bs, max_pages), dtype=torch.int32, device=self.device
            )

        return KVReadTables(
            full=_zeros(),
            sliding_window=_zeros() if self._swa_v2p_table is not None else None,
        )

    # -- per-batch view --------------------------------------------------------

    def build_index_table(
        self,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        max_pages: Optional[int] = None,
        into: Optional[KVReadTables] = None,
    ) -> KVIndexTable:
        """The one per-batch entry point.

        Non-unified: the raw ``(req_to_token, req_pool_indices)`` passthrough,
        no tensor ops and no copies. Unified: fills each row's live prefix and
        returns the table WHOLE, so a caller needing a stable pointer (a
        captured graph bakes it) passes its own tables in ``into``;
        ``into=None`` allocates of width ``max_pages`` instead.
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
        if into is not None:
            out_full = into.full
            out_swa = into.sliding_window
            # A caller-owned table may be padded wider than req_to_token's span
            # (trtllm_mla / flashmla pad to a page-count bound); the columns
            # past it have no source to read, so stop there.
            width = min(
                out_full.shape[1] if max_pages is None else max_pages,
                -(-self.req_to_token.shape[1] // self.page_size),
            )
        else:
            assert max_pages is not None, (
                "KVIndexTranslator.build_index_table: allocating needs max_pages "
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
    ) -> None:
        """`build_index_table(into=...)` for a caller that owns a bare block
        table rather than a KVReadTables: trtllm_mla / flashmla consume that
        table directly, its rows already being the index table's rows.
        """
        assert (
            self.is_translating
        ), "KVIndexTranslator.fill_read_table on a pool that needs no translation"
        self.build_index_table(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            into=KVReadTables(full=out, sliding_window=None),
        )

    def index_table_for_batch(self, forward_batch) -> KVIndexTable:
        """Eager per-batch view, memoized in one slot keyed by batch identity
        so multi-consumer metadata builds share a build. The next batch
        replaces the slot; consumers only read during their own build. The
        captured path does not memoize -- it refreshes its buffers per
        replay."""
        memo = self._index_table_memo
        if memo is not None and memo[0]() is forward_batch:
            return memo[1]
        max_pages = None
        if self.is_translating:
            # `seq_lens_cpu` is a non-None but STALE slice on a gpu_only
            # batch; `seq_lens_sum` is the signal that it is live. A stale max
            # under-sizes the table and the tail then reads as the sink.
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

    def assert_backends_carry_translator(self, backends) -> None:
        """Boot guard: under the unified pool every backend a forward can reach
        must carry THIS translator."""
        if not self.is_translating:
            return
        for backend in backends:
            if backend is None:
                continue
            assert backend.kv_index_translator is self, (
                f"{type(backend).__name__} does not carry the runner's "
                "KVIndexTranslator. A backend (or wrapper) reachable under "
                "--enable-unified-memory must forward `kv_index_translator`, or "
                "read-index producers silently skip the virtual->kernel-facing "
                "translation."
            )

    # -- write loc (phase 1; phase 2 lives in build_index_table) ----------------

    def rebind_write_loc(self, forward_batch) -> None:
        """Phase 1 of the WRITE contract: translate the batch's write loc to
        FULL-side kernel-facing ids exactly once, at ForwardBatch
        construction. No-op on non-unified pools.

        REBIND, never mutate: the translate returns a FRESH tensor, so the
        ScheduleBatch's aliased tensor stays VIRTUAL for the radix / accept /
        in-flight machinery that reads it.
        """
        self._index_table_memo = None
        if not self.is_translating or forward_batch.out_cache_loc is None:
            return
        forward_batch.out_cache_loc = self._translate_full(forward_batch.out_cache_loc)

    def sliding_window_write_loc_for(
        self, out_cache_loc: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """This batch's sliding-window write loc, or None when there is no loc
        this forward or the pool has no sliding-window id space."""
        if out_cache_loc is None or self._swa_write_loc_from_full is None:
            return None
        return self._swa_write_loc_from_full(out_cache_loc)

    def _swa_write_loc_unified(self, kernel_loc: torch.Tensor) -> torch.Tensor:
        """Sliding-window write loc, derived pointwise from FULL-side
        kernel-facing values (phase 2 of the write contract).
        """
        full_stride = self.page_size * self._full_page_multiplier
        offset = kernel_loc % full_stride  # == virtual_token % page_size
        # An unmapped physical page reads back as -1; clamp it rather than let
        # the gather wrap onto the v2p table's last element.
        virt_page = self._full_p2v_table[kernel_loc // full_stride].clamp_(min=0)
        swa_stride = self.page_size * self._swa_page_multiplier
        return (self._swa_v2p_table[virt_page] * swa_stride + offset).clamp_(min=0)

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
