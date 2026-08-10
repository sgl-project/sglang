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
"""The read-path id-space choke point for the unified memory pool.

ALL knowledge of kernel-facing KV id spaces lives here: which translate the
active pool wants (dense vs physical), the v2p tables, the multipliers, and
the canonical per-batch page tables built from them. Attention backends
consume a `KVIndexBatchView` and stay id-space-blind:

    non-unified: table = req_to_token, rows = req_pool_indices — the EXACT
                 objects backends read today; zero tensor ops, strict
                 passthrough (byte-identical to the non-unified path today).
    unified:     table = the canonical page table (page-granular int32,
                 entries already kernel-facing), rows = arange(bs).

Design invariant making ONE table sufficient: every kernel-facing id space is
page-affine over virtual ids with in-page offsets preserved —
``entry(p) = f(p)``, ``token(t) = entry * ps + t % ps`` — so the same table
serves padded-2D page-table backends directly (fa3/trtllm block tables ARE
its rows) and token-level CSR builders via the affine reconstruction
(``SRC_PAGE_SIZE`` on the gather kernels).

SWA models get a PARALLEL swa table built DIRECTLY from virtual
``req_to_token`` through the swa sub-pool's own v2p — never chained through
full-physical (matching the write rail: the swa loc is computed from the
still-virtual loc). In-kernel ``full_to_swa`` gathers remain a static-pool
mechanism only; the view carries that legacy mapping for them.

Capture contract (mirroring the write-loc rebind): capture-stable buffers are
zero-filled — entry 0 is the page-0 sink in EVERY id space, so capture
batches (zero-filled inputs) are safe by construction; replay prep refreshes
the live prefix IN PLACE and never rebinds a captured buffer; stale tails are
the fa3 contract (consumers bound reads by cache_seqlens). int32 entry bounds
are guaranteed at pool construction (`_assert_dense_id_bound`).
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.kvcache.kernel_page_table import build_kernel_page_table
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


class KVIndexBatchView(msgspec.Struct, frozen=True):
    """What a backend's index builders consume for one batch — nothing more.

    A builder gathers ``table[rows[b], pos]`` exactly as it gathers
    ``req_to_token[req_pool_indices[b], pos]`` today; ``src_page_size`` tells
    token-level consumers how to reconstruct token ids
    (``token = table_entry * src_page_size + pos % src_page_size``; 1 means
    the table is already token-granular). No backend branches on id spaces.
    """

    table: torch.Tensor  # 2-D id table the index builders gather from
    rows: torch.Tensor  # row per batch lane (req_pool_indices | arange(bs))
    row_stride: int  # table row stride, in elements
    src_page_size: int  # granularity of `table` entries (1 = token-granular)
    kernel_facing: bool  # True: entries are already kernel-facing ids
    swa_table: Optional[torch.Tensor]  # unified SWA models: the swa canonical
    full_to_swa_map: Optional[torch.Tensor]  # static SWA pools: legacy mapping


class KVIndexSource:
    """Built once per ModelRunner; the ONLY code that probes the allocator's
    id-space capabilities (dense translate, v2p tables, multipliers)."""

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

        # Capability probe — by TYPE, once. The unified composites (and only
        # they) carry the kernel-facing id surface added in D3; anything else
        # is a static pool and gets the strict passthrough.
        self.enabled = isinstance(
            token_to_kv_pool_allocator,
            (UnifiedMambaTokenToKVPoolAllocator, UnifiedSWATokenToKVPoolAllocator),
        )
        if self.enabled:
            alloc = token_to_kv_pool_allocator
            self._full_v2p = alloc.full_v2p_page_table
            self._full_mult = alloc.kernel_page_multiplier
            self._translate_full = alloc.translate_kv_loc_dense
            if isinstance(alloc, UnifiedSWATokenToKVPoolAllocator):
                self._swa_v2p = alloc.swa_v2p_page_table
                self._swa_mult = alloc.swa_kernel_page_multiplier
                self._translate_swa = token_to_kv_pool.translate_loc_from_full_to_swa
            else:
                self._swa_v2p = None
                self._swa_mult = 1
                self._translate_swa = None
            self._static_full_to_swa = None
        else:
            self._full_v2p = None
            self._full_mult = 1
            self._translate_full = None
            self._swa_v2p = None
            self._swa_mult = 1
            self._translate_swa = None
            self._static_full_to_swa = (
                token_to_kv_pool.full_to_swa_index_mapping
                if isinstance(token_to_kv_pool, SWAKVPool)
                else None
            )

        # Lazily-grown arange for the unified view's `rows`; replaced by the
        # capture-sized buffer (stable pointer) once capture buffers exist.
        self._rows: Optional[torch.Tensor] = None
        self._cap_full: Optional[torch.Tensor] = None
        self._cap_swa: Optional[torch.Tensor] = None

    # -- capture-stable buffers ------------------------------------------------

    def ensure_capture_buffers(self, *, max_bs: int, max_context_len: int) -> None:
        """Idempotent. Zero-filled ``(max_bs, ceil(ctx/ps))`` int32 per kind —
        zeros are the sink in every id space, so a captured graph replaying
        over a never-refreshed buffer reads the page-0 sink, not garbage."""
        if not self.enabled or self._cap_full is not None:
            return
        max_pages = -(-max_context_len // self.page_size)
        self._cap_full = torch.zeros(
            (max_bs, max_pages), dtype=torch.int32, device=self.device
        )
        if self._swa_v2p is not None:
            self._cap_swa = torch.zeros(
                (max_bs, max_pages), dtype=torch.int32, device=self.device
            )
        if self._rows is None or self._rows.numel() < max_bs:
            self._rows = torch.arange(
                max_bs, dtype=torch.int64, device=self.device
            )

    # -- per-batch view --------------------------------------------------------

    def batch_view(
        self,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        max_pages: Optional[int] = None,
        captured: bool = False,
    ) -> KVIndexBatchView:
        """The one per-batch entry point.

        Non-unified: returns the raw ``(req_to_token, req_pool_indices)``
        passthrough — no tensor ops, no copies. Unified: builds the canonical
        table(s) — a fresh zeros table of width ``max_pages`` (eager), or the
        live-prefix refresh of the capture buffers (``captured=True``; the
        returned ``table`` is the WHOLE buffer so its pointer is
        capture-bakeable).

        ``seq_lens`` is deliberately a caller-supplied tensor rather than
        read off a batch: mode knowledge stays with the caller. In
        particular, TARGET_VERIFY builders pass the WIDENED lengths
        (``seq_lens + draft_len``) so the table covers the draft slots —
        building a verify table from un-widened lengths leaves the draft
        pages on the sink (pinned by TestVerifyWidenedLengths).
        """
        if not self.enabled:
            return KVIndexBatchView(
                table=self.req_to_token,
                rows=req_pool_indices,
                row_stride=self.req_to_token.stride(0),
                src_page_size=1,
                kernel_facing=False,
                swa_table=None,
                full_to_swa_map=self._static_full_to_swa,
            )

        bs = int(req_pool_indices.numel())
        if captured:
            assert self._cap_full is not None, (
                "KVIndexSource.batch_view(captured=True) before "
                "ensure_capture_buffers()"
            )
            out_full = self._cap_full
            out_swa = self._cap_swa
            width = out_full.shape[1] if max_pages is None else max_pages
        else:
            assert max_pages is not None, (
                "KVIndexSource.batch_view: eager path needs max_pages "
                "(from the batch's seq_lens_cpu max)"
            )
            width = max_pages
            out_full = torch.zeros(
                (bs, width), dtype=torch.int32, device=self.device
            )
            out_swa = (
                torch.zeros((bs, width), dtype=torch.int32, device=self.device)
                if self._swa_v2p is not None
                else None
            )

        build_kernel_page_table(
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            v2p=self._full_v2p,
            multiplier=self._full_mult,
            page_size=self.page_size,
            max_pages=width,
            out=out_full,
        )
        if out_swa is not None:
            # Directly from VIRTUAL ids through the swa side's own v2p — never
            # chained through full-physical.
            build_kernel_page_table(
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                v2p=self._swa_v2p,
                multiplier=self._swa_mult,
                page_size=self.page_size,
                max_pages=width,
                out=out_swa,
            )
        return KVIndexBatchView(
            table=out_full,
            rows=self._rows_for(bs),
            row_stride=out_full.stride(0),
            src_page_size=self.page_size,
            kernel_facing=True,
            swa_table=out_swa,
            full_to_swa_map=None,
        )

    def rebind_write_loc(self, forward_batch) -> None:
        """The WRITE half of the id-space contract, folded
        onto the choke point.

        Translate the forward batch's write loc to KERNEL-FACING ids exactly
        once, at ForwardBatch preparation. REBIND, never mutate: the translate
        returns a FRESH tensor, so the ScheduleBatch's aliased tensor stays
        VIRTUAL (radix/accept/inflight machinery reads it). ORDER-CRITICAL for
        hybrid SWA: one virtual id maps to TWO kernel-facing ids — the swa
        rail must be computed from the still-VIRTUAL loc BEFORE the full-side
        rebind replaces it. No-op (byte-identical) on non-unified pools.

        Also stashes this source on the batch for the read-rail producers
        (the deepseek MHA mixin) that translate req_to_token-derived indices
        at production.
        """
        if not self.enabled or forward_batch.out_cache_loc is None:
            return
        if self._translate_swa is not None:
            # int32 — the shared read-index kernel convention (dtype preserved).
            forward_batch.swa_out_cache_loc = self._translate_swa(
                forward_batch.out_cache_loc
            )
        forward_batch.out_cache_loc = self._translate_full(
            forward_batch.out_cache_loc
        )
        forward_batch.out_cache_loc_is_physical = True
        forward_batch._kv_index_source = self

    def build_into(
        self,
        *,
        out: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Fill a backend-owned padded 2-D block table's live prefix with
        FULL-side canonical entries (trtllm_mla / flashmla consume such tables
        directly — their rows ARE the canonical rows).

        Prefix-only: columns past each row's live pages keep the backend's own
        fill (-1 sentinel or stale-but-unread values) — that tail is the
        consumer kernel's contract, not ours. Unified-only: callers dispatch on
        ``self.enabled`` and keep their static builder otherwise.
        """
        assert self.enabled, "KVIndexSource.build_into on a passthrough source"
        # Cap at the widest legal column: the table may be padded past the
        # context (alignment constraints); a page can only START inside
        # req_to_token.
        max_pages = min(
            out.shape[1],
            -(-self.req_to_token.shape[1] // self.page_size),
        )
        return build_kernel_page_table(
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            v2p=self._full_v2p,
            multiplier=self._full_mult,
            page_size=self.page_size,
            max_pages=max_pages,
            out=out,
        )

    def view_for_forward_batch(self, forward_batch) -> KVIndexBatchView:
        """Eager per-batch view, stashed on the ForwardBatch so TBO children
        and multi-consumer forwards build it once. The captured path does NOT
        stash — it rebuilds into the capture-stable buffers at every replay
        prep via ``batch_view(captured=True)``."""
        view = forward_batch.kv_index_view
        if view is not None:
            return view
        max_pages = None
        if self.enabled:
            slc = forward_batch.seq_lens_cpu
            if slc is not None and slc.numel() > 0:
                max_seq = int(slc.max())
            else:
                # gpu_only batches carry no CPU mirror; bound by the table.
                max_seq = self.req_to_token.shape[1]
            max_pages = max(-(-max_seq // self.page_size), 1)
        view = self.batch_view(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            max_pages=max_pages,
        )
        forward_batch.kv_index_view = view
        return view

    def _rows_for(self, bs: int) -> torch.Tensor:
        if self._rows is None or self._rows.numel() < bs:
            self._rows = torch.arange(
                max(bs, 64), dtype=torch.int64, device=self.device
            )
        return self._rows[:bs]

    # -- token-level translate surface (the mixin / local-attn consumers) ------

    def translate_full(
        self, kv_indices: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Virtual token ids -> kernel-facing full-side ids (identity when the
        source is disabled — callers never branch)."""
        if not self.enabled:
            assert out is None, "passthrough translate takes no out="
            return kv_indices
        return self._translate_full(kv_indices, out=out)

