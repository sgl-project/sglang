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

A KV slot can be named in two id spaces:

  * **virtual** - what `req_to_token` stores. Keeps naming the same logical
    slot even after the pool moves data around.
  * **physical** - where that slot sits in the pool right now, which is also
    what a kernel indexes the per-layer K/V views with ("kernel-facing" in
    older names means the same id).

The two coincide on a plain pool, so nothing here does any work there.

A pool can hold several sub-pools (full attention, sliding window), each with
its own physical ids; this runner's translator names each one its forwards
touch with an `IdSpace` (`space`). Every id of an iteration is translated once
per sub-pool, by the iteration's `KVLocPlan` (`plan`, `own_plan`): its write
window, and a page table over its rows. A backend never translates; it asks
for the sub-pool it indexes (`write_ids`, and the readers below, by
`IdSpaceKind`), and the readers answer "what do I gather from, and which row
is mine?" with a `KVIndexTable`:

    ids[row_ids[b], pos]

    plain pool : ids = req_to_token, row_ids = req_pool_indices (those very
                 objects - no copy, no kernel)
    unified    : ids = the plan's array of physical page ids,
                 row_ids = arange(batch_size)

Backends call their own copy a *page table* (fa3) or a *block table*
(trtllm); here it is the **index table**. `read_source` hands a gather kernel
the table, `pack_read_stream` packs a CSR stream from it (the table's first
reader gets its stream from the launch that builds the table), and
`copy_page_table` copies it into a captured graph's table.

Converting only ever rewrites the page number and keeps the in-page offset, so
one page-granular table serves both kinds of consumer: a block-table backend
uses its rows as-is, and one that wants flat per-token ids rebuilds them as

    token_id = entry * entry_page_size + pos % entry_page_size

The one translation outside a plan is DCP's read production
(`translate_dcp_read_ids`): which ids a rank reads is only known where the
DCP index kernels select them.
"""

from __future__ import annotations

from typing import Dict, Optional

import msgspec
import torch

from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashinfer_kv_indices_triton,
)
from sglang.kernels.ops.kvcache.kv_read_table import (
    build_kv_read_table,
    build_kv_read_table_and_stream,
)
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWAAllocatorBase,
)
from sglang.srt.mem_cache.allocator.unified_mamba import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.kv_loc_plan import (
    IdSpace,
    IdSpaceKind,
    KVLocPlan,
    pad_with_sink,
    window_read_extent,
)
from sglang.srt.mem_cache.unified_draft_pool import fused_draft_host_allocator
from sglang.srt.runtime_context import get_parallel


class KVIndexTable(msgspec.Struct, frozen=True):
    """Collection of what one batch gathers from."""

    ids: torch.Tensor  # 2-D array of KV ids to gather from
    row_ids: torch.Tensor  # which row belongs to batch lane b
    row_stride: int  # stride between rows of `ids`, in elements
    entry_page_size: int  # what one entry covers: 1 = a token, N = a page of N
    is_translated: bool  # entries are already physical ids


class KVReadStream(msgspec.Struct, frozen=True):
    """A CSR stream a paged wrapper plans over: ``seq_lens[b]`` ids of lane
    ``b`` from ``kv_start_idx[b]``, landing at ``indptr[b]`` in ``out``."""

    seq_lens: torch.Tensor
    indptr: torch.Tensor
    out: torch.Tensor
    kv_start_idx: Optional[torch.Tensor]


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

        is_unified_target = (
            isinstance(
                token_to_kv_pool_allocator,
                (UnifiedMambaTokenToKVPoolAllocator, UnifiedSWAAllocatorBase),
            )
            and token_to_kv_pool_allocator.get_kvcache() is token_to_kv_pool
        )
        host_allocator = fused_draft_host_allocator(token_to_kv_pool)
        is_fused_draft = (
            host_allocator is not None and host_allocator is token_to_kv_pool_allocator
        )
        self.is_translating = is_unified_target or is_fused_draft
        # The sub-pools this runner's forwards write and read, each with how
        # it names the iteration's slots (`IdSpace`).
        self._spaces: Dict[IdSpaceKind, IdSpace] = {}
        if self.is_translating:
            alloc = token_to_kv_pool_allocator
            self._capture_page_size = alloc.page_size
            self._full_v2p_table = alloc.full_v2p_page_table
            self._translate_full = alloc.translate_kv_loc
            # DCP read ids stay WIDENED to the consumer: selecting this rank's
            # share changes the length, so only the production site can do it.
            self.defer_read_translate = get_parallel().attn_dcp_size > 1
            # Keyed by the allocator, which owns the tables: a fused draft's
            # runner shares the target's full sub-pool, and so its key.
            self._spaces[IdSpaceKind.FULL] = IdSpace(
                key=(IdSpaceKind.FULL, id(alloc)),
                # The WRITE loc is the one id that arrives DCP-WIDENED: read
                # indices are collapsed by the DCP index kernels, `out_cache_loc`
                # still carries the owner rule in `loc % dcp_size`.
                write=alloc.translate_write_loc,
                read_v2p=None if self.defer_read_translate else self._full_v2p_table,
            )
            routes_window_layers = is_unified_target or isinstance(
                token_to_kv_pool, BaseSWAKVPool
            )
            if isinstance(alloc, UnifiedSWAAllocatorBase) and routes_window_layers:
                # Straight from the virtual ids through the sliding-window
                # sub-pool's own table.
                self._spaces[IdSpaceKind.SLIDING_WINDOW] = IdSpace(
                    key=(IdSpaceKind.SLIDING_WINDOW, id(alloc)),
                    write=alloc.translate_loc_from_full_to_swa,
                    read_v2p=(
                        None if self.defer_read_translate else alloc.swa_v2p_page_table
                    ),
                )
        else:
            self._capture_page_size = page_size
            self._full_v2p_table = None
            self._translate_full = None
            self.defer_read_translate = False
            self._spaces[IdSpaceKind.FULL] = IdSpace(key=(IdSpaceKind.FULL, None))
            # `translate_loc_from_full_to_swa` is abstract on `BaseSWAKVPool`,
            # which is also what the backends' `_resolve_swa_kv_pool` keys on.
            # Reads stay in `req_to_token`; the backends map them themselves.
            if isinstance(token_to_kv_pool, BaseSWAKVPool) and (
                not isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
                or token_to_kv_pool.request_window is None
            ):
                self._spaces[IdSpaceKind.SLIDING_WINDOW] = IdSpace(
                    key=(IdSpaceKind.SLIDING_WINDOW, id(token_to_kv_pool)),
                    write=token_to_kv_pool.translate_loc_from_full_to_swa,
                )

        self._rows: Optional[torch.Tensor] = (
            torch.arange(req_to_token.shape[0], dtype=torch.int64, device=device)
            if self.is_translating
            else None
        )

    def space(self, kind: IdSpaceKind) -> Optional[IdSpace]:
        """How this runner's ``kind`` sub-pool names the iteration's slots, or
        None when its pool has no such sub-pool."""
        return self._spaces.get(kind)

    def write_ids(
        self, forward_batch, kind: IdSpaceKind = IdSpaceKind.FULL
    ) -> Optional[torch.Tensor]:
        """The ids ``forward_batch`` writes in this runner's ``kind`` sub-pool:
        its plan's, for the part of the window it writes, as long as its
        `out_cache_loc` (lanes past the plan's -- a padded batch's, a captured
        graph's buffer -- name the sink). None when there is no write loc or
        no such sub-pool."""
        plan = getattr(forward_batch, "kv_loc_plan", None)
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if plan is None or out_cache_loc is None:
            return None
        ids = plan.write_ids(
            self, kind=kind, cols=getattr(forward_batch, "kv_loc_cols", None)
        )
        return pad_with_sink(ids, out_cache_loc.shape[0])

    def capture_token_capacity(self, max_token_pool_size: int) -> int:
        """Host capture rows are indexed by request-token IDs, not kernel IDs.

        Unified IDs span the whole virtual table even when admission is capped.
        DCP widens allocator pages; the runner's page size stays physical.
        """
        if self.is_translating:
            return self._full_v2p_table.numel() * self._capture_page_size
        return max_token_pool_size + self.page_size

    # -- the iteration's plan ----------------------------------------------------

    def plan(
        self,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        write_virtual: Optional[torch.Tensor],
        read_extent: int = 0,
        write_slots: Optional[torch.Tensor] = None,
    ) -> KVLocPlan:
        """This iteration's ids, translated once. ``write_virtual`` is the
        iteration's write window (``[batch, window]`` flattened when its
        forwards write columns of it); ``read_extent`` is how far past
        ``seq_lens`` the iteration's reads reach. ``write_slots`` replaces
        ``write_virtual`` for a runner's own buffer (`bind_runner_slots`)."""
        return KVLocPlan(
            source=self,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            write_virtual=write_virtual,
            read_extent=read_extent,
            write_slots=write_slots,
        )

    def own_plan(self, forward_batch, *, runner_slots: bool = False) -> KVLocPlan:
        """The plan of a forward that is its own iteration, from its own
        fields: its ``out_cache_loc`` is the write window (virtual, or with
        ``runner_slots`` a runner's own buffer), and its reads reach
        `window_read_extent` past its lengths."""
        write_ids = forward_batch.out_cache_loc
        seq_lens = getattr(forward_batch, "seq_lens", None)
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        encoder_lens = getattr(forward_batch, "encoder_lens", None)
        if (
            encoder_lens is not None
            and seq_lens is not None
            and self.reads_are_translated
        ):
            # An encoder-decoder row holds its encoder tokens ahead of the
            # decoder's, and its self-attention reads past `seq_lens`.
            seq_lens = seq_lens + encoder_lens
            encoder_lens_cpu = getattr(forward_batch, "encoder_lens_cpu", None)
            seq_lens_cpu = (
                None
                if seq_lens_cpu is None or encoder_lens_cpu is None
                else seq_lens_cpu + torch.tensor(encoder_lens_cpu).to(seq_lens_cpu)
            )
        return self.plan(
            req_pool_indices=getattr(forward_batch, "req_pool_indices", None),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            write_virtual=None if runner_slots else write_ids,
            write_slots=write_ids if runner_slots else None,
            read_extent=window_read_extent(
                getattr(forward_batch, "forward_mode", None),
                getattr(forward_batch, "spec_info", None),
                write_ids,
                getattr(forward_batch, "batch_size", 0),
            ),
        )

    def bind_runner_slots(self, forward_batch) -> None:
        """Bind a batch a runner builds over its own buffers, to capture a
        graph or to warm up with. Its write ids are the runner's buffer, used
        as it is; its reads plan over the runner's own rows and lengths."""
        self.own_plan(forward_batch, runner_slots=True).bind(forward_batch, self)

    def _build_iteration_table(
        self,
        plan: KVLocPlan,
        space: IdSpace,
        *,
        rows: Optional[int],
        previous: Optional[KVIndexTable],
        stream: Optional[KVReadStream] = None,
    ) -> KVIndexTable:
        """A plan's page table in one sub-pool: the passthrough where its reads
        stay virtual, else one build over ``[0, seq_lens + read_extent)``
        through its page table. Rows past the plan's batch (a captured graph's
        padded lanes) read the sink; a second, wider request copies the built
        rows instead of building them again. ``stream``, from the reader the
        build is for, is packed by the same launch."""
        if space.read_v2p is None:
            return self._passthrough_table(plan.req_pool_indices)
        bs = int(plan.req_pool_indices.numel())
        rows = max(bs, rows or 0)
        if previous is not None:
            assert stream is None, "a stream is packed by the table's first build"
            out = torch.zeros(
                (rows, previous.ids.shape[1]), dtype=torch.int32, device=self.device
            )
            out[: previous.ids.shape[0]].copy_(previous.ids)
        else:
            row_pages = -(-self.req_to_token.shape[1] // self.page_size)
            slc = plan.seq_lens_cpu
            if slc is not None and slc.numel() > 0:
                max_seq = int(slc.max()) + plan.read_extent
                width = min(max(-(-max_seq // self.page_size), 1), row_pages)
            else:
                width = row_pages
            # A fresh table: the build writes every column of the batch's rows,
            # the sink past each row's live prefix included.
            out = torch.empty((rows, width), dtype=torch.int32, device=self.device)
            if stream is not None:
                build_kv_read_table_and_stream(
                    req_to_token=self.req_to_token,
                    req_pool_indices=plan.req_pool_indices,
                    seq_lens=plan.seq_lens,
                    v2p=space.read_v2p,
                    page_size=self.page_size,
                    max_pages=width,
                    out=out,
                    stream_lens=stream.seq_lens,
                    stream_indptr=stream.indptr,
                    stream_out=stream.out,
                    stream_kv_start_idx=stream.kv_start_idx,
                    seq_len_delta=plan.read_extent,
                )
            else:
                if rows > bs:
                    out[bs:].zero_()
                build_kv_read_table(
                    req_to_token=self.req_to_token,
                    req_pool_indices=plan.req_pool_indices,
                    seq_lens=plan.seq_lens,
                    v2p=space.read_v2p,
                    page_size=self.page_size,
                    max_pages=width,
                    out=out,
                    zero_tail=True,
                    seq_len_delta=plan.read_extent,
                )
        return KVIndexTable(
            ids=out,
            row_ids=self._rows[:rows],
            row_stride=out.stride(0),
            entry_page_size=self.page_size,
            is_translated=True,
        )

    # -- readers: derive from the plan, never translate -------------------------

    def _reads_translated(self, kind: IdSpaceKind) -> bool:
        space = self._spaces.get(kind)
        return space is not None and space.read_v2p is not None

    def read_source(
        self,
        plan: KVLocPlan,
        *,
        req_pool_indices: torch.Tensor,
        bs: int,
        kind: IdSpaceKind = IdSpaceKind.FULL,
    ) -> KVIndexTable:
        """Where lane ``b``'s ids in the ``kind`` sub-pool are gathered from:
        the plan's table when this runner reads it translated (lane ``b`` is
        the plan's row ``b``, padded lanes reading the sink), ``req_to_token``
        at the caller's ``req_pool_indices`` otherwise. For a gather kernel
        that takes a table, its rows and its entry granularity."""
        if self._reads_translated(kind):
            return self._plan_table(plan, kind=kind, rows=bs)
        return self._passthrough_table(req_pool_indices)

    def read_table(
        self,
        plan: KVLocPlan,
        *,
        kind: IdSpaceKind = IdSpaceKind.FULL,
        rows: Optional[int] = None,
    ) -> KVIndexTable:
        """The page table this runner reads in its ``kind`` sub-pool this
        iteration: the plan's when it reads translated ids (``rows`` past the
        plan's batch, a padded batch's lanes, reading the sink), the
        ``req_to_token`` passthrough otherwise (a static or private pool, or
        DCP, where the producing kernel selects this rank's share)."""
        if self._reads_translated(kind):
            return self._plan_table(plan, kind=kind, rows=rows)
        return self._passthrough_table(plan.req_pool_indices)

    def _plan_table(
        self,
        plan: KVLocPlan,
        *,
        kind: IdSpaceKind,
        rows: Optional[int] = None,
        stream: Optional[KVReadStream] = None,
    ):
        assert plan.is_read_by(self), (
            "a translating reader must read through the plan of its own "
            "req_to_token rows"
        )
        return plan.read_table(kind=kind, rows=rows, stream=stream)

    def _passthrough_table(self, req_pool_indices: torch.Tensor) -> KVIndexTable:
        return KVIndexTable(
            ids=self.req_to_token,
            row_ids=req_pool_indices,
            row_stride=self.req_to_token.stride(0),
            entry_page_size=1,
            is_translated=False,
        )

    def pack_read_stream(
        self,
        plan: KVLocPlan,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        indptr: torch.Tensor,
        out: torch.Tensor,
        kv_start_idx: Optional[torch.Tensor] = None,
        kind: IdSpaceKind = IdSpaceKind.FULL,
        token_mapping: Optional[torch.Tensor] = None,
    ) -> bool:
        """Fill ``out``'s CSR rows with the ids in the ``kind`` sub-pool a paged
        wrapper plans over and report whether they are physical: one gather
        from `read_source`. A ``False`` return means the ids are still VIRTUAL
        full-attention ids (a static pool, or DCP), for the caller to finish.
        A static SWA pool can pass its full->swa table as ``token_mapping`` to
        fuse that translation into the gather."""
        bs = int(seq_lens.numel())
        if self._reads_translated(kind) and not plan.has_read_table(kind):
            # The plan's first reader: the build that makes its table packs
            # this stream from the same gather, in the same launch.
            self._plan_table(
                plan,
                kind=kind,
                rows=bs,
                stream=KVReadStream(
                    seq_lens=seq_lens,
                    indptr=indptr,
                    out=out,
                    kv_start_idx=kv_start_idx,
                ),
            )
            return True
        src = self.read_source(
            plan, req_pool_indices=req_pool_indices, bs=bs, kind=kind
        )
        assert token_mapping is None or not src.is_translated
        create_flashinfer_kv_indices_triton[(bs,)](
            src.ids,
            src.row_ids[:bs],
            seq_lens,
            indptr,
            kv_start_idx,
            out,
            src.row_stride,
            ENTRY_PAGE_SIZE=src.entry_page_size,
            token_mapping=token_mapping,
        )
        return src.is_translated or token_mapping is not None

    def copy_page_table(
        self,
        plan: KVLocPlan,
        *,
        out: torch.Tensor,
        kind: IdSpaceKind = IdSpaceKind.FULL,
    ) -> None:
        """Copy the plan's page table in the ``kind`` sub-pool into ``out``, a
        capture-stable table a captured graph reads (one row per lane, padded
        lanes reading the sink). Columns past a row's live prefix keep stale
        values, which the kernels never read past their own lengths."""
        assert self._reads_translated(kind), (
            "copy_page_table: reads stay virtual here (a non-unified pool, or "
            "DCP, where the caller selects this rank's share itself)"
        )
        table = self._plan_table(plan, kind=kind, rows=out.shape[0])
        width = min(out.shape[1], table.ids.shape[1])
        out[:, :width].copy_(table.ids[: out.shape[0], :width])

    # -- dispositions ----------------------------------------------------------

    @property
    def reads_are_translated(self) -> bool:
        """Whether a read this translator fills comes out physical. False
        on a non-unified pool, and under DCP, where the ids stay VIRTUAL for
        ``translate_dcp_read_ids`` to finish."""
        return self._reads_translated(IdSpaceKind.FULL)

    @property
    def full_v2p_table(self) -> Optional[torch.Tensor]:
        """The full-attention virtual->physical PAGE table, or None when this
        pool needs no translation.

        For the DCP page-table builders, whose gather is over a rank's cyclic
        slice rather than a row prefix, so a plan's row-prefix table cannot
        serve them.
        """
        return self._full_v2p_table

    def bind_and_verify_backends(self, backends) -> None:
        """Boot: make every reachable backend carry THIS translator.

        Model-layer producers read it off `get_attn_backend()`, so an unset
        attribute is an unreachable hook, not "no translation needed".
        """
        for backend in backends:
            if backend is None:
                continue
            if backend.kv_index_translator is None:
                backend.kv_index_translator = self
                continue
            assert backend.kv_index_translator is self, (
                f"{type(backend).__name__} carries a KVIndexTranslator that is "
                "not this runner's. A wrapper must forward the inner backend's "
                "copy, not build its own."
            )

    def bind_own_plan(self, forward_batch) -> None:
        """Bind a batch built outside `ForwardBatch.init_new` that is its own
        iteration to a plan of its own fields (`own_plan`). A forward that
        shares its iteration's window takes that plan instead."""
        self.own_plan(forward_batch).bind(forward_batch, self)

    # -- DCP read production ---------------------------------------------------

    @property
    def needs_read_translate(self) -> bool:
        """Whether `translate_dcp_read_ids` is anything but the identity, so a
        hot path can skip the call rather than round-trip a no-op copy."""
        return self.is_translating or get_parallel().attn_dcp_size > 1

    def translate_dcp_read_ids(self, widened_ids: torch.Tensor) -> torch.Tensor:
        """Widened logical READ ids -> physical ids, for either pool.

        The one hook every DCP read-index production site calls; on a static
        pool `widened // dcp_size` IS the whole virtual->physical translation.
        """
        dcp_size = get_parallel().attn_dcp_size
        if dcp_size > 1:
            widened_ids = widened_ids // dcp_size
        if not self.is_translating:
            return widened_ids
        return self._translate_full(widened_ids)
