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
"""One scheduler iteration's KV slot ids, per sub-pool.

Under the unified pool a KV slot has a virtual id (what the scheduler allocates
and `req_to_token` stores) and, in each sub-pool of the pool, a physical id
(what that sub-pool's kernels index). Every forward of one iteration -- the
target's, and a fused draft's, which lives in the target's pages -- writes the
same slots and reads the same rows, and compaction does not move a page while
the iteration's forwards are in flight.

    writes   `write_ids`    the iteration's write window, translated once per
                            sub-pool, here; every writer binds it
    reads    `read_table`   the rows' page table over [0, seq_lens + extent),
                            built here only when a reader of these rows reads
                            the table form, and then shared by every reader

A read is translated exactly once on its way to the kernel that consumes it,
and only by the translator's read primitives (`KVIndexTranslator`): from the
shared table when there is one, otherwise in the reader's own gather, straight
into the reader's own buffer -- so no table is built that no reader needs. A
consumer never translates and never sees which kind of pool it reads.

A sub-pool is named by its `IdSpace`: how it maps the iteration's virtual ids,
and the page table its reads go through. A consumer asks for ids in the space
it indexes -- its runner's translator holds one per sub-pool its pool has
(`KVIndexTranslator.space`) -- and the plan derives each space's write ids on
first use. A space that indexes virtual ids as they are (a static pool's, a
private draft pool's) costs nothing.

Who builds the plan: `ForwardBatch.init_new` for a forward that is the only one
in its iteration; a speculative worker, once per iteration, as soon as the
iteration's slots are allocated, handing the same plan to the draft and target
forwards and to the direct KV writers; a graph runner, for a batch it builds
over its own buffers to capture or warm up with (`write_slots`).
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Callable, Dict, Hashable, Optional, Union

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_index_translator import (
        KVIndexTable,
        KVIndexTranslator,
        KVReadStream,
    )


class IdSpaceKind(enum.Enum):
    """A sub-pool of a runner's KV pool, as the consumers that index it name
    it."""

    # Full-attention layers: the pool's own token ids.
    FULL = enum.auto()
    # The sliding-window layers of a hybrid pool: the same token ids through
    # the sliding-window sub-pool's table.
    SLIDING_WINDOW = enum.auto()


class IdSpace(msgspec.Struct, frozen=True):
    """How one sub-pool names the iteration's slots.

    ``write`` maps the iteration's virtual token ids to this sub-pool's ids
    (None: they index it as they are). ``read_v2p`` is the page table its reads
    go through (None: reads stay in `req_to_token`, for the consumer to
    finish). Runners whose pools share a sub-pool share its ``key``, and with
    it a plan's ids for it."""

    key: Hashable
    write: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    read_v2p: Optional[torch.Tensor] = None


def window_read_extent(forward_mode, spec_info, write_ids, batch_size: int) -> int:
    """How far past ``seq_lens`` a forward reads: the window it writes, when
    its ``seq_lens`` do not count that window yet -- a verify, whose rows read
    at most ``draft_token_num`` past them (a ragged verify writes fewer for
    some rows); a speculative draft decode, which reads back its own earlier
    steps; and a draft extend, whose lengths take its window only after its
    batch is built. Zero for every other forward, whose ``seq_lens`` already
    include what it writes."""
    if spec_info is None or write_ids is None or not batch_size:
        return 0
    if forward_mode.is_target_verify():
        return spec_info.draft_token_num
    if forward_mode.is_decode() or forward_mode.is_draft_extend_v2():
        return write_ids.numel() // batch_size
    return 0


class TokenSpan(msgspec.Struct, frozen=True):
    """A contiguous run ``[start, stop)`` of a write window's flat tokens:
    one half of a batch split in two (two-batch overlap). A ``slice`` in
    `Cols` names columns of every row; this names tokens of the flattened
    window, which a reader takes as a view."""

    start: int
    stop: int


# Part of a `[batch, window]` write window: a slice of its columns, a
# contiguous run of its flat tokens, or a flat index into it (-1: the sink).
Cols = Union[slice, TokenSpan, torch.Tensor]


def pad_with_sink(ids: Optional[torch.Tensor], n: int) -> Optional[torch.Tensor]:
    """``ids`` padded to ``n`` entries with the sink (id 0 in every space), for
    a batch whose write ids cover more lanes than its plan's (a padded batch,
    a captured graph's buffer)."""
    if ids is None or ids.shape[0] >= n:
        return ids
    padded = ids.new_zeros(n)
    padded[: ids.shape[0]].copy_(ids)
    return padded


class KVLocPlan:
    """The ids of one iteration, in every sub-pool, each computed once."""

    def __init__(
        self,
        *,
        source: KVIndexTranslator,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        write_virtual: Optional[torch.Tensor],
        read_extent: int = 0,
        write_slots: Optional[torch.Tensor] = None,
    ):
        self._source = source
        self.req_pool_indices = req_pool_indices
        # Committed lengths when the plan is built; every read this iteration
        # stays within `seq_lens + read_extent` (the window it writes).
        self.seq_lens = seq_lens
        self.seq_lens_cpu = seq_lens_cpu
        self.read_extent = read_extent
        full = source.space(IdSpaceKind.FULL)
        if write_slots is not None:
            # A runner's own write buffer, for a graph capture or a warmup run:
            # in its own pool's ids by construction, naming the sink until a
            # replay fills it, and kept by address in a captured graph. Used
            # as it is.
            assert write_virtual is None
            self.write_virtual = None if full.write is not None else write_slots
            self.write_physical = write_slots
        else:
            # Aliases the ScheduleBatch's tensor, which stays virtual for the
            # radix tree, the accept path and lazy compaction's in-flight write
            # set.
            self.write_virtual = write_virtual
            self.write_physical = (
                full.write(write_virtual)
                if full.write is not None and write_virtual is not None
                else write_virtual
            )
        self._full_key = full.key
        # The window in each sub-pool's ids, and each sub-pool's read table,
        # by space key.
        self._write_ids: Dict[Hashable, Optional[torch.Tensor]] = {
            full.key: self.write_physical
        }
        self._read_tables: Dict[Hashable, KVIndexTable] = {}

    # -- writes ----------------------------------------------------------------

    def bind(self, batch, reader: KVIndexTranslator, *, cols: Optional[Cols] = None):
        """Give ``batch`` (a ForwardBatch, or a view standing in for one) this
        plan, the part of its window it writes (``cols``), and its write ids in
        the full-attention ids `reader`'s pool indexes. The one way a forward
        gets its write ids; nothing here translates. A consumer of another
        sub-pool takes its ids from the plan (`KVIndexTranslator.write_ids`)."""
        batch.kv_loc_plan = self
        batch.kv_loc_cols = cols
        batch.out_cache_loc = self.write_ids(reader, cols=cols)
        batch.out_cache_loc_virtual = (
            self.virtual_write_ids(cols=cols)
            if self.is_translated_for(reader)
            else None
        )
        # A forward with no write loc writes nothing and stays unmarked.
        batch.out_cache_loc_is_physical = batch.out_cache_loc is not None

    def bind_replay(self, batch, reader: KVIndexTranslator, *, slots: torch.Tensor):
        """`bind`, for the batch a captured graph replays this iteration's
        forward with. Its write ids are ``slots``, the runner's capture-stable
        buffer that already holds this plan's write ids and names the sink past
        them. The virtual mirror stays with the live batch."""
        self.bind(batch, reader)
        batch.out_cache_loc = slots
        batch.out_cache_loc_virtual = None

    def write_ids(
        self,
        reader: KVIndexTranslator,
        *,
        kind: IdSpaceKind = IdSpaceKind.FULL,
        cols: Optional[Cols] = None,
    ) -> Optional[torch.Tensor]:
        """The write window as `reader`'s ``kind`` sub-pool indexes it, or None
        when its pool has no such sub-pool. ``cols`` selects part of a
        ``[batch, window]`` window, for a forward that writes only part of it:
        a slice of columns (a draft step ahead of the verify window it opens),
        or a flat index into the window, -1 naming the sink (a ragged verify's
        packed rows)."""
        space = reader.space(kind)
        if space is None:
            return None
        if space.key not in self._write_ids:
            self._write_ids[space.key] = self._window_in(space)
        return self._cols(self._write_ids[space.key], cols)

    def _window_in(self, space: IdSpace) -> Optional[torch.Tensor]:
        if self.write_virtual is None:
            # No window this iteration, or a runner's own slots, which name
            # the sink in every other sub-pool too.
            if self.write_physical is None:
                return None
            return torch.zeros_like(self.write_physical)
        if space.write is None:
            return self.write_virtual
        return space.write(self.write_virtual)

    def virtual_write_ids(self, *, cols: Optional[Cols] = None):
        """The same columns in the virtual space: the ids the scheduler's
        bookkeeping (lazy compaction's in-flight write set, the radix tree)
        names."""
        return self._cols(self.write_virtual, cols)

    def _cols(self, ids: Optional[torch.Tensor], cols: Optional[Cols]):
        if ids is None or cols is None:
            return ids
        if isinstance(cols, TokenSpan):
            return ids[cols.start : cols.stop]
        if isinstance(cols, torch.Tensor):
            return torch.where(cols >= 0, ids[cols.clamp(min=0)], 0)
        bs = int(self.req_pool_indices.numel())
        if bs == 1:
            # One row: its columns are a slice of the window itself.
            return ids[cols]
        return ids.view(bs, -1)[:, cols].reshape(-1)

    def cols_slice(self, cols: Optional[Cols], tokens: slice) -> Cols:
        """The ``tokens`` of a forward that writes ``cols``: the columns of
        one half of a batch split in two (two-batch overlap). A contiguous
        half stays a `TokenSpan`; only columns spread over the rows become a
        flat index into the window. Tokens past the window are left out; the
        consumer pads them with the sink."""
        if isinstance(cols, torch.Tensor):
            return cols[tokens]
        window = (
            self.write_virtual
            if self.write_virtual is not None
            else self.write_physical
        )
        if cols is None or isinstance(cols, TokenSpan):
            base = 0 if cols is None else cols.start
            end = window.numel() if cols is None else cols.stop
            start, stop, step = tokens.indices(end - base)
            if step == 1:
                return TokenSpan(base + start, base + max(start, stop))
        index = torch.arange(window.numel(), device=window.device)
        return self._cols(index, cols)[tokens]

    def reads_from(
        self,
        source: KVIndexTranslator,
        *,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        read_extent: int,
    ) -> KVLocPlan:
        """This plan's write ids, with reads planned over ``source``'s own
        `req_to_token` rows and lengths -- a draft that reads a compact table
        of its own while writing the iteration's window. The write ids are
        shared, not translated again."""
        plan = KVLocPlan(
            source=source,
            req_pool_indices=self.req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            write_virtual=None,
            read_extent=read_extent,
        )
        plan.write_virtual = self.write_virtual
        plan.write_physical = self.write_physical
        plan._write_ids = self._write_ids
        return plan

    # -- reads -----------------------------------------------------------------

    def has_read_table(self, kind: IdSpaceKind = IdSpaceKind.FULL) -> bool:
        """Whether a reader has had the ``kind`` table built yet."""
        return self._source.space(kind).key in self._read_tables

    def read_table(
        self,
        *,
        kind: IdSpaceKind = IdSpaceKind.FULL,
        rows: Optional[int] = None,
        stream: Optional[KVReadStream] = None,
        into: Optional[torch.Tensor] = None,
    ) -> KVIndexTable:
        """The rows' page table in the ``kind`` sub-pool over ``[0, seq_lens +
        read_extent)``, built on first use and shared by every reader of the
        iteration. ``rows`` past the plan's batch (a captured graph's padded
        lanes) are appended reading the sink, by copying, never by translating
        again; a reader that asks for ``rows`` gets exactly that many, a view of
        a table a wider reader built. On a sub-pool whose reads stay virtual
        (static, or DCP, where the producing kernel selects this rank's share)
        it is the `req_to_token` passthrough. ``stream`` is the first reader's
        CSR stream (no table yet), packed from the gather that builds the
        table; ``into``, the first reader's own capture-stable table, built in
        place to serve as the plan's for this iteration."""
        space = self._source.space(kind)
        table = self._read_tables.get(space.key)
        if table is None or (
            table.is_translated and rows is not None and table.ids.shape[0] < rows
        ):
            table = self._source.build_iteration_table(
                self, space, rows=rows, previous=table, stream=stream, into=into
            )
            self._read_tables[space.key] = table
        else:
            assert stream is None and into is None, (
                "a stream or a destination goes to the table's first build"
            )
        if rows is not None and table.ids.shape[0] > rows:
            # Built wider, for a captured graph's padded lanes: this reader's
            # kernels size their batch by the table's rows.
            table = msgspec.structs.replace(
                table, ids=table.ids[:rows], row_ids=table.row_ids[:rows]
            )
        return table

    def is_read_by(self, reader: KVIndexTranslator) -> bool:
        """Whether `reader` reads this plan's tables: it indexes the plan's
        physical ids and gathers the same `req_to_token` rows. A reader with a
        table of its own (a draft's compact `req_to_token`) plans its own
        reads."""
        return (
            self.is_translated_for(reader)
            and reader.req_to_token is self._source.req_to_token
        )

    def is_translated_for(self, reader: KVIndexTranslator) -> bool:
        """Whether `reader`'s pool indexes this plan's physical ids: the
        target's, and a fused draft's, which lives in the target's full
        sub-pool. A pass-through reader (a static pool, a private draft pool)
        indexes the virtual ids."""
        full = reader.space(IdSpaceKind.FULL)
        return full.write is not None and full.key == self._full_key
