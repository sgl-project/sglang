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
"""One scheduler iteration's KV slot ids, translated once.

Under the unified pool a KV slot has a virtual id (what the scheduler allocates
and `req_to_token` stores) and a physical id (what kernels index). Every forward
of one iteration -- the target's, and a fused draft's, which lives in the
target's pages -- reads and writes the same slots through the same
virtual->physical table, and compaction does not move a page while the
iteration's forwards are in flight. So each id is translated once, here, and
every consumer reads the result:

    writes   `write_ids`      the iteration's write window
    reads    `read_table`     the rows' page table over [0, seq_lens + extent)

A consumer never translates. It asks the plan for ids in the space its pool
indexes: physical for a translating pool (the target, a fused draft), virtual
for a pool that indexes virtual ids (a static pool, a private draft pool). On a
static pool the plan does no work at all.

Who builds the plan: `ForwardBatch.init_new` for a forward that is the only one
in its iteration; a speculative worker, once per iteration, as soon as the
iteration's slots are allocated, handing the same plan to the draft and target
forwards and to the direct KV writers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_index_translator import (
        KVIndexTable,
        KVIndexTranslator,
    )


class KVLocPlan:
    """The ids of one iteration, both spaces, each computed once."""

    def __init__(
        self,
        *,
        source: KVIndexTranslator,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        write_virtual: Optional[torch.Tensor],
        read_extent: int = 0,
    ):
        self._source = source
        self.req_pool_indices = req_pool_indices
        # Committed lengths when the plan is built; every read this iteration
        # stays within `seq_lens + read_extent` (the window it writes).
        self.seq_lens = seq_lens
        self.seq_lens_cpu = seq_lens_cpu
        self.read_extent = read_extent
        # Aliases the ScheduleBatch's tensor, which stays virtual for the radix
        # tree, the accept path and lazy compaction's in-flight write set.
        self.write_virtual = write_virtual
        self.write_physical = (
            source._translate_write_full(write_virtual)
            if source.is_translating and write_virtual is not None
            else write_virtual
        )
        self._swa_write: Optional[torch.Tensor] = None
        self._read_table: Optional[KVIndexTable] = None

    # -- writes ----------------------------------------------------------------

    def bind(self, batch, reader: KVIndexTranslator, *, cols: Optional[slice] = None):
        """Give ``batch`` (a ForwardBatch, or a view standing in for one) its
        write ids from this plan, in the space `reader`'s pool indexes, and
        the plan itself for its reads. The one way a forward gets its write
        ids; nothing here translates."""
        batch.kv_loc_plan = self
        batch.out_cache_loc = self.write_ids(reader, cols=cols)
        batch.out_cache_loc_virtual = (
            self.virtual_write_ids(cols=cols)
            if self.is_translated_for(reader)
            else None
        )
        # A forward with no write loc writes nothing and stays unmarked.
        batch.out_cache_loc_is_physical = batch.out_cache_loc is not None

    def write_ids(
        self, reader: KVIndexTranslator, *, cols: Optional[slice] = None
    ) -> Optional[torch.Tensor]:
        """The write window as `reader`'s pool indexes it. ``cols`` selects
        columns of a ``[batch, window]`` window, for a forward that writes only
        part of it (a draft step ahead of the verify window it opens)."""
        ids = (
            self.write_physical
            if self.is_translated_for(reader)
            else self.write_virtual
        )
        return self._cols(ids, cols)

    def virtual_write_ids(self, *, cols: Optional[slice] = None):
        """The same columns in the virtual space: the ids the scheduler's
        bookkeeping (lazy compaction's in-flight write set, the radix tree)
        names."""
        return self._cols(self.write_virtual, cols)

    def _cols(self, ids: Optional[torch.Tensor], cols: Optional[slice]):
        if ids is None or cols is None:
            return ids
        bs = int(self.req_pool_indices.numel())
        return ids.view(bs, -1)[:, cols].reshape(-1)

    def swa_write_ids(self, *, cols: Optional[slice] = None) -> Optional[torch.Tensor]:
        """The sliding-window sub-pool's write ids for the same columns, or
        None when the pool has no sliding-window space. Derived once for the
        whole window, from its virtual ids: one lookup, not an inverse lookup
        of the physical ones."""
        if self._swa_write is None and self.write_virtual is not None:
            self._swa_write = self._source._swa_write_ids(
                virtual=self.write_virtual, physical=self.write_physical
            )
        return self._cols(self._swa_write, cols)

    # -- reads -----------------------------------------------------------------

    def read_table(self, *, rows: Optional[int] = None) -> KVIndexTable:
        """The rows' page table over ``[0, seq_lens + read_extent)``, built on
        first use and shared by every reader of the iteration. ``rows`` past
        the plan's batch (a captured graph's padded lanes) are appended reading
        the sink, by copying, never by translating again. On a pool that reads
        virtual ids (static, or DCP, where the producing kernel selects this
        rank's share) it is the ``req_to_token`` passthrough."""
        table = self._read_table
        if table is None or (
            table.is_translated and rows is not None and table.ids.shape[0] < rows
        ):
            self._read_table = table = self._source._build_iteration_table(
                self, rows=rows, previous=table
            )
        return table

    def is_translated_for(self, reader: KVIndexTranslator) -> bool:
        """Whether `reader`'s pool indexes this plan's physical ids: the
        target's, and a fused draft's, which translates through the target's
        table. A pass-through reader (a static pool, a private draft pool)
        indexes the virtual ids."""
        return reader.is_translating and reader.full_v2p_table is (
            self._source.full_v2p_table
        )
