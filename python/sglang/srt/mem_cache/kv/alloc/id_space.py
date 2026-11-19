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
"""How a sub-pool names its units.

`UnitIds` is the record a band keeps: which virtual unit sits at which physical
unit, and the translation of token ids between the two. `IdSpace` is how a
consumer names a sub-pool's slots through that record for one iteration's reads
and writes; `IdSpaceKind` holds the names consumers ask a token sub-pool by.
"""

from __future__ import annotations

import enum
from typing import Callable, Dict, Hashable, Optional

import msgspec
import torch
from torch.profiler import record_function

from sglang.kernels.ops.memory.virtual_slot import bind_inplace, translate_token_ids
from sglang.srt.runtime_context import get_parallel


class IdSpaceKind(str, enum.Enum):
    """The names of the two token sub-pools today's consumers ask for by
    role. Id spaces are keyed by sub-pool name, so any sub-pool's name works
    wherever a kind does."""

    # Full-attention layers: the pool's own token ids.
    FULL = "full"
    # The sliding-window layers of a hybrid pool: the same token ids through
    # the sliding-window sub-pool's table.
    SLIDING_WINDOW = "swa"


class IdSpace(msgspec.Struct, frozen=True):
    """How one sub-pool names the iteration's slots.

    ``write`` maps the iteration's virtual token ids to this sub-pool's ids
    (None: they index it as they are). ``read_v2p`` is the page table its reads
    go through (None: reads stay in `req_to_token`, for the consumer to
    finish). ``table`` is the sub-pool's page table even when reads stay
    virtual (DCP), for builders that gather a rank's share themselves.
    Runners whose pools share a sub-pool share its ``key``, and with it a
    plan's ids for it."""

    key: Hashable
    write: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    read_v2p: Optional[torch.Tensor] = None
    table: Optional[torch.Tensor] = None
    prepare_read: Optional[Callable[[], None]] = None

    def read_mapping(self):
        if self.prepare_read is not None:
            self.prepare_read()
        return self.read_v2p

    def ids(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        """``virtual_ids`` as this sub-pool indexes them."""
        return virtual_ids if self.write is None else self.write(virtual_ids)


def state_id_space(spaces: Dict[str, IdSpace], sub_pool: str) -> IdSpace:
    """How a recurrent-state sub-pool names the slots the request table holds:
    through its allocator's table when the allocator hands out virtual slot ids,
    else as they are."""
    space = spaces.get(sub_pool)
    return IdSpace(key=(sub_pool, None)) if space is None else space


class UnitIds:
    """How one sub-pool names its units.

    A unit has a virtual id, which the request table and the tree hold and which
    stays put while the band moves the unit, and a physical id, its position in
    the band now. `virtual_to_physical` and `physical_to_virtual` record the
    pairing: unit 0 is the padding sink in both, and each table's trailing row
    is the -1 sentinel. A token id is ``unit * unit_tokens + offset`` in either
    space, so translating one rewrites the unit and keeps the offset.

    The owner sub-pool mints its virtual units; a window binds the owner's, so
    its virtual count is the owner's; a state sub-pool's units are slots. The
    band that places the units writes both tables; consumers read them.
    """

    def __init__(
        self,
        *,
        num_virtual_ids: int,
        num_physical_units: int,
        unit_tokens: int,
        shards_under_dcp: bool,
        device: str,
    ):
        assert num_virtual_ids > 0, "virtual page count must be positive"
        # Under decode context parallelism `unit_tokens` stays the PHYSICAL rows
        # one unit occupies: read ids arrive DCP-collapsed, and only write ids
        # carry the widened owner rule (`translate_write`).
        self.unit_tokens = unit_tokens
        self.shards_under_dcp = shards_under_dcp
        self.num_virtual_ids = num_virtual_ids
        self.virtual_to_physical = torch.full(
            (num_virtual_ids + 1,), -1, dtype=torch.int64, device=device
        )
        self.physical_to_virtual = torch.full(
            (num_physical_units + 1,), -1, dtype=torch.int64, device=device
        )

    def reset(self) -> None:
        """Nothing bound but the padding sink."""
        self.virtual_to_physical.fill_(-1)
        self.virtual_to_physical[0] = 0
        self.virtual_to_physical[-1] = -1  # trailing sentinel
        self.physical_to_virtual.fill_(-1)
        self.physical_to_virtual[0] = 0
        self.physical_to_virtual[-1] = -1

    def bind(self, virtual_ids: torch.Tensor, physical_ids: torch.Tensor) -> None:
        """Pair unit-granular virtual ids with physical ids."""
        with record_function("UnitIds.bind"):
            bind_inplace(
                virtual_ids,
                physical_ids,
                self.virtual_to_physical,
                self.physical_to_virtual,
            )

    def is_bound(self, virtual_unit: int) -> bool:
        if virtual_unit < 0 or virtual_unit >= self.num_virtual_ids:
            return False
        return int(self.virtual_to_physical[virtual_unit].item()) != -1

    def translate(
        self,
        virtual_tokens: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Virtual token ids -> physical token ids, the ids kernels index the
        per-layer views with:

            physical(t) = v2p[t // unit_tokens] * unit_tokens + t % unit_tokens

        Under DCP the input is the DCP-collapsed id (`widened // dcp_size`), so
        this works on the physical unit size. An unmapped, out-of-range or
        negative id resolves to 0, the sink every kernel skips. ``out=`` writes
        in place into a caller-owned buffer, required under cuda-graph capture:
        the captured graph records the gather against a fixed ``data_ptr``.
        int64 out; a consumer whose kernel ABI wants int32 narrows where it
        fills that buffer.
        """
        with record_function("UnitIds.translate"):
            return self._translate(virtual_tokens, dcp_size=1, out=out)

    def translate_write(
        self,
        widened_loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        out_width: Optional[int] = None,
    ) -> torch.Tensor:
        """Widened virtual write loc (`out_cache_loc`) -> physical id.

        Reads arrive already DCP-collapsed, but `out_cache_loc` does not: it
        still carries the owner rule in `loc % dcp_size`. Ids this rank does not
        own go to id 0, the padding sink every write kernel skips.
        """
        parallel = get_parallel()
        dcp_size = parallel.attn_dcp_size if self.shards_under_dcp else 1
        with record_function("UnitIds.translate_write"):
            return self._translate(
                widened_loc,
                dcp_size=dcp_size,
                dcp_rank=parallel.attn_dcp_rank,
                out=out,
                out_width=out_width,
            )

    def _translate(
        self,
        loc: torch.Tensor,
        *,
        dcp_size: int,
        dcp_rank: int = 0,
        out: Optional[torch.Tensor] = None,
        out_width: Optional[int] = None,
    ) -> torch.Tensor:
        """Share ``translate_token_ids``'s output dtype and shape contract."""
        return translate_token_ids(
            loc=loc,
            v2p=self.virtual_to_physical,
            page_size=self.unit_tokens,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            out=out,
            out_width=out_width,
        )

    def token_space(
        self, key: Hashable, *, widened_writes: bool, defer_reads: bool
    ) -> IdSpace:
        """How a token sub-pool names an iteration's slots: virtual token ids
        map through this record. ``widened_writes``: the write window arrives
        DCP-widened (the owner's); ``defer_reads`` keeps reads virtual (DCP
        selects a rank's share where it builds them)."""
        return IdSpace(
            key=key,
            write=self.translate_write if widened_writes else self.translate,
            read_v2p=None if defer_reads else self.virtual_to_physical,
            table=self.virtual_to_physical,
        )

    def slot_space(self, key: Hashable) -> IdSpace:
        """How a state sub-pool names the slots the request table holds: one
        slot per unit, so the write is exactly a gather through ``read_v2p``,
        which a fused index kernel can reproduce in its own launch."""
        table = self.virtual_to_physical
        return IdSpace(
            key=key,
            write=lambda slots: table[slots].to(torch.int32),
            read_v2p=table,
            table=table,
        )


class WindowRead(msgspec.Struct, frozen=True, kw_only=True):
    """What a sliding-window layer reads of each request, and how its kernel
    masks the read to each query's window. Every attention mode (extend,
    target verify, draft extend; eager or replayed from a graph) takes its
    window read from `window_read`, so the read range and the mask never
    disagree."""

    # Position of the first token read, per request.
    start: torch.Tensor
    # Tokens read from ``start`` on, per request.
    lens: torch.Tensor
    # The layer's window: a query at position p sees keys from p - window on.
    window: int
    # The window the kernel is planned with, or -1 when the read is a prefix
    # that the caller masks itself (a ragged extend's paged pass).
    window_left: int

    def tree_mask(
        self, mask: torch.Tensor, *, width: int, read_lens_sum: int
    ) -> torch.Tensor:
        """A verify tree's mask, laid out over this read.

        ``mask`` is laid out over each request's whole sequence: ``width``
        rows of ``seq_len + width`` columns, the tree's own in the last
        ``width``. A tree row's query sits at its depth in the tree, not at
        its row index, so a kernel's row-indexed window would misplace it;
        the window is folded into the mask instead, and the kernel is planned
        without one. Each row sees only the prefix columns and tree ancestors
        inside its own window. ``read_lens_sum`` is ``lens.sum()``,
        known on the host."""
        device = mask.device
        lens = self.lens.to(torch.int64)
        seq_lens = self.start.to(torch.int64) + lens
        bs = lens.numel()
        ar = torch.arange(width, device=device)
        # The tree block of each request, [bs, width, width]. Rows a graph
        # replay pads past the live batch may index past the mask: clamp.
        full_row = seq_lens + width
        full_base = torch.cumsum(full_row * width, 0) - full_row * width
        tree_idx = (
            full_base[:, None, None]
            + ar[None, :, None] * full_row[:, None, None]
            + seq_lens[:, None, None]
            + ar[None, None, :]
        )
        tree = mask[tree_idx.clamp(max=mask.numel() - 1)].to(torch.bool)
        # Each row sees itself and its ancestors: its depth is one less.
        depth = tree.sum(-1) - 1
        tree = tree & (depth[:, :, None] - depth[:, None, :] <= self.window)
        read_row = lens + width
        read_end = torch.cumsum(read_row * width, 0)
        total = read_lens_sum * width + width * width * bs
        t = torch.arange(total, device=device)
        req = torch.searchsorted(read_end, t, right=True).clamp(max=bs - 1)
        local = t - (read_end - read_row * width)[req]
        row = local // read_row[req]
        col = local % read_row[req]
        prefix = col < lens[req]
        tree_col = (col - lens[req]).clamp(min=0, max=width - 1)
        return torch.where(
            prefix,
            col >= lens[req] + depth[req, row] - self.window,
            tree[req, row, tree_col],
        )


def window_read(
    *,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    window: int,
    prefix_only: bool,
) -> WindowRead:
    """The window read of a batch whose requests hold ``seq_lens`` tokens, of
    which ``prefix_lens`` are cached before this forward's queries.

    ``prefix_only``: the read covers the cached prefix alone (the queries' own
    keys arrive another way, as in a ragged extend); it keeps the prefix's last
    ``window`` tokens. Otherwise the read covers the cached prefix and the
    queries' keys: the last ``window`` tokens before the first query, through
    the last token, and the kernel masks each query to its own window."""
    if prefix_only:
        start = torch.clamp(prefix_lens - window, min=0)
        return WindowRead(
            start=start, lens=prefix_lens - start, window=window, window_left=-1
        )
    lens = torch.minimum(seq_lens, window + seq_lens - prefix_lens)
    return WindowRead(
        start=seq_lens - lens, lens=lens, window=window, window_left=window
    )


class ReplicatedTokenIds:
    """Draft coordinates over every logical token of an owner DCP page."""

    def __init__(self, owner: UnitIds, factor: int):
        self.owner = owner
        self.factor = factor
        self.expanded = torch.empty(
            ((owner.virtual_to_physical.numel() - 1) * factor + 1,),
            dtype=torch.int64,
            device=owner.virtual_to_physical.device,
        )

    def refresh(self):
        from sglang.kernels.ops.memory.virtual_slot import expand_page_mapping

        expand_page_mapping(self.owner.virtual_to_physical, self.expanded, self.factor)

    def write(self, ids):
        return translate_token_ids(
            loc=ids,
            v2p=self.owner.virtual_to_physical,
            page_size=self.owner.unit_tokens * self.factor,
            dcp_size=1,
            dcp_rank=0,
        )

    def space(self, key):
        return IdSpace(
            key=(key, "replicated", self.factor),
            write=self.write,
            read_v2p=self.expanded,
            table=self.expanded,
            prepare_read=self.refresh,
        )
