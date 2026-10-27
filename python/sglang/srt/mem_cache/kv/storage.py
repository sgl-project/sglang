"""The unified pool as host and peer transfers see it.

`KVStorage` is what HiCache and PD read: the sub-pools in the order host pools
and transfers list them, each with its external names, its layout view, its
model layers, the device memory a peer registers (`regions`), and how a host
load takes and returns device room. Sub-pools carry names, never a lifetime;
code outside the KV package loops over them instead of naming a combination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.kv.memory.host import page_envelope_pair
from sglang.srt.mem_cache.kv.plan import KVStoragePlan, SubPoolExternal, WireEntry
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.srt.mem_cache.kv.state_storage import StateSlotsExhausted

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class StorageSubPool(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    external: SubPoolExternal
    # Each routed model layer id -> its index in the sub-pool's view.
    layer_indices: Dict[int, int]
    # One slot per request rather than per token: hosts mirror it as state.
    holds_state: bool


class SubPoolUsage(msgspec.Struct, frozen=True, kw_only=True):
    """A sub-pool's occupancy as observers report it, in its own units: tokens,
    or slots for a sub-pool that holds state. ``metrics`` is the family its
    numbers are published under."""

    name: str
    metrics: str
    capacity: int
    available: int


class Region(msgspec.Struct, frozen=True, kw_only=True):
    """A sub-pool's device memory as a PD peer registers it: one entry per
    buffer, and per entry the slices a peer of another TP size reads. A whole
    page envelope has no slices: both peers must lay it out identically."""

    ptrs: Tuple[int, ...]
    lens: Tuple[int, ...]
    item_lens: Tuple[int, ...]
    dim_per_tensor: Tuple[int, ...] = ()
    conv_shard_groups: Tuple[Any, ...] = ()
    slice_outer_counts: Tuple[int, ...] = ()
    layer_ids: Tuple[int, ...] = ()
    # The token KV byte layout both peers must share, for views that have one.
    kv_layout: Optional[str] = None


class KVStorage:
    """The plan's sub-pools over one unified buffer, their views and the chain
    allocator that hands out their ids. Token KV views and layer indices come
    from the KV pool's router, the state view from the allocator's state."""

    def __init__(
        self,
        *,
        plan: KVStoragePlan,
        router: KVLayerRouter,
        allocator,
        start_layer: int,
        requests=None,
    ):
        self._plan = plan
        self._allocator = allocator
        self.start_layer = start_layer
        # The request table: rows and the ids they hold.
        self.requests = requests
        state = allocator.state
        self._views: Dict[str, Any] = {}
        sub_pools = []
        for s in plan.owner_first():
            holds_state = isinstance(s.spec, MambaSubPoolSpec)
            if holds_state:
                self._views[s.name] = state.pool
                layer_indices = dict(state.layer_map)
            else:
                self._views[s.name] = router.view(s.name)
                layer_indices = {
                    layer_id: router.route(layer_id).index
                    for layer_id in router.sub_pool(s.name).layer_ids
                }
            sub_pools.append(
                StorageSubPool(
                    name=s.name,
                    external=s.external,
                    layer_indices=layer_indices,
                    holds_state=holds_state,
                )
            )
        self._sub_pools = tuple(sub_pools)

    @property
    def allocator(self):
        """The chain allocator, for what does not go through this storage yet:
        admission and eviction (capacity), the tree's frees."""
        return self._allocator

    # -- the request lifecycle --
    #
    # A take that may need room first calls ``make_room(num_tokens)``: the
    # caller evicts from its prefix tree there, before any id is taken.

    def acquire(self, reqs: List[Req]) -> Optional[List[int]]:
        """Bind the requests' rows and, with a state sub-pool, their state and
        tracking slots; None when the table has too few free rows. A request
        already holding a row (a later chunk) keeps it, and one already holding
        a state slot (a radix hit) keeps that. If the slots run out after the
        rows were bound, the rows and slots this call took go back before
        `StateSlotsExhausted` surfaces."""
        table = self.requests
        state = self._allocator.state
        fresh_rows = [req.kv.req_pool_idx is None for req in reqs]
        rows = table.alloc(reqs)
        if rows is None or state is None:
            return rows
        try:
            state.acquire(table, reqs, rows, fresh_rows)
        except StateSlotsExhausted:
            for req, fresh in zip(reqs, fresh_rows):
                if fresh:
                    table.free(req)
            raise
        return rows

    def take_tokens(
        self, num_tokens: int, *, make_room: Callable[[int], Any]
    ) -> Optional[torch.Tensor]:
        """Owner token ids, one per token (page size 1); None when they do not
        fit."""
        make_room(num_tokens)
        return self._allocator.alloc(num_tokens)

    def take_extend(
        self,
        *,
        rows: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        num_tokens: int,
        make_room: Callable[[int], Any],
        window_tail: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Owner token ids for ``[prefix_len, seq_len)`` of each request's
        ``rows`` entry, by page, continuing each request's last page from
        ``last_loc``; None when they do not fit. With ``window_tail`` (one
        request whose KV a peer sends), a window binds pages for its last
        ``window_tail`` tokens only."""
        demand = self._allocator.get_extend_allocation_demand(
            prefix_lens_cpu,
            seq_lens_cpu,
            conservative_num_tokens=num_tokens
            + len(seq_lens_cpu) * self._allocator.page_size,
            shard_size=1,
        )
        make_room(demand)
        if window_tail is not None:
            return self._allocator.alloc_extend_swa_tail(
                prefix_lens=prefix_lens,
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                last_loc=last_loc,
                extend_num_tokens=num_tokens,
                swa_tail_len=window_tail,
            )
        return self._allocator.alloc_extend(
            prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, num_tokens
        )

    def take_decode(
        self,
        *,
        rows: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        make_room: Callable[[int], Any],
    ) -> Optional[torch.Tensor]:
        """One owner token id per ``rows`` entry, at ``seq_len - 1``, taking a
        page only when that request's last page is full; None when they do not
        fit."""
        make_room(len(seq_lens) * self._allocator.page_size)
        return self._allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    def free_tokens(self, ids: torch.Tensor) -> None:
        """Give back owner token ids no request holds any more."""
        self._allocator.free(ids)

    def release_row(self, req: Req) -> None:
        """Give back the request's row; its ids went back before."""
        self.requests.free(req)

    def take_rows(self, num_rows: int) -> Optional[List[int]]:
        """Rows bound to no request (beam search members); None when too few
        are free."""
        return self.requests.alloc_rows(num_rows)

    def release_rows(self, rows: List[int]) -> None:
        """Give back rows `take_rows` handed out; their ids went back before."""
        self.requests.free_rows(rows)

    @property
    def plan(self) -> KVStoragePlan:
        """The plan this storage was built from (read-only)."""
        return self._plan

    @property
    def owner(self) -> str:
        """The sub-pool that owns the token id space: the anchor of host
        stacks, and the transfer's KV entries."""
        return self._plan.owner().name

    def sub_pools(self) -> Tuple[StorageSubPool, ...]:
        """Every sub-pool, owner first."""
        return self._sub_pools

    def sub_pool_names(self) -> Tuple[str, ...]:
        """Every sub-pool's name, owner first: what a transfer that reaches
        every sub-pool pins."""
        return tuple(s.name for s in self._sub_pools)

    def tree_components(self) -> Tuple[str, ...]:
        """The prefix tree's component per sub-pool, in the order the tree
        drives their eviction."""
        return tuple(s.external.tree_component for s in self._plan.drive_order())

    def view(self, sub_pool: str) -> Any:
        return self._views[sub_pool]

    def accounting_violations(self) -> List[str]:
        """What the idle leak check reports beyond the per-sub-pool counts: the
        bytes the chain's bands account for that do not add up. Empty when
        healthy."""
        return self._allocator.verify_byte_accounting()

    def usage(self) -> Tuple[SubPoolUsage, ...]:
        """Every sub-pool's capacity and free units now, owner first."""
        units = self._allocator.usage()
        return tuple(
            SubPoolUsage(
                name=s.name,
                metrics=s.external.metrics,
                capacity=units[s.name][0],
                available=units[s.name][1],
            )
            for s in self._sub_pools
        )

    def page_envelope_pair(self) -> Optional[Tuple[str, str]]:
        """The two sub-pools whose host pools may share one page-envelope
        backing, owner first (kv/memory/host.py)."""
        return page_envelope_pair(self._plan)

    def wire_entries(self) -> Tuple[WireEntry, ...]:
        return self._plan.wire_entries()

    def regions(self, sub_pool: str) -> Region:
        view = self._views[sub_pool]
        ptrs, lens, item_lens = view.get_contiguous_buf_infos()
        if isinstance(self._plan.sub_pool(sub_pool).spec, MambaSubPoolSpec):
            return Region(
                ptrs=tuple(ptrs),
                lens=tuple(lens),
                item_lens=tuple(item_lens),
                dim_per_tensor=tuple(view.get_state_dim_per_tensor()),
                conv_shard_groups=tuple(view.get_state_conv_shard_groups()),
                slice_outer_counts=tuple(view.get_state_slice_outer_counts()),
                layer_ids=tuple(view.get_state_layer_ids()),
            )
        spec = self._plan.sub_pool(sub_pool).spec
        return Region(
            ptrs=tuple(ptrs),
            lens=tuple(lens),
            item_lens=tuple(item_lens),
            kv_layout=(
                view.kv_cache_layout if isinstance(spec, MHASubPoolSpec) else None
            ),
        )

    def device_room(self, sub_pool: str) -> int:
        """Free units a host load can take in the sub-pool now."""
        if sub_pool == self._allocator.roles.window:
            return self._allocator.swa_available_size()
        assert sub_pool == self._allocator.roles.state, sub_pool
        return self._allocator.state_slots.schedulable_available_size()

    def host_ids(
        self, sub_pool: str
    ) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """How a host transfer turns the controller's ids for the sub-pool into
        the rows its view indexes, or None when they index it as they are (a
        window's host loads bind physical pages). Called on the transfer
        stream; pins keep the rows in place until the copy is acknowledged."""
        roles = self._allocator.roles
        if sub_pool == roles.owner:
            return self._allocator.full_attn_allocator.ids.translate
        if sub_pool == roles.state:
            return self._allocator.state.slots.translate
        return None

    def add_pin_source(
        self, sub_pools: Sequence[str], busy: Callable[[], bool]
    ) -> None:
        """Keep the named sub-pools' device rows where they are whenever
        ``busy()`` holds: a DMA engine (a PD transfer, a HiCache copy) reads
        or writes the addresses it was handed. Register each initiator once,
        over the queues that live across iterations; tree locks are not pins."""
        self._allocator.add_pin_source(sub_pools, busy)

    def host_load_alloc(
        self, sub_pool: str
    ) -> Optional[Tuple[Callable[[int], Any], Callable[[Any], Any]]]:
        """How a host load takes device units in the sub-pool and gives back
        the ones it did not use, or None when loads take the owner's token
        ids. A window binds physical pages before the copy."""
        roles = self._allocator.roles
        if sub_pool == roles.window:
            window = self._allocator.swa_attn_allocator
            return window.alloc_physical, window.cancel_physical_reservation
        if sub_pool == roles.state:
            slots = self._allocator.state_slots
            return slots.alloc, slots.free
        return None
