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

from sglang.srt.mem_cache.kv.advance import TrackingAdvance, advance_tracking
from sglang.srt.mem_cache.kv.decl import StateLifetime, WindowLifetime
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.kv.memory.host import page_envelope_pair
from sglang.srt.mem_cache.kv.plan import (
    KVStoragePlan,
    SubPoolExternal,
    TreeSlot,
    WireEntry,
)
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


class ResumeUnits(msgspec.Struct, frozen=True, kw_only=True):
    """Physical units and the exact token span they restore on a peer.

    ``token_boundary`` is the continuation point, including for a state
    checkpoint. It comes from the handoff, not from the request's later length.
    """

    unit_ids: torch.Tensor
    token_start: int
    token_boundary: int


class KVStorage:
    """The plan's sub-pools over one unified buffer, their views and the chain
    allocator that hands out their ids. Token KV views and layer indices come
    from the KV pool's router, the state view from the allocator's state."""

    def __init__(
        self,
        *,
        plan: KVStoragePlan,
        sized=None,
        router: KVLayerRouter,
        allocator,
        start_layer: int,
        requests=None,
        scratch=None,
        request_memory=None,
    ):
        self._plan = plan
        self.sized = sized
        self._allocator = allocator
        self.start_layer = start_layer
        # The request table: rows and the ids they hold.
        self.requests = requests
        self.scratch = scratch
        self._request_memory = request_memory
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

    def host_layout(self, name: str):
        """The sub-pool's host layout and persistent device buffers."""
        return self.view(name).host_layout()

    @property
    def request_memory(self):
        """Fixed REQUEST parts, copied together when a request is retracted."""
        return self._request_memory

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
        if rows is None:
            return None
        if self._request_memory is not None:
            self._request_memory.reset_rows(
                [row for row, fresh in zip(rows, fresh_rows) if fresh]
            )
        if state is None:
            return rows
        try:
            state.acquire(table, reqs, rows, fresh_rows, reset_request_parts=False)
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

    def advance(
        self,
        req: Req,
        frontier: Optional[int] = None,
        *,
        window: Optional[int] = None,
        retain_floor: Optional[int] = None,
        shares_prefixes: bool = False,
        eviction_interval: int = 1,
        tracking: Optional[TrackingAdvance] = None,
    ) -> None:
        """Apply checkpoint tracking and retire rows outside the read window.

        Tracking carries the completed forward's boundary. Window retirement
        uses ``frontier`` and is clamped by the prefix tree's retention floor.
        """
        if tracking is not None:
            advance_tracking(self, req, tracking)
        if window is None:
            return
        if frontier is None:
            raise ValueError("window retirement needs a token frontier")

        from sglang.srt.mem_cache.common import free_swa_out_of_window_slots

        free_swa_out_of_window_slots(
            req,
            frontier,
            sliding_window_size=window,
            page_size=self._allocator.page_size,
            req_to_token_pool=self.requests,
            token_to_kv_pool_allocator=self._allocator,
            supports_prefix_sharing=shares_prefixes,
            retain_floor=retain_floor,
            eviction_interval=eviction_interval,
        )

    def release_row(self, req: Req) -> None:
        """Give back the request's row; its ids went back before."""
        self.requests.free(req)

    def take_rows(self, num_rows: int) -> Optional[List[int]]:
        """Rows bound to no request (beam search members); None when too few
        are free."""
        rows = self.requests.alloc_rows(num_rows)
        if rows is not None and self._request_memory is not None:
            self._request_memory.reset_rows(rows)
        return rows

    def release_rows(self, rows: List[int]) -> None:
        """Give back rows `take_rows` handed out; their ids went back before."""
        self.requests.free_rows(rows)

    # -- what the prefix tree's effects reach --

    def tree_sub_pool(self, component: str) -> str:
        """The sub-pool whose values the prefix tree's ``component`` holds."""
        for s in self._sub_pools:
            if s.external.tree_component == component:
                return s.name
        raise KeyError(f"no sub-pool backs the tree component {component!r}")

    def _band(self, sub_pool: str):
        """The band that hands out ``sub_pool``'s rows on their own: the
        owner's when a window shares its token ids, the window's, or the
        whole allocator for an owner alone."""
        if self._plan.sub_pool(sub_pool).external.tree_component == "SWA":
            return self._allocator.swa_attn_allocator
        if self._allocator.has_window_pool:
            return self._allocator.full_attn_allocator
        return self._allocator

    def available_units(self, sub_pool: str) -> int:
        """Rows ``take_units`` can hand out without eviction."""
        return self._band(sub_pool).available_size()

    def take_units(self, sub_pool: str, count: int) -> Optional[torch.Tensor]:
        """``count`` rows of ``sub_pool`` alone (a load the tree stages); None
        when they do not fit."""
        return self._band(sub_pool).alloc(count)

    def release_taken(self, sub_pool: str, ids: torch.Tensor) -> None:
        """Give back rows `take_units` handed out."""
        self._band(sub_pool).free(ids)

    def free_units(self, sub_pool: str, ids: torch.Tensor) -> None:
        """Give back a sub-pool's rows the tree held, as page-exact segments of
        owner token ids: the owner's rows alone, or a window's rows of those
        token ids."""
        if self._plan.sub_pool(sub_pool).external.tree_component == "SWA":
            self._allocator.free_swa_segment(ids, start_pos=0)
        else:
            self._band(sub_pool).free_segment(ids, start_pos=0)

    def free_token_rows(self, ids: torch.Tensor) -> None:
        """Give back every sub-pool's rows of these owner token ids."""
        self._allocator.free_segment(ids, start_pos=0)

    def free_owner_rows(self, ids: torch.Tensor) -> None:
        """Give back the owner's rows of token ids whose window rows went back
        before."""
        self._allocator.free_full_segment(ids, start_pos=0)

    def window_ids(self, sub_pool: str, owner_ids: torch.Tensor) -> torch.Tensor:
        """The window's row ids of these owner token ids, as transfers and the
        tree's window values name them."""
        return self._allocator.translate_swa_indices_for_transfer(owner_ids).to(
            torch.int64
        )

    def bind_window(
        self, sub_pool: str, owner_ids: torch.Tensor, window_ids: torch.Tensor
    ) -> None:
        """Let these owner token ids name these window rows."""
        self._allocator.set_full_to_swa_mapping(owner_ids, window_ids)

    def rehome_window(
        self, sub_pool: str, kept_ids: torch.Tensor, incoming_ids: torch.Tensor
    ) -> torch.Tensor:
        """Hand the window rows of ``incoming_ids`` to ``kept_ids`` (the same
        tokens, already cached under other ids) and return them by their new
        name. Nothing is allocated or freed: the window band rebinds the kept
        ids' pages to the incoming pages' memory and drops the incoming ones."""
        window = self._allocator.swa_attn_allocator
        kept_pages, incoming_pages = _page_pairs(
            kept_ids, incoming_ids, self._allocator.page_size
        )
        physical = window.ids.virtual_to_physical[incoming_pages]
        # `> 0` strict: -1 = dropped, 0 = the padding sink. The incoming ids
        # were just allocated by the request in flight, so every page is live;
        # anything else would hand the cached tokens the sink's zeros.
        assert bool((physical > 0).all()), (
            f"incoming window pages must all be live, got {physical.tolist()}"
        )
        window.ids.bind(kept_pages, physical)
        window.ids.virtual_to_physical.index_fill_(0, incoming_pages, -1)
        window.clear_inverse_history()
        return self.window_ids(sub_pool, kept_ids)

    def release_rehomed_owner_rows(self, ids: torch.Tensor) -> None:
        """Give back owner token ids whose window rows `rehome_window` moved."""
        self._allocator.free(ids)

    # The scheduler chooses tracking boundaries;
    # storage owns ping-pong slots and the request table's matching columns.

    def take_tracking_slot(self, req, position: int) -> bool:
        """Give ``req`` a tracking slot at ``position`` of its ping-pong pair;
        False when no state slot is free (tracking then stays in place)."""
        slot = self._allocator.state_slots.alloc(1)
        if slot is None:
            return False
        self.requests.set_mamba_ping_pong_slot(req, position, slot[0])
        return True

    def release_tracking_slot(self, req, position: int) -> None:
        """Give back the tracking slot at ``position`` of ``req``'s pair and
        empty that position."""
        self._allocator.state_slots.free(
            req.kv.mamba_ping_pong_track_buffer[position].unsqueeze(0)
        )
        self.requests.set_mamba_ping_pong_slot(req, position, -1)

    def release_state_slot(self, req) -> None:
        """Give back the state slot ``req`` took before its KV (a request that
        does not run after all)."""
        self._allocator.state_slots.free(req.kv.mamba_pool_idx.unsqueeze(-1))
        req.kv.mamba_pool_idx = None

    # -- the prefix tree's state effects: the tree decides, the storage executes --

    def take_state_slots(self, sub_pool: str, count: int) -> Optional[torch.Tensor]:
        """``count`` free state slots of ``sub_pool``; None when none are free."""
        return self._allocator.state_slots.alloc(count)

    def free_state_slots(self, sub_pool: str, slots: torch.Tensor) -> None:
        self._allocator.state_slots.free(slots)

    def copy_state(self, sub_pool: str, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy the state in slots ``src`` into slots ``dst``."""
        ids = self._allocator.state_ids(sub_pool)
        self._allocator.state.pool.copy_from(ids.ids(src), ids.ids(dst))

    def release_state(
        self, sub_pool: str, req: Req, *, keep: Optional[int] = None
    ) -> None:
        """Give back a finished request's state slots, keeping its tracking
        slot ``keep`` when the tree took that one."""
        self._allocator.state.release(
            self.requests, req, mamba_ping_pong_track_buffer_to_keep=keep
        )

    def donate_tracking_slot(
        self, sub_pool: str, req: Req, new_slot: torch.Tensor
    ) -> torch.Tensor:
        """Hand the tree the request's kept tracking slot, giving the request
        ``new_slot`` in its place."""
        return self.requests.donate_mamba_ping_pong_slot(req, new_slot)

    def quantizes_checkpoints(self, sub_pool: str) -> bool:
        """Whether the tree's state checkpoints live in a quantized pool of
        their own rather than in state slots."""
        return self._allocator.state.ckpt_pool is not None

    def take_checkpoint_slot(self, sub_pool: str) -> Optional[torch.Tensor]:
        return self._allocator.state.ckpt_pool.alloc(1)

    def store_checkpoint(
        self, sub_pool: str, active: torch.Tensor, checkpoint: torch.Tensor
    ) -> None:
        """Quantize the state in slots ``active`` into ``checkpoint``."""
        state = self._allocator.state
        state.ckpt_pool.store_from_active(state.pool, active.view(-1), checkpoint)

    def free_checkpoints(self, sub_pool: str, checkpoints: torch.Tensor) -> None:
        self._allocator.state.ckpt_pool.free(checkpoints)

    def settle_replay(self, sub_pool: str, slot: torch.Tensor) -> int:
        """At a request's finish: the tokens its slot's state lags behind the
        request (its replay ring's unflushed depth, 0 without one); the lag
        restarts at zero."""
        write_pos = self._allocator.state.pool.replayssm_write_pos
        if write_pos is None:
            return 0
        slot = self._allocator.state_ids(sub_pool).ids(slot)
        lag = int(write_pos[slot].item())
        write_pos[slot] = 0
        return lag

    def restart_replay(self, sub_pool: str, slots: torch.Tensor) -> None:
        """A loaded state is current: its replay lag restarts at zero."""
        write_pos = self._allocator.state.pool.replayssm_write_pos
        if write_pos is not None:
            write_pos[self._allocator.state_ids(sub_pool).ids(slots)] = 0

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

    def tree_slots(self, *, window_reach: int = 0) -> Tuple[TreeSlot, ...]:
        """The prefix tree's slot per sub-pool, in the order the tree drives
        their eviction; ``window_reach`` widens the window a prefix keeps for a
        reader beyond the window's own layers (a fused draft)."""
        return self._plan.tree_slots(window_reach=window_reach)

    def view(self, sub_pool: str) -> Any:
        return self._views[sub_pool]

    def accounting_violations(self) -> List[str]:
        """Byte-accounting violations beyond the idle per-sub-pool leak counts."""
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

    def resume_units(
        self,
        sub_pool: str,
        req: Req,
        *,
        token_start: int,
        token_boundary: int,
        window: Optional[int] = None,
    ) -> ResumeUnits:
        if not 0 <= token_start <= token_boundary:
            raise ValueError("invalid continuation token span")
        planned = self._plan.sub_pool(sub_pool)
        if isinstance(planned.lifetime, StateLifetime):
            grid = planned.lifetime.checkpoint_interval
            if grid is not None and token_boundary % grid:
                raise ValueError(
                    "state handoff must name an explicit checkpoint boundary"
                )
            ids = self.requests.req_index_to_mamba_index_mapping[
                req.kv.req_pool_idx
            ].reshape(-1)
            physical = self._allocator.state_id_spaces()[sub_pool].ids(ids)
            return ResumeUnits(
                unit_ids=physical,
                token_start=token_boundary,
                token_boundary=token_boundary,
            )
        begin = token_start
        if isinstance(planned.lifetime, WindowLifetime):
            retained = planned.lifetime.window
            if retained is None:
                retained = window
            if retained is None:
                raise ValueError("window handoff requires its retention length")
            begin = max(begin, token_boundary - retained)
        begin = begin // planned.unit * planned.unit
        tokens = self.requests.req_to_token[req.kv.req_pool_idx, begin:token_boundary]
        physical = (
            self._allocator.translate_kv_indices_for_transfer(tokens)
            if sub_pool == self.owner
            else self._allocator.id_spaces()[sub_pool].ids(tokens.to(torch.int64))
        )
        return ResumeUnits(
            unit_ids=physical[:: planned.unit] // planned.unit,
            token_start=begin,
            token_boundary=token_boundary,
        )

    def regions(self, sub_pool: str, *, part: Optional[str] = None) -> Region:
        derived = [
            d.part
            for d in self._plan.derived_parts
            if d.sub_pool == sub_pool
            and (d.part.name == part or d.wire == part)
            and part is not None
        ]
        request = [
            d.part
            for d in self._plan.request_parts
            if (d.part.name == part or d.wire == part) and part is not None
        ]
        if derived:
            return self._allocator.arena.page_parts[sub_pool].regions(derived)
        if request:
            return self._request_memory.regions(request)
        if part not in (None, "kv", "state"):
            raise KeyError(f"sub-pool {sub_pool!r} has no transfer part {part!r}")
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


def _page_pairs(
    kept_ids: torch.Tensor, incoming_ids: torch.Tensor, page_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Page ids of two token ranges that address the same logical tokens.

    Dedupes by first occurrence with one shared mask rather than
    `torch.unique`: unique sorts by id, and allocation hands out virtual ids in
    no particular order, so sorting would pair page k of one range with an
    unrelated page of the other. One mask keeps the pairing positional, hence
    logical.
    """
    kept = kept_ids.detach().to(torch.int64) // page_size
    incoming = incoming_ids.detach().to(torch.int64) // page_size
    assert kept.numel() == incoming.numel(), (
        f"rehoming window rows needs a 1:1 token correspondence, got "
        f"{kept.numel()} kept vs {incoming.numel()} incoming"
    )
    starts = torch.ones_like(kept, dtype=torch.bool)
    starts[1:] = kept[1:] != kept[:-1]
    incoming_starts = torch.ones_like(incoming, dtype=torch.bool)
    incoming_starts[1:] = incoming[1:] != incoming[:-1]
    assert torch.equal(starts, incoming_starts), (
        "the two ranges break into pages at different offsets, so no "
        "page-granular ownership transfer expresses the token mapping"
    )
    return kept[starts], incoming[starts]
