"""The pools built without unified memory, as a request's lifecycle and a PD
peer see them.

A request's rows and token ids are taken and given back, and a PD peer
registers device memory, through one storage interface, the unified pool's
`kv/storage.py::KVStorage`. Pools built without unified memory answer the same
calls through `StaticKVStorage`, which also holds what only they need: the NPU
DeepSeek-V4 c128 pages, the rotation bases of KV page interleave, and the state
entries each static pool family registers. All of it goes when those pools do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.kv.advance import TrackingAdvance, advance_tracking
from sglang.srt.mem_cache.kv.plan import (
    TREE_STATE,
    TREE_TOKEN,
    TREE_WINDOW,
    TreeSlot,
    WireEntry,
)
from sglang.srt.mem_cache.kv.state_storage import StateSlotsExhausted
from sglang.srt.mem_cache.kv.storage import Region, ResumeUnits, SubPoolUsage
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.kv.state_storage import StateStorage
    from sglang.srt.mem_cache.kv.storage import KVStorage
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class StaticPoolShape(msgspec.Struct, frozen=True, kw_only=True):
    """What the static pools' usage needs and only the scheduler knows: which
    pools the model has, and the token capacities they were sized to."""

    has_window: bool
    has_state: bool
    # The token pool's capacity without a window, in the allocator's units.
    token_capacity: int
    full_tokens_per_layer: Optional[int] = None
    swa_tokens_per_layer: Optional[int] = None
    # The prefix tree's numbers: the window layers' attention window, and the
    # token interval a cached recurrent state is valid at.
    window: Optional[int] = None
    checkpoint_interval: Optional[int] = None


def kv_storage_of(
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    *,
    tree_cache: Optional[BasePrefixCache] = None,
    batch: Optional[ScheduleBatch] = None,
    shape: Optional[StaticPoolShape] = None,
    draft_token_to_kv_pool=None,
    num_hidden_layers: Optional[int] = None,
) -> KVStorage | StaticKVStorage:
    """What a request's lifecycle, the pool observers and a PD peer go through:
    a unified pool's own storage, else the static pools behind
    `StaticKVStorage` (the keyword arguments feed only what the static pools
    need: ``tree_cache`` and ``batch`` their allocation, ``shape`` the
    observers' usage, the draft pool and the target's layer count the state
    entries they register with a draft's)."""
    storage = token_to_kv_pool_allocator.storage
    if storage is not None:
        return storage
    return StaticKVStorage(
        req_to_token_pool,
        token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        batch=batch,
        shape=shape,
        draft_token_to_kv_pool=draft_token_to_kv_pool,
        num_hidden_layers=num_hidden_layers,
    )


def acquire_rows(
    table: ReqToTokenPool, state: Optional[StateStorage], reqs: List[Req]
) -> Optional[List[int]]:
    """Bind the requests' rows in a static table and, with recurrent state,
    their state and tracking slots; None when the table has too few free rows.
    A request already holding a row or a slot keeps it. If the slots run out
    after the rows were bound, the rows and slots taken go back before
    `StateSlotsExhausted` surfaces."""
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


class StaticKVStorage:
    """The static request table, token allocator and KV pool behind
    `KVStorage`'s request lifecycle and PD transfer calls; see that class for
    what each call does."""

    # Static pools are not divided into sub-pools: the KV pool answers as the
    # owner, and each state entry it registers under a key of its own.
    owner = "kv"

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        *,
        tree_cache: Optional[BasePrefixCache] = None,
        batch: Optional[ScheduleBatch] = None,
        shape: Optional[StaticPoolShape] = None,
        draft_token_to_kv_pool=None,
        num_hidden_layers: Optional[int] = None,
    ):
        self.requests = req_to_token_pool
        self._allocator = token_to_kv_pool_allocator
        self._tree_cache = tree_cache
        self._batch = batch
        self._shape = shape
        self._draft_token_to_kv_pool = draft_token_to_kv_pool
        self._num_hidden_layers = num_hidden_layers
        self._owner_region: Optional[Region] = None
        self._state_entries: Optional[_StaticStateEntries] = None

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
        allocator = self._allocator
        state = allocator.state
        if state is not None and sub_pool == state.sub_pool:
            slots = self.requests.req_index_to_mamba_index_mapping[
                req.kv.req_pool_idx
            ].reshape(-1)
            return ResumeUnits(
                unit_ids=allocator.state_ids(sub_pool).ids(slots),
                token_start=token_boundary,
                token_boundary=token_boundary,
            )
        if sub_pool not in (self.owner, "swa"):
            raise KeyError(sub_pool)
        page_size = allocator.page_size
        pool = allocator.get_kvcache()
        request_window = pool.request_window if sub_pool == "swa" else None
        begin = token_start
        if sub_pool == "swa":
            retained = pool.sliding_window if request_window is not None else window
            if retained is None:
                raise ValueError("window handoff requires its retention length")
            begin = max(begin, token_boundary - retained)
        begin = begin // page_size * page_size
        if request_window is not None:
            units = torch.as_tensor(
                request_window.page_indices(
                    int(req.kv.req_pool_idx), begin, token_boundary
                )
            )
        else:
            tokens = self.requests.req_to_token[
                req.kv.req_pool_idx, begin:token_boundary
            ]
            physical = (
                allocator.translate_swa_indices_for_transfer(tokens)
                if sub_pool == "swa"
                else allocator.translate_kv_indices_for_transfer(tokens)
            )
            units = physical[::page_size] // page_size
        return ResumeUnits(
            unit_ids=units, token_start=begin, token_boundary=token_boundary
        )

    def acquire(self, reqs: List[Req]) -> Optional[List[int]]:
        return acquire_rows(self.requests, self._allocator.state, reqs)

    def take_tokens(
        self, num_tokens: int, *, make_room: Callable[[int], Any]
    ) -> Optional[torch.Tensor]:
        make_room(num_tokens)
        return self._allocator.alloc(num_tokens)

    def take_extend(
        self,
        *,
        rows: Optional[torch.Tensor],
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        num_tokens: int,
        make_room: Callable[[int], Any],
        window_tail: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        # Over estimate the number of tokens: assume each request needs a new
        # page (one page per CLASS per request under sharding — the min-class
        # availability floor must cover every request's ceil(K_i/N) rounding,
        # matching the N*ps per-request admission reserve).
        allocator = self._allocator
        shard_size = page_interleave_shard_size(allocator)
        conservative_num_tokens = num_tokens + len(seq_lens_cpu) * (
            allocator.page_size * shard_size
        )
        make_room(
            allocator.get_extend_allocation_demand(
                prefix_lens_cpu,
                seq_lens_cpu,
                conservative_num_tokens=conservative_num_tokens,
                shard_size=shard_size,
            )
        )

        is_dsv4 = rows is not None and hasattr(allocator, "c128_attn_allocator")
        extra_alloc_kwargs = {}
        kv_shard_rotation_bases = None
        if shard_size > 1:
            kv_shard_rotation_bases = _kv_shard_rotation_bases(
                tree_cache=self._tree_cache,
                batch=self._batch,
                prefix_lens_cpu=prefix_lens_cpu,
            )
            extra_alloc_kwargs["rotation_bases"] = kv_shard_rotation_bases
        if is_dsv4:
            c128_num_pages = allocator.c128_num_pages_needed(
                prefix_lens_cpu, seq_lens_cpu
            )
            allocator.ensure_c128_capacity(self._tree_cache, c128_num_pages)
            extra_alloc_kwargs["req_pool_indices"] = rows
            # Per-call per-req table for the C128 KV last_loc lookup.
            if self._batch is not None:
                extra_alloc_kwargs["req_to_token_pool"] = self._batch.req_to_token_pool

        if window_tail is not None:
            out = allocator.alloc_extend_swa_tail(
                prefix_lens=prefix_lens,
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                last_loc=last_loc,
                extend_num_tokens=num_tokens,
                swa_tail_len=window_tail,
                **extra_alloc_kwargs,
            )
        else:
            out = allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                num_tokens,
                **extra_alloc_kwargs,
            )
        if is_dsv4:
            bundle = out
            out = None if bundle is None else bundle.out_full_loc
            if self._batch is not None:
                self._batch.out_cache_loc_dsv4 = bundle

        if out is not None and kv_shard_rotation_bases is not None:
            # The allocator resolved None entries (new chains) in place from the
            # least-full class at each request's turn; record the bases for the
            # radix insert to stamp onto new tree nodes (UnifiedTreeNode.rotation_base).
            for req, base in zip(self._batch.reqs, kv_shard_rotation_bases):
                req.kv_rotation_base = base
        return out

    def take_decode(
        self,
        *,
        rows: Optional[torch.Tensor],
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        make_room: Callable[[int], Any],
    ) -> Optional[torch.Tensor]:
        allocator = self._allocator
        # Over estimate the number of tokens: assume each request needs a new page.
        make_room(len(seq_lens) * allocator.page_size)

        # DSV4-NPU allocator also needs req_pool_indices for C128 KV allocation and
        # returns a DSV4OutCacheLoc bundle; hasattr-gated so others stay unchanged.
        is_dsv4 = rows is not None and hasattr(allocator, "c128_attn_allocator")
        extra_alloc_kwargs = {}
        if is_dsv4:
            c128_num_pages = allocator.c128_num_pages_needed(
                (seq_lens_cpu - 1).clamp(min=0), seq_lens_cpu
            )
            allocator.ensure_c128_capacity(self._tree_cache, c128_num_pages)
            extra_alloc_kwargs["req_pool_indices"] = rows
            # Per-call per-req C128 table for the last_loc lookup.
            if self._batch is not None:
                extra_alloc_kwargs["req_to_token_pool"] = self._batch.req_to_token_pool

        out = allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc, **extra_alloc_kwargs
        )
        if is_dsv4:
            bundle = out
            out = None if bundle is None else bundle.out_full_loc
            if self._batch is not None:
                self._batch.out_cache_loc_dsv4 = bundle
        return out

    def free_tokens(self, ids: torch.Tensor) -> None:
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
        component_type=None,
    ) -> None:
        if tracking is not None:
            advance_tracking(self, req, tracking)
        if window is None:
            return
        if frontier is None:
            raise ValueError("window retirement needs a token frontier")

        from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        component_type = ComponentType.SWA if component_type is None else component_type
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
            component_type=component_type,
        )

    def release_row(self, req: Req) -> None:
        # The DSV4-NPU table's free() also releases the c4/c128 state pages.
        self.requests.free(req)

    def take_rows(self, num_rows: int) -> Optional[List[int]]:
        return self.requests.alloc_rows(num_rows)

    def release_rows(self, rows: List[int]) -> None:
        self.requests.free_rows(rows)

    # -- what the prefix tree's effects reach --

    # A second window component (buffer mode's) reaches the same window rows.
    _TREE_SUB_POOLS = {
        "FULL": "kv",
        "SWA": "swa",
        "AUXILIARY_SWA": "swa",
        "MAMBA": "mamba",
    }

    def tree_sub_pool(self, component: str) -> str:
        return self._TREE_SUB_POOLS[component]

    def _band(self, sub_pool: str):
        if sub_pool == "swa":
            return self._allocator.swa_attn_allocator
        if self._allocator.has_window_pool:
            return self._allocator.full_attn_allocator
        return self._allocator

    def available_units(self, sub_pool: str) -> int:
        return self._band(sub_pool).available_size()

    def take_units(self, sub_pool: str, count: int) -> Optional[torch.Tensor]:
        return self._band(sub_pool).alloc(count)

    def release_taken(self, sub_pool: str, ids: torch.Tensor) -> None:
        self._band(sub_pool).free(ids)

    def free_units(self, sub_pool: str, ids: torch.Tensor) -> None:
        if sub_pool == "swa":
            self._allocator.free_swa_segment(ids, start_pos=0)
        else:
            self._band(sub_pool).free_segment(ids, start_pos=0)

    def free_token_rows(self, ids: torch.Tensor) -> None:
        self._allocator.free_segment(ids, start_pos=0)

    def free_owner_rows(self, ids: torch.Tensor) -> None:
        self._allocator.free_full_segment(ids, start_pos=0)

    def window_ids(self, sub_pool: str, owner_ids: torch.Tensor) -> torch.Tensor:
        return self._allocator.translate_swa_indices_for_transfer(owner_ids).to(
            torch.int64
        )

    def bind_window(
        self, sub_pool: str, owner_ids: torch.Tensor, window_ids: torch.Tensor
    ) -> None:
        self._allocator.set_full_to_swa_mapping(owner_ids, window_ids)

    def rehome_window(
        self, sub_pool: str, kept_ids: torch.Tensor, incoming_ids: torch.Tensor
    ) -> torch.Tensor:
        # The static window is reached through a token -> window row mapping:
        # point the kept ids at the incoming rows and clear the incoming ids.
        window_ids = self.window_ids(sub_pool, incoming_ids)
        self._allocator.clear_full_to_swa_mapping(incoming_ids)
        self._allocator.set_full_to_swa_mapping(kept_ids, window_ids)
        return window_ids

    def release_rehomed_owner_rows(self, ids: torch.Tensor) -> None:
        self._allocator.free_full_segment(ids, start_pos=0)

    # -- a running request's state tracking: the scheduler decides when a
    # request crosses a tracking boundary; the storage hands out and takes back
    # the slots its ping-pong pair needs, and keeps the table's columns in step.

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

    def accounting_violations(self) -> List[str]:
        return self._allocator.verify_byte_accounting()

    def usage(self) -> Tuple[SubPoolUsage, ...]:
        shape, allocator = self._shape, self._allocator
        if shape.has_window:
            (full_capacity, full_available), (swa_capacity, swa_available) = (
                allocator.swa_capacity_and_available(
                    full_capacity=shape.full_tokens_per_layer,
                    swa_capacity=shape.swa_tokens_per_layer,
                )
            )
            usage = [
                SubPoolUsage(
                    name="full",
                    metrics="full",
                    capacity=full_capacity,
                    available=full_available,
                ),
                SubPoolUsage(
                    name="swa",
                    metrics="swa",
                    capacity=swa_capacity,
                    available=swa_available,
                ),
            ]
        else:
            # A state model's token capacity is what its requests can schedule.
            capacity = (
                self.requests.schedulable_token_capacity(allocator.size)
                if shape.has_state
                else shape.token_capacity
            )
            usage = [
                SubPoolUsage(
                    name="full",
                    metrics="full",
                    capacity=capacity,
                    available=allocator.available_size(),
                )
            ]
        if shape.has_state:
            usage.append(
                SubPoolUsage(
                    name="mamba",
                    metrics="mamba",
                    capacity=allocator.state.pool.size,
                    available=allocator.state_slots.available_size(),
                )
            )
        return tuple(usage)

    # -- what a PD peer registers --

    def view(self, sub_pool: str) -> Any:
        """For the owner, the KV pool whose layout a PD peer checks: the token
        KV view of a pool that wraps one beside recurrent state, else the pool
        itself. For a state entry, the view it registers, where it has one of
        its own (a window sub-pool's)."""
        states = self._states()
        if sub_pool == self.owner:
            return states.owner_view
        return states.views[sub_pool]()

    def regions(self, sub_pool: str, *, part: Optional[str] = None) -> Region:
        if sub_pool == self.owner:
            if self._owner_region is None:
                self._owner_region = _static_owner_region(self._allocator.get_kvcache())
            return self._owner_region
        return self._states().regions[sub_pool]

    def wire_entries(self) -> Tuple[WireEntry, ...]:
        return self._states().wire_entries

    def sub_pool_names(self) -> Tuple[str, ...]:
        return (self.owner,)

    def tree_slots(self, *, window_reach: int = 0) -> Tuple[TreeSlot, ...]:
        shape = self._shape
        window = max(shape.window, window_reach) if shape.window is not None else None
        return (
            (TreeSlot(component="FULL", family=TREE_TOKEN),)
            + (
                (TreeSlot(component="SWA", family=TREE_WINDOW, window=window),)
                if shape.has_window
                else ()
            )
            + (
                (
                    TreeSlot(
                        component="MAMBA",
                        family=TREE_STATE,
                        checkpoint_interval=shape.checkpoint_interval,
                    ),
                )
                if shape.has_state
                else ()
            )
        )

    def add_pin_source(
        self, sub_pools: Sequence[str], busy: Callable[[], bool]
    ) -> None:
        """Static pools never move a row, so a transfer has nothing to pin."""

    def _states(self) -> _StaticStateEntries:
        if self._state_entries is None:
            state = self._allocator.state
            self._state_entries = _static_state_entries(
                self._allocator.get_kvcache(),
                draft_token_to_kv_pool=self._draft_token_to_kv_pool,
                total_kv_layers=self._num_hidden_layers,
                state_pool=None if state is None else state.pool,
            )
        return self._state_entries


class _StaticStateEntries(msgspec.Struct, frozen=True, kw_only=True):
    owner_view: Any
    regions: Dict[str, Region]
    # A state entry's view, for the entries that have one, read when asked.
    views: Dict[str, Callable[[], Any]]
    wire_entries: Tuple[WireEntry, ...]


def _static_owner_region(token_to_kv_pool) -> Region:
    """The KV entries a static pool registers."""
    from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

    if isinstance(token_to_kv_pool, MiniMaxSparseKVPool):
        ptrs, lens, item_lens = token_to_kv_pool.get_sparse_kv_buf_infos()
    else:
        ptrs, lens, item_lens = token_to_kv_pool.get_contiguous_buf_infos()
    from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool

    # A pool wrapping one token KV view (its other layers hold state), or one
    # whose main page rows pair by layer id, reports per-entry layer ids for the
    # peers to pair by.
    layer_ids = (
        token_to_kv_pool.get_kv_layer_ids()
        if token_to_kv_pool.layer_router.wrapped_view() is not None
        or isinstance(token_to_kv_pool, MiniMaxM3Nvfp4KVPool)
        else ()
    )
    return Region(
        ptrs=tuple(ptrs),
        lens=tuple(lens),
        item_lens=tuple(item_lens),
        layer_ids=tuple(layer_ids),
    )


def _is_dsv4_npu(pool) -> bool:
    from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
        DSV4NPUTokenToKVPool,
    )

    return isinstance(pool, DSV4NPUTokenToKVPool)


def _static_state_entries(
    token_to_kv_pool,
    *,
    draft_token_to_kv_pool,
    total_kv_layers: Optional[int],
    state_pool,
) -> _StaticStateEntries:
    """A static pool's state entries in the order PD peers register them; a
    draft's state rides in the target's entries where the pool family shares
    them."""
    from sglang.srt.disaggregation.base.conn import StateType
    from sglang.srt.disaggregation.utils import remap_draft_layer_ids
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
    from sglang.srt.mem_cache.memory_pool import (
        DSATokenToKVPool,
        MHATokenToKVPool,
        MHATokenToKVPoolMXFP8,
        MiniMaxSparseKVPool,
    )
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    entries: List[Tuple[Any, Region, Any]] = []
    types: List[Any] = []
    owner_view = token_to_kv_pool

    def add(
        state_type,
        data_ptrs,
        data_lens,
        item_lens,
        dim_per_tensor=None,
        conv_shard_groups=None,
        slice_outer_counts=None,
        layer_ids=None,
        *,
        kv_layout=None,
        view: Optional[Callable[[], Any]] = None,
    ) -> None:
        types.append(state_type)
        entries.append(
            (
                state_type,
                Region(
                    ptrs=tuple(data_ptrs),
                    lens=tuple(data_lens),
                    item_lens=tuple(item_lens),
                    dim_per_tensor=tuple(dim_per_tensor or ()),
                    conv_shard_groups=tuple(conv_shard_groups or ()),
                    slice_outer_counts=tuple(slice_outer_counts or ()),
                    layer_ids=tuple(layer_ids or ()),
                    kv_layout=kv_layout,
                ),
                view,
            )
        )

    def append_dsa_tail(pool) -> None:
        if not pool.kpool_use_compress:
            return
        tail_ptrs, tail_lens, tail_item_lens = pool.get_compress_tail_buf_infos()
        if tail_ptrs:
            add(
                StateType.DSA_TAIL,
                tail_ptrs,
                tail_lens,
                tail_item_lens,
            )

    if isinstance(token_to_kv_pool, MHATokenToKVPoolMXFP8):
        add(
            StateType.BLOCK_SCALE,
            *token_to_kv_pool.get_kv_scale_buf_infos(),
        )

    if isinstance(token_to_kv_pool, MiniMaxSparseKVPool):
        if token_to_kv_pool.index_kv_pool is not None:
            raise NotImplementedError(
                "PD disaggregation for MiniMax sparse layers with index value "
                "(index_kv_pool) is not yet supported; only K-only sparse layers are."
            )
        if token_to_kv_pool.index_k_pool is not None:
            dp, dl, il = token_to_kv_pool.get_index_k_state_buf_infos()
            add(StateType.MINIMAX_INDEX_K, dp, dl, il)
        dense_ptrs, dense_lens, dense_item_lens = (
            token_to_kv_pool.get_dense_kv_state_buf_infos()
        )
        # An all-sparse pool has no dense layers; registering an empty component
        # would make every transfer carry a state with nothing to move.
        if dense_ptrs:
            add(StateType.MINIMAX_DENSE_KV, dense_ptrs, dense_lens, dense_item_lens)
    elif hasattr(token_to_kv_pool, "get_state_buf_infos"):
        data_ptrs, data_lens, item_lens = token_to_kv_pool.get_state_buf_infos()

        router = token_to_kv_pool.layer_router
        has_window_sub_pool = router.sub_pool(IdSpaceKind.SLIDING_WINDOW) is not None
        # A pool with a sliding-window sub-pool describes its window state here,
        # and so does DeepSeek-V4's, its heterogeneous state list per entry.
        if has_window_sub_pool or isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool):
            swa_kv_layout = None
            if has_window_sub_pool and isinstance(
                router.view(IdSpaceKind.SLIDING_WINDOW), MHATokenToKVPool
            ):
                swa_kv_layout = router.view(IdSpaceKind.SLIDING_WINDOW).kv_cache_layout
            add(
                StateType.SWA,
                data_ptrs,
                data_lens,
                item_lens,
                kv_layout=swa_kv_layout,
                # Read when asked, as the window's view was: not every pool
                # with window state keeps a window pool of its own.
                view=lambda: token_to_kv_pool.swa_kv_pool,
            )
            # MXFP8 KV: each sub-pool's block scales ride as their own component
            # so they inherit the index payload of the KV they describe.
            # Only a pool with a window sub-pool owns a full sub-pool here;
            # DeepSeek-V4 describes its state per entry.
            if has_window_sub_pool:
                # Global and SWA caches may use different dtypes.
                for pool, state_type in (
                    (token_to_kv_pool.full_kv_pool, StateType.BLOCK_SCALE),
                    (token_to_kv_pool.swa_kv_pool, StateType.BLOCK_SCALE_SWA),
                ):
                    if isinstance(pool, MHATokenToKVPoolMXFP8):
                        add(state_type, *pool.get_kv_scale_buf_infos())
            # unified_kv: the SWA ring lives in the unified buffers (no separate
            # swa_kv_pool) and is addressed per-row, so ship it as SWA_RING.
            if getattr(token_to_kv_pool, "_unified_kv", False) and hasattr(
                token_to_kv_pool, "get_unified_swa_ring_buf_infos"
            ):
                ring_ptrs, ring_lens, ring_item_lens = (
                    token_to_kv_pool.get_unified_swa_ring_buf_infos()
                )
                if ring_ptrs:
                    add(
                        StateType.SWA_RING,
                        ring_ptrs,
                        ring_lens,
                        ring_item_lens,
                    )
            if hasattr(token_to_kv_pool, "get_request_state_buf_infos"):
                c128_ptrs, c128_lens, c128_item_lens = (
                    token_to_kv_pool.get_request_state_buf_infos()
                )
                if c128_ptrs:
                    add(
                        StateType.DSV4_REQUEST_STATE,
                        c128_ptrs,
                        c128_lens,
                        c128_item_lens,
                    )
        elif router.wrapped_view() is not None:
            dim = (
                token_to_kv_pool.get_state_dim_per_tensor()
                if hasattr(token_to_kv_pool, "get_state_dim_per_tensor")
                else None
            )
            # Peers read the token KV view's layout, not the wrapping pool's.
            owner_view = token_to_kv_pool.full_kv_pool
            conv_shard_groups = (
                token_to_kv_pool.get_state_conv_shard_groups()
                if hasattr(token_to_kv_pool, "get_state_conv_shard_groups")
                else None
            )
            slice_outer_counts = (
                token_to_kv_pool.get_state_slice_outer_counts()
                if hasattr(token_to_kv_pool, "get_state_slice_outer_counts")
                else None
            )
            # Global layer ids let the sender pair src/dst entries when the
            # prefill PP stage registers only its own subset of mamba layers.
            layer_ids = token_to_kv_pool.get_state_layer_ids()
            add(
                StateType.MAMBA,
                data_ptrs,
                data_lens,
                item_lens,
                dim,
                conv_shard_groups,
                slice_outer_counts,
                layer_ids,
            )
            # Hybrid DSA pools keep their index cache and kpool tail in the
            # full-attention sub-pool rather than in the Mamba state above.
            if getattr(token_to_kv_pool, "use_dsa", False):
                dsa_pool = token_to_kv_pool.full_kv_pool
                dsa_ptrs, dsa_lens, dsa_item_lens = dsa_pool.get_state_buf_infos()
                add(
                    StateType.DSA,
                    dsa_ptrs,
                    dsa_lens,
                    dsa_item_lens,
                )
                append_dsa_tail(dsa_pool)
            if isinstance(token_to_kv_pool, QSATokenToKVPool):
                qsa_ptrs, qsa_lens, qsa_item_lens = (
                    token_to_kv_pool.get_qsa_pending_state_buf_infos()
                )
                qsa_layer_ids = token_to_kv_pool.get_qsa_pending_state_layer_ids()
                compressed_ptrs, compressed_lens, compressed_item_lens = (
                    token_to_kv_pool.get_qsa_compressed_state_buf_infos()
                )
                compressed_layer_ids = (
                    token_to_kv_pool.get_qsa_compressed_state_layer_ids()
                )
                if isinstance(draft_token_to_kv_pool, QSATokenToKVPool):
                    if total_kv_layers is None:
                        raise ValueError(
                            "QSA draft state transfer requires total_kv_layers"
                        )
                    draft_ptrs, draft_lens, draft_item_lens = (
                        draft_token_to_kv_pool.get_qsa_pending_state_buf_infos()
                    )
                    draft_layer_ids = remap_draft_layer_ids(
                        draft_token_to_kv_pool.get_qsa_pending_state_layer_ids(),
                        total_kv_layers,
                    )
                    qsa_ptrs += draft_ptrs
                    qsa_lens += draft_lens
                    qsa_item_lens += draft_item_lens
                    qsa_layer_ids += draft_layer_ids

                    (
                        draft_compressed_ptrs,
                        draft_compressed_lens,
                        draft_compressed_item_lens,
                    ) = draft_token_to_kv_pool.get_qsa_compressed_state_buf_infos()
                    draft_compressed_layer_ids = remap_draft_layer_ids(
                        draft_token_to_kv_pool.get_qsa_compressed_state_layer_ids(),
                        total_kv_layers,
                    )
                    compressed_ptrs += draft_compressed_ptrs
                    compressed_lens += draft_compressed_lens
                    compressed_item_lens += draft_compressed_item_lens
                    compressed_layer_ids += draft_compressed_layer_ids
                add(
                    StateType.QSA_PENDING,
                    qsa_ptrs,
                    qsa_lens,
                    qsa_item_lens,
                    layer_ids=qsa_layer_ids,
                )
                add(
                    StateType.QSA_COMPRESSED,
                    compressed_ptrs,
                    compressed_lens,
                    compressed_item_lens,
                    layer_ids=compressed_layer_ids,
                )
        elif isinstance(token_to_kv_pool, (DSATokenToKVPool, NPUMLATokenToKVPool)):
            tail_ptrs, tail_lens, tail_item_lens = [], [], []
            if isinstance(token_to_kv_pool, DSATokenToKVPool):
                tail_ptrs, tail_lens, tail_item_lens = (
                    token_to_kv_pool.get_compress_tail_buf_infos()
                )
            if draft_token_to_kv_pool is not None and isinstance(
                draft_token_to_kv_pool, DSATokenToKVPool
            ):
                (
                    draft_data_ptrs,
                    draft_data_lens,
                    draft_item_lens,
                ) = draft_token_to_kv_pool.get_state_buf_infos()
                data_ptrs = data_ptrs + draft_data_ptrs
                data_lens = data_lens + draft_data_lens
                item_lens = item_lens + draft_item_lens
                draft_tail_ptrs, draft_tail_lens, draft_tail_item_lens = (
                    draft_token_to_kv_pool.get_compress_tail_buf_infos()
                )
                tail_ptrs = tail_ptrs + draft_tail_ptrs
                tail_lens = tail_lens + draft_tail_lens
                tail_item_lens = tail_item_lens + draft_tail_item_lens
            # An NPU MLA pool sends its index and tail with the KV (PD reads
            # its buffer groups from the pool), so it adds no state entry.
            if not isinstance(token_to_kv_pool, NPUMLATokenToKVPool):
                add(StateType.DSA, data_ptrs, data_lens, item_lens)
                if tail_ptrs:
                    add(StateType.DSA_TAIL, tail_ptrs, tail_lens, tail_item_lens)

    if is_npu() and _is_dsv4_npu(token_to_kv_pool):
        from sglang.srt.disaggregation.ascend.conn import AscendStateType

        c128_ptrs, c128_lens, c128_item_lens = token_to_kv_pool.get_c128_kv_buf_infos()
        if c128_ptrs:
            add(
                AscendStateType.DSV4_C128,
                c128_ptrs,
                c128_lens,
                c128_item_lens,
            )

        # On A5 (CYCLE cache_mode), C4 state uses request-local ring rows rather
        # than SWA pages.  Register it separately so P and D can independently
        # map logical positions when their local ring sizes differ.
        from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

        if is_npu_arch35():
            c4_ptrs, c4_lens, c4_item_lens = token_to_kv_pool.get_c4_state_buf_infos()
            if c4_ptrs:
                add(
                    AscendStateType.DSV4_C4_STATE,
                    c4_ptrs,
                    c4_lens,
                    c4_item_lens,
                )

    # DSV4 NextN shares the target allocator, so target and draft use the same
    # local SWA indices. Keep draft buffers in a separate positional component
    # to avoid mixing them into the target's heterogeneous state layout, while
    # reusing the existing SWA transport dispatch on both GPU and NPU.
    if isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool) and isinstance(
        draft_token_to_kv_pool, DeepSeekV4TokenToKVPool
    ):
        if not draft_token_to_kv_pool.compression_ratios or not all(
            ratio == 0 for ratio in draft_token_to_kv_pool.compression_ratios
        ):
            raise RuntimeError(
                "DSV4 draft state transfer expects SWA-only NextN layers"
            )
        if token_to_kv_pool._unified_kv != draft_token_to_kv_pool._unified_kv:
            raise RuntimeError(
                "DSV4 target and draft pools must use the same unified-KV mode"
            )

        if token_to_kv_pool._unified_kv:
            target_geometry = (
                token_to_kv_pool.unified_swa_window,
                token_to_kv_pool.unified_swa_ring_size,
                token_to_kv_pool.unified_swa_pages,
            )
            draft_geometry = (
                draft_token_to_kv_pool.unified_swa_window,
                draft_token_to_kv_pool.unified_swa_ring_size,
                draft_token_to_kv_pool.unified_swa_pages,
            )
            if target_geometry != draft_geometry:
                raise RuntimeError(
                    "DSV4 target and draft pools must share SWA ring geometry: "
                    f"target={target_geometry}, draft={draft_geometry}"
                )
            draft_ptrs, draft_lens, draft_item_lens = (
                draft_token_to_kv_pool.get_unified_swa_ring_buf_infos()
            )
            draft_state_type = StateType.SWA_RING
        else:
            if (
                token_to_kv_pool.full_to_swa_index_mapping
                is not draft_token_to_kv_pool.full_to_swa_index_mapping
            ):
                raise RuntimeError(
                    "DSV4 target and draft pools must share the SWA index mapping"
                )
            # Under encoder replay the target window and the draft ring address SWA
            # by request slot; equal rows per slot let them share one page list.
            request_window = token_to_kv_pool.request_window
            target_geometry = (
                token_to_kv_pool.page_size,
                token_to_kv_pool.sliding_window,
                request_window.capacity if request_window is not None else None,
            )
            draft_geometry = (
                draft_token_to_kv_pool.page_size,
                draft_token_to_kv_pool.sliding_window,
                draft_token_to_kv_pool.swa_req_ring_size,
            )
            if target_geometry != draft_geometry:
                raise RuntimeError(
                    "DSV4 target and draft pools must share SWA geometry: "
                    f"target={target_geometry}, draft={draft_geometry}"
                )
            draft_ptrs, draft_lens, draft_item_lens = (
                draft_token_to_kv_pool.get_state_buf_infos()
            )
            draft_state_type = StateType.SWA

        if draft_ptrs:
            add(
                draft_state_type,
                draft_ptrs,
                draft_lens,
                draft_item_lens,
            )
        row_ptrs, row_lens, row_item_lens = (
            draft_token_to_kv_pool.get_swa_valid_from_buf_infos()
        )
        if row_ptrs:
            add(StateType.REQUEST_ROW, row_ptrs, row_lens, row_item_lens)

    # The recurrent state pool, when the KV pool does not describe it (a
    # window KV pool beside a state sub-pool).
    if StateType.MAMBA not in types and state_pool is not None:
        data_ptrs, data_lens, item_lens = state_pool.get_contiguous_buf_infos()
        if data_ptrs:
            dim = state_pool.get_state_dim_per_tensor()
            conv_shard_groups = state_pool.get_state_conv_shard_groups()
            slice_outer_counts = state_pool.get_state_slice_outer_counts()
            add(
                StateType.MAMBA,
                data_ptrs,
                data_lens,
                item_lens,
                dim,
                conv_shard_groups,
                slice_outer_counts,
            )

    regions = {}
    views = {}
    wire_entries = []
    for i, (state_type, region, view) in enumerate(entries):
        key = f"state{i}"
        regions[key] = region
        if view is not None:
            views[key] = view
        wire_entries.append(WireEntry(sub_pool=key, part="", state_type=state_type))
    return _StaticStateEntries(
        owner_view=owner_view,
        regions=regions,
        views=views,
        wire_entries=tuple(wire_entries),
    )


def _kv_shard_rotation_bases(
    tree_cache: BasePrefixCache, batch: ScheduleBatch, prefix_lens_cpu: torch.Tensor
) -> list:
    """Per-request rotation bases ``b_i`` of the batch's chains, host-only.

    The owner class of position-page P is ``(b_i + P) % shard_size``. Rules:

    - Read through ``req.last_node`` at alloc time, never a value cached on
      the request: a ``checkpoint`` can rebind a chunked request onto
      another chain's canonical locs between chunks, changing the base. The
      read goes through ``tree_cache.rotation_base_of`` because the node
      handle is tree-specific (a NodeId on the unified tree).
    - A request without a cached prefix starts a new chain: None here — the
      allocator draws from the least-full class at that request's turn (so
      the draw sees earlier requests' pops in the same batch) and resolves
      the entry in place.
    - A radix-disabled cache keeps no nodes past the root, which carries no
      base; its chunked continuations fall back to the base recorded on the
      request at the previous chunk's alloc (no cross-request reuse, no
      rebind there).
    """
    assert batch is not None and len(batch.reqs) == len(prefix_lens_cpu)
    bases = []
    for i, req in enumerate(batch.reqs):
        if int(prefix_lens_cpu[i]) == 0:
            bases.append(None)
            continue
        node_base = tree_cache.rotation_base_of(req.last_node)
        if node_base is not None:
            bases.append(node_base)
        else:
            assert req.kv_rotation_base is not None, (
                "sharded extend with a cached prefix but no rotation base: "
                "req.last_node carries none and the request recorded none "
                "at a previous alloc"
            )
            bases.append(req.kv_rotation_base)
    return bases
