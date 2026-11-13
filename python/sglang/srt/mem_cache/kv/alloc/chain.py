"""One allocator over the chain of bands in a `KVArena`.

Every sub-pool is a band allocator, an end or a float of the chain. The owner
band mints the virtual token ids; a window band binds the owner's virtual
pages; a state band hands out per-request slots. Every lifecycle call (frees,
groups, events, pins, flushes, verification) is one loop over the bands, and
every capacity question is one layout search over them, whatever the chain's
shape.
"""

from __future__ import annotations

import functools
import logging
import math
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import msgspec
import torch
from torch.profiler import record_function

from sglang.kernels.ops.memory.allocator import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.environ import InvariantCheckLevel
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.capacity import observe_only_read
from sglang.srt.mem_cache.kv.alloc.band import (
    Band,
    FloatBand,
    allocator_operation,
    chain_byte_accounting_violations,
    flush_deferred_free_group,
    settle_pending_for_recovery,
)
from sglang.srt.mem_cache.kv.alloc.demand import (
    DemandPurpose,
    KVDemand,
    demand_inputs,
    smallest_admissible_chunk,
)
from sglang.srt.mem_cache.kv.alloc.forward_steps import ForwardSteps
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
from sglang.srt.mem_cache.kv.alloc.invariants import (
    WINDOW_SEGMENT_BOUND,
    WINDOW_SEGMENT_RELEASED,
)
from sglang.srt.mem_cache.kv.alloc.page_usage import Consumer
from sglang.srt.mem_cache.kv.alloc.pricing import (
    NO_MOVE,
    ChainSnapshot,
    TakePlan,
    feasible,
    first_plan,
    move_order,
)
from sglang.srt.mem_cache.kv.alloc.pricing import plan as plan_layout
from sglang.srt.mem_cache.kv.alloc.pricing import (
    reclaimed_snapshot,
    snapshot_chain,
)
from sglang.srt.mem_cache.kv.alloc.signals import NetworkWaiter
from sglang.srt.mem_cache.kv.alloc.stage_capacity import (
    StageCapacity,
    StageGeometry,
    check_stage_pools,
    simulated_snapshot,
    stage_geometry,
)
from sglang.srt.mem_cache.kv.memory.arena import KVArena
from sglang.srt.mem_cache.utils import get_num_new_pages
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.invariants import expect, resolve_level

# Rows an unreleased network transfer touches, fixed for one relief.
_NETWORK_ROWS = "network-rows"

# A band's growth average keeps 1 - 1/_GROWTH_DECAY of itself per take, in
# units of 1/_GROWTH_SCALE page.
_GROWTH_DECAY = 16
_GROWTH_SCALE = 64

logger = logging.getLogger(__name__)


def _units_key(demands: Mapping[str, int]) -> Tuple[Tuple[str, int], ...]:
    """A demand of units per band as a question pipeline stages compare."""
    return tuple(sorted(demands.items()))


def _one_operation(method):
    """Run a chain allocation as one allocator operation over every member
    (`band.py::allocator_operation`): its host reads come before its first
    stream wait, whichever band waits first."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        chain = getattr(self, "allocator", self)
        with allocator_operation(chain.members):
            return method(self, *args, **kwargs)

    return wrapper


class _ScopedPhysicalTakes:
    """Materialize a reservation in any band order without changing its fit.

    Live-page deltas count completed takes, including owner/state fast paths.
    Existing cancellation returns those pages and restores the pending demand.
    An unrelated free conservatively restores demand as well. No device ids
    are inspected, and leaving the scope never rolls back queued transfers.
    """

    def __init__(self, allocator, demands):
        self.allocator = allocator
        self.demands = {name: n for name, n in demands.items() if n > 0}
        # Make the rearrangement this reservation needs before any take is
        # issued, so the takes then fit in place, in whatever order they come.
        # Under pipeline stages, which decide on snapshots that fix no page, none is
        # fixed here: a host load's reservation ends inside the scope that took it.
        if not allocator.lay_out_for(self.demands):
            raise RuntimeError(
                "physical reservation needs reclamation before its scope"
            )
        snapshot = allocator.pricing_snapshot()
        if allocator.plan_in_place(self.demands) is None:
            raise RuntimeError("a realized layout does not hold its reservation")
        self.initial_live = {band.name: band.live for band in snapshot.bands}
        self.applying = False

    @_one_operation
    def take(self, band, pages, consumer=Consumer.FORWARD):
        snapshot = self.allocator.pricing_snapshot()
        remaining = {
            current.name: self.demands.get(current.name, 0)
            - (current.live - self.initial_live[current.name])
            for current in snapshot.bands
        }
        if pages > remaining[band.sub_pool_name] or any(
            n < 0 for n in remaining.values()
        ):
            return None
        # Planned again from the current layout: the takes so far consumed
        # prefixes of earlier plans, so the rest still fits in place.
        layout = self.allocator.plan_in_place(
            {name: n for name, n in remaining.items() if n > 0}
        )
        if layout is None:
            return None
        self.applying = True
        try:
            return band.take_planned_pages(
                layout.takes[band.sub_pool_name].prefix(pages), consumer
            )
        finally:
            self.applying = False


class ChainRoles(msgspec.Struct, frozen=True, kw_only=True):
    """Sub-pool names by role: the virtual-id owner, the sliding window that
    binds the owner's virtual pages, and the per-request state slots."""

    owner: str
    window: Optional[str] = None
    state: Optional[str] = None
    windows: Tuple[str, ...] = ()


def _chain_order(bands: Sequence[Band]) -> Tuple[Band]:
    """Low byte to high byte: the grow-up end, the floats, the grow-down end."""
    rank = {"up": 0, "float": 1, "down": 2}
    return tuple(sorted(bands, key=lambda band: rank[band.grow_direction]))


def _wire_chain(chain: Sequence[Band]) -> None:
    if len(chain) == 2:
        low, high = chain
        low.bind_peer(high)
        high.bind_peer(low)
        return
    for low, high in zip(chain, chain[1:]):
        low.bind_high_peer(high)
        high.bind_low_peer(low)


def take_new_pages(
    owner: Band,
    num_new_pages: int,
    page_tokens: Callable[[torch.Tensor], torch.Tensor],
    *,
    copy_new_pages: bool,
    physical_pages: Optional[torch.Tensor] = None,
    consumer: Consumer = Consumer.FORWARD,
) -> Optional[torch.Tensor]:
    """Fill tail pages before taking new virtual pages for each request.

    Bind every new page in the owner band before translating its token ids;
    otherwise its v2p entry remains -1. ``consumer``: who writes the new
    pages first (`Band.acquire`)."""
    if num_new_pages > len(owner.free_virtual_ids):
        return None
    # Physical-capacity pre-check; on shortfall the chain lays itself out.
    need_tokens = num_new_pages * owner.page_size
    if physical_pages is None and need_tokens > owner.available_size():
        if not owner.relieve(need_tokens):
            return None
    # The free list is only rebound, not mutated in place, so its consumed
    # prefix still names the pages the kernel takes.
    new_virtual_pages = None
    if num_new_pages > 0:
        new_virtual_pages = owner.free_virtual_ids[:num_new_pages]
        if copy_new_pages:
            new_virtual_pages = new_virtual_pages.clone()
    out_indices = page_tokens(owner.free_virtual_ids)
    if new_virtual_pages is not None:
        if physical_pages is None:
            if owner.place_pages(new_virtual_pages, num_new_pages, consumer) is None:
                return None
        else:
            owner.ids.bind(new_virtual_pages, physical_pages)
            owner._maybe_zero_pages(physical_pages)
    # A rebind moves the capacity epoch; a step within its pages takes none.
    if num_new_pages > 0:
        owner.free_virtual_ids = owner.free_virtual_ids[num_new_pages:]
    return out_indices


class ChainAllocator(BaseTokenToKVPoolAllocator):
    """Token, window and state allocation over one shared byte buffer."""

    # Every sub-pool takes its rows from one shared arena.
    pools_share_memory = True

    hands_out_virtual_ids = True
    # State slots acquire allocates for a fresh request (its working slot and
    # initial tracking slots); the storage binds it from the state policy.
    fresh_request_state_slots = 1
    # Bound at construction: the sizing plan, the request rows' memory, and
    # (once the storage binds them) each window's retained tail by sub-pool.
    plan = None
    request_memory = None
    retained_windows: Optional[Dict[str, int]] = None
    # Under pipeline parallelism, every stage's geometry, which every capacity
    # question is asked of (`install_stage_capacity`); None on a single stage.
    stage_capacity: Optional[StageCapacity] = None
    # Every stage's simulated snapshot, kept until the capacity epoch moves.
    _stage_memo: Tuple[Optional[int], Tuple[ChainSnapshot, ...]] = (None, ())

    # Base init writes `self.size`; the chain computes its capacity.
    @property
    def size(self) -> int:
        """Token capacity for observers: held plus `available_size` with no window
        (the leak census's available term cancels), else the smaller side share."""
        if not self._windows:
            return self.available_size() + self.full_attn_allocator.allocated_count()
        return min(self._size_full, self._size_swa)

    @size.setter
    def size(self, value) -> None:
        pass

    def __init__(
        self,
        *,
        arena: KVArena,
        roles: ChainRoles,
        kvcache,
        views: Dict[str, Any],
        device: str,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = True,
        full_cap_tokens: Optional[int] = None,
        full_max_total_num_tokens: Optional[int] = None,
        swa_max_total_num_tokens: Optional[int] = None,
    ):
        # Token bands share widened virtual pages. A replicated draft carries
        # every token inside its entries; target views select the rank's rows.
        shards_under_dcp = True
        dcp_size = get_parallel().attn_dcp_size
        super().__init__(
            size=0,
            page_size=page_size * dcp_size,
            # The chain hands out units of the arena's bytes.
            dtype=torch.uint8,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.arena = arena
        self.roles = roles
        # Views used by each band's move, zero and copy operations.
        self.views = dict(views)
        # The ends compact lazily, behind the forwards that may still read the
        # pages they move: unified memory has no eager compaction.
        assert lazy_compaction, "the unified pool's ends compact lazily"
        self.lazy_compaction = lazy_compaction
        band = dict(
            arena=arena,
            device=device,
            need_sort=need_sort,
            forward_stream=forward_stream,
        )
        self.full_attn_allocator = Band(
            sub_pool_name=roles.owner,
            is_id_owner=True,
            page_size=page_size,
            shards_under_dcp=shards_under_dcp,
            lazy_compaction=lazy_compaction,
            **band,
        )
        owner = self.full_attn_allocator
        owner.max_live_tokens = (
            None
            if full_cap_tokens is None
            else full_cap_tokens * (owner.page_size // owner.pool_page_size)
        )
        self.window_allocators: Dict[str, Band] = {}
        window_names = roles.windows or (
            () if roles.window is None else (roles.window,)
        )
        for name in window_names:
            floats = arena.band(name).grow_direction == "float"
            self.window_allocators[name] = (FloatBand if floats else Band)(
                sub_pool_name=name,
                is_id_owner=False,
                page_size=page_size,
                shards_under_dcp=shards_under_dcp,
                lazy_compaction=lazy_compaction and not floats,
                virtual_num_pages=self.full_attn_allocator.ids.num_virtual_ids,
                **band,
            )
        self.swa_attn_allocator = self.window_allocators.get(roles.window)
        self.mamba_allocator: Optional[Band] = None
        if roles.state is not None:
            self.mamba_allocator = Band(
                sub_pool_name=roles.state,
                is_id_owner=True,
                page_size=1,  # state is per request, orthogonal to KV paging
                lazy_compaction=lazy_compaction,
                **band,
            )
        # Role order: owner, window, state. Every fan-out walks this tuple.
        self.members: Tuple[Band, ...] = tuple(
            b
            for b in (
                self.full_attn_allocator,
                *self.window_allocators.values(),
                self.mamba_allocator,
            )
            if b is not None
        )
        # A lazy chain hands a freed page out again from the schedule stream
        # (zeroing, compaction), so its freed pages wait for the steps in flight.
        self.forward_steps: Optional[ForwardSteps] = (
            ForwardSteps(self.members) if lazy_compaction else None
        )
        for band in self.members:
            band.chain_allocator = self
        by_name = {band.sub_pool_name: band for band in self.members}
        self.bands = tuple(by_name[band.name] for band in arena.bands)
        _wire_chain(self.bands)
        self._band_positions = {b.sub_pool_name: i for i, b in enumerate(self.bands)}
        # Moving average of each band's growth past its holes, `_GROWTH_SCALE`
        # per page; read from the plans alone, so every rank keeps the same one.
        self._growth: Dict[str, int] = {b.sub_pool_name: 0 for b in self.bands}
        self._snapshot_key: Optional[int] = None
        self._snapshot_memo: Optional[ChainSnapshot] = None
        self._window_capacity_from_sizing = full_max_total_num_tokens is not None
        self._set_token_shares(full_max_total_num_tokens, swa_max_total_num_tokens)

        if self.swa_attn_allocator is not None:
            self._size_window_host_pools()
        # Translated before the pool when the pool indexes host copies physically.
        self.free_page_reps_group: Optional[List[torch.Tensor]] = None
        self.full_page_reps_group: List[torch.Tensor] = []
        self.full_free_group: List[torch.Tensor] = []
        if self._keeps_leak_census():
            # Empty (not None) for the leak checker.
            self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
            self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[unified-memory-pool] ChainAllocator ready: %s, joint available=%d",
            ", ".join(
                f"{b.sub_pool_name}({b.grow_direction}) max_slots={b.max_slots} "
                f"entry_bytes={b.entry_bytes}"
                for b in self.bands
            ),
            self.available_size(),
        )

    def _set_token_shares(
        self,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
    ) -> None:
        """Each side's nominal token share: the sizing's rows per rank in logical
        tokens, else the empty pool's; without a window, the owner's id space."""
        owner, window = self.full_attn_allocator, self.swa_attn_allocator
        if window is None:
            # A logical token capacity, widened like `size`, not a row count.
            self._size_full = (owner.max_slots - 1) * get_parallel().attn_dcp_size
            self._size_swa = 0
            return
        if (full_max_total_num_tokens is None) != (swa_max_total_num_tokens is None):
            raise ValueError(
                "full_max_total_num_tokens and swa_max_total_num_tokens must "
                "either both be set or both be omitted"
            )
        if full_max_total_num_tokens is not None:
            self._size_full = (
                int(full_max_total_num_tokens) * owner.page_size // owner.pool_page_size
            )
            self._size_swa = (
                int(swa_max_total_num_tokens)
                * window.page_size
                // window.pool_page_size
            )
        else:
            self._size_full = owner.available_size()
            self._size_swa = min(
                window.available_size(), len(owner.free_virtual_ids) * self.page_size
            )

    def _keeps_leak_census(self) -> bool:
        """Whether the leak checker's census applies: not where it would mix
        physical and virtual ids (a float, a state end, several windows)."""
        return (
            self.mamba_allocator is None
            and len(self._windows) <= 1
            and not any(band.grow_direction == "float" for band in self.bands)
        )

    def _size_window_host_pools(self) -> None:
        """Host views advertise physical rows, before the host layout widens IDs."""
        for band, tokens in (
            (self.full_attn_allocator, self.size_full),
            *((window, self.size_swa) for window in self._windows),
        ):
            pool = self.views[band.sub_pool_name]
            rows = tokens * band.pool_page_size // band.page_size
            pool.host_capacity_tokens = rows
            pool.host_capacity_bytes = rows * band.entry_bytes

    @property
    def has_window_pool(self) -> bool:
        return self.swa_attn_allocator is not None

    @property
    def _windows(self) -> Tuple[Band, ...]:
        return tuple(self.window_allocators.values())

    # -- the pricing snapshot --

    def _snapshot_key_now(self) -> int:
        """The chain's capacity epoch and, per band, its compaction mode, which
        is not a capacity field."""
        epoch = bits = 0
        for band in self.bands:
            epoch += band._capacity_epoch
            bits = (bits << 1) | bool(band.lazy_compaction)
        return (epoch << len(self.bands)) | bits

    def pricing_snapshot(self) -> ChainSnapshot:
        """The bands' snapshot every capacity decision prices on, kept until
        the epoch moves."""
        key = self._snapshot_key_now()
        if key != self._snapshot_key:
            self._snapshot_memo = snapshot_chain(
                self.bands,
                page_size=self.page_size,
                total_bytes=self.arena.total_bytes,
                positions=self._band_positions,
            )
            self._snapshot_key = key
        return self._snapshot_memo

    def network_held_bytes(self) -> Tuple[Tuple[int, int], ...]:
        """Merged byte ranges outside the bands that unreleased network sends
        still read (a last-resort move's fences); no device read, alike on every
        rank."""
        ranges = sorted(
            (use.lo, use.hi)
            for band in self.bands
            if band.usage.has_fences()
            for use in band.usage.network_fences()
        )
        merged: List[Tuple[int, int]] = []
        for lo, hi in ranges:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return tuple(merged)

    def held_snapshot(self) -> ChainSnapshot:
        """The pricing snapshot with `network_held_bytes` held, so a realization
        first tries the layouts in which no take waits for a send."""
        snapshot = self.pricing_snapshot()
        held = self.network_held_bytes()
        if not held:
            return snapshot
        return msgspec.structs.replace(snapshot, held=held)

    def plan_in_place(self, demands: Dict[str, int]) -> Optional[TakePlan]:
        """Takes that fit where the bands stand: around send-held bytes when that
        serves, else through them (such a take waits on the host for the send)."""
        held = self.held_snapshot()
        found = plan_layout(held, demands, NO_MOVE)
        if found is None and held.held:
            found = plan_layout(self.pricing_snapshot(), demands, NO_MOVE)
        return found

    def _snapshot_memo_violations(self) -> List[str]:
        """Idle-time check: a kept snapshot that differs from a fresh one means a
        capacity field changed without moving the epoch. Empty == healthy."""
        if self._snapshot_key != self._snapshot_key_now():
            return []
        fresh = snapshot_chain(
            self.bands,
            page_size=self.page_size,
            total_bytes=self.arena.total_bytes,
        )
        if fresh == self._snapshot_memo:
            return []
        return [f"[chain] stale pricing snapshot: {self._snapshot_memo} != {fresh}"]

    def _cap_full_available(self, tokens):
        return self.full_attn_allocator.cap_available(tokens)

    # -- pipeline stages --
    #
    # Every stage holds the same units in its own arena and asks each capacity
    # question of every stage's simulated snapshot, so every stage decides alike.

    def stage_geometry(self) -> StageGeometry:
        """This stage's arena bytes and bands, as every stage prices them."""
        id_owners = tuple(
            band.sub_pool_name
            for band in (self.full_attn_allocator, self.mamba_allocator)
            if band is not None
        )
        return stage_geometry(
            self.pricing_snapshot(),
            id_owners=id_owners,
            window_tokens=self._size_swa,
        )

    def _stage_pools(self) -> Tuple[Tuple[str, ...], int]:
        return tuple(sorted(band.sub_pool_name for band in self.bands)), self.page_size

    def agree_on_stage_shares(self, group) -> None:
        """Every stage in ``group`` joins, whatever its bands: all must plan the same
        sub-pools, then with a window each keeps the smallest FULL and window shares,
        so request limits and the HiCache host pools sized from them agree."""
        every = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(
            every,
            (self._stage_pools(), self._size_full, self._size_swa),
            group=group,
        )
        check_stage_pools([pools for pools, _, _ in every])
        if self._windows:
            self._size_full = min(full for _, full, _ in every)
            self._size_swa = min(window for _, _, window in every)
            self._size_window_host_pools()

    def install_stage_capacity(
        self,
        stage: int,
        geometries: Sequence[StageGeometry],
        *,
        checks: bool = False,
    ) -> None:
        """Answer capacity questions as stage ``stage`` of ``geometries`` (stage order),
        asking every stage; each keeps the smallest nominal window share."""
        if geometries[stage] != self.stage_geometry():
            raise ValueError(f"stage {stage}'s geometry is not this chain's")
        check_stage_pools([(g.sub_pools(), g.page_size) for g in geometries])
        if self._windows:
            self._size_swa = min(geometry.window_tokens for geometry in geometries)
        self.stage_capacity = StageCapacity(stage, geometries, checks=checks)
        self._stage_memo = (None, ())

    def _logical_counts(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Live pages per band, and virtual pages in use per id owner: what
        every stage holds alike."""
        live = {
            band.sub_pool_name: band.allocated_count() // band.page_size
            for band in self.bands
        }
        used = {
            band.sub_pool_name: band.ids.num_virtual_ids
            - band.min_page_index
            - len(band.free_virtual_ids)
            for band in (self.full_attn_allocator, self.mamba_allocator)
            if band is not None
        }
        return live, used

    def _stage_snapshots(self) -> Tuple[ChainSnapshot, ...]:
        """The snapshots each capacity question is asked of: the pricing snapshot on a
        single stage, else every stage's simulated snapshot, in stage order."""
        capacity = self.stage_capacity
        if capacity is None:
            return (self.pricing_snapshot(),)
        key = self._snapshot_key_now()
        if self._stage_memo[0] != key:
            live, used = self._logical_counts()
            self._stage_memo = (
                key,
                tuple(
                    simulated_snapshot(geometry, live, used)
                    for geometry in capacity.geometries
                ),
            )
        return self._stage_memo[1]

    def _ask(
        self,
        op: str,
        key: Callable[[], Any],
        on_stage: Callable[[ChainSnapshot], Any],
        combine: Callable = all,
    ):
        """Question ``(op, key())`` answered by ``on_stage`` on the pricing snapshot,
        or under pipeline stages ``combine`` of its answers on every stage's
        snapshot; ``key`` is built only for the stages' digest."""
        capacity = self.stage_capacity
        if capacity is None:
            return on_stage(self.pricing_snapshot())
        answer = combine(on_stage(s) for s in self._stage_snapshots())
        if capacity.checks:
            capacity.note(op, key(), answer)
        return answer

    def _decided_take(self, demands: Dict[str, int]) -> Optional[bool]:
        """Whether a take of ``demands`` (units per band) succeeds on every pipeline
        stage; None on a single stage, where the take itself decides."""
        if self.stage_capacity is None:
            return None
        demands = {n: d for n, d in demands.items() if d > 0}
        return self._ask(
            "take",
            lambda: _units_key(demands),
            lambda s: self._units_fit_on(s, demands),
        )

    def _refused_decided_take(self, decided: Optional[bool], demands) -> None:
        # The only fixed pages under pipeline stages are a host load's reservation,
        # held only inside its own scope, which laid out every take of that load.
        if decided:
            raise RuntimeError(
                f"this pipeline stage cannot take {demands}, which every "
                "stage decided it takes"
            )

    def stage_counts(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """The units every stage holds alike (`_logical_counts`)."""
        return self._logical_counts()

    # -- capacity questions --
    #
    # Sub-pools share bytes and demands mix per-token and per-request units, so
    # decisions ask the demand itself; the scalar reads below are observer estimates.

    def units_fit(self, demands: Dict[str, int]) -> bool:
        """Whether a take of these units (pages of a token sub-pool, slots of the
        state sub-pool) succeeds now by the take's own checks, on every PP stage."""
        return self._ask(
            "units_fit",
            lambda: _units_key(demands),
            lambda snapshot: self._units_fit_on(snapshot, demands),
        )

    def _units_fit_on(self, snapshot: ChainSnapshot, demands: Dict[str, int]) -> bool:
        owner = snapshot.band(self.full_attn_allocator.sub_pool_name)
        owner_pages = demands.get(owner.name, 0)
        if owner_pages > owner.free_virtual_pages:
            return False
        if self._full_cap_reclaim(owner_pages * self.page_size):
            return False
        # A window binds the owner's virtual pages: it can never hold more.
        bound = owner.num_virtual_pages - owner.min_page
        if any(demands.get(w.sub_pool_name, 0) > bound for w in self._windows):
            return False
        state = self.mamba_allocator
        if (
            state is not None
            and demands.get(state.sub_pool_name, 0)
            > snapshot.band(state.sub_pool_name).free_virtual_pages
        ):
            return False
        return feasible(snapshot, {n: d for n, d in demands.items() if d > 0})

    def _estimated_pages(self, names: Tuple[str, ...]) -> int:
        """Pages each of ``names`` could take together by free bytes, index space, id
        room and FULL cap; it ignores alignment, fixed and pending pages, so may err."""
        # The least over every stage's simulated snapshot, so every stage agrees.
        return min(
            self._estimated_pages_on(snapshot, names)
            for snapshot in self._stage_snapshots()
        )

    def _estimated_pages_on(
        self, snapshot: ChainSnapshot, names: Tuple[str, ...]
    ) -> int:
        owner_name = self.full_attn_allocator.sub_pool_name
        owner_snap = snapshot.band(owner_name)
        # Every band's first page lies past the padding sink.
        sink = min(b.min_page * b.page_bytes for b in snapshot.bands)
        free_bytes = (
            snapshot.total_bytes
            - sink
            - sum(b.live * b.page_bytes for b in snapshot.bands)
        )
        named = [snapshot.bands[snapshot.index(name)] for name in names]
        pages = max(0, free_bytes) // sum(b.page_bytes for b in named)
        for b in named:
            # Its id owner's room: an owner's own free ids; a window binds
            # the token owner's.
            if b.name in self.window_pool_names():
                id_room = owner_snap.num_virtual_pages - owner_snap.min_page
            else:
                id_room = b.free_virtual_pages
            pages = min(pages, b.num_pages - b.min_page - b.live, id_room)
        owner = self.full_attn_allocator
        if owner.sub_pool_name in names and owner.max_live_tokens is not None:
            room = max(0, owner.max_live_tokens - owner.allocated_count())
            pages = min(pages, room // self.page_size)
        return max(0, pages)

    def estimated_units_available(self, sub_pool: str) -> int:
        """An estimate of what ``sub_pool`` alone could still take, in tokens (slots for
        state), for observers; reading it inside a capacity decision is a violation."""
        observe_only_read("ChainAllocator.estimated_units_available")
        band = self.bands[self._band_positions[sub_pool]]
        pages = self._estimated_pages((sub_pool,))
        return pages if band is self.mamba_allocator else pages * self.page_size

    def state_slots_fit(self, count: int) -> bool:
        """Whether ``count`` state slots can be taken now."""
        return count <= 0 or self.units_fit({self.mamba_allocator.sub_pool_name: count})

    def estimated_state_slots_available(self) -> int:
        """`estimated_units_available` of the state sub-pool: observers only."""
        return self.estimated_units_available(self.mamba_allocator.sub_pool_name)

    def held_units(self, sub_pool: str) -> int:
        """Units sub-pool ``sub_pool`` holds now: tokens (slots for state)."""
        return self.bands[self._band_positions[sub_pool]].allocated_count()

    # The scalar reads the shared allocator interface requires: estimates
    # (`estimated_units_available`), never for decisions.

    def available_size(self) -> int:
        """An estimate of the tokens one more token allocation could take, the owner
        and every window taking each token together. Observers only."""
        observe_only_read("ChainAllocator.available_size")
        names = tuple(
            b.sub_pool_name for b in (self.full_attn_allocator, *self._windows)
        )
        return self._estimated_pages(names) * self.page_size

    def owner_available_size(self):
        """An estimate of the token sub-pool's availability. Observers only."""
        return self.full_available_size()

    def full_available_size(self) -> int:
        return self.estimated_units_available(self.full_attn_allocator.sub_pool_name)

    def swa_available_size(self) -> int:
        if self.swa_attn_allocator is None:
            raise AttributeError("a chain without a window has no window tokens")
        return self.estimated_units_available(self.swa_attn_allocator.sub_pool_name)

    def window_available_size(self, sub_pool: str) -> int:
        """An estimate of window sub-pool ``sub_pool``'s availability."""
        if sub_pool not in self.window_pool_names():
            raise KeyError(f"no window sub-pool {sub_pool!r}")
        return self.estimated_units_available(sub_pool)

    def conserve_full_available_size(self) -> int:
        """The sizing's nominal owner share minus the owner tokens held: the
        static split, not what a take can serve."""
        return self._size_full - self.full_attn_allocator.allocated_count()

    def conserve_swa_available_size(self) -> int:
        """The sizing's nominal window share minus the window tokens held."""
        return self._size_swa - self.swa_attn_allocator.allocated_count()

    @property
    def size_full(self) -> int:
        cap = self.full_attn_allocator.max_live_tokens
        if cap is None or not self._windows:
            return self._size_full
        return min(self._size_full, cap)

    @property
    def size_swa(self) -> int:
        return self._size_swa

    @property
    def draft_virtual_id_space(self) -> int:
        owner = self.full_attn_allocator
        dcp_size = get_parallel().attn_dcp_size if owner.shards_under_dcp else 1
        return (owner.max_slots - 1) * dcp_size

    def _full_cap_reclaim(self, full_tokens: int, *, empty_pool=False) -> int:
        owner = self.full_attn_allocator
        if owner.max_live_tokens is None:
            return 0
        demand = int(-(-full_tokens // self.page_size)) * self.page_size
        live = 0 if empty_pool else owner.allocated_count()
        return (
            -(-max(0, live + demand - owner.max_live_tokens) // self.page_size)
            * self.page_size
        )

    def owner_tokens_only_owner_frees_serve(self, owner_pages: int) -> int:
        """Of a take of ``owner_pages`` owner pages, the tokens only freeing owner
        tokens makes room for (past the FULL live-token cap or the owner's free ids,
        the most over PP stages), so an eviction can count them before it walks."""
        owner = self.full_attn_allocator.sub_pool_name

        def on_stage(snapshot: ChainSnapshot) -> int:
            short_ids = max(0, owner_pages - snapshot.band(owner).free_virtual_pages)
            return max(
                self._full_cap_reclaim(owner_pages * self.page_size),
                short_ids * self.page_size,
            )

        return self._ask(
            "owner_frees_serve", lambda: owner_pages, on_stage, combine=max
        )

    def window_pool_names(self) -> Tuple[str, ...]:
        return tuple(window.sub_pool_name for window in self._windows)

    def target_window_pool(self) -> Optional[str]:
        """The target's window (the longest window its layers declare)."""
        return self.roles.window

    def sub_pool_roles(self) -> Tuple[str, Tuple[str, ...], Optional[str]]:
        """The token sub-pool's name, every window sub-pool's, and the state
        sub-pool's (None without one)."""
        state = self.mamba_allocator
        return (
            self.full_attn_allocator.sub_pool_name,
            self.window_pool_names(),
            None if state is None else state.sub_pool_name,
        )

    def page_demands(self, demand: KVDemand) -> Dict[str, int]:
        """Pages per band for a token demand: the owner's tokens, each
        window's, and state slots."""
        demand = demand.checked(self.window_pool_names())
        page = self.page_size
        demands = {
            self.full_attn_allocator.sub_pool_name: -(-int(demand.full_tokens) // page)
        }
        for name, tokens in demand.window_tokens.items():
            demands[name] = -(-int(tokens) // page)
        if self.mamba_allocator is not None:
            demands[self.mamba_allocator.sub_pool_name] = int(demand.state_slots)
        return demands

    def ensure_capacity(self, demand: KVDemand) -> bool:
        """Whether an allocation of this demand succeeds now: the question the
        take itself answers, so a yes here is a successful take."""
        return self.units_fit(self.page_demands(demand))

    def admits(
        self,
        demand: KVDemand,
        *,
        evictable: Optional[Dict[str, int]] = None,
        empty_pool: bool = False,
    ) -> bool:
        """Whether ``demand`` fits once ``evictable`` cached entries (tokens per
        sub-pool, slots for state) are evicted, or in an empty pool. Every
        admission question asks this, so an admitted demand is one the take
        serves. Token counts may be fractional expected values."""
        window_tokens = dict(demand.window_tokens)
        if set(window_tokens) != set(self.window_pool_names()):
            raise ValueError(
                f"admission must name every window: got {sorted(window_tokens)}, "
                f"the pool keeps {sorted(self.window_pool_names())}"
            )
        evictable = {} if empty_pool or evictable is None else evictable
        units = self._admission_units(demand, evictable, empty_pool)

        def key():
            return (
                demand.full_tokens,
                tuple(sorted(window_tokens.items())),
                demand.state_slots,
                tuple(sorted(evictable.items())),
                empty_pool,
            )

        if units is None:
            return self._ask("admits", key, lambda snapshot: False)
        demands, reclaim = units
        return self._ask(
            "admits",
            key,
            lambda snapshot: self._admits_on(
                snapshot, demand.state_slots, demands, reclaim, empty_pool
            ),
        )

    def _admission_units(
        self, demand: KVDemand, evictable: Dict[str, int], empty_pool: bool
    ) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
        """The units ``demand`` takes and those ``evictable`` frees, per band;
        None when it fits no stage (negative counts, the FULL cap)."""
        full_tokens = demand.full_tokens
        if full_tokens < 0 or any(t < 0 for t in demand.window_tokens.values()):
            return None
        owner = self.full_attn_allocator.sub_pool_name
        # Expected decode lengths may be fractional; a page holds whole tokens.
        full_tokens = math.ceil(full_tokens)
        if self._full_cap_reclaim(full_tokens, empty_pool=True):
            return None
        if self._full_cap_reclaim(full_tokens, empty_pool=empty_pool) > evictable.get(
            owner, 0
        ):
            return None
        page = self.page_size
        demands = {owner: -(-full_tokens // page)}
        reclaim = {owner: evictable.get(owner, 0) // page}
        for name in self.window_pool_names():
            demands[name] = -(-math.ceil(demand.window_tokens[name]) // page)
            reclaim[name] = evictable.get(name, 0) // page
        state = self.mamba_allocator
        if state is not None:
            demands[state.sub_pool_name] = int(demand.state_slots)
            reclaim[state.sub_pool_name] = int(evictable.get(state.sub_pool_name, 0))
        return demands, reclaim

    def _admits_on(
        self,
        snapshot: ChainSnapshot,
        state_slots,
        demands: Dict[str, int],
        reclaim: Dict[str, int],
        empty_pool: bool,
    ) -> bool:
        """`admits` on one stage's ``snapshot``: its ids, then its layout."""
        owner = snapshot.band(self.full_attn_allocator.sub_pool_name)
        owner_pages = owner.num_virtual_pages - owner.min_page
        # A window binds the owner's virtual pages: it can never hold more.
        if any(demands[name] > owner_pages for name in self.window_pool_names()):
            return False
        free_ids = (
            owner_pages
            if empty_pool
            else owner.free_virtual_pages + reclaim[owner.name]
        )
        if demands[owner.name] > free_ids:
            return False
        if self.mamba_allocator is not None:
            state = snapshot.band(self.mamba_allocator.sub_pool_name)
            free_slots = (
                state.num_virtual_pages - state.min_page
                if empty_pool
                else state.free_virtual_pages + reclaim[state.name]
            )
            if state_slots > free_slots:
                return False
        snapshot = reclaimed_snapshot(snapshot, reclaim, empty_pool=empty_pool)
        return feasible(snapshot, {n: d for n, d in demands.items() if d > 0})

    def prealloc_ceiling_fits(self, demand: KVDemand) -> bool | None:
        demand = demand.checked(self.window_pool_names())
        return self.admits(demand, empty_pool=True)

    def window_capacity_and_available(self, sub_pool: str) -> Tuple[int, int]:
        """Window ``sub_pool``'s tokens held plus its estimated availability, and that
        estimate. Observers only."""
        band = self._window_band(sub_pool)
        available = self.estimated_units_available(sub_pool)
        return available + band.allocated_count(), available

    def _window_band(self, sub_pool: str):
        for window in self._windows:
            if window.sub_pool_name == sub_pool:
                return window
        raise KeyError(f"no window sub-pool {sub_pool!r}")

    def prealloc_fits(
        self,
        demand: KVDemand,
        *,
        evictable: Dict[str, int],
        full_budget_tokens: int,
        window_budget_tokens: Optional[Dict[str, int]] = None,
    ) -> bool:
        """A decode-node preallocation on the admission rule: its tokens, window
        tails and state slots priced together. The scheduler's budgets count from
        zero (`capacity.shares_memory`): each is minus what it holds back for pending
        work, charged as pending demand, never as a per-side cap, since the sides
        share bytes."""
        demand = demand.checked(self.window_pool_names())
        windows = dict(demand.window_tokens)
        for name, budget in (window_budget_tokens or {}).items():
            windows[name] += max(0, -budget)
        return self.admits(
            KVDemand(
                demand.full_tokens + max(0, -full_budget_tokens),
                windows,
                demand.state_slots,
            ),
            evictable=evictable,
        )

    def prealloc_shortfall(self, demand: KVDemand) -> str | None:
        """None when a preallocation of ``demand`` fits now, else the shortfall;
        eviction, if any, came first (`mem_cache.capacity`)."""
        if self.units_fit(self.page_demands(demand)):
            return None
        return (
            "Unified memory reclamation insufficient: "
            f"needed=(full={demand.full_tokens}, "
            f"windows={dict(demand.window_tokens)}, "
            f"state_slots={demand.state_slots})"
        )

    def fits_after_frees(self, demands: Dict[str, int]) -> bool:
        """Whether ``demands`` fit once the caller's queued frees apply, so an
        eviction walk sees the room it made and stops."""
        self.flush_deferred_full_frees()
        return self.units_fit(demands)

    def set_network_waiter(self, waiter: Optional[NetworkWaiter]) -> None:
        """Wait on the host for network transfers a move cannot avoid: every
        band of the chain asks ``waiter``."""
        for band in self.bands:
            band.network_waiter = waiter

    def usage(self) -> Dict[str, Tuple[int, int]]:
        """Each sub-pool's (capacity, available units), by sub-pool name: the
        units it holds plus its estimated availability, and that estimate
        (`estimated_units_available`). The held units are exact, so a leak
        census comparing what is held against the capacity stays exact.
        Observers only."""
        usage = {}
        for band in self.bands:
            name = band.sub_pool_name
            free = self.estimated_units_available(name)
            usage[name] = (band.allocated_count() + free, free)
        return usage

    def _ceiling_windows(self, sliding_window_size: int | None) -> Dict[str, int]:
        """The tail a lone request keeps of each window: the storage's retained
        lengths once bound, else the window's stated length or the scheduler's."""
        if self.retained_windows is not None:
            return dict(self.retained_windows)
        windows = {}
        for window in self._windows:
            stated = None
            if self.plan is not None:
                lifetime = self.plan.sub_pool(window.sub_pool_name).lifetime
                stated = getattr(lifetime, "window", None)
            windows[window.sub_pool_name] = (
                stated if stated is not None else sliding_window_size
            )
        return windows

    def max_new_tokens_for_memory(
        self,
        input_tokens: int,
        max_new_tokens: int,
        *,
        token_capacity: int,
        sliding_window_size: int | None,
        chunk_size: int | None,
        prefill_retire_lag: int,
        eviction_interval: int,
        checkpoint_interval: int = 0,
        decode_retire_lag: int = 0,
        chunk_align: int | None = None,
    ) -> int | None:
        """The most new tokens, at most ``max_new_tokens``, a lone request of
        ``input_tokens`` can be admitted with in an empty pool, within
        ``token_capacity``; None when even the prompt cannot. Priced as
        `DemandPurpose.CEILING`: all of its decode, a fresh request's state
        slots, window retirement (`demand.window_retirement`), and the smallest
        chunk admission may cut its prompt to (`smallest_admissible_chunk`)."""
        page = self.page_size
        paged_input = -(-input_tokens // page) * page
        max_new_tokens = min(
            max_new_tokens, max(0, token_capacity - paged_input - page - 1)
        )
        windows = self._ceiling_windows(sliding_window_size)
        state_slots = (
            0 if self.mamba_allocator is None else self.fresh_request_state_slots
        )
        # Below the grid no shorter chunk is admissible: admission runs the
        # request at the configured chunk or not at all.
        chunk = smallest_admissible_chunk(chunk_size, page, chunk_align) or chunk_size

        def fits(new_tokens):
            return self.admits(
                demand_inputs(
                    DemandPurpose.CEILING,
                    input_tokens,
                    new_tokens,
                    windows=windows,
                    page_size=page,
                    chunk_limit=chunk,
                    prefill_retire_lag=prefill_retire_lag,
                    eviction_interval=eviction_interval,
                    checkpoint_interval=checkpoint_interval,
                    decode_retire_lag=decode_retire_lag,
                    state_slots=state_slots,
                ).demand(),
                empty_pool=True,
            )

        if not fits(0):
            return None
        if fits(max_new_tokens):
            return max_new_tokens
        low, high = 0, max_new_tokens
        while low < high:
            mid = (low + high + 1) // 2
            if fits(mid):
                low = mid
            else:
                high = mid - 1
        return low

    def get_extend_allocation_demand(
        self,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        *,
        conservative_num_tokens: int,
        shard_size: int,
    ) -> int:
        """The new pages' tokens an extend takes; the conservative count when
        the extend is sharded, or the owner alone takes them."""
        if shard_size != 1 or not self._windows:
            return conservative_num_tokens
        return self.page_size * get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )

    # A shared-memory pool never calls the prefix cache (`mem_cache.capacity`
    # drives eviction), so the inherited separate-pool hooks raise.
    def _asks_the_cache(self, *args, **kwargs):
        raise TypeError(
            "a shared-memory pool does not call the prefix cache; ask "
            "sglang.srt.mem_cache.capacity instead"
        )

    admit_preallocation = _asks_the_cache
    create_prefill_budget = _asks_the_cache
    evict_to_free_tokens = _asks_the_cache
    check_decode_capacity = _asks_the_cache
    reclaim_for_prealloc = _asks_the_cache
    reclaim_for_load_back = _asks_the_cache

    def resize(self, config) -> None:
        if (
            not self.arena.backed_after_capture
            or config.unified_memory_pool_bytes != self.arena.total_bytes
        ):
            raise RuntimeError("the arena must be backed before allocator.resize")
        for band in self.members:
            if (
                band.allocated_count()
                or band.usage.has_pending()
                or band._pending_until_retire
            ):
                raise RuntimeError(
                    f"cannot resize non-empty band {band.sub_pool_name!r}"
                )
        owner = self.full_attn_allocator
        owner.max_live_tokens = (
            None
            if config.unified_full_cap_tokens is None
            else config.unified_full_cap_tokens
            * (owner.page_size // owner.pool_page_size)
        )
        for band in self.members:
            band._set_capacity(
                self.arena.num_entries(band.sub_pool_name),
                virtual_num_pages=None
                if band.is_id_owner
                else owner.ids.num_virtual_ids,
            )
        self._set_token_shares(
            (
                config.full_max_total_num_tokens
                if self._window_capacity_from_sizing
                else None
            ),
            (
                config.swa_max_total_num_tokens
                if self._window_capacity_from_sizing
                else None
            ),
        )
        self.get_kvcache().size = (
            config.full_max_total_num_tokens
            if self.swa_attn_allocator is None
            else owner.max_slots - 1
        )
        for band in self.members:
            view = self.views[band.sub_pool_name]
            view.size = band.max_slots - band.pool_page_size
            if band.sub_pool_name == self.roles.state:
                view.host_capacity_tokens = view.size
                view.host_capacity_bytes = view.size * band.entry_bytes
            else:
                view.num_pages = band.num_pages
        if self.swa_attn_allocator is not None:
            self._size_window_host_pools()
        else:
            view = self.views[owner.sub_pool_name]
            view.host_capacity_tokens = config.full_max_total_num_tokens
            view.host_capacity_bytes = (
                config.full_max_total_num_tokens * owner.entry_bytes
            )

    # -- state slots --

    def flush_deferred_full_frees(self) -> None:
        """Expose grouped owner frees without ending the caller's free group."""
        flush_deferred_free_group(
            self,
            (
                self.free_group,
                self.free_page_reps_group,
                self.full_page_reps_group,
                self.full_free_group,
            ),
        )

    # -- translation --

    def debug_print(self) -> str:
        return ", ".join(
            f"#{b.sub_pool_name}-available={b.available_size()}" for b in self.bands
        ) + (f", #joint-available={self.available_size()}")

    def get_kvcache(self):
        return self._kvcache

    def translate_kv_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Owner read path: virtual TOKEN ids -> physical TOKEN ids; an unmapped
        or negative id lands on 0, the sink. ``out=`` writes in place, for
        cuda-graph buffer stability."""
        return self.full_attn_allocator.ids.translate(loc, out=out)

    @property
    def full_v2p_page_table(self) -> torch.Tensor:
        """Page-level virtual->physical table of the owner; kernels that build a
        block table straight from req_to_token gather through it."""
        return self.full_attn_allocator.ids.virtual_to_physical

    @property
    def full_p2v_page_table(self) -> torch.Tensor:
        """Page-level physical->virtual table of the owner."""
        return self.full_attn_allocator.ids.physical_to_virtual

    @property
    def swa_v2p_page_table(self) -> torch.Tensor:
        """Page-level virtual->physical table of the window."""
        return self.swa_attn_allocator.ids.virtual_to_physical

    def id_spaces(self, *, defer_reads: bool = False) -> Dict[str, IdSpace]:
        """Every token sub-pool's id space, keyed by sub-pool name, in chain
        order: the owner's and each window's virtual ids map through that
        band's unit record. ``defer_reads`` keeps reads virtual (DCP selects a
        rank's share where it builds them); writes always translate, and only
        the owner's write window arrives DCP-widened."""
        owner = self.full_attn_allocator
        spaces = {
            owner.sub_pool_name: owner.ids.token_space(
                (owner.sub_pool_name, id(self)),
                widened_writes=True,
                defer_reads=defer_reads,
            )
        }
        for window in self._windows:
            spaces[window.sub_pool_name] = window.ids.token_space(
                (window.sub_pool_name, id(self)),
                widened_writes=True,
                defer_reads=defer_reads,
            )
        return spaces

    def state_id_spaces(self) -> Dict[str, IdSpace]:
        """The state band's id space: its slots."""
        band = self.mamba_allocator
        if band is None:
            return {}
        return {band.sub_pool_name: band.ids.slot_space((band.sub_pool_name, id(self)))}

    def translate_write_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        out_width: Optional[int] = None,
    ) -> torch.Tensor:
        """Widened virtual WRITE loc -> physical id; see `UnitIds.translate_write`."""
        return self.full_attn_allocator.ids.translate_write(
            loc, out=out, out_width=out_width
        )

    def translate_kv_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Virtual TOKEN ids -> owner PHYSICAL token ids for the PD transfer
        engine, which addresses page envelopes by physical id; virtual ids
        address real bytes too, so a missed translation corrupts silently."""
        # Defensive: argument validation rejects decode context parallelism with
        # PD disaggregation, so reaching it means a config path got past that.
        assert (
            not self.full_attn_allocator.shards_under_dcp
            or get_parallel().attn_dcp_size == 1
        ), (
            "PD-disaggregation transfer with the unified memory pool does not "
            "support decode context parallelism: the transfer ships whole page "
            "envelopes, which hold only this rank's shard of each widened page."
        )
        return self.full_attn_allocator.ids.translate(kv_indices.to(torch.int64))

    def translate_swa_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Virtual TOKEN ids -> window PHYSICAL token ids for the PD transfer
        engine: the window read translate, on int64 ids."""
        return self.swa_attn_allocator.ids.translate(kv_indices.to(torch.int64))

    @property
    def cpu_copy_carries_mamba(self) -> bool:
        return self.roles.state is not None

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        """Retraction backup: `Req.offload_kv_cache` hands over `req_to_token`
        rows, which hold VIRTUAL ids here; each sub-pool's view copies its
        physical ones."""
        from sglang.srt.mem_cache.kv.memory.snapshot import copy_to_cpu

        return copy_to_cpu(
            self, indices, state_indices=mamba_indices, request_row=req_pool_index
        )

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        from sglang.srt.mem_cache.kv.memory.snapshot import restore_from_cpu

        restore_from_cpu(
            self,
            kv_cache_cpu,
            indices,
            state_indices=mamba_indices,
            request_row=req_pool_index,
        )

    # -- ordering with transfers and forwards --

    def fix_pages(self, sub_pool: str, key: Hashable, pages: Sequence[int]) -> None:
        """Keep these physical pages of ``sub_pool`` where they are, under
        ``key``, until `unfix_pages`: a transfer that holds their physical ids,
        or a page that must not be touched. Moves go around them, and every
        plan keeps them inside their band."""
        self.bands[self._band_positions[sub_pool]].fix_pages(key, pages)

    def unfix_pages(
        self, sub_pool: str, key: Hashable, pages: Optional[Sequence[int]] = None
    ) -> None:
        """Release the pages of ``sub_pool`` fixed under ``key``: every one, or
        ``pages``."""
        self.bands[self._band_positions[sub_pool]].unfix_pages(key, pages)

    def end_load_reservations(self) -> None:
        """Submitted host loads order moves of the rows they write through
        their recorded copies: no member keeps rows reserved for a load."""
        for member in self.members:
            member.end_load_reservation()

    def prewarm_compaction(self) -> None:
        """Prewarm the page-move path of every member that compacts."""
        for member in self.members:
            member.prewarm_compaction()

    def compaction_can_run(self) -> bool:
        """Whether bytes freed on either side can serve the other side's next
        allocation now: its moves are unblocked."""
        return all(
            not band.moves_blocked()
            for band in (self.full_attn_allocator, *self._windows)
        )

    def shared_bytes(self, full_tokens: int, swa_tokens: int) -> int:
        """Shared-buffer bytes of FULL and SWA token counts, in whole pages."""
        page_size = self.page_size
        full_pages = (full_tokens + page_size - 1) // page_size
        swa_pages = (swa_tokens + page_size - 1) // page_size
        return (
            full_pages * self.full_attn_allocator.entry_bytes_per_page
            + swa_pages * self.swa_attn_allocator.entry_bytes_per_page
        )

    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:
        """Forward the per-batch `forward_done` event to every member."""
        with record_function("ChainAllocator.set_latest_forward_done_event"):
            for member in self.members:
                member.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
        state_slots_virtual: Optional[torch.Tensor] = None,
        *,
        token_writes_unknown: bool = False,
        state_writes_unknown: bool = False,
    ) -> None:
        """Hand the forward's metadata to every member; token bands translate the
        write set through their own v2p. The state band, written in place by
        state kernels rather than through `out_cache_loc`, takes the virtual
        state slots the forward writes. ``token_writes_unknown`` /
        ``state_writes_unknown``: the token / state bands cannot read this
        forward's write ids before it ends."""
        with record_function("ChainAllocator.set_inflight_forward"):
            for member in self.members:
                is_state = member is self.mamba_allocator
                member.set_inflight_forward(
                    forward_done,
                    state_slots_virtual if is_state else out_cache_loc_virtual,
                    writes_unknown=(
                        state_writes_unknown if is_state else token_writes_unknown
                    ),
                )

    def settle_pending_for_recovery(self, bands=None):
        """The entry of an urgent recovery over ``bands`` (every member by
        default): one host read ahead, then every pending page of those bands
        settles (`band.py::settle_pending_for_recovery`)."""
        return settle_pending_for_recovery(self.members if bands is None else bands)

    def retire_forward(self, forward_done: torch.cuda.Event) -> None:
        """The scheduler processed this forward's result; see `Band.retire_forward`."""
        for member in self.members:
            member.retire_forward(forward_done)

    def flush_opportunistic(self) -> int:
        """Non-urgent flush of every member, ends before floats; sync-free."""
        with record_function("ChainAllocator.flush_opportunistic"):
            if all(
                m._free_phys_pages.numel() == 0 and not m.usage.has_pending()
                for m in self.members
            ):
                return 0
            ends_first = sorted(self.members, key=lambda m: m.grow_direction == "float")
            return sum(m.flush_opportunistic() for m in ends_first)

    def verify_byte_accounting(self) -> List[str]:
        return (
            chain_byte_accounting_violations(list(self.bands))
            + self._snapshot_memo_violations()
        )

    # -- alloc --

    @_one_operation
    def _allocate(
        self,
        *,
        full_tokens: int,
        num_owner_pages: int,
        owner_take: Callable[[Optional[torch.Tensor]], Optional[torch.Tensor]],
        window_tail_pages: Optional[Mapping[str, int]] = None,
        consumer: Consumer = Consumer.FORWARD,
    ) -> Optional[torch.Tensor]:
        """Take the owner's and every window's new pages in one planned layout,
        then bind the windows to the owner's new virtual pages; None exactly when
        no reachable layout serves the demand. With ``window_tail_pages`` (naming
        every window), each window binds only the last that many new pages."""
        owner = self.full_attn_allocator
        demands = {owner.sub_pool_name: num_owner_pages}
        for window in self._windows:
            demands[window.sub_pool_name] = (
                num_owner_pages
                if window_tail_pages is None
                else window_tail_pages[window.sub_pool_name]
            )
        decided = self._decided_take(demands)
        if decided is False:
            return None
        if self._full_cap_reclaim(full_tokens) or num_owner_pages > len(
            owner.free_virtual_ids
        ):
            self._refused_decided_take(decided, demands)
            return None
        physical = self._plan_and_take(
            demands, bind_at_frontier=True, consumer=consumer
        )
        if physical is None:
            self._refused_decided_take(decided, demands)
            return None
        # free_virtual_ids is rebound (slice/cat), never mutated in place, so
        # this prefix view still names the pages the owner is about to consume.
        new_virtual_pages = owner.free_virtual_ids[:num_owner_pages]
        # None: the owner takes at its frontier and binds in one kernel.
        out = owner_take(physical.get(owner.sub_pool_name))
        if out is None:
            raise RuntimeError(
                "ChainAllocator: the owner refused pages its planned layout "
                "reserved -- internal-state inconsistency"
            )
        for window in self._windows:
            if window.sub_pool_name not in physical:
                continue
            pages = physical[window.sub_pool_name]
            num_pages = demands[window.sub_pool_name]
            # The extend kernel hands new pages out in token order, so the tail is
            # a slice, not a data-dependent filter that would block the host.
            bind = new_virtual_pages[new_virtual_pages.numel() - num_pages :]
            if pages is None:
                if window.place_pages(bind, num_pages, consumer) is None:
                    raise RuntimeError(
                        f"{window.sub_pool_name}: the planned frontier take of "
                        f"{num_pages} pages was refused"
                    )
                continue
            assert bind.numel() == pages.numel()
            window.ids.bind(bind, pages)
            window._maybe_zero_pages(pages)
        return out

    @_one_operation
    def _plan_and_take(
        self,
        demands: Dict[str, int],
        *,
        bind_at_frontier: bool = False,
        consumer: Consumer = Consumer.FORWARD,
    ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """Physical pages for ``demands`` (pages per band) in a planned layout;
        None exactly when `feasible` says no, and a plan that does not realize
        raises. With ``bind_at_frontier``, see `realize`."""
        demands = {name: n for name, n in demands.items() if n > 0}
        if not demands:
            return {}
        with self._relief(demands) as plan:
            if plan is None:
                return None
            return self.realize(
                plan, bind_at_frontier=bind_at_frontier, consumer=consumer
            )

    @_one_operation
    def realize(
        self,
        plan: TakePlan,
        *,
        bind_at_frontier: bool = False,
        consumer: Consumer = Consumer.FORWARD,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Lay the chain out as ``plan`` (made from its current snapshot) says and
        take each demanded band's physical pages; a refusal is an internal error.
        With ``bind_at_frontier``, a band whose take only extends its frontier
        maps to None: the caller takes and binds them in one kernel."""
        self._lay_out(plan)
        self._note_growth(plan)
        taken = {}
        for band in self.bands:
            take = plan.takes.get(band.sub_pool_name)
            if take is None or not take.count():
                continue
            if bind_at_frontier and band.takes_at_frontier(take):
                taken[band.sub_pool_name] = None
                continue
            physical = band.take_planned_pages(take, consumer)
            if physical is None:
                raise RuntimeError(
                    f"{band.sub_pool_name}: the planned take of {take.count()} "
                    "pages was refused"
                )
            taken[band.sub_pool_name] = physical
        return taken

    def _note_growth(self, plan: TakePlan) -> None:
        """Fold one plan's extensions (pages a take adds past the band's holes)
        into every band's growth average."""
        for name, average in self._growth.items():
            take = plan.takes.get(name)
            extended = 0
            if take is not None and take.pages is not None:
                extended = take.pages[1] - take.pages[0]
            self._growth[name] = (
                average - average // _GROWTH_DECAY + extended * _GROWTH_SCALE
            )

    @_one_operation
    def lay_out_for(self, demands: Dict[str, int]) -> bool:
        """Make the least rearrangement after which ``demands`` (pages per band)
        fit without moving a page, for takes issued later in any order; False
        exactly when `feasible` says no."""
        demands = {name: n for name, n in demands.items() if n > 0}
        with self._relief(demands) as plan:
            if plan is None:
                return False
            self._lay_out(plan)
            return True

    @contextmanager
    def _relief(self, demands: Dict[str, int]):
        """The plan serving ``demands``, and the scope it is laid out in.

        Plans but the last resort keep out of the bytes unreleased network sends
        still read (`held_snapshot`). A rearrangement reads every band's host
        state before the layout's first stream wait, and keeps rows a network
        transfer touches in place when some layout serves that way; only
        otherwise does it move them, and a copy or reuse of their bytes then
        waits on the host for the engine (`set_network_waiter`). The choice
        reads logical releases only, alike on every rank."""
        in_place = plan_layout(
            self.held_snapshot(), demands, NO_MOVE, growth=self._growth
        )
        if in_place is not None:
            yield in_place
            return
        if not feasible(self.pricing_snapshot(), demands):
            # No layout serves it: refused before any pending page settles.
            yield None
            return
        # Settle pending pages first, so the plan and its realization see holes
        # and live pages only (a float left with no live page parks).
        with self.settle_pending_for_recovery(self.bands):
            yield from self._relief_plan(demands)

    def _relief_plan(self, demands: Dict[str, int]):
        held = False
        try:
            for band in self.bands:
                if band.has_network_rows():
                    rows = band.network_rows()
                    if rows:
                        band.fix_pages(_NETWORK_ROWS, sorted(rows))
                        band.holds_fixed = True
                        held = True
            held_world = self.held_snapshot()
            found = (
                first_plan(held_world, demands, growth=self._growth)
                if held or held_world.held
                else None
            )
            if found is None and held:
                for band in self.bands:
                    band.unfix_pages(_NETWORK_ROWS)
                    band.holds_fixed = False
                held = False
            if found is None:
                found = first_plan(
                    self.pricing_snapshot(), demands, growth=self._growth
                )
            yield found
        finally:
            for band in self.bands:
                if held:
                    band.unfix_pages(_NETWORK_ROWS)
                band.holds_fixed = False

    def _lay_out(self, plan: TakePlan) -> None:
        """The plan's rearrangement: compact the listed ends, then move the
        floats into their target blocks."""
        bands = {band.sub_pool_name: band for band in self.bands}
        for name in plan.flush:
            band = bands[name]
            band.flush_for_allocation()
            low_byte = band._byte_low_frontier()
            high_byte = band._byte_high_frontier()
            live = band.occupied_pages() if band.lazy_compaction else None
            if live is not None:
                e = band.entry_bytes_per_page
                fixed = band.fixed_extent()
                if band.grow_direction == "up":
                    top = band.min_page_index + live
                    if fixed is not None:
                        top = max(top, fixed[1])
                    packed = (band.min_page_index * e, top * e)
                else:
                    bottom = band.num_pages - live
                    if fixed is not None:
                        bottom = min(bottom, fixed[0])
                    packed = (bottom * e, band.num_pages * e)
                if (low_byte, high_byte) != packed:
                    raise RuntimeError(
                        f"{name}: an urgent flush left {(low_byte, high_byte)}, "
                        f"the plan needs {packed}"
                    )
        blocks = {
            name: (
                first * bands[name].entry_bytes_per_page,
                end * bands[name].entry_bytes_per_page,
            )
            for name, (first, end) in plan.targets.items()
        }
        spans = {
            name: (bands[name]._byte_low_frontier(), bands[name]._byte_high_frontier())
            for name in plan.targets
        }
        for name in move_order(spans, blocks):
            bands[name].relocate_into(*plan.targets[name])

    @contextmanager
    def scoped_physical_takes(self, demand: KVDemand):
        """Consume a pre-realized reservation across independently issued H2D
        takes, on every chain shape: without the scope a floating window's take
        can land where the later state slot no longer fits."""
        if any(band._physical_take_scope is not None for band in self.members):
            raise RuntimeError("physical reservation scopes cannot be nested")
        scope = _ScopedPhysicalTakes(self, self.page_demands(demand))
        for band in self.members:
            band._physical_take_scope = scope
        try:
            yield
        finally:
            for band in self.members:
                band._physical_take_scope = None

    def _take_owner_tokens(self, need_size, physical, consumer=Consumer.FORWARD):
        owner = self.full_attn_allocator
        if physical is None:
            return owner.alloc(need_size, consumer)
        return take_new_pages(
            owner,
            need_size // self.page_size,
            lambda pages: owner._expand_pages_to_tokens(
                pages[: need_size // self.page_size]
            ),
            copy_new_pages=False,
            physical_pages=physical,
            consumer=consumer,
        )

    def alloc(
        self, need_size: int, consumer: Consumer = Consumer.FORWARD
    ) -> Optional[torch.Tensor]:
        """``need_size`` owner token ids; ``consumer`` writes their pages
        first (`Band.acquire`)."""
        with record_function("ChainAllocator.alloc"):
            return self._allocate(
                full_tokens=need_size,
                num_owner_pages=need_size // self.page_size,
                owner_take=lambda physical: self._take_owner_tokens(
                    need_size, physical, consumer
                ),
                consumer=consumer,
            )

    def _paged_take(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int,
        consumer: Consumer = Consumer.FORWARD,
    ) -> Callable[[Optional[torch.Tensor]], Optional[torch.Tensor]]:
        def extend_tokens(free_pages: torch.Tensor) -> torch.Tensor:
            bs = len(prefix_lens)
            out_indices = torch.empty(
                (extend_num_tokens,), dtype=torch.int64, device=self.device
            )
            # The kernel computes `page_id * page_size + offset`, so the free
            # virtual pages yield virtual token ids.
            with record_function("ChainAllocator.alloc_extend.kernel"):
                alloc_extend_kernel[(bs,)](
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    free_pages,
                    out_indices,
                    self.full_attn_allocator.page_size,
                )
            return out_indices

        return lambda physical=None: take_new_pages(
            self.full_attn_allocator,
            num_new_pages,
            extend_tokens,
            copy_new_pages=False,
            physical_pages=physical,
            consumer=consumer,
        )

    @_one_operation
    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        consumer: Consumer = Consumer.FORWARD,
    ) -> Optional[torch.Tensor]:
        """Paged extend; returns virtual TOKEN ids. A window binds exactly the
        virtual pages the owner consumed. ``consumer`` writes the new pages
        first (`Band.acquire`)."""
        with record_function("ChainAllocator.alloc_extend"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
            need_tokens = num_new_pages * self.page_size
            return self._allocate(
                full_tokens=need_tokens,
                num_owner_pages=num_new_pages,
                owner_take=self._paged_take(
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    num_new_pages,
                    consumer,
                ),
                consumer=consumer,
            )

    @_one_operation
    def alloc_extend_swa_tail(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        window_tail_lens: Mapping[str, int],
        consumer: Consumer = Consumer.FORWARD,
    ) -> Optional[torch.Tensor]:
        """Decode-node prealloc: owner KV for the whole sequence, window KV for
        the live window tail only. ``window_tail_lens`` names every window's
        own tail by sub-pool.

        The owner and the window SHARE one virtual id space (a virtual page names
        an owner-physical page and, if bound, a window-physical one), so the tail
        is expressed by binding the window for the TAIL's new virtual pages only.
        A new page left unbound has no window-physical page, which reads as the
        sink and is skipped by `free`'s `> 0` mask -- exactly the out-of-window
        state window retirement produces. Admission prices the owner's new pages
        and only the new pages in the window tail; a partial prefix page is
        already bound and costs no new window page. ``consumer`` writes the
        new pages first (`Band.acquire`).
        """
        assert len(prefix_lens_cpu) == 1
        tails = dict(
            KVDemand(extend_num_tokens, window_tail_lens)
            .checked(self.window_pool_names())
            .window_tokens
        )
        assert all(0 <= tail <= extend_num_tokens for tail in tails.values())
        with record_function("ChainAllocator.alloc_extend_swa_tail"):
            prefix_len = int(prefix_lens_cpu[0])
            seq_len = int(seq_lens_cpu[0])
            first_new_page = (prefix_len + self.page_size - 1) // self.page_size
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )

            def new_tail_pages(tail_len: int) -> int:
                # Only NEW pages need binding; a tail page carried in from the
                # prefix is already bound on the window.
                if not tail_len:
                    return 0
                first_tail_page = (seq_len - tail_len) // self.page_size
                return (seq_len + self.page_size - 1) // self.page_size - max(
                    first_new_page, first_tail_page
                )

            tail_pages = {
                window.sub_pool_name: new_tail_pages(tails[window.sub_pool_name])
                for window in self._windows
            }
            return self._allocate(
                full_tokens=num_new_pages * self.page_size,
                num_owner_pages=num_new_pages,
                owner_take=self._paged_take(
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    num_new_pages,
                    consumer,
                ),
                window_tail_pages=tail_pages,
                consumer=consumer,
            )

    @_one_operation
    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Paged decode: one new token per request, consuming a page only when the
        decode wraps."""
        with record_function("ChainAllocator.alloc_decode"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )

            def decode_tokens(free_pages: torch.Tensor) -> torch.Tensor:
                bs = len(seq_lens)
                out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
                with record_function("ChainAllocator.alloc_decode.kernel"):
                    alloc_decode_kernel[(bs,)](
                        seq_lens,
                        last_loc,
                        free_pages,
                        out_indices,
                        self.full_attn_allocator.page_size,
                    )
                return out_indices

            need_tokens = num_new_pages * self.page_size
            return self._allocate(
                full_tokens=need_tokens,
                num_owner_pages=num_new_pages,
                owner_take=lambda physical: take_new_pages(
                    self.full_attn_allocator,
                    num_new_pages,
                    decode_tokens,
                    copy_new_pages=True,
                    physical_pages=physical,
                ),
            )

    @_one_operation
    def alloc_state_slots(self, need_size: int) -> Optional[torch.Tensor]:
        """Per-request state slots (VIRTUAL slot ids), laid out with the token
        pages. Acquired as a schedule-stream copy's, since a state copy may write
        a new slot before any forward runs."""
        state = self.mamba_allocator
        if need_size <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        demands = {state.sub_pool_name: need_size}
        decided = self._decided_take(demands)
        if decided is False:
            return None
        if need_size > len(state.free_virtual_ids):
            self._refused_decided_take(decided, demands)
            return None
        physical = self._plan_and_take(
            demands, bind_at_frontier=True, consumer=Consumer.SCHEDULE_COPY
        )
        if physical is None:
            self._refused_decided_take(decided, demands)
            return None
        slots = state.free_virtual_ids[:need_size]
        pages = physical[state.sub_pool_name]
        if pages is None:
            if state.place_pages(slots, need_size, Consumer.SCHEDULE_COPY) is None:
                raise RuntimeError(
                    f"{state.sub_pool_name}: the planned frontier take of "
                    f"{need_size} slots was refused"
                )
        else:
            state.ids.bind(slots, pages)
            state._maybe_zero_pages(pages)
        state.free_virtual_ids = state.free_virtual_ids[need_size:]
        return state._expand_pages_to_tokens(slots)

    def is_slot_allocated(self, slot: int) -> bool:
        """Token-slot surface = the owner (which owns the virtual ids)."""
        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()

    # -- free --

    def _clear_inverse_history(self) -> None:
        for member in self.members:
            member.clear_inverse_history()

    def free(self, free_index: torch.Tensor) -> None:
        with record_function("ChainAllocator.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if self.free_group is not None:
                self.free_group.append(self._copy_for_free_group(free_index))
                return
            # Order is not load-bearing: each band's v2p IS the mapping. Only a
            # window needs the tombstone filter; the owner's ids are all bound.
            v = free_index.detach().to(torch.int64)
            for window in self._windows:
                window_v2p_pages = window.ids.virtual_to_physical[v // self.page_size]
                # `> 0` strict: -1 = tombstoned, 0 = padding-sink page; both skipped.
                live_tokens = v[window_v2p_pages > 0]
                if live_tokens.numel() > 0:
                    window.free(live_tokens)
            self.full_attn_allocator.free(v)
            self._clear_inverse_history()

    def free_swa(self, free_index: torch.Tensor) -> None:
        self.free_window(self.roles.window, free_index)

    def free_window(self, name: str, free_index: torch.Tensor) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        window = self.window_allocators[name]
        v = free_index.detach().to(torch.int64)
        live = v[window.ids.virtual_to_physical[v // self.page_size] > 0]
        if live.numel():
            window.free(live, _pages=live if self.page_size == 1 else None)
            window.clear_inverse_history()

    def free_swa_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        self.free_window_segment(self.roles.window, free_index, start_pos=start_pos)

    def free_window_segment(
        self, name: str, free_index: torch.Tensor, *, start_pos: int
    ) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        window = self.window_allocators[name]
        reps = window._page_reps(free_index.detach().to(torch.int64), start_pos)
        self._free_window_page_reps(window, reps)
        window.clear_inverse_history()

    def _free_window_page_reps(self, window: Band, reps: torch.Tensor) -> None:
        v_pages = window.virtual_pages_of(reps)
        # Signal-only invariant: avoid launching its lookup and comparison
        # when detection is disabled. This invariant has no recovery action.
        if resolve_level() >= InvariantCheckLevel.WARN:
            expect(
                WINDOW_SEGMENT_BOUND,
                window.ids.virtual_to_physical[v_pages] > 0,
                msg="caller wants free_full_segment",
            )
        window.free(reps, _pages=v_pages)

    def free_full(self, free_index: torch.Tensor) -> None:
        """Release owner ids after primary-window retirement.

        Independently retained windows may still own those ids and are released too.
        """
        if not self._windows:
            self.free(free_index)
            return
        if free_index is None or free_index.numel() == 0:
            return
        if self.free_group is not None:
            self.full_free_group.append(self._copy_for_free_group(free_index))
            return
        if len(self._windows) > 1:
            self.free(free_index)
        else:
            self.full_attn_allocator.free(free_index.detach().to(torch.int64))
            self.full_attn_allocator.clear_inverse_history()

    def free_full_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        """Release a kv-row segment after all its windows were released."""
        if not self._windows:
            BaseTokenToKVPoolAllocator.free_full_segment(
                self, free_index, start_pos=start_pos
            )
            return
        if free_index is None or free_index.numel() == 0:
            return
        reps = self.full_attn_allocator._page_reps(
            free_index.detach().to(torch.int64), start_pos
        )
        # Signal-only invariant: avoid launching its lookup and comparison
        # when detection is disabled. This invariant has no recovery action.
        if resolve_level() >= InvariantCheckLevel.WARN:
            pages = reps // self.page_size
            for window in self._windows:
                expect(
                    WINDOW_SEGMENT_RELEASED,
                    window.ids.virtual_to_physical[pages] <= 0,
                    msg="caller wants free_segment",
                )
        if self.free_page_reps_group is None:
            self._release_page_reps((), (reps,))
        else:
            self.full_page_reps_group.append(reps)

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        if full_indices.numel() == 0:
            return
        assert full_indices.numel() == swa_indices.numel()
        full_pages = full_indices.to(torch.int64) // self.page_size
        swa_pages = swa_indices.to(torch.int64) // self.page_size
        window = self.swa_attn_allocator
        window.ids.bind(full_pages, swa_pages)

    def clear_full_to_swa_mapping(self, full_indices: torch.Tensor) -> None:
        # Paired with set_full_to_swa_mapping: a shared id space has no mapping tensor.
        return

    # -- free-group --

    def free_group_begin(self) -> None:
        super().free_group_begin()
        self.free_page_reps_group = []
        self.full_page_reps_group = []
        self.full_free_group = []

    def free_group_end(self) -> None:
        pending, self.free_page_reps_group = self.free_page_reps_group, None
        full_pending, self.full_page_reps_group = self.full_page_reps_group, []
        full_free_group, self.full_free_group = self.full_free_group, []
        super().free_group_end()
        if full_free_group:
            self.free_full(torch.cat(full_free_group))
        if pending or full_pending:
            self._release_page_reps(pending, full_pending)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        """Fixed-shape counterpart of `free()` for a segment whose window pages
        are all bound; see `Band._page_reps`."""
        if free_index is None or free_index.numel() == 0:
            return
        if not self._windows and self.page_size == 1:
            # Without a window a page-1 segment is the plain free; the reps path
            # frees the same ids in another order.
            self.free(free_index)
            return
        reps = self.full_attn_allocator._page_reps(
            free_index.detach().to(torch.int64), start_pos
        )
        if self.free_page_reps_group is None:
            self._release_page_reps((reps,), ())
        else:
            self.free_page_reps_group.append(reps)

    def _release_page_reps(
        self, both: Sequence[torch.Tensor], full_only: Sequence[torch.Tensor]
    ) -> None:
        """Release fully bound segments and segments with no remaining windows."""
        pieces = (*both, *full_only)
        reps = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
        if both:
            window_reps = both[0] if len(both) == 1 else torch.cat(tuple(both))
            for window in self._windows:
                self._free_window_page_reps(window, window_reps)
        self.full_attn_allocator.free(
            reps, _pages=self.full_attn_allocator.virtual_pages_of(reps)
        )
        self._clear_inverse_history()

    def clear(self) -> None:
        for member in self.members:
            member.clear()
        if self.state is not None:
            # The state slots and the side states; the request table keeps ids.
            self.state.clear()
        self.free_group = None
        self.free_page_reps_group = None
        self.full_page_reps_group = []
        self.full_free_group = []
