"""One allocator over the chain of bands in a `KVArena`.

Every sub-pool is a band allocator, an end or a float of the chain. The owner
band mints the virtual token ids; a window band binds the owner's virtual
pages; a state band hands out per-request slots. Every lifecycle call (frees,
groups, events, pins, flushes, verification) is one loop over the bands.
What still differs between chain shapes is the interim capacity policy in
`interim.py`, chosen from the band geometry.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import msgspec
import torch
from torch.profiler import record_function

from sglang.srt.environ import InvariantCheckLevel
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.mem_cache.allocator.swa import (
    UNIFIED_SWA_SEGMENT_BOUND,
    UNIFIED_SWA_SEGMENT_RELEASED,
)
from sglang.srt.mem_cache.kv.alloc.band import (
    Band,
    FloatBand,
    chain_byte_accounting_violations,
    flush_deferred_free_group,
    full_tokens_before_mamba_recheck,
    relieve_for_alloc,
)
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
from sglang.srt.mem_cache.kv.alloc.interim import (
    ChainAdmissionRules,
    interim_policy_for,
)
from sglang.srt.mem_cache.kv.alloc.pricing import (
    ChainSnapshot,
    TakePlan,
    feasible,
    first_plan,
    move_order,
    no_relief_plan,
    reclaimed_snapshot,
    snapshot_chain,
)
from sglang.srt.mem_cache.kv.memory.arena import KVArena
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import get_num_new_pages, next_power_of_2
from sglang.srt.utils.invariants import expect, resolve_level

logger = logging.getLogger(__name__)


class _ScopedPhysicalTakes:
    """Materialize a reservation in any band order without changing its fit.

    Live-page deltas count completed takes, including owner/state fast paths.
    Existing cancellation returns those pages and restores the pending demand.
    An unrelated free conservatively restores demand as well. No device ids
    are inspected, and leaving the scope never rolls back queued transfers.
    """

    def __init__(self, allocator, demands):
        self.allocator = allocator
        self.demands = dict(demands)
        # Admission and reclaim answered with every rearrangement the chain can
        # make; make the one this reservation needs before any take is issued,
        # so the takes then fit in place, in whatever order they come.
        if no_relief_plan(
            allocator.pricing_snapshot(), tuple(self.demands.items())
        ) is None and not allocator.lay_out_for(self.demands):
            raise RuntimeError(
                "physical reservation needs reclamation before its scope"
            )
        snapshot = allocator.pricing_snapshot()
        if no_relief_plan(snapshot, tuple(self.demands.items())) is None:
            raise RuntimeError("a realized layout does not hold its reservation")
        self.initial_live = {band.name: band.live for band in snapshot.bands}
        self.applying = False

    def take(self, band, pages):
        snapshot = self.allocator.pricing_snapshot()
        remaining = {
            current.name: self.demands.get(current.name, 0)
            - (current.live - self.initial_live[current.name])
            for current in snapshot.bands
        }
        if pages > remaining[band.sub_pool_name]:
            return None
        plan = no_relief_plan(snapshot, tuple(remaining.items()))
        if plan is None:
            return None
        take = plan.takes[band.sub_pool_name].prefix(
            pages, page_bytes=band.entry_bytes_per_page
        )
        self.applying = True
        try:
            return band.take_planned_physical_pages(take)
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
) -> Optional[torch.Tensor]:
    """Fill tail pages before taking new virtual pages for each request.

    Bind every new page in the owner band before translating its token ids;
    otherwise its v2p entry remains -1.
    """
    if num_new_pages > len(owner.free_virtual_ids):
        return None
    # Physical-capacity pre-check; on shortfall run the relief ladder.
    need_tokens = num_new_pages * owner.page_size
    if physical_pages is None and need_tokens > owner.available_size():
        if not relieve_for_alloc(owner, need_tokens):
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
            if owner.place_pages(new_virtual_pages, num_new_pages) is None:
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

    hands_out_virtual_ids = True

    # Base init writes `self.size`; capacity is computed by the chain's policy.
    @property
    def size(self) -> int:
        return self._policy.size(self)

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
        lazy_compaction: bool = False,
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
        by_name = {band.sub_pool_name: band for band in self.members}
        self.bands = tuple(by_name[band.name] for band in arena.bands)
        _wire_chain(self.bands)
        self._band_positions = {b.sub_pool_name: i for i, b in enumerate(self.bands)}
        self._snapshot_key: Optional[int] = None
        self._snapshot_memo: Optional[ChainSnapshot] = None
        self._window_capacity_from_sizing = full_max_total_num_tokens is not None
        self._policy = interim_policy_for(
            self,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
        )

        if self.swa_attn_allocator is not None:
            self._size_window_host_pools()
        # Translated before the pool when the pool indexes host copies physically.
        self.free_page_reps_group: Optional[List[torch.Tensor]] = None
        self.full_page_reps_group: List[torch.Tensor] = []
        self.full_free_group: List[torch.Tensor] = []
        if self._policy.keeps_leak_census:
            # Empty (not None) for the leak checker.
            self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
            self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[unified-memory-pool] ChainAllocator ready (%s): %s, joint available=%d",
            self._policy.unit_name,
            ", ".join(
                f"{b.sub_pool_name}({b.grow_direction}) max_slots={b.max_slots} "
                f"entry_bytes={b.entry_bytes}"
                for b in self.bands
            ),
            self.available_size(),
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

    # -- capacity views (the interim policy of the chain's shape) --

    def _snapshot_key_now(self) -> int:
        """The chain's capacity epoch and, per band, its compaction mode and
        whether a pin source is busy: neither is a capacity field. A band's
        blocked bit is its load pin, a capacity field, or a busy pin source;
        sources are callables, so the epoch alone misses a flip."""
        epoch = bits = 0
        for band in self.bands:
            epoch += band._capacity_epoch
            pinned = band.pinned_by_source()
            bits = (bits << 2) | (2 if band.lazy_compaction else 0) | pinned
        return (epoch << (2 * len(self.bands))) | bits

    def pricing_snapshot(self) -> ChainSnapshot:
        """The bands' snapshot every capacity decision prices on, kept until
        the epoch moves or a pin source's answer changes."""
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

    def available_size(self) -> int:
        return self._cap_full_available(self._policy.available_size(self))

    def full_available_size(self) -> int:
        return self._cap_full_available(self._policy.full_available_size(self))

    def swa_available_size(self) -> int:
        return self._policy.swa_available_size(self)

    def conserve_full_available_size(self) -> int:
        return self._policy.conserve_full_available_size(self)

    def conserve_swa_available_size(self) -> int:
        return self._policy.conserve_swa_available_size(self)

    def schedulable_full_available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def schedulable_swa_available_size(self) -> int:
        return self.swa_attn_allocator.schedulable_available_size()

    @property
    def current_full_capacity(self) -> int:
        return self._policy.current_full_capacity(self)

    @property
    def current_swa_capacity(self) -> int:
        return self._policy.current_swa_capacity(self)

    @property
    def size_full(self) -> int:
        return self._policy.size_full(self)

    @property
    def size_swa(self) -> int:
        return self._policy.size_swa(self)

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

    def page_demands(
        self,
        full_tokens: int,
        swa_tokens: int,
        *,
        auxiliary_swa_tokens: Optional[int] = None,
        state_slots: int = 0,
    ) -> Dict[str, int]:
        """Pages per band for a token demand: the owner's and the window's
        tokens, every further window's (the owner's unless given), and state
        slots."""
        page = self.page_size
        demands = {self.full_attn_allocator.sub_pool_name: -(-int(full_tokens) // page)}
        for window in self._windows:
            tokens = (
                swa_tokens
                if window is self.swa_attn_allocator
                else full_tokens
                if auxiliary_swa_tokens is None
                else auxiliary_swa_tokens
            )
            demands[window.sub_pool_name] = -(-int(tokens) // page)
        if self.mamba_allocator is not None:
            demands[self.mamba_allocator.sub_pool_name] = int(state_slots)
        return demands

    def ensure_capacity(
        self,
        full_tokens: int,
        swa_tokens: int,
        *,
        auxiliary_swa_tokens: Optional[int] = None,
        state_slots: int = 0,
    ) -> bool:
        """Whether an allocation of this demand succeeds now: the question the
        take itself answers, so a yes here is a successful take."""
        if full_tokens < 0 or swa_tokens < 0:
            return False
        if self._full_cap_reclaim(full_tokens):
            return False
        demands = self.page_demands(
            full_tokens,
            swa_tokens,
            auxiliary_swa_tokens=auxiliary_swa_tokens,
            state_slots=state_slots,
        )
        owner = self.full_attn_allocator
        if demands[owner.sub_pool_name] > len(owner.free_virtual_ids):
            return False
        # A window binds the owner's virtual pages: it can never hold more.
        owner_pages = owner.ids.num_virtual_ids - owner.min_page_index
        if any(demands[w.sub_pool_name] > owner_pages for w in self._windows):
            return False
        state = self.mamba_allocator
        if state is not None and state_slots > len(state.free_virtual_ids):
            return False
        return feasible(
            self.pricing_snapshot(), {n: d for n, d in demands.items() if d > 0}
        )

    def _fits_after_reclaim(
        self,
        full_tokens,
        swa_tokens,
        *,
        auxiliary_swa_tokens=None,
        state_slots=0,
        full_evictable_tokens=0,
        swa_evictable_tokens=0,
        auxiliary_swa_evictable_tokens=0,
        state_evictable_slots=0,
        empty_pool=False,
        **_,
    ) -> bool:
        """Whether some layout serves the demand once the credited cached
        entries are evicted: the take's question, asked of the chain after the
        reclaim (or of an empty pool)."""
        page = self.page_size
        # Expected decode lengths may be fractional; a page holds whole tokens.
        demands = self.page_demands(
            max(0, math.ceil(full_tokens)),
            max(0, math.ceil(swa_tokens)),
            auxiliary_swa_tokens=(
                None
                if auxiliary_swa_tokens is None
                else max(0, math.ceil(auxiliary_swa_tokens))
            ),
            state_slots=int(state_slots),
        )
        owner = self.full_attn_allocator
        reclaim = {owner.sub_pool_name: int(full_evictable_tokens) // page}
        for window in self._windows:
            reclaim[window.sub_pool_name] = (
                int(swa_evictable_tokens)
                if window is self.swa_attn_allocator
                else int(auxiliary_swa_evictable_tokens)
            ) // page
        state = self.mamba_allocator
        if state is not None:
            reclaim[state.sub_pool_name] = int(state_evictable_slots)
        free_ids = (
            owner.ids.num_virtual_ids - owner.min_page_index
            if empty_pool
            else len(owner.free_virtual_ids) + reclaim[owner.sub_pool_name]
        )
        if demands[owner.sub_pool_name] > free_ids:
            return False
        snapshot = reclaimed_snapshot(
            self.pricing_snapshot(), reclaim, empty_pool=empty_pool
        )
        return feasible(snapshot, {n: d for n, d in demands.items() if d > 0})

    def prefill_state_fits(self, full_tokens, swa_tokens, state_slots, **kwargs):
        """Check native state demand without charging it to the FULL-token cap."""
        if self._full_cap_reclaim(full_tokens, empty_pool=True):
            return False
        if self._full_cap_reclaim(full_tokens) > kwargs.get("full_evictable_tokens", 0):
            return False
        # Admission never promises what the take refuses.
        return self._policy.prefill_state_fits(
            self, full_tokens, swa_tokens, state_slots, **kwargs
        ) and self._fits_after_reclaim(
            full_tokens, swa_tokens, state_slots=state_slots, **kwargs
        )

    def can_reserve(self, full_tokens, swa_tokens, **kwargs) -> bool:
        if self._full_cap_reclaim(full_tokens, empty_pool=True):
            return False
        if self._full_cap_reclaim(
            full_tokens, empty_pool=kwargs.get("empty_pool", False)
        ) > kwargs.get("full_evictable_tokens", 0):
            return False
        # Admission never promises what the take refuses.
        return self._policy.can_reserve(
            self, full_tokens, swa_tokens, **kwargs
        ) and self._fits_after_reclaim(full_tokens, swa_tokens, **kwargs)

    def reclaim_plan(self, full_tokens, swa_tokens, **kwargs):
        cap_reclaim = self._full_cap_reclaim(full_tokens)
        if cap_reclaim > kwargs.get("full_evictable_tokens", 0):
            return None
        plan = self._policy.reclaim_plan(self, full_tokens, swa_tokens, **kwargs)
        if plan is None:
            return None
        return max(plan[0], cap_reclaim), plan[1]

    def prealloc_fits_assumes_reclaim(self) -> bool:
        return self._policy.prealloc_fits_assumes_reclaim(self)

    def prealloc_ceiling_fits(self, full_tokens: int, swa_tokens: int) -> bool | None:
        return self._policy.prealloc_ceiling_fits(self, full_tokens, swa_tokens)

    def prealloc_fits(self, tree_cache, full_tokens, swa_tokens, **kwargs) -> bool:
        return self._policy.prealloc_fits(
            self, tree_cache, full_tokens, swa_tokens, **kwargs
        )

    def reclaim_for_prealloc(
        self, tree_cache, full_tokens: int, swa_tokens: int, **kwargs
    ) -> str | None:
        return self._policy.reclaim_for_prealloc(
            self, tree_cache, full_tokens, swa_tokens, **kwargs
        )

    def admission_rules(self) -> ChainAdmissionRules:
        return self._policy.admission_rules(self)

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        from sglang.srt.mem_cache.prefill_budget import ChainPrefillBudget

        return ChainPrefillBudget(
            self,
            tree_cache,
            rules=self.admission_rules(),
            num_mixed_decode_tokens=num_mixed_decode_tokens,
        )

    def usage(self) -> Dict[str, Tuple[int, int]]:
        """Each sub-pool's (capacity, available units), by sub-pool name."""
        return self._policy.usage(self)

    def swa_capacity_and_available(self, *, full_capacity, swa_capacity):
        return self._policy.swa_capacity_and_available(
            self, full_capacity=full_capacity, swa_capacity=swa_capacity
        )

    def max_new_tokens_for_memory(self, input_tokens, max_new_tokens, **kwargs):
        return self._policy.max_new_tokens_for_memory(
            self, input_tokens, max_new_tokens, **kwargs
        )

    def get_extend_allocation_demand(self, prefix_lens_cpu, seq_lens_cpu, **kwargs):
        return self._policy.get_extend_allocation_demand(
            self, prefix_lens_cpu, seq_lens_cpu, **kwargs
        )

    def evict_to_free_tokens(self, tree_cache, num_tokens: int, **kwargs):
        return self._policy.evict_to_free_tokens(self, tree_cache, num_tokens, **kwargs)

    def check_decode_capacity(
        self, *, num_tokens: int, tree_cache, requests=None, spec_algorithm=None
    ) -> bool:
        """Whether the next decode step's allocation fits after eviction: the
        same question the decode take answers."""
        self.evict_to_free_tokens(tree_cache, num_tokens)
        return self.ensure_capacity(num_tokens, num_tokens)

    def resize(self, config) -> None:
        if (
            not self.arena.backed_after_capture
            or config.unified_memory_pool_bytes != self.arena.total_bytes
        ):
            raise RuntimeError("the arena must be backed before allocator.resize")
        for band in self.members:
            if band.allocated_count() or band._pending_reuse:
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
        self._policy = interim_policy_for(
            self,
            full_max_total_num_tokens=(
                config.full_max_total_num_tokens
                if self._window_capacity_from_sizing
                else None
            ),
            swa_max_total_num_tokens=(
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

    def _flush_targets(self):
        return self._policy.flush_targets(self)

    def _ask_float_for_room(self, need_tokens: int) -> None:
        self._policy.ask_float_for_room(self, need_tokens)

    # -- state slots --

    def mamba_full_cache_donor(self):
        return self._policy.mamba_full_cache_donor(self)

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

    def full_tokens_before_mamba_recheck(self, target_size: int) -> int:
        return full_tokens_before_mamba_recheck(
            self.full_attn_allocator, self.mamba_allocator, target_size
        )

    def prepare_mamba_allocation(self, target_size: int) -> None:
        self._policy.prepare_mamba_allocation(self, target_size)

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

    def add_pin_source(
        self, sub_pools: Sequence[str], busy: Callable[[], bool]
    ) -> None:
        """Keep the named sub-pools' pages in place whenever ``busy()`` holds:
        a DMA initiator whose queues outlive an iteration, registered once. A
        transfer that reaches every sub-pool names every one: a band it leaves
        out would relocate its own pages under the same in-flight copy."""
        assert self.lazy_compaction, (
            "Pin sources (PD disaggregation, HiCache) with the unified memory "
            "pool require lazy compaction (eager free-path compaction moves "
            "pages under in-flight transfers)."
        )
        for name in sub_pools:
            self.bands[self._band_positions[name]].add_pin_source(busy)

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

    def set_hicache_transfer_done_event(self, transfer_key: Hashable, event) -> None:
        # Every member a pin covers must also order its moves after the copy.
        for member in self.members:
            member.set_hicache_transfer_done_event(transfer_key, event)

    def prewarm_compaction(self) -> None:
        """Prewarm the page-move path of every member that compacts."""
        for member in self.members:
            member.prewarm_compaction()

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
    ) -> None:
        """Hand the forward's metadata to every member; token bands translate the
        write set through their own v2p. The state band, written in place by
        state kernels rather than through `out_cache_loc`, takes the virtual
        state slots the forward writes. ``token_writes_unknown``: the token
        bands cannot read this forward's write ids before it ends."""
        with record_function("ChainAllocator.set_inflight_forward"):
            for member in self.members:
                is_state = member is self.mamba_allocator
                member.set_inflight_forward(
                    forward_done,
                    state_slots_virtual if is_state else out_cache_loc_virtual,
                    writes_unknown=token_writes_unknown and not is_state,
                )

    def retire_forward(self, forward_done: torch.cuda.Event) -> None:
        """The scheduler processed this forward's result; see `Band.retire_forward`."""
        for member in self.members:
            member.retire_forward(forward_done)

    def flush_opportunistic(self) -> int:
        """Non-urgent flush of every member, ends before floats; sync-free."""
        with record_function("ChainAllocator.flush_opportunistic"):
            if all(
                m._free_phys_pages.numel() == 0 and not m._pending_reuse
                for m in self.members
            ):
                return 0
            ends_first = sorted(self.members, key=lambda m: m.grow_direction == "float")
            return sum(m.flush_opportunistic() for m in ends_first)

    def verify_byte_accounting(self) -> List[str]:
        return (
            chain_byte_accounting_violations(list(self.bands))
            + self._policy.memo_violations(self)
            + self._snapshot_memo_violations()
        )

    # -- alloc --

    def _allocate(
        self,
        *,
        full_tokens: int,
        swa_tokens: int,
        num_owner_pages: int,
        owner_take: Callable[[Optional[torch.Tensor]], Optional[torch.Tensor]],
        window_pages: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = lambda out, new_pages: new_pages,
    ) -> Optional[torch.Tensor]:
        """Take the owner's new pages and every window's pages in one planned
        layout, then bind the windows to the owner's new virtual pages. None
        exactly when no reachable layout serves the demand."""
        owner = self.full_attn_allocator
        if self._full_cap_reclaim(full_tokens) or num_owner_pages > len(
            owner.free_virtual_ids
        ):
            return None
        demands = {owner.sub_pool_name: num_owner_pages}
        for window in self._windows:
            demands[window.sub_pool_name] = (
                -(-swa_tokens // self.page_size)
                if window is self.swa_attn_allocator
                else num_owner_pages
            )
        physical = self._plan_and_take(demands)
        if physical is None:
            return None
        # free_virtual_ids is rebound (slice/cat), never mutated in place, so
        # this prefix view still names the pages the owner is about to consume.
        new_virtual_pages = owner.free_virtual_ids[:num_owner_pages]
        out = owner_take(physical.get(owner.sub_pool_name))
        if out is None:
            raise RuntimeError(
                "ChainAllocator: the owner refused pages its planned layout "
                "reserved -- internal-state inconsistency"
            )
        for window in self._windows:
            pages = physical.get(window.sub_pool_name)
            if pages is None:
                continue
            bind = (
                window_pages(out, new_virtual_pages)
                if window is self.swa_attn_allocator
                else new_virtual_pages
            )
            assert bind.numel() == pages.numel()
            window.ids.bind(bind, pages)
            window._maybe_zero_pages(pages)
        return out

    def _plan_and_take(
        self, demands: Dict[str, int]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Physical pages for ``demands`` (pages per band), laid out by the
        chain's own procedure: in place, then with the ends compacted, then with
        the floats relocated. None exactly when no reachable layout serves them
        (`feasible` says no); a planned layout that does not realize raises."""
        demands = {name: n for name, n in demands.items() if n > 0}
        if not demands:
            return {}
        plan = first_plan(self.pricing_snapshot(), demands)
        if plan is None:
            return None
        return self.realize(plan)

    def realize(self, plan: TakePlan) -> Dict[str, torch.Tensor]:
        """Lay the chain out as ``plan``, made from its current snapshot, says
        and take the planned pages: the listed ends compact, the floats move
        into their target blocks in an order where none is copied into a block
        another still holds, then every band takes its pages in chain order.
        Returns each demanded band's physical pages. A plan of the current
        snapshot always realizes; a refusal is an internal error."""
        self._lay_out(plan)
        taken = {}
        for band in self.bands:
            take = plan.takes.get(band.sub_pool_name)
            if take is None:
                continue
            extended = 0 if take.pages is None else take.pages[1] - take.pages[0]
            pages = take.reuse + extended
            if not pages:
                continue
            if band.grow_direction == "float":
                physical = band.take_physical_pages(pages, placement=take.pages)
            else:
                physical = band.take_physical_pages(pages)
            if physical is None:
                raise RuntimeError(
                    f"{band.sub_pool_name}: the planned take of {pages} pages "
                    "was refused"
                )
            taken[band.sub_pool_name] = physical
        return taken

    def lay_out_for(self, demands: Dict[str, int]) -> bool:
        """Make the least rearrangement that serves ``demands`` (pages per
        band), so that they then fit without moving a page; False exactly when
        no reachable layout serves them (`feasible` says no). For takes issued
        later and in any order, such as a host load's."""
        demands = {name: n for name, n in demands.items() if n > 0}
        plan = first_plan(self.pricing_snapshot(), demands)
        if plan is None:
            return False
        self._lay_out(plan)
        return True

    def _lay_out(self, plan: TakePlan) -> None:
        """The plan's rearrangement: compact the listed ends, then move the
        floats into their target blocks."""
        bands = {band.sub_pool_name: band for band in self.bands}
        for name in plan.flush:
            band = bands[name]
            band.flush_for_allocation()
            low_byte = band._byte_low_frontier()
            high_byte = band._byte_high_frontier()
            live = band.live_page_count if band.lazy_compaction else None
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
    def scoped_physical_takes(
        self, full_tokens, swa_tokens, *, auxiliary_swa_tokens=None, state_slots=0
    ):
        """Consume a pre-realized reservation across independently issued H2D takes."""
        if len(self._windows) <= 1:
            yield
            return
        if any(band._physical_take_scope is not None for band in self.members):
            raise RuntimeError("physical reservation scopes cannot be nested")
        scope = _ScopedPhysicalTakes(
            self,
            self._policy._demands(
                self, full_tokens, swa_tokens, auxiliary_swa_tokens, state_slots
            ),
        )
        for band in self.members:
            band._physical_take_scope = scope
        try:
            yield
        finally:
            for band in self.members:
                band._physical_take_scope = None

    def _take_owner_tokens(self, need_size, physical):
        owner = self.full_attn_allocator
        if physical is None:
            return owner.alloc(need_size)
        return take_new_pages(
            owner,
            need_size // self.page_size,
            lambda pages: owner._expand_pages_to_tokens(
                pages[: need_size // self.page_size]
            ),
            copy_new_pages=False,
            physical_pages=physical,
        )

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with record_function("ChainAllocator.alloc"):
            return self._allocate(
                full_tokens=need_size,
                swa_tokens=need_size,
                num_owner_pages=need_size // self.page_size,
                owner_take=lambda physical: self._take_owner_tokens(
                    need_size, physical
                ),
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
                    next_power_of_2(bs),
                    self.full_attn_allocator.page_size,
                )
            return out_indices

        return lambda physical=None: take_new_pages(
            self.full_attn_allocator,
            num_new_pages,
            extend_tokens,
            copy_new_pages=False,
            physical_pages=physical,
        )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ) -> Optional[torch.Tensor]:
        """Paged extend; returns virtual TOKEN ids. A window binds exactly the
        virtual pages the owner consumed."""
        with record_function("ChainAllocator.alloc_extend"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
            need_tokens = num_new_pages * self.page_size
            return self._allocate(
                full_tokens=need_tokens,
                swa_tokens=need_tokens,
                num_owner_pages=num_new_pages,
                owner_take=self._paged_take(
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    num_new_pages,
                ),
            )

    def alloc_extend_swa_tail(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        swa_tail_len: int,
    ) -> Optional[torch.Tensor]:
        """Decode-node prealloc: owner KV for the whole sequence, window KV for
        the live window tail only.

        The owner and the window SHARE one virtual id space (a virtual page names
        an owner-physical page and, if bound, a window-physical one), so the tail
        is expressed by binding the window for the TAIL's new virtual pages only.
        A new page left unbound has no window-physical page, which reads as the
        sink and is skipped by `free`'s `> 0` mask -- exactly the out-of-window
        state window retirement produces. Admission prices the owner's new pages
        and only the new pages in the window tail; a partial prefix page is
        already bound and costs no new window page.
        """
        assert len(prefix_lens_cpu) == 1
        assert 0 <= swa_tail_len <= extend_num_tokens
        with record_function("ChainAllocator.alloc_extend_swa_tail"):
            prefix_len = int(prefix_lens_cpu[0])
            seq_len = int(seq_lens_cpu[0])
            first_new_page = (prefix_len + self.page_size - 1) // self.page_size
            first_tail_page = (seq_len - swa_tail_len) // self.page_size
            num_swa_pages = (
                (seq_len + self.page_size - 1) // self.page_size
                - max(first_new_page, first_tail_page)
                if swa_tail_len
                else 0
            )
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )

            def tail_pages(out: torch.Tensor, new_pages: torch.Tensor) -> torch.Tensor:
                # Only NEW pages need binding; a tail page carried in from the
                # prefix is already bound on the window. The extend kernel hands
                # the new pages out in token order, so the tail's new pages are
                # the last `num_swa_pages`: a slice, not a data-dependent filter
                # that would block the host on the device.
                return new_pages[new_pages.numel() - num_swa_pages :]

            return self._allocate(
                full_tokens=num_new_pages * self.page_size,
                swa_tokens=num_swa_pages * self.page_size,
                num_owner_pages=num_new_pages,
                owner_take=self._paged_take(
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    num_new_pages,
                ),
                window_pages=tail_pages,
            )

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
                        next_power_of_2(bs),
                        self.full_attn_allocator.page_size,
                    )
                return out_indices

            need_tokens = num_new_pages * self.page_size
            return self._allocate(
                full_tokens=need_tokens,
                swa_tokens=need_tokens,
                num_owner_pages=num_new_pages,
                owner_take=lambda physical: take_new_pages(
                    self.full_attn_allocator,
                    num_new_pages,
                    decode_tokens,
                    copy_new_pages=True,
                    physical_pages=physical,
                ),
            )

    def alloc_state_slots(self, need_size: int) -> Optional[torch.Tensor]:
        """Per-request state slots (VIRTUAL slot ids), laid out by the same
        plan as token pages."""
        state = self.mamba_allocator
        if need_size <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if need_size > len(state.free_virtual_ids):
            return None
        physical = self._plan_and_take({state.sub_pool_name: need_size})
        if physical is None:
            return None
        slots = state.free_virtual_ids[:need_size]
        state.free_virtual_ids = state.free_virtual_ids[need_size:]
        pages = physical[state.sub_pool_name]
        state.ids.bind(slots, pages)
        state._maybe_zero_pages(pages)
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
        v_pages = reps // self.page_size
        # Signal-only invariant: avoid launching its lookup and comparison
        # when detection is disabled. This invariant has no recovery action.
        if resolve_level() >= InvariantCheckLevel.WARN:
            expect(
                UNIFIED_SWA_SEGMENT_BOUND,
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
            for window in self._windows:
                expect(
                    UNIFIED_SWA_SEGMENT_RELEASED,
                    window.ids.virtual_to_physical[reps // self.page_size] <= 0,
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
        self.swa_attn_allocator.ids.bind(full_pages, swa_pages)

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
        self.full_attn_allocator.free(reps, _pages=reps // self.page_size)
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
