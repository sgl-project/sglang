"""One allocator over the chain of bands in a `KVArena`.

Every sub-pool is a band allocator, an end or a float of the chain. The owner
band mints the virtual token ids; a window band binds the owner's virtual
pages; a state band hands out per-request slots. Every lifecycle call (frees,
groups, events, pins, flushes, verification) is one loop over the bands, and
every capacity question is one layout search over them, whatever the chain's
shape.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
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
)
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
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
from sglang.srt.mem_cache.kv.memory.arena import KVArena
from sglang.srt.mem_cache.kv.plan import window_tree_component
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import get_num_new_pages, next_power_of_2
from sglang.srt.utils.invariants import expect, resolve_level

# Rows an unreleased network transfer touches, fixed for one relief.
_NETWORK_ROWS = "network-rows"

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult

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
        self.demands = {name: n for name, n in demands.items() if n > 0}
        # Admission and reclaim answered with every rearrangement the chain can
        # make; make the one this reservation needs before any take is issued,
        # so the takes then fit in place, in whatever order they come.
        if plan_layout(
            allocator.pricing_snapshot(), self.demands, NO_MOVE
        ) is None and not allocator.lay_out_for(self.demands):
            raise RuntimeError(
                "physical reservation needs reclamation before its scope"
            )
        snapshot = allocator.pricing_snapshot()
        if plan_layout(snapshot, self.demands, NO_MOVE) is None:
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
        if pages > remaining[band.sub_pool_name] or any(
            n < 0 for n in remaining.values()
        ):
            return None
        # Planned again from the current layout: the takes so far consumed
        # prefixes of earlier plans, so the rest still fits in place.
        layout = plan_layout(
            snapshot, {name: n for name, n in remaining.items() if n > 0}, NO_MOVE
        )
        if layout is None:
            return None
        self.applying = True
        try:
            return band.take_planned_pages(
                layout.takes[band.sub_pool_name].prefix(pages)
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
) -> Optional[torch.Tensor]:
    """Fill tail pages before taking new virtual pages for each request.

    Bind every new page in the owner band before translating its token ids;
    otherwise its v2p entry remains -1.
    """
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

    # Base init writes `self.size`; the chain computes its capacity.
    @property
    def size(self) -> int:
        """Token capacity. Without a window, the tokens held plus the same
        availability `available_size` reads, so the leak census's available
        term cancels; with one, the sizing's (or the empty pool's) per-side
        token shares."""
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
        for band in self.members:
            band.chain_allocator = self
        by_name = {band.sub_pool_name: band for band in self.members}
        self.bands = tuple(by_name[band.name] for band in arena.bands)
        _wire_chain(self.bands)
        self._band_positions = {b.sub_pool_name: i for i, b in enumerate(self.bands)}
        self._snapshot_key: Optional[int] = None
        self._snapshot_memo: Optional[ChainSnapshot] = None
        # Largest-fit reads, each kept until the capacity epoch moves.
        self._fit_memo: Dict[Hashable, Tuple[tuple, int]] = {}
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
        """Each side's nominal token share: the sizing's physical rows per rank
        widened to the logical tokens the bands allocate, else what the empty
        pool offers each side. Without a window, the owner's whole id space."""
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

    # -- capacity reads --
    #
    # Every read answers from the same layout search as the take, on the
    # current snapshot. A sub-pool's availability is the largest demand of
    # that sub-pool ALONE (every other sub-pool demanding nothing) that a take
    # can serve now: tokens for a token sub-pool, slots for the state sub-pool.
    # Sub-pools share bytes, so the availabilities of several sub-pools do not
    # add up: a per-sub-pool shortfall (demand minus availability) that is zero
    # everywhere does not prove a joint demand fits; ask `ensure_capacity`.

    def _units_fit(self, demands: Dict[str, int]) -> bool:
        """Whether a take of these units (pages of a token sub-pool, slots of
        the state sub-pool) succeeds now: the FULL cap, the virtual ids each
        id owner has left, then the layout search the take itself runs."""
        owner = self.full_attn_allocator
        owner_pages = demands.get(owner.sub_pool_name, 0)
        if owner_pages > len(owner.free_virtual_ids):
            return False
        if self._full_cap_reclaim(owner_pages * self.page_size):
            return False
        # A window binds the owner's virtual pages: it can never hold more.
        bound = owner.ids.num_virtual_ids - owner.min_page_index
        if any(demands.get(w.sub_pool_name, 0) > bound for w in self._windows):
            return False
        state = self.mamba_allocator
        if state is not None and demands.get(state.sub_pool_name, 0) > len(
            state.free_virtual_ids
        ):
            return False
        return feasible(
            self.pricing_snapshot(), {n: d for n, d in demands.items() if d > 0}
        )

    def _largest_fit(self, query: Hashable, names: Tuple[str, ...], hi: int) -> int:
        """The most units that fit when each of ``names`` takes that many.
        A fit is downward closed (a smaller demand of the same shape fits
        whenever a larger one does), so a binary search finds the boundary."""
        key = (self._snapshot_key_now(), self.full_attn_allocator.max_live_tokens, hi)
        memo = self._fit_memo.get(query)
        if memo is not None and memo[0] == key:
            return memo[1]
        low, high = 0, max(0, hi)
        while low < high:
            mid = (low + high + 1) // 2
            if self._units_fit(dict.fromkeys(names, mid)):
                low = mid
            else:
                high = mid - 1
        self._fit_memo[query] = (key, low)
        return low

    def _id_bound(self, band: Band) -> int:
        """Units a sub-pool's demand can never exceed: its id owner's room."""
        if band is self.mamba_allocator:
            return len(band.free_virtual_ids)
        owner = self.full_attn_allocator
        if band is owner:
            return len(owner.free_virtual_ids)
        return owner.ids.num_virtual_ids - owner.min_page_index

    def units_available(self, sub_pool: str) -> int:
        """The sub-pool's availability (see the contract above): a take of at
        most this many units of it alone succeeds now, and one more does not."""
        band = self.bands[self._band_positions[sub_pool]]
        pages = self._largest_fit(sub_pool, (sub_pool,), self._id_bound(band))
        return pages if band is self.mamba_allocator else pages * self.page_size

    def available_size(self) -> int:
        """Tokens one more token allocation can take now: the owner and every
        window take each new token together."""
        names = tuple(
            b.sub_pool_name for b in (self.full_attn_allocator, *self._windows)
        )
        hi = self._id_bound(self.full_attn_allocator)
        return self._largest_fit("tokens", names, hi) * self.page_size

    def full_available_size(self) -> int:
        return self.units_available(self.full_attn_allocator.sub_pool_name)

    def swa_available_size(self) -> int:
        if self.swa_attn_allocator is None:
            raise AttributeError("a chain without a window has no window tokens")
        return self.units_available(self.swa_attn_allocator.sub_pool_name)

    def state_slots_available(self) -> int:
        return self.units_available(self.mamba_allocator.sub_pool_name)

    def conserve_full_available_size(self) -> int:
        """The sizing's nominal owner share minus the owner tokens held: the
        static split, not what a take can serve."""
        return self._size_full - self.full_attn_allocator.allocated_count()

    def conserve_swa_available_size(self) -> int:
        """The sizing's nominal window share minus the window tokens held."""
        return self._size_swa - self.swa_attn_allocator.allocated_count()

    def schedulable_full_available_size(self) -> int:
        return self.full_available_size()

    def schedulable_swa_available_size(self) -> int:
        return self.swa_available_size()

    @property
    def current_full_capacity(self) -> int:
        """Owner tokens held now plus what the owner alone could still take."""
        return self.full_available_size() + self.full_attn_allocator.allocated_count()

    @property
    def current_swa_capacity(self) -> int:
        return self.swa_available_size() + self.swa_attn_allocator.allocated_count()

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

    def window_component(self, window: Band) -> ComponentType:
        """The prefix tree slot a window sub-pool goes by: the target's window
        is the tree's first window slot; every further window is named by its
        sub-pool."""
        if window is self.swa_attn_allocator:
            return ComponentType.SWA
        return ComponentType.of(window_tree_component(window.sub_pool_name))

    def _window_demand(self, full_tokens, swa_tokens, window_tokens=None):
        """A token demand per window sub-pool: ``window_tokens`` names any
        window's tokens; otherwise the target's window takes ``swa_tokens``
        and every further window the owner's."""
        window_tokens = window_tokens or {}
        return {
            window.sub_pool_name: window_tokens.get(
                window.sub_pool_name,
                swa_tokens if window is self.swa_attn_allocator else full_tokens,
            )
            for window in self._windows
        }

    def page_demands(
        self,
        full_tokens: int,
        swa_tokens: int,
        *,
        window_tokens: Optional[Mapping[str, int]] = None,
        state_slots: int = 0,
    ) -> Dict[str, int]:
        """Pages per band for a token demand: the owner's tokens, each window's
        (`_window_demand`), and state slots."""
        page = self.page_size
        demands = {self.full_attn_allocator.sub_pool_name: -(-int(full_tokens) // page)}
        for name, tokens in self._window_demand(
            full_tokens, swa_tokens, window_tokens
        ).items():
            demands[name] = -(-int(tokens) // page)
        if self.mamba_allocator is not None:
            demands[self.mamba_allocator.sub_pool_name] = int(state_slots)
        return demands

    def ensure_capacity(
        self,
        full_tokens: int,
        swa_tokens: int,
        *,
        window_tokens: Optional[Mapping[str, int]] = None,
        state_slots: int = 0,
    ) -> bool:
        """Whether an allocation of this demand succeeds now: the question the
        take itself answers, so a yes here is a successful take."""
        if full_tokens < 0 or swa_tokens < 0:
            return False
        return self._units_fit(
            self.page_demands(
                full_tokens,
                swa_tokens,
                window_tokens=window_tokens,
                state_slots=state_slots,
            )
        )

    def admission_windows(
        self, tree_cache
    ) -> Tuple[Tuple[str, ComponentType, int, int], ...]:
        """Each window sub-pool's name, its prefix-tree window slot, the
        history its cache retains (what a request is charged at most) and its
        evictable tokens."""
        windows = []
        for window in self._windows:
            component = self.window_component(window)
            windows.append(
                (
                    window.sub_pool_name,
                    component,
                    tree_cache.window_retain_size(component),
                    (
                        tree_cache.swa_evictable_size()
                        if component == ComponentType.SWA
                        else tree_cache.window_evictable_size(component)
                    ),
                )
            )
        return tuple(windows)

    def admission_evictable(self, tree_cache) -> Dict[str, int]:
        """What the prefix tree could free per sub-pool: tokens, or state
        slots for the state sub-pool."""
        evictable = {
            self.full_attn_allocator.sub_pool_name: tree_cache.full_evictable_size()
        }
        for name, _, _, tokens in self.admission_windows(tree_cache):
            evictable[name] = tokens
        if self.mamba_allocator is not None:
            evictable[self.mamba_allocator.sub_pool_name] = (
                tree_cache.mamba_evictable_size() if tree_cache.supports_mamba() else 0
            )
        return evictable

    def admits(
        self,
        full_tokens: float,
        window_tokens: Dict[str, float],
        state_slots: int = 0,
        *,
        evictable: Optional[Dict[str, int]] = None,
        empty_pool: bool = False,
    ) -> bool:
        """Whether a demand fits once ``evictable`` cached entries are evicted
        (or in an empty pool): FULL tokens against the FULL cap and the owner's
        ids, each window sub-pool's tokens (a window missing from
        ``window_tokens`` is charged nothing), state slots in full, then one
        layout search over the shared bytes. Every admission question asks
        this, so an admitted demand is one the take serves."""
        if full_tokens < 0 or any(t < 0 for t in window_tokens.values()):
            return False
        evictable = {} if empty_pool or evictable is None else evictable
        owner = self.full_attn_allocator
        # Expected decode lengths may be fractional; a page holds whole tokens.
        full_tokens = math.ceil(full_tokens)
        if self._full_cap_reclaim(full_tokens, empty_pool=True):
            return False
        if self._full_cap_reclaim(full_tokens, empty_pool=empty_pool) > evictable.get(
            owner.sub_pool_name, 0
        ):
            return False
        page = self.page_size
        demands = {owner.sub_pool_name: -(-full_tokens // page)}
        reclaim = {owner.sub_pool_name: evictable.get(owner.sub_pool_name, 0) // page}
        owner_pages = owner.ids.num_virtual_ids - owner.min_page_index
        for window in self._windows:
            name = window.sub_pool_name
            demands[name] = -(-math.ceil(window_tokens.get(name, 0)) // page)
            # A window binds the owner's virtual pages: it can never hold more.
            if demands[name] > owner_pages:
                return False
            reclaim[name] = evictable.get(name, 0) // page
        free_ids = (
            owner_pages
            if empty_pool
            else len(owner.free_virtual_ids) + reclaim[owner.sub_pool_name]
        )
        if demands[owner.sub_pool_name] > free_ids:
            return False
        state = self.mamba_allocator
        if state is not None:
            demands[state.sub_pool_name] = int(state_slots)
            reclaim[state.sub_pool_name] = int(evictable.get(state.sub_pool_name, 0))
            free_slots = (
                state.ids.num_virtual_ids - state.min_page_index
                if empty_pool
                else len(state.free_virtual_ids) + reclaim[state.sub_pool_name]
            )
            if state_slots > free_slots:
                return False
        snapshot = reclaimed_snapshot(
            self.pricing_snapshot(), reclaim, empty_pool=empty_pool
        )
        return feasible(snapshot, {n: d for n, d in demands.items() if d > 0})

    def prealloc_fits_assumes_reclaim(self) -> bool:
        """Whether decode-node preallocation reclaims the evictable entries its
        admission credits before it takes: chains with a window do."""
        return bool(self._windows)

    def prealloc_ceiling_fits(self, full_tokens: int, swa_tokens: int) -> bool | None:
        return self.admits(
            full_tokens,
            self._window_demand(full_tokens, swa_tokens),
            empty_pool=True,
        )

    def prealloc_fits(
        self,
        tree_cache,
        full_tokens,
        swa_tokens,
        *,
        full_budget_tokens: int,
        swa_budget_tokens: int | None = None,
        state_slots: int = 0,
    ) -> bool:
        """A decode-node preallocation on the admission rule: its tokens and
        the ``state_slots`` it takes with them, priced together since they
        share bytes. The budgets are the scheduler's per-side token views
        after decode headroom and retraction: the request stays within them,
        and what they hold back from this side's reachable tokens is pending
        demand, charged alongside the request."""
        if full_tokens > full_budget_tokens or (
            swa_budget_tokens is not None and swa_tokens > swa_budget_tokens
        ):
            return False
        evictable = self.admission_evictable(tree_cache)
        if not self.prealloc_fits_assumes_reclaim():
            # Nothing is reclaimed before the take: credit no eviction.
            evictable = dict.fromkeys(evictable, 0)
        owner = self.full_attn_allocator.sub_pool_name
        full_tokens += max(
            0, self.full_available_size() + evictable[owner] - full_budget_tokens
        )
        windows = self._window_demand(full_tokens, swa_tokens)
        if swa_budget_tokens is not None and self.swa_attn_allocator is not None:
            name = self.swa_attn_allocator.sub_pool_name
            windows[name] += max(
                0, self.swa_available_size() + evictable[name] - swa_budget_tokens
            )
        return self.admits(full_tokens, windows, state_slots, evictable=evictable)

    def reclaim_for_load_back(
        self, tree_cache, full_tokens: int, swa_tokens: int, **demand
    ) -> str | None:
        """Reclaim before a host load takes its rows; None means ready."""
        return self.reclaim_for_prealloc(tree_cache, full_tokens, swa_tokens, **demand)

    def reclaim_for_prealloc(
        self,
        tree_cache,
        full_tokens: int,
        swa_tokens: int,
        *,
        window_tokens: Optional[Mapping[str, int]] = None,
        state_slots: int = 0,
    ) -> str | None:
        """Free room for a preallocation; None means it is ready, else a
        description of the shortfall for the caller to attach to its request."""
        demand = dict(window_tokens=window_tokens, state_slots=state_slots)
        self.evict_to_free_tokens(
            tree_cache, full_tokens, swa_num_tokens=swa_tokens, **demand
        )
        if self.ensure_capacity(full_tokens, swa_tokens, **demand):
            return None
        return (
            "Unified memory reclamation insufficient: "
            f"needed=(full={full_tokens}, swa={swa_tokens}, "
            f"windows={dict(window_tokens or {})}, state_slots={state_slots})"
        )

    def _reclaimable_units(self, tree_cache) -> Dict[str, int]:
        """What the prefix tree could free per sub-pool, in pages (slots for
        the state sub-pool)."""
        units = {}
        state = self.mamba_allocator
        for name, n in self.admission_evictable(tree_cache).items():
            units[name] = (
                n
                if state is not None and name == state.sub_pool_name
                else n // self.page_size
            )
        return units

    def reclaim(self, tree_cache, demands: Dict[str, int]) -> bool:
        """Evict cached entries until ``demands`` (pages per sub-pool, slots
        for state) can be taken (`_reclaim`); return whether they can."""
        self._reclaim(tree_cache, demands)
        return self._units_fit({n: d for n, d in demands.items() if d > 0})

    def reclaim_for_demand(self, tree_cache, demand: EvictParams) -> EvictResult:
        """An allocation's demand as the prefix tree states it, reclaimed as
        one demand (`_reclaim`): the sub-pools share bytes."""
        page = self.page_size
        owner = self.full_attn_allocator
        swa = self.swa_attn_allocator
        state = self.mamba_allocator
        demands = {owner.sub_pool_name: -(-demand.num_tokens // page)}
        for window in self._windows:
            tokens = (
                demand.swa_num_tokens
                if window is swa
                else demand.window_num_tokens.get(self.window_component(window), 0)
            )
            demands[window.sub_pool_name] = -(-tokens // page)
        if state is not None:
            demands[state.sub_pool_name] = demand.mamba_num
        return self._reclaim(tree_cache, demands)

    def _reclaim(self, tree_cache, demands: Dict[str, int]) -> EvictResult:
        """Evict cached entries until ``demands`` can be taken, and report what
        was evicted.

        A demand that fits now, rearranged if need be, evicts nothing; nor does
        one that would not fit with every evictable entry gone, since eviction
        would only lose cache: the caller refuses it or retracts. Otherwise
        victims go one at a time in the cache's order, and every walk stops at
        the first victim after which the demand fits (`_fits_after_frees`), so
        the order is a heuristic only: first the entries of each sub-pool whose
        own demand does not fit now, whose freed pages it then takes in place;
        then token entries by LRU, whose eviction frees their whole node; then
        the state checkpoints left; then window entries. Admission credits the
        same evictable entries (`admits`), so an admitted demand is one this
        reclaim makes room for."""
        from sglang.srt.mem_cache.base_prefix_cache import EvictResult

        result = EvictResult()
        demands = {n: d for n, d in demands.items() if d > 0}
        if self._units_fit(demands):
            return result
        if not self._admits_pages(demands, self._reclaimable_units(tree_cache)):
            return result
        state = self.mamba_allocator
        order = [
            {n for n, d in demands.items() if not self._units_fit({n: d})},
            {self.full_attn_allocator.sub_pool_name},
            set() if state is None else {state.sub_pool_name},
            {w.sub_pool_name for w in self._windows},
        ]
        for names in order:
            if self._fits_after_frees(demands):
                break
            reclaimable = self._reclaimable_units(tree_cache)
            self._evict(
                tree_cache, {n: reclaimable.get(n, 0) for n in names}, demands, result
            )
        return result

    def _admits_pages(
        self, demands: Dict[str, int], reclaimable: Dict[str, int]
    ) -> bool:
        """`admits` for a demand already in pages (slots for state)."""
        owner = self.full_attn_allocator
        page = self.page_size
        state = self.mamba_allocator
        windows = {
            w.sub_pool_name: demands.get(w.sub_pool_name, 0) * page
            for w in self._windows
        }
        evictable = {
            name: n if state is not None and name == state.sub_pool_name else n * page
            for name, n in reclaimable.items()
        }
        return self.admits(
            demands.get(owner.sub_pool_name, 0) * page,
            windows,
            0 if state is None else demands.get(state.sub_pool_name, 0),
            evictable=evictable,
        )

    def _evict(
        self, tree_cache, units: Dict[str, int], demands: Dict[str, int], result
    ) -> None:
        """Ask the prefix tree for ``units`` (pages, slots for state) per
        sub-pool, the walks stopping once ``demands`` can be taken; add what it
        evicted to ``result``."""
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        page = self.page_size
        swa = self.swa_attn_allocator
        state = self.mamba_allocator
        params = EvictParams(
            num_tokens=units.get(self.full_attn_allocator.sub_pool_name, 0) * page,
            swa_num_tokens=0 if swa is None else units.get(swa.sub_pool_name, 0) * page,
            window_num_tokens={
                self.window_component(w): units[w.sub_pool_name] * page
                for w in self._windows
                if w is not swa and units.get(w.sub_pool_name, 0)
            },
            mamba_num=0 if state is None else units.get(state.sub_pool_name, 0),
        )
        if (
            params.num_tokens
            or params.swa_num_tokens
            or params.window_num_tokens
            or params.mamba_num
        ):
            step = tree_cache.evict_until(
                params, satisfied=lambda: self._fits_after_frees(demands)
            )
            result.num_tokens_evicted += step.num_tokens_evicted
            result.swa_num_tokens_evicted += step.swa_num_tokens_evicted
            for component, tokens in step.window_num_tokens_evicted.items():
                result.window_num_tokens_evicted[component] = (
                    result.window_num_tokens_evicted.get(component, 0) + tokens
                )
            result.mamba_num_evicted += step.mamba_num_evicted

    def _fits_after_frees(self, demands: Dict[str, int]) -> bool:
        """The reclaim walks' stop: the frees a walk queued in the caller's
        free group are applied first, or the walk would not see the room it
        made and would evict on."""
        self.flush_deferred_full_frees()
        return self._units_fit(demands)

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        from sglang.srt.mem_cache.prefill_budget import ChainPrefillBudget

        return ChainPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    def usage(self) -> Dict[str, Tuple[int, int]]:
        """Each sub-pool's (capacity, available units), by sub-pool name: the
        units it holds plus its availability, and its availability."""
        owner = self.full_attn_allocator
        usage = {
            owner.sub_pool_name: (
                self.current_full_capacity,
                self.full_available_size(),
            )
        }
        for window in self._windows:
            free = self.units_available(window.sub_pool_name)
            usage[window.sub_pool_name] = (window.allocated_count() + free, free)
        if self.mamba_allocator is not None:
            usage[self.mamba_allocator.sub_pool_name] = (
                self.state.pool.size,
                self.state_slots_available(),
            )
        return usage

    def swa_capacity_and_available(self, *, full_capacity, swa_capacity):
        return (
            (self.current_full_capacity, self.full_available_size()),
            (self.current_swa_capacity, self.swa_available_size()),
        )

    def max_new_tokens_for_memory(
        self,
        input_tokens: int,
        max_new_tokens: int,
        *,
        token_capacity: int,
        sliding_window_size: int | None,
        chunk_size: int | None,
    ) -> int | None:
        """The most new tokens a lone request of ``input_tokens`` can be
        admitted with in an empty pool, at most ``max_new_tokens``; None when
        even the prompt alone cannot. ``token_capacity`` is already the FULL
        cap this pool applies."""
        from sglang.srt.mem_cache.prefill_budget import estimate_swa_kv_tokens

        windows = {
            window.sub_pool_name: (
                sliding_window_size
                if window is self.swa_attn_allocator or self.storage is None
                else self.storage.plan.sub_pool(window.sub_pool_name).lifetime.window
            )
            for window in self._windows
        }
        state_slots = 0 if self.mamba_allocator is None else 1

        def fits(new_tokens):
            return self.admits(
                input_tokens + new_tokens + self.page_size,
                {
                    name: estimate_swa_kv_tokens(
                        input_tokens,
                        new_tokens,
                        sliding_window_size=window,
                        page_size=self.page_size,
                        allocation_limit=chunk_size,
                    )
                    for name, window in windows.items()
                },
                state_slots,
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

    def evict_to_free_tokens(
        self,
        tree_cache,
        num_tokens: int,
        *,
        swa_num_tokens: Optional[int] = None,
        window_tokens: Optional[Mapping[str, int]] = None,
        state_slots: int = 0,
    ) -> bool | None:
        """Evict cached entries until this token demand (every window taking
        ``swa_num_tokens``, else the same tokens) can be taken, the way static
        pools would (`reclaim`). Return whether it can, or None without
        a prefix cache."""
        if tree_cache is None or not tree_cache.supports_prefix_sharing():
            return None
        self.flush_deferred_full_frees()
        return self.reclaim(
            tree_cache,
            self.page_demands(
                num_tokens,
                num_tokens if swa_num_tokens is None else swa_num_tokens,
                window_tokens=window_tokens,
                state_slots=state_slots,
            ),
        )

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
            if band.allocated_count() or band.usage.has_pending():
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
        if self.swa_attn_allocator is not None:
            # Backends ask for the sliding-window layers' ids by their role:
            # the target's window.
            spaces.setdefault(
                IdSpaceKind.SLIDING_WINDOW,
                spaces[self.swa_attn_allocator.sub_pool_name],
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
        with self._relief(demands) as plan:
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
            if take is None or not take.count():
                continue
            physical = band.take_planned_pages(take)
            if physical is None:
                raise RuntimeError(
                    f"{band.sub_pool_name}: the planned take of {take.count()} "
                    "pages was refused"
                )
            taken[band.sub_pool_name] = physical
        return taken

    def lay_out_for(self, demands: Dict[str, int]) -> bool:
        """Make the least rearrangement that serves ``demands`` (pages per
        band), so that they then fit without moving a page; False exactly when
        no reachable layout serves them (`feasible` says no). For takes issued
        later and in any order, such as a host load's."""
        demands = {name: n for name, n in demands.items() if n > 0}
        with self._relief(demands) as plan:
            if plan is None:
                return False
            self._lay_out(plan)
            return True

    @contextmanager
    def _relief(self, demands: Dict[str, int]):
        """The plan serving ``demands``, and the scope it is laid out in.

        In place first: nothing moves, nothing is read. Otherwise every band's
        write sets, transfer pages, free list and recent holes are read on the
        host before the first stream wait of the layout. Rows an unreleased
        network transfer reads or writes (and pages pending on one) then stay
        where they are, as fixed pages, if some layout serves the demand that
        way: moving one would wait the engine's completion flag on the GPU,
        for as long as the remote peer takes. Only when no such layout exists
        does the plan move them, waiting their flags. `feasible` prices every
        page as movable either way, so it says yes exactly when one of the two
        layouts exists; the choice reads logical releases only, alike on every
        rank."""
        snapshot = self.pricing_snapshot()
        in_place = plan_layout(snapshot, demands, NO_MOVE)
        if in_place is not None:
            yield in_place
            return
        for band in self.bands:
            band.begin_relief()
        held = False
        try:
            for band in self.bands:
                if band.has_network_rows():
                    rows = band.network_rows()
                    if rows:
                        band.fix_pages(_NETWORK_ROWS, sorted(rows))
                        held = True
            found = first_plan(self.pricing_snapshot(), demands) if held else None
            if found is None and held:
                for band in self.bands:
                    band.unfix_pages(_NETWORK_ROWS)
                held = False
            if found is None:
                found = first_plan(self.pricing_snapshot(), demands)
            yield found
        finally:
            for band in self.bands:
                if held:
                    band.unfix_pages(_NETWORK_ROWS)
                band.end_relief()

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
    def scoped_physical_takes(
        self, full_tokens, swa_tokens, *, window_tokens=None, state_slots=0
    ):
        """Consume a pre-realized reservation across independently issued H2D takes."""
        if len(self._windows) <= 1:
            yield
            return
        if any(band._physical_take_scope is not None for band in self.members):
            raise RuntimeError("physical reservation scopes cannot be nested")
        scope = _ScopedPhysicalTakes(
            self,
            self.page_demands(
                full_tokens,
                swa_tokens,
                window_tokens=window_tokens,
                state_slots=state_slots,
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
