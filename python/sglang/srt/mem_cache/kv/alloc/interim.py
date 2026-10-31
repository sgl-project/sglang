"""Capacity policies selected from unified chain geometry.

`ChainAllocator` owns the bands and lifecycle operations. Each policy supplies
joint pricing, admission hooks, reclaim and per-side capacity views for one
chain shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv.alloc.band import (
    FloatBand,
    _float_open_short_side,
    relieve_for_alloc,
)
from sglang.srt.mem_cache.kv.alloc.pricing import (
    ChainSnapshot,
    FloatChain,
    TokenEnds,
    ends_available_tokens,
    ends_can_reserve,
    ends_fit,
    ends_realization,
    float_available_tokens,
    float_byte_shortfall,
    float_can_reserve,
    float_end_certificate,
    float_high_target,
    float_token_id_capacity,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import get_num_new_pages

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator


def _snapshot(c: ChainAllocator) -> ChainSnapshot:
    return c.pricing_snapshot()


def _state_usage(c: ChainAllocator) -> Dict[str, Tuple[int, int]]:
    """The state sub-pool's (slots, free slots)."""
    return {c.roles.state: (c.state.pool.size, c.state_slots.available_size())}


class _InterimPolicy:
    """Capacity behavior one chain shape shares with no other.

    Hooks a shape does not specialize fall back to the base allocator's.
    """

    # Window chains price a token allocation before the owner takes it; a chain
    # without a window lets the owner band relieve its own shortfall.
    prices_before_take = True
    # The leak checker's census: False sets `free_pages` to None, its skip
    # contract, where a census would mix physical and virtual ids.
    keeps_leak_census = True

    def get_extend_allocation_demand(self, c: ChainAllocator, *args, **kwargs) -> int:
        return BaseTokenToKVPoolAllocator.get_extend_allocation_demand(
            c, *args, **kwargs
        )

    def prealloc_fits_assumes_reclaim(self, c: ChainAllocator) -> bool:
        """Whether decode-node preallocation reclaims the evictable entries its
        admission credits (`reclaim_for_prealloc`) before it takes."""
        return False

    def usage(self, c: ChainAllocator) -> Dict[str, Tuple[int, int]]:
        """Each sub-pool's (capacity, available units) as observers report
        them, by sub-pool name: tokens, or slots for a state sub-pool."""
        raise NotImplementedError

    def memo_violations(self, c: ChainAllocator) -> List[str]:
        return []


class _OwnerEndPolicy(_InterimPolicy):
    """The token owner, optionally opposite a per-request state end."""

    prices_before_take = False

    def __init__(self, c: ChainAllocator):
        pass

    @property
    def unit_name(self) -> str:
        return "owner end with optional state"

    def size(self, c: ChainAllocator) -> int:
        # TOKENS. MUST use the SAME available view as `available_size()`, so the
        # available term cancels out of the leak invariant.
        return c.available_size() + c.full_attn_allocator.allocated_count()

    # Realizable-with-compaction view, so the retract gate / evict / schedule_policy
    # do not over-retract while the state peer holds drainable holes.
    def available_size(self, c: ChainAllocator) -> int:
        return c.full_attn_allocator.schedulable_available_size()

    def full_available_size(self, c: ChainAllocator) -> int:
        return c.full_attn_allocator.schedulable_available_size()

    def size_full(self, c: ChainAllocator) -> int:
        # Widened like `size`: a logical token capacity, not a row count.
        return (c.full_attn_allocator.max_slots - 1) * get_parallel().attn_dcp_size

    def usage(self, c: ChainAllocator) -> Dict[str, Tuple[int, int]]:
        return {
            c.roles.owner: (c.size, c.available_size()),
            **(_state_usage(c) if c.roles.state is not None else {}),
        }


class _WindowPolicy(_InterimPolicy):
    """A window band binds the owner's virtual pages: the joint views."""

    def __init__(
        self,
        c: ChainAllocator,
        *,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
    ):
        if (full_max_total_num_tokens is None) != (swa_max_total_num_tokens is None):
            raise ValueError(
                "full_max_total_num_tokens and swa_max_total_num_tokens must "
                "either both be set or both be omitted"
            )
        fa, sa = c.full_attn_allocator, c.swa_attn_allocator
        legacy_capacities = full_max_total_num_tokens is not None
        self._empty_shared_gap_bytes = fa._current_gap_bytes()
        if legacy_capacities:
            # Sizing supplies physical rows per rank. All policy capacity and
            # usage views count the widened logical tokens the bands allocate.
            self._size_full = (
                int(full_max_total_num_tokens) * fa.page_size // fa.pool_page_size
            )
            self._size_swa = (
                int(swa_max_total_num_tokens) * sa.page_size // sa.pool_page_size
            )
        else:
            self._size_full = fa.available_size()
            self._size_swa = min(
                sa.available_size(), len(fa.free_virtual_ids) * c.page_size
            )
        self._full_max_total_num_tokens = self._size_full
        self._swa_max_total_num_tokens = self._size_swa
        # Epoch-keyed memo for the joint capacity view (any chain member's
        # mutation invalidates -- see `Band._chain_capacity_epoch`).
        self._joint_avail_memo_epoch: Optional[int] = None
        self._joint_avail_memo_tokens: int = 0

    def size(self, c: ChainAllocator) -> int:
        return min(self._size_full, self._size_swa)

    def size_full(self, c: ChainAllocator) -> int:
        cap = c.full_attn_allocator.max_live_tokens
        return self._size_full if cap is None else min(self._size_full, cap)

    def size_swa(self, c: ChainAllocator) -> int:
        return self._size_swa

    def available_size(self, c: ChainAllocator) -> int:
        """Tokens available for `alloc(N)` / `alloc_extend(N)` (TOKENS)."""
        epoch = c.full_attn_allocator._chain_capacity_epoch()
        if self._joint_avail_memo_epoch != epoch:
            self._joint_avail_memo_tokens = self._compute_available_size(c)
            self._joint_avail_memo_epoch = epoch
        return self._joint_avail_memo_tokens

    # Slot-conservation views for the leak invariant only; the byte-coordinated
    # value would flag spurious leaks. `allocated_count()` is in TOKENS.
    def conserve_full_available_size(self, c: ChainAllocator) -> int:
        return self._full_max_total_num_tokens - c.full_attn_allocator.allocated_count()

    def conserve_swa_available_size(self, c: ChainAllocator) -> int:
        return self._swa_max_total_num_tokens - c.swa_attn_allocator.allocated_count()

    # Per-side views read by scheduling / eviction: the static-conserve cap bounds
    # the lending side, `schedulable_*` the side grown into the shared gap.
    def full_available_size(self, c: ChainAllocator) -> int:
        return min(
            c.conserve_full_available_size(),
            c.full_attn_allocator.schedulable_available_size(),
        )

    def swa_available_size(self, c: ChainAllocator) -> int:
        return min(
            c.conserve_swa_available_size(),
            c.swa_attn_allocator.schedulable_available_size(),
        )

    def current_full_capacity(self, c: ChainAllocator) -> int:
        return c.full_available_size() + c.full_attn_allocator.allocated_count()

    def current_swa_capacity(self, c: ChainAllocator) -> int:
        return c.swa_available_size() + c.swa_attn_allocator.allocated_count()

    def get_extend_allocation_demand(
        self,
        c: ChainAllocator,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        *,
        conservative_num_tokens: int,
        shard_size: int,
    ) -> int:
        if shard_size != 1:
            return conservative_num_tokens
        return c.page_size * get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=c.page_size,
            prefix_lens=prefix_lens_cpu,
        )

    def memo_violations(self, c: ChainAllocator) -> List[str]:
        """Idle-time twin of `Band._capacity_memo_violations`
        for the joint view. Empty == healthy."""
        if (
            self._joint_avail_memo_epoch
            != c.full_attn_allocator._chain_capacity_epoch()
        ):
            return []
        actual = self._compute_available_size(c)
        if self._joint_avail_memo_tokens == actual:
            return []
        return [
            f"[joint] stale available_size memo: "
            f"cached={self._joint_avail_memo_tokens}, actual={actual}"
        ]


class _TokenEndsPolicy(_WindowPolicy):
    """Two token ends: the owner and its window, with asymmetric shared-byte
    reservations."""

    def __init__(self, c: ChainAllocator, **kwargs):
        super().__init__(c, **kwargs)
        self._ends_memo: Optional[Tuple[tuple, TokenEnds]] = None

    @property
    def unit_name(self) -> str:
        return "two token ends"

    def _ends(self, c: ChainAllocator) -> TokenEnds:
        constants = (self._empty_shared_gap_bytes, self._size_full, self._size_swa)
        memo = self._ends_memo
        if memo is None or memo[0] != constants:
            ends = TokenEnds(
                owner=c.roles.owner,
                window=c.roles.window,
                empty_gap_bytes=constants[0],
                size_full=constants[1],
                size_swa=constants[2],
            )
            memo = self._ends_memo = (constants, ends)
        return memo[1]

    def _compute_available_size(self, c: ChainAllocator) -> int:
        """Joint byte budget in TOKENS: each chain alloc(1) consumes one
        full-side AND one swa-side page under the same virtual id."""
        return ends_available_tokens(_snapshot(c), self._ends(c))

    def flush_targets(self, c: ChainAllocator):
        """Flush ALL members, including ones that are not short themselves: a
        one-sided hole is unusable, and compacting it yields SHARED gap."""
        return (c.full_attn_allocator, c.swa_attn_allocator)

    def ask_float_for_room(self, c: ChainAllocator, need_tokens: int) -> None:
        """No float in a two-END chain -- nothing can slide."""
        return None

    def prealloc_fits_assumes_reclaim(self, c):
        return True

    def can_reserve(
        self,
        c: ChainAllocator,
        full_tokens: int | float,
        swa_tokens: int | float,
        *,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
        auxiliary_swa_evictable_tokens: int = 0,
        state_evictable_slots: int = 0,
        empty_pool: bool = False,
        require_token_slack: bool = False,
    ) -> bool:
        """Check pending FULL/SWA demand against the shared byte envelope."""
        return ends_can_reserve(
            _snapshot(c),
            self._ends(c),
            full_tokens,
            swa_tokens,
            full_evictable_tokens=full_evictable_tokens,
            swa_evictable_tokens=swa_evictable_tokens,
            empty_pool=empty_pool,
            require_token_slack=require_token_slack,
        )

    def ensure_capacity(
        self, c: ChainAllocator, full_tokens: int, swa_tokens: int
    ) -> bool:
        """Gate one allocation and compact both sides on shortfall."""
        if full_tokens < 0 or swa_tokens < 0:
            return False
        if full_tokens == 0 and swa_tokens == 0:
            return True
        page_size = c.page_size
        num_full_pages = (int(full_tokens) + page_size - 1) // page_size
        num_swa_pages = (int(swa_tokens) + page_size - 1) // page_size
        ends = self._ends(c)
        realization = ends_realization(
            _snapshot(c), ends, num_full_pages, num_swa_pages
        )
        if not realization.flush_ends:
            return realization.fits
        c.full_attn_allocator.flush_for_allocation()
        c.swa_attn_allocator.flush_for_allocation()
        return ends_fit(
            _snapshot(c), ends, num_full_pages, num_swa_pages, compacted=False
        )

    def swa_capacity_and_available(
        self, c: ChainAllocator, *, full_capacity, swa_capacity
    ):
        return (
            (c.current_full_capacity, c.full_available_size()),
            (c.current_swa_capacity, c.swa_available_size()),
        )

    def usage(self, c: ChainAllocator) -> Dict[str, Tuple[int, int]]:
        return {
            c.roles.owner: (c.current_full_capacity, c.full_available_size()),
            c.roles.window: (c.current_swa_capacity, c.swa_available_size()),
        }


class _FloatChainPolicy(_WindowPolicy):
    """A slot end, the window floating in the middle, and the owner end:

        [ state (grow-up END) | window (FLOAT middle) | owner (grow-down END) ]

    The ends never relocate, so they take the per-request state pool and the
    unbounded per-step grower; the window is capped with the cheapest slots to
    move, and its out-of-window tombstones become float holes recycled in place.
    """

    keeps_leak_census = False

    @property
    def unit_name(self) -> str:
        return "ends with a float"

    # -- capacity --

    def __init__(self, c: ChainAllocator, **kwargs):
        super().__init__(c, **kwargs)
        self._float_chain: Optional[FloatChain] = None

    def _chain(self, c: ChainAllocator) -> FloatChain:
        chain = getattr(self, "_float_chain", None)
        if chain is None:
            roles = c.roles
            chain = self._float_chain = FloatChain(
                owner=roles.owner, window=roles.window, state=roles.state
            )
        return chain

    def can_reserve(
        self,
        c: ChainAllocator,
        full_tokens: int | float,
        swa_tokens: int | float,
        *,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
        auxiliary_swa_evictable_tokens: int = 0,
        state_evictable_slots: int = 0,
        empty_pool: bool = False,
        require_token_slack: bool = False,
    ) -> bool:
        return float_can_reserve(
            _snapshot(c),
            self._chain(c),
            full_tokens,
            swa_tokens,
            full_evictable_tokens=full_evictable_tokens,
            swa_evictable_tokens=swa_evictable_tokens,
            empty_pool=empty_pool,
        )

    def ensure_capacity(
        self, c: ChainAllocator, full_tokens: int, swa_tokens: int
    ) -> bool:
        if full_tokens < 0 or swa_tokens < 0:
            return False
        if c.can_reserve(full_tokens, swa_tokens):
            return True
        if full_tokens != swa_tokens:
            for allocator in c._flush_targets():
                allocator.flush_for_allocation()
            if c.can_reserve(full_tokens, swa_tokens):
                return True
            _float_open_short_side(
                c.swa_attn_allocator,
                {
                    c.full_attn_allocator: -(-full_tokens // c.page_size),
                    c.swa_attn_allocator: -(-swa_tokens // c.page_size),
                    c.mamba_allocator: 0,
                },
            )
            return c.can_reserve(full_tokens, swa_tokens)
        need_tokens = int(-(-full_tokens // c.page_size)) * c.page_size
        chain = self._chain(c)
        if not float_token_id_capacity(_snapshot(c), chain, need_tokens):
            return False
        if need_tokens <= c.available_size():
            return True
        if float_end_certificate(_snapshot(c), chain, need_tokens):
            # The certificate preserves FLOAT holes in place. Absorbing FLOAT
            # boundaries first would describe a different allocation layout.
            c.full_attn_allocator.flush_for_allocation()
            c.mamba_allocator.flush_for_allocation()
            return need_tokens <= c.available_size()
        if float_high_target(_snapshot(c), chain, need_tokens) is not None:
            c.full_attn_allocator.flush_for_allocation()
            c.mamba_allocator.flush_for_allocation()
            if not float_token_id_capacity(_snapshot(c), chain, need_tokens):
                return False
            if need_tokens <= c.available_size():
                return True
            # Recompute from actual frontiers: no credit for another END flush.
            target = float_high_target(
                _snapshot(c), chain, need_tokens, flush_ends=False
            )
            if target is None:
                return False
            c.swa_attn_allocator.make_room(side="high", min_bytes=target)
            return float_token_id_capacity(_snapshot(c), chain, need_tokens) and (
                need_tokens <= c.available_size()
            )
        if float_byte_shortfall(_snapshot(c), chain, need_tokens) > 0:
            return False
        # Today's last resort, imperative: where the float's holes sit decides it.
        return relieve_for_alloc(c, need_tokens)

    def _compute_available_size(self, c: ChainAllocator) -> int:
        """Joint TOKENS for equal FULL/SWA demand, using the same page predicate
        as tail allocation. FULL takes the high band before SWA binds its pages."""
        return float_available_tokens(_snapshot(c), self._chain(c))

    def flush_targets(self, c: ChainAllocator):
        """All three members, float FIRST: its zero-copy boundary absorption must
        land before the deficit math prices a relocation it already covered."""
        return (
            c.swa_attn_allocator,
            c.full_attn_allocator,
            c.mamba_allocator,
        )

    def _alloc_demand(self, c: ChainAllocator, need_tokens: int):
        """Demand VECTOR for one chain allocation, in PAGES per band. A token
        never draws a state slot, so mamba is an explicit 0, not an omission."""
        need_n = -(-need_tokens // c.page_size)
        return {
            c.full_attn_allocator: need_n,
            c.swa_attn_allocator: need_n,
            c.mamba_allocator: 0,
        }

    def ask_float_for_room(self, c: ChainAllocator, need_tokens: int) -> None:
        """Chain shortfall: hand the demand vector to the shared policy;
        the float is whichever demanded band floats."""
        demand = self._alloc_demand(c, need_tokens)
        flt = None
        for b in demand:
            if isinstance(b, FloatBand):
                flt = b
        _float_open_short_side(flt, demand)

    def swa_capacity_and_available(
        self, c: ChainAllocator, *, full_capacity, swa_capacity
    ):
        return (
            (self._size_full, c.conserve_full_available_size()),
            (self._size_swa, c.conserve_swa_available_size()),
        )

    def usage(self, c: ChainAllocator) -> Dict[str, Tuple[int, int]]:
        # The static caps the slot-conservation views count down from.
        return {
            c.roles.owner: (
                self._full_max_total_num_tokens,
                c.conserve_full_available_size(),
            ),
            c.roles.window: (
                self._swa_max_total_num_tokens,
                c.conserve_swa_available_size(),
            ),
            **_state_usage(c),
        }


class _MultiWindowPolicy(_WindowPolicy):
    """Conservative admission for independently retained window bands.

    Admission can credit evictable pages while reservation checks only current
    geometry. Relief and eviction finish before any physical take is published.
    Inherited per-side budgets are distinct from equal-demand joint availability.
    """

    keeps_leak_census = False
    unit_name = "independent windows"

    def _demands(
        self, c, full_tokens, swa_tokens, auxiliary_swa_tokens=None, state_slots=0
    ):
        auxiliary_swa_tokens = (
            full_tokens if auxiliary_swa_tokens is None else auxiliary_swa_tokens
        )
        page = c.page_size
        # Request/load-back state is allocated before the token bands.
        state_demand = ((c.roles.state, int(state_slots)),) if c.roles.state else ()
        return state_demand + tuple(
            (
                b.sub_pool_name,
                int(
                    -(
                        -(
                            full_tokens
                            if b is c.full_attn_allocator
                            else swa_tokens
                            if b is c.swa_attn_allocator
                            else auxiliary_swa_tokens
                        )
                        // page
                    )
                ),
            )
            for b in (c.full_attn_allocator, *c._windows)
        )

    def can_reserve(
        self,
        c,
        full_tokens,
        swa_tokens,
        *,
        auxiliary_swa_tokens=None,
        full_evictable_tokens=0,
        swa_evictable_tokens=0,
        auxiliary_swa_evictable_tokens=0,
        state_slots=0,
        state_evictable_slots=0,
        empty_pool=False,
        require_token_slack=False,
    ):
        from sglang.srt.mem_cache.kv.alloc.pricing import (
            NO_MOVE,
            plan,
            reclaimed_snapshot,
        )

        if full_tokens < 0 or swa_tokens < 0 or state_slots < 0:
            return False
        if require_token_slack and (
            full_tokens >= self._size_full or swa_tokens >= self._size_swa
        ):
            return False
        snapshot = _snapshot(c)
        credits = {
            c.roles.owner: int(full_evictable_tokens) // c.page_size,
            **{
                b.sub_pool_name: int(
                    swa_evictable_tokens
                    if b is c.swa_attn_allocator
                    else auxiliary_swa_evictable_tokens
                )
                // c.page_size
                for b in c._windows
            },
        }
        if c.roles.state is not None:
            credits[c.roles.state] = int(state_evictable_slots)
        if empty_pool or any(credits.values()):
            snapshot = reclaimed_snapshot(snapshot, credits, empty_pool=empty_pool)
        if (
            int(-(-full_tokens // c.page_size))
            > snapshot.band(c.roles.owner).free_virtual_pages
        ):
            return False
        if state_slots and (
            c.roles.state is None
            or state_slots > snapshot.band(c.roles.state).free_virtual_pages
        ):
            return False
        return (
            plan(
                snapshot,
                dict(
                    self._demands(
                        c, full_tokens, swa_tokens, auxiliary_swa_tokens, state_slots
                    )
                ),
                NO_MOVE,
            )
            is not None
        )

    def _compute_available_size(self, c):
        pages = min(
            len(c.full_attn_allocator.free_virtual_ids),
            *(
                b.available_size() // c.page_size
                for b in (c.full_attn_allocator, *c._windows)
            ),
        )
        # The in-place plan is monotone in each band's demand, so this is the
        # largest equal demand that fits without moving a page.
        low, high = 0, pages
        while low < high:
            mid = (low + high + 1) // 2
            if self.can_reserve(c, mid * c.page_size, mid * c.page_size):
                low = mid
            else:
                high = mid - 1
        return low * c.page_size

    def ensure_capacity(
        self, c, full_tokens, swa_tokens, *, auxiliary_swa_tokens=None, state_slots=0
    ):
        if self.can_reserve(
            c,
            full_tokens,
            swa_tokens,
            auxiliary_swa_tokens=auxiliary_swa_tokens,
            state_slots=state_slots,
        ):
            return True
        for band in self.flush_targets(c):
            band.flush_for_allocation()
        if self.can_reserve(
            c,
            full_tokens,
            swa_tokens,
            auxiliary_swa_tokens=auxiliary_swa_tokens,
            state_slots=state_slots,
        ):
            return True
        demand = {
            next(b for b in c.bands if b.sub_pool_name == name): n
            for name, n in self._demands(
                c, full_tokens, swa_tokens, auxiliary_swa_tokens, state_slots
            )
        }
        for band in c.bands:
            if isinstance(band, FloatBand):
                _float_open_short_side(band, demand)
                if self.can_reserve(
                    c,
                    full_tokens,
                    swa_tokens,
                    auxiliary_swa_tokens=auxiliary_swa_tokens,
                    state_slots=state_slots,
                ):
                    return True
        return False

    def flush_targets(self, c):
        return tuple(sorted(c.bands, key=lambda b: b.grow_direction != "float"))

    def ask_float_for_room(self, c, need_tokens):
        self.ensure_capacity(c, need_tokens, need_tokens)

    def swa_capacity_and_available(self, c, *, full_capacity, swa_capacity):
        return (
            (self._size_full, c.conserve_full_available_size()),
            (self._size_swa, c.conserve_swa_available_size()),
        )

    def usage(self, c):
        return {
            b.sub_pool_name: (
                b.allocated_count() + b.available_size(),
                b.available_size(),
            )
            for b in c.bands
        }

    def prealloc_fits_assumes_reclaim(self, c):
        return True


def interim_policy_for(
    c: ChainAllocator,
    *,
    full_max_total_num_tokens: Optional[int],
    swa_max_total_num_tokens: Optional[int],
) -> _InterimPolicy:
    """The policy of the chain's shape: its floats, and whether it has a slot
    end beside the owner."""
    if len(c._windows) > 1:
        return _MultiWindowPolicy(
            c,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
        )
    has_float = any(band.grow_direction == "float" for band in c.bands)
    has_window = c.swa_attn_allocator is not None
    has_slots = c.mamba_allocator is not None
    if has_window and not has_slots and not has_float:
        return _TokenEndsPolicy(
            c,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
        )
    if has_window and has_slots and has_float:
        return _FloatChainPolicy(
            c,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
        )
    if not has_window and not has_float:
        return _OwnerEndPolicy(c)
    raise ValueError(
        "unified memory has no capacity policy for a chain of "
        f"{[band.sub_pool_name for band in c.bands]}"
    )
