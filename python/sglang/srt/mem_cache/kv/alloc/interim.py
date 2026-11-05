"""Capacity policies selected from unified chain geometry.

`ChainAllocator` owns the bands and lifecycle operations. Each policy supplies
joint pricing, admission hooks, reclaim and per-side capacity views for one
chain shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv.alloc.band import (
    FloatBand,
    _float_open_short_side,
    flush_deferred_free_group,
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
    ends_reclaim_plan,
    float_available_tokens,
    float_byte_shortfall,
    float_can_reserve,
    float_end_certificate,
    float_fit,
    float_high_target,
    float_reclaim_satisfied,
    float_token_id_capacity,
)
from sglang.srt.mem_cache.swa_memory_pool import filter_window_cpu_copy
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel
from sglang.srt.utils.common import get_num_new_pages
from sglang.srt.utils.rank_consensus_checker import rank_consensus

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator


class ChainAdmissionRules(msgspec.Struct, frozen=True, kw_only=True):
    """Which admission rule a unified chain uses.

    `window` says how window tokens are admitted: "none" (no window sub-pool),
    "split" (checked apart from the owner's tokens) or "joint" (owner and window
    tokens in one reservation over the shared bytes).
    """

    window: Literal["none", "split", "joint"]
    # Full-token equivalents one new state slot takes from the shared bytes; 0: no state.
    state_slot_cost: int = 0


def _snapshot(c: ChainAllocator) -> ChainSnapshot:
    return c.pricing_snapshot()


def _virtual_tokens_to_physical(virtual_tokens: torch.Tensor, band) -> torch.Tensor:
    """Virtual token ids -> the band's physical token ids, page-aware. A page
    the band does not bind yields a negative id."""
    page_size = band.page_size
    if page_size == 1:
        return band.ids.virtual_to_physical[virtual_tokens]
    physical_pages = band.ids.virtual_to_physical[virtual_tokens // page_size]
    return physical_pages * page_size + virtual_tokens % page_size


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

    def prealloc_fits_assumes_reclaim(self, c: ChainAllocator) -> bool:
        return BaseTokenToKVPoolAllocator.prealloc_fits_assumes_reclaim(c)

    def prealloc_ceiling_fits(
        self, c: ChainAllocator, full_tokens: int, swa_tokens: int
    ) -> bool | None:
        return BaseTokenToKVPoolAllocator.prealloc_ceiling_fits(
            c, full_tokens, swa_tokens
        )

    def prealloc_fits(self, c: ChainAllocator, tree_cache, *args, **kwargs) -> bool:
        return BaseTokenToKVPoolAllocator.prealloc_fits(c, tree_cache, *args, **kwargs)

    def admission_rules(self, c: ChainAllocator) -> ChainAdmissionRules:
        raise NotImplementedError

    def max_new_tokens_for_memory(self, c: ChainAllocator, *args, **kwargs):
        return BaseTokenToKVPoolAllocator.max_new_tokens_for_memory(c, *args, **kwargs)

    def get_extend_allocation_demand(self, c: ChainAllocator, *args, **kwargs) -> int:
        return BaseTokenToKVPoolAllocator.get_extend_allocation_demand(
            c, *args, **kwargs
        )

    def evict_to_free_tokens(self, c: ChainAllocator, tree_cache, num_tokens: int):
        return BaseTokenToKVPoolAllocator.evict_to_free_tokens(
            c, tree_cache, num_tokens
        )

    def mamba_full_cache_donor(self, c: ChainAllocator):
        return None

    def usage(self, c: ChainAllocator) -> Dict[str, Tuple[int, int]]:
        """Each sub-pool's (capacity, available units) as observers report
        them, by sub-pool name: tokens, or slots for a state sub-pool."""
        raise NotImplementedError

    def resize(self, c: ChainAllocator, config) -> None:
        raise NotImplementedError(
            "post-capture resize is planned only for a full + sliding-window chain"
        )

    def memo_violations(self, c: ChainAllocator) -> List[str]:
        return []

    # Whether a retraction copy carries the request's state; otherwise the
    # scheduler backs the state up on its own.
    cpu_copy_carries_state = False

    def get_cpu_copy(
        self, c: ChainAllocator, indices, *, mamba_indices, req_pool_index
    ):
        raise NotImplementedError

    def load_cpu_copy(
        self, c: ChainAllocator, cpu_copy, indices, *, mamba_indices, req_pool_index
    ) -> None:
        raise NotImplementedError


class _TokenAndSlotEndsPolicy(_InterimPolicy):
    """A token end (the owner) and a slot end (per-request state)."""

    prices_before_take = False

    def __init__(self, c: ChainAllocator):
        pass

    @property
    def unit_name(self) -> str:
        return "token-and-slot ends"

    def size(self, c: ChainAllocator) -> int:
        # TOKENS. MUST use the SAME available view as `available_size()`, so the
        # available term cancels out of the leak invariant.
        return (
            c.full_attn_allocator.schedulable_available_size()
            + c.full_attn_allocator.allocated_count()
        )

    # Realizable-with-compaction view, so the retract gate / evict / schedule_policy
    # do not over-retract while the state peer holds drainable holes.
    def available_size(self, c: ChainAllocator) -> int:
        return c.full_attn_allocator.schedulable_available_size()

    def full_available_size(self, c: ChainAllocator) -> int:
        return c.full_attn_allocator.schedulable_available_size()

    def size_full(self, c: ChainAllocator) -> int:
        # Widened like `size`: a logical token capacity, not a row count.
        return (c.full_attn_allocator.max_slots - 1) * get_parallel().attn_dcp_size

    def mamba_full_cache_donor(self, c: ChainAllocator):
        return c

    def usage(self, c: ChainAllocator) -> Dict[str, Tuple[int, int]]:
        return {
            c.roles.owner: (c.size, c.available_size()),
            **_state_usage(c),
        }

    def prepare_mamba_allocation(self, c: ChainAllocator, target_size: int) -> None:
        """Make deferred Full reclaim visible to the Mamba capacity view."""
        c.flush_deferred_full_frees()
        if target_size > c.mamba_allocator.schedulable_available_size():
            return
        if target_size > c.mamba_allocator.available_size():
            relieve_for_alloc(c.mamba_allocator, target_size)

    def admission_rules(self, c: ChainAllocator) -> ChainAdmissionRules:
        return ChainAdmissionRules(
            window="none", state_slot_cost=self._state_slot_cost(c)
        )

    def _state_slot_cost(self, c: ChainAllocator) -> int:
        """Full-token-equivalents of shared-gap bytes ONE mamba state consumes; the
        prefill planner reserves this so admission stays inside the JOINT budget,
        rounded UP. The `dcp_size` factor is there because that budget is in widened
        tokens, one of which is `entry_bytes / dcp_size` local bytes.
        """
        return -(
            -c.mamba_allocator.entry_bytes_per_page
            * get_parallel().attn_dcp_size
            // c.full_attn_allocator.entry_bytes
        )

    def resize(self, c: ChainAllocator, config) -> None:
        BaseTokenToKVPoolAllocator.resize(c, config)

    cpu_copy_carries_state = True

    def get_cpu_copy(
        self, c: ChainAllocator, indices, *, mamba_indices, req_pool_index
    ):
        owner = c.views[c.roles.owner]
        kv_cpu = owner.get_cpu_copy(
            c.full_attn_allocator.ids.translate(indices.to(torch.int64)),
            req_pool_index=req_pool_index,
        )
        state_cpu = None
        if mamba_indices is not None:
            state_cpu = c.state.pool.get_cpu_copy(
                c.state_ids(c.roles.state).ids(mamba_indices)
            )
        return kv_cpu, state_cpu

    def load_cpu_copy(
        self, c: ChainAllocator, cpu_copy, indices, *, mamba_indices, req_pool_index
    ) -> None:
        kv_cpu, state_cpu = cpu_copy
        c.views[c.roles.owner].load_cpu_copy(
            kv_cpu,
            c.full_attn_allocator.ids.translate(indices.to(torch.int64)),
            req_pool_index=req_pool_index,
        )
        if state_cpu is not None and mamba_indices is not None:
            c.state.pool.load_cpu_copy(
                state_cpu, c.state_ids(c.roles.state).ids(mamba_indices)
            )


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
            self._size_full = int(full_max_total_num_tokens)
            self._size_swa = int(swa_max_total_num_tokens)
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
        return self._size_full

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

    def get_cpu_copy(
        self, c: ChainAllocator, indices, *, mamba_indices, req_pool_index
    ):
        # `indices` are virtual token ids; each side copies its physical ones,
        # and the window only the tokens it still binds.
        full_phys = _virtual_tokens_to_physical(indices, c.full_attn_allocator)
        swa_phys = _virtual_tokens_to_physical(indices, c.swa_attn_allocator)
        full_cpu = c.views[c.roles.owner].get_cpu_copy(
            full_phys, req_pool_index=req_pool_index
        )
        valid = swa_phys >= 0
        swa_cpu = None
        if bool(valid.any().item()):
            swa_cpu = c.views[c.roles.window].get_cpu_copy(swa_phys[valid])
        return {"full": full_cpu, "swa": swa_cpu, "swa_mask": valid.cpu()}

    def load_cpu_copy(
        self, c: ChainAllocator, cpu_copy, indices, *, mamba_indices, req_pool_index
    ) -> None:
        full_phys = _virtual_tokens_to_physical(indices, c.full_attn_allocator)
        c.views[c.roles.owner].load_cpu_copy(
            cpu_copy["full"], full_phys, req_pool_index=req_pool_index
        )
        if cpu_copy.get("swa") is None:
            return
        window = c.views[c.roles.window]
        swa_phys = _virtual_tokens_to_physical(indices, c.swa_attn_allocator)
        old_swa_mask = cpu_copy["swa_mask"].to(indices.device)
        assert old_swa_mask.shape == indices.shape
        row_mask = (swa_phys >= 0)[old_swa_mask].cpu()
        swa_phys = swa_phys[old_swa_mask][row_mask.to(indices.device)]
        if swa_phys.numel() == 0:
            return
        window.load_cpu_copy(
            filter_window_cpu_copy(
                window, swa_kv_cpu=cpu_copy["swa"], row_mask=row_mask
            ),
            swa_phys,
        )


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

    def prealloc_fits_assumes_reclaim(self, c: ChainAllocator) -> bool:
        return True

    def prealloc_ceiling_fits(
        self, c: ChainAllocator, full_tokens: int, swa_tokens: int
    ) -> bool | None:
        return c.can_reserve(full_tokens, swa_tokens, empty_pool=True)

    def reclaim_for_prealloc(
        self, c: ChainAllocator, tree_cache, full_tokens: int, swa_tokens: int
    ) -> str | None:
        """Reclaim both sides together: freeing FULL pages can open SWA room
        and the reverse, so the shared envelope is the only gate worth
        re-checking."""
        ready = c.evict_to_free_tokens(
            tree_cache, full_tokens, swa_num_tokens=swa_tokens
        )
        if ready is None:
            ready = c.ensure_capacity(full_tokens, swa_tokens)
        if ready:
            return None
        return (
            "Unified FULL/SWA byte reclamation insufficient: "
            f"needed=({full_tokens}, {swa_tokens})"
        )

    def prealloc_fits(
        self,
        c: ChainAllocator,
        tree_cache,
        full_tokens: int,
        swa_tokens: int,
        *,
        full_budget_tokens: int,
        swa_budget_tokens: int | None = None,
    ) -> bool:
        """Price both sides against the shared byte envelope.

        There is no per-side capacity for the scheduler's budget to be
        compared against, so the gap between that budget and what this side
        can currently hand out is folded back into the demand; `can_reserve`
        then prices the whole ask in bytes. Reachable only for hybrid-SWA
        models, so the tree's `full_*` accounting is the full-attention one.
        """
        full_evictable_tokens = tree_cache.full_evictable_size()
        swa_evictable_tokens = tree_cache.swa_evictable_size()
        full_tokens += (
            c.full_available_size() + full_evictable_tokens - full_budget_tokens
        )
        if swa_budget_tokens is not None:
            swa_tokens += (
                c.swa_available_size() + swa_evictable_tokens - swa_budget_tokens
            )
        return c.can_reserve(
            full_tokens,
            swa_tokens,
            full_evictable_tokens=full_evictable_tokens,
            swa_evictable_tokens=swa_evictable_tokens,
        )

    def reclaim_plan(
        self,
        c: ChainAllocator,
        full_tokens: int | float,
        swa_tokens: int | float,
        *,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
        empty_pool: bool = False,
        prefer_swa: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """Return cumulative FULL/SWA eviction targets, or None if impossible."""
        return ends_reclaim_plan(
            _snapshot(c),
            self._ends(c),
            full_tokens,
            swa_tokens,
            full_evictable_tokens=full_evictable_tokens,
            swa_evictable_tokens=swa_evictable_tokens,
            empty_pool=empty_pool,
            prefer_swa=prefer_swa,
        )

    def can_reserve(
        self,
        c: ChainAllocator,
        full_tokens: int | float,
        swa_tokens: int | float,
        *,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
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

    def admission_rules(self, c: ChainAllocator) -> ChainAdmissionRules:
        return ChainAdmissionRules(window="joint")

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

    def max_new_tokens_for_memory(
        self,
        c: ChainAllocator,
        input_tokens: int,
        max_new_tokens: int,
        *,
        token_capacity: int,
        sliding_window_size: int | None,
        chunk_size: int | None,
    ) -> int | None:
        from sglang.srt.mem_cache.prefill_budget import estimate_swa_kv_tokens

        # The search does not change the chain: price every candidate on one snapshot.
        snap, ends = _snapshot(c), self._ends(c)

        def fits(candidate):
            return ends_can_reserve(
                snap,
                ends,
                input_tokens + candidate + c.page_size,
                estimate_swa_kv_tokens(
                    input_tokens,
                    candidate,
                    sliding_window_size=sliding_window_size,
                    page_size=c.page_size,
                    allocation_limit=chunk_size,
                ),
                empty_pool=True,
                require_token_slack=True,
            )

        if not fits(0):
            return None
        if fits(max_new_tokens):
            return max_new_tokens
        lo, hi = 0, max_new_tokens
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    @rank_consensus(same_params=["num_tokens", "swa_num_tokens"], same_results=True)
    def evict_to_free_tokens(
        self,
        c: ChainAllocator,
        tree_cache,
        num_tokens: int,
        *,
        swa_num_tokens: Optional[int] = None,
    ) -> bool | None:
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        if tree_cache is None or not tree_cache.supports_prefix_sharing():
            return
        # Read per call: storage attach/detach and PD role switches move both leaves.
        memory = get_memory()
        prefer_swa = (
            memory.enable_hierarchical_cache
            and memory.hicache_storage_backend is not None
        ) or get_disagg().disaggregation_mode == "decode"
        required_swa = num_tokens if swa_num_tokens is None else swa_num_tokens
        flush_deferred_free_group(
            c,
            (
                c.free_group,
                c.free_page_reps_group,
                c.full_free_group,
                c.full_page_reps_group,
            ),
        )
        reclaim_plan = c.reclaim_plan(
            num_tokens,
            required_swa,
            full_evictable_tokens=tree_cache.full_evictable_size(),
            swa_evictable_tokens=tree_cache.swa_evictable_size(),
            prefer_swa=prefer_swa,
        )
        if reclaim_plan is None:
            return
        full_reclaim, swa_reclaim = reclaim_plan
        if full_reclaim or swa_reclaim:
            tree_cache.evict_for_alloc(
                EvictParams(num_tokens=full_reclaim, swa_num_tokens=swa_reclaim),
                allocation_reclaim_satisfied=lambda: (
                    c.reclaim_plan(num_tokens, required_swa) == (0, 0)
                ),
            )
        # A zero-reclaim plan can still depend on compaction before allocation.
        return c.ensure_capacity(num_tokens, required_swa)

    def resize(self, c: ChainAllocator, config) -> None:
        """Synchronize allocator capacity after the shared pool is finalized."""
        if not c.arena.backed_after_capture:
            raise RuntimeError("ChainAllocator.resize requires post-capture backing")
        final_pool_bytes = config.unified_memory_pool_bytes
        if final_pool_bytes is None:
            raise ValueError(
                "ChainAllocator.resize requires config.unified_memory_pool_bytes"
            )
        if int(final_pool_bytes) != c.arena.total_bytes:
            raise RuntimeError(
                "the arena must be backed before allocator.resize: "
                f"config has {int(final_pool_bytes)} bytes but the pool exposes "
                f"{c.arena.total_bytes} bytes"
            )
        fa, sa = c.full_attn_allocator, c.swa_attn_allocator
        for allocator in (fa, sa):
            if allocator.allocated_count() != 0 or allocator._pending_reuse:
                raise RuntimeError(
                    f"cannot resize non-empty unified allocator "
                    f"{allocator.sub_pool_name!r}"
                )
        fa._set_capacity(c.arena.num_entries(fa.sub_pool_name))
        sa._set_capacity(
            c.arena.num_entries(sa.sub_pool_name),
            virtual_num_pages=fa.ids.num_virtual_ids,
        )
        self._empty_shared_gap_bytes = fa._current_gap_bytes()
        self._size_full = fa.available_size()
        self._size_swa = min(
            sa.available_size(),
            len(fa.free_virtual_ids) * c.page_size,
        )
        self._full_max_total_num_tokens = self._size_full
        self._swa_max_total_num_tokens = self._size_swa
        c.get_kvcache().size = fa.max_slots - 1
        for allocator, host_capacity in (
            (fa, self._size_full),
            (sa, self._size_swa),
        ):
            pool = c.views[allocator.sub_pool_name]
            pool.num_pages = allocator.num_pages
            pool.size = allocator.num_pages * c.page_size - c.page_size
            pool.host_capacity_tokens = host_capacity
            pool.host_capacity_bytes = host_capacity * allocator.entry_bytes


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

    def prealloc_fits(
        self,
        c: ChainAllocator,
        tree_cache,
        full_tokens: int,
        swa_tokens: int,
        *,
        full_budget_tokens: int,
        swa_budget_tokens: int | None = None,
    ) -> bool:
        """Price the pair on the float chain's grid, then against the budgets.

        Each side's `available_size` takes `schedulable_available_size()`,
        which credits the peer's drainable holes, so the two are backed by the
        same bytes and a pair that fits each side alone can fail together. The
        budgets still apply on top: they carry decode headroom this allocator
        cannot see.
        """
        page_size = c.page_size
        if not float_fit(
            _snapshot(c),
            self._chain(c),
            -(-full_tokens // page_size),
            -(-swa_tokens // page_size),
        ):
            return False
        return full_tokens <= full_budget_tokens and (
            swa_budget_tokens is None or swa_tokens <= swa_budget_tokens
        )

    def reclaim_for_prealloc(
        self, c: ChainAllocator, tree_cache, full_tokens: int, swa_tokens: int
    ) -> str | None:
        """Free room for a decode-node preallocation; None means it is ready.

        Only the sliding-window side is reclaimed here; the full side is priced
        by the caller's budget.
        """
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        available = c.swa_available_size()
        if available < swa_tokens:
            tree_cache.evict_for_alloc(
                EvictParams(swa_num_tokens=swa_tokens - available)
            )
            available = c.swa_available_size()
        if available < swa_tokens:
            return (
                f"SWA eviction insufficient: needed={swa_tokens}, available={available}"
            )
        return None

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
        need_tokens = -(-int(full_tokens) // c.page_size) * c.page_size
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

    def mamba_full_cache_donor(self, c: ChainAllocator):
        return c

    def prepare_mamba_allocation(self, c: ChainAllocator, target_size: int) -> None:
        """Expose Full reclaim, then move the SWA float away from Mamba."""
        c.flush_deferred_full_frees()
        if target_size <= c.mamba_allocator.available_size():
            return
        c.full_attn_allocator.flush_for_allocation()
        relieve_for_alloc(c.mamba_allocator, target_size)

    def _state_slot_cost(self, c: ChainAllocator) -> int:
        """Full-token-equivalents one mamba/conv slot removes from the shared buffer:
        a tri-pool token costs e_f + e_s bytes, and the quotient is rounded UP."""
        e_tok = c.full_attn_allocator.entry_bytes + c.swa_attn_allocator.entry_bytes
        return -(-c.mamba_allocator.entry_bytes_per_page // e_tok)

    def admission_rules(self, c: ChainAllocator) -> ChainAdmissionRules:
        # State competes for the shared gap too; keep the split token admission
        # plus the state-slot gate until one reservation prices all three.
        return ChainAdmissionRules(
            window="split", state_slot_cost=self._state_slot_cost(c)
        )

    def swa_capacity_and_available(
        self, c: ChainAllocator, *, full_capacity, swa_capacity
    ):
        return (
            (full_capacity, c.conserve_full_available_size()),
            (swa_capacity, c.conserve_swa_available_size()),
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

    @rank_consensus(same_params=["num_tokens"], same_results=True)
    def evict_to_free_tokens(
        self, c: ChainAllocator, tree_cache, num_tokens: int
    ) -> bool | None:
        if tree_cache is None or not tree_cache.supports_prefix_sharing():
            return
        if num_tokens <= 0:
            return True
        if c.free_group is not None:
            c.flush_deferred_full_frees()
        num_tokens = -(-num_tokens // c.page_size) * c.page_size
        chain = self._chain(c)
        # Immediate capacity needs no reclaim plan or layout preparation.
        if (
            float_token_id_capacity(_snapshot(c), chain, num_tokens)
            and num_tokens <= c.available_size()
        ):
            return True

        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        full_reclaim = tree_cache.full_evictable_size()
        swa_reclaim = tree_cache.swa_evictable_size()
        state_reclaim = tree_cache.mamba_evictable_size()
        fa, sa, ma = (
            c.full_attn_allocator,
            c.swa_attn_allocator,
            c.mamba_allocator,
        )
        if (
            num_tokens // c.page_size
            > fa.free_virtual_ids.numel() + full_reclaim // c.page_size
            or num_tokens // c.page_size
            > sa.num_pages
            - sa.min_page_index
            - sa._live_pages()
            + swa_reclaim // c.page_size
            or float_byte_shortfall(
                _snapshot(c),
                chain,
                num_tokens,
                full_reclaim=full_reclaim,
                swa_reclaim=swa_reclaim,
                state_reclaim=state_reclaim,
            )
            > 0
        ):
            return False
        if c.ensure_capacity(num_tokens, num_tokens):
            return True

        # Quotas are cumulative logical counts within each phase. Token reclaim
        # precedes state reclaim so Mamba-first tree ordering cannot skip it.
        state_freed = 0
        if full_reclaim or swa_reclaim:
            result = tree_cache.evict_for_alloc(
                EvictParams(num_tokens=num_tokens, swa_num_tokens=num_tokens),
                allocation_reclaim_satisfied=lambda: float_reclaim_satisfied(
                    _snapshot(c), chain, num_tokens
                ),
            )
            state_freed = result.mamba_num_evicted
            if (
                result.num_tokens_evicted
                or result.swa_num_tokens_evicted
                or state_freed
            ) and c.ensure_capacity(num_tokens, num_tokens):
                return True

        pair_bytes = (
            num_tokens
            // c.page_size
            * (fa.entry_bytes_per_page + sa.entry_bytes_per_page)
        )
        state_quota = max(0, -(-pair_bytes // ma.entry_bytes) - state_freed)
        if state_quota and tree_cache.mamba_evictable_size():
            result = tree_cache.evict_for_alloc(
                EvictParams(mamba_num=state_quota),
                allocation_reclaim_satisfied=lambda: float_reclaim_satisfied(
                    _snapshot(c), chain, num_tokens
                ),
            )
            if (
                result.num_tokens_evicted
                or result.swa_num_tokens_evicted
                or result.mamba_num_evicted
            ):
                return c.ensure_capacity(num_tokens, num_tokens)
        return (
            float_token_id_capacity(_snapshot(c), chain, num_tokens)
            and num_tokens <= c.available_size()
        )


def interim_policy_for(
    c: ChainAllocator,
    *,
    full_max_total_num_tokens: Optional[int],
    swa_max_total_num_tokens: Optional[int],
) -> _InterimPolicy:
    """The policy of the chain's shape: its floats, and whether it has a slot
    end beside the owner."""
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
    if not has_window and has_slots and not has_float:
        return _TokenAndSlotEndsPolicy(c)
    raise ValueError(
        "unified memory has no capacity policy for a chain of "
        f"{[band.sub_pool_name for band in c.bands]}"
    )
