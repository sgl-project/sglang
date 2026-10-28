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
from sglang.srt.mem_cache.kv.alloc.pricing import ChainSnapshot, snapshot_chain
from sglang.srt.mem_cache.kv.memory.arena import KVArena
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import get_num_new_pages, next_power_of_2
from sglang.srt.utils.invariants import expect, resolve_level

logger = logging.getLogger(__name__)


class ChainRoles(msgspec.Struct, frozen=True, kw_only=True):
    """Sub-pool names by role: the virtual-id owner, the sliding window that
    binds the owner's virtual pages, and the per-request state slots."""

    owner: str
    window: Optional[str] = None
    state: Optional[str] = None


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
) -> Optional[torch.Tensor]:
    """Fill tail pages before taking new virtual pages for each request.

    Bind every new page in the owner band before translating its token ids;
    otherwise its v2p entry remains -1.
    """
    if num_new_pages > len(owner.free_virtual_ids):
        return None
    # Physical-capacity pre-check; on shortfall run the relief ladder.
    need_tokens = num_new_pages * owner.page_size
    if need_tokens > owner.available_size():
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
        if owner.place_pages(new_virtual_pages, num_new_pages) is None:
            return None  # defensive; the pre-check should have prevented it
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
        full_max_total_num_tokens: Optional[int] = None,
        swa_max_total_num_tokens: Optional[int] = None,
    ):
        # Only the owner shards under decode context parallelism, and only in a
        # chain without a window: window chains reject it at argument
        # validation and never read the width.
        shards_under_dcp = roles.window is None
        dcp_size = get_parallel().attn_dcp_size if shards_under_dcp else 1
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
        self.swa_attn_allocator: Optional[Band] = None
        if roles.window is not None:
            floats = arena.band(roles.window).grow_direction == "float"
            # A float never runs the lazy event pipeline: frees mark holes and
            # allocations reuse them in place.
            self.swa_attn_allocator = (FloatBand if floats else Band)(
                sub_pool_name=roles.window,
                is_id_owner=False,  # binds the virtual pages the owner mints
                page_size=page_size,
                lazy_compaction=lazy_compaction and not floats,
                virtual_num_pages=self.full_attn_allocator.ids.num_virtual_ids,
                **band,
            )
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
                self.swa_attn_allocator,
                self.mamba_allocator,
            )
            if b is not None
        )
        self.bands = _chain_order(self.members)
        _wire_chain(self.bands)
        self._band_positions = {b.sub_pool_name: i for i, b in enumerate(self.bands)}
        self._snapshot_key: Optional[int] = None
        self._snapshot_memo: Optional[ChainSnapshot] = None
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
        """A window chain's host pools are sized in tokens."""
        for band, tokens in (
            (self.full_attn_allocator, self.size_full),
            (self.swa_attn_allocator, self.size_swa),
        ):
            pool = self.views[band.sub_pool_name]
            pool.host_capacity_tokens = tokens
            pool.host_capacity_bytes = tokens * band.entry_bytes

    @property
    def has_window_pool(self) -> bool:
        return self.swa_attn_allocator is not None

    @property
    def _windows(self) -> Tuple[Band, ...]:
        return () if self.swa_attn_allocator is None else (self.swa_attn_allocator,)

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

    def available_size(self) -> int:
        return self._policy.available_size(self)

    def full_available_size(self) -> int:
        return self._policy.full_available_size(self)

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

    def ensure_capacity(self, full_tokens: int, swa_tokens: int) -> bool:
        return self._policy.ensure_capacity(self, full_tokens, swa_tokens)

    def can_reserve(self, full_tokens, swa_tokens, **kwargs) -> bool:
        return self._policy.can_reserve(self, full_tokens, swa_tokens, **kwargs)

    def reclaim_plan(self, full_tokens, swa_tokens, **kwargs):
        return self._policy.reclaim_plan(self, full_tokens, swa_tokens, **kwargs)

    def prealloc_fits_assumes_reclaim(self) -> bool:
        return self._policy.prealloc_fits_assumes_reclaim(self)

    def prealloc_ceiling_fits(self, full_tokens: int, swa_tokens: int) -> bool | None:
        return self._policy.prealloc_ceiling_fits(self, full_tokens, swa_tokens)

    def prealloc_fits(self, tree_cache, full_tokens, swa_tokens, **kwargs) -> bool:
        return self._policy.prealloc_fits(
            self, tree_cache, full_tokens, swa_tokens, **kwargs
        )

    def reclaim_for_prealloc(
        self, tree_cache, full_tokens: int, swa_tokens: int
    ) -> str | None:
        return self._policy.reclaim_for_prealloc(
            self, tree_cache, full_tokens, swa_tokens
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

    def resize(self, config) -> None:
        self._policy.resize(self, config)

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
                widened_writes=False,
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
        return self._policy.cpu_copy_carries_state

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        """Retraction backup: `Req.offload_kv_cache` hands over `req_to_token`
        rows, which hold VIRTUAL ids here; each sub-pool's view copies its
        physical ones."""
        return self._policy.get_cpu_copy(
            self, indices, mamba_indices=mamba_indices, req_pool_index=req_pool_index
        )

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        self._policy.load_cpu_copy(
            self,
            kv_cache_cpu,
            indices,
            mamba_indices=mamba_indices,
            req_pool_index=req_pool_index,
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
    ) -> None:
        """Hand the forward's metadata to every member; token bands translate the
        write set through their own v2p, and the state band, written by state
        kernels rather than through `out_cache_loc`, gets None."""
        with record_function("ChainAllocator.set_inflight_forward"):
            for member in self.members:
                member.set_inflight_forward(
                    forward_done,
                    None if member is self.mamba_allocator else out_cache_loc_virtual,
                )

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
        owner_take: Callable[[], Optional[torch.Tensor]],
        window_pages: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = lambda out, new_pages: new_pages,
    ) -> Optional[torch.Tensor]:
        """Price before taking owner ids, then bind their new window pages."""
        owner = self.full_attn_allocator
        if not self._policy.prices_before_take:
            # The owner prices and relieves its own shortfall.
            return owner_take()
        if not self.ensure_capacity(full_tokens, swa_tokens):
            return None
        # free_virtual_ids is rebound (slice/cat), never mutated in place, so
        # this prefix view still names the pages the owner is about to consume.
        new_virtual_pages = owner.free_virtual_ids[:num_owner_pages]
        out = owner_take()
        assert out is not None, (
            "ChainAllocator: the owner refused after the joint pre-check "
            "passed -- internal-state inconsistency"
        )
        bind = window_pages(out, new_virtual_pages)
        for window in self._windows:
            window.alloc_with_virtual(bind)
        return out

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with record_function("ChainAllocator.alloc"):
            return self._allocate(
                full_tokens=need_size,
                swa_tokens=need_size,
                num_owner_pages=need_size // self.page_size,
                owner_take=lambda: self.full_attn_allocator.alloc(need_size),
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
    ) -> Callable[[], Optional[torch.Tensor]]:
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

        return lambda: take_new_pages(
            self.full_attn_allocator,
            num_new_pages,
            extend_tokens,
            copy_new_pages=False,
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
            if not self._policy.prices_before_take:
                return self._paged_take(
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    get_num_new_pages(
                        seq_lens=seq_lens_cpu,
                        page_size=self.full_attn_allocator.page_size,
                        prefix_lens=prefix_lens_cpu,
                    ),
                )()
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
                if swa_tail_len == 0 or new_pages.numel() == 0:
                    return new_pages[:0]
                tail = torch.unique(out[-swa_tail_len:] // self.page_size)
                # Only NEW pages need binding; a tail page carried in from the
                # prefix is already bound on the window.
                return new_pages[torch.isin(new_pages, tail)]

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
                owner_take=lambda: take_new_pages(
                    self.full_attn_allocator,
                    num_new_pages,
                    decode_tokens,
                    copy_new_pages=True,
                ),
            )

    def alloc_state_slots(self, need_size: int) -> Optional[torch.Tensor]:
        """Per-request state slots: the state band prices and relieves its own
        shortfall (VIRTUAL slot ids)."""
        return self.mamba_allocator.alloc(need_size)

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
        """Window tombstone path: release window-physical, keep the virtual id
        and owner-physical live; `v2p_page[v_page] = -1` IS the tombstone."""
        if free_index is None or free_index.numel() == 0:
            return
        v = free_index.detach().to(torch.int64)
        ps = self.page_size
        for window in self._windows:
            # `> 0` strict: -1 = tombstoned, page 0 = padding sink (never freeable).
            live = v[window.ids.virtual_to_physical[v // ps] > 0]
            if live.numel() == 0:
                continue
            if ps == 1:
                # token == page and the live filter just deduped against the v2p
                # table, so these ARE unique page ids -- same skip as `_free_lazy`.
                window.free(live, _pages=live)
            else:
                window.free(live)
            window.clear_inverse_history()

    def free_swa_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        """free_swa() for a kv-row segment whose window pages are all bound: page
        reps come from `start_pos`, so it is fixed-shape at every page size."""
        if free_index is None or free_index.numel() == 0:
            return
        for window in self._windows:
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
        """Release the owner-physical page and the virtual id, leaving the window
        alone -- the caller already tombstoned it. Without a window this is free()."""
        if not self._windows:
            self.free(free_index)
            return
        if free_index is None or free_index.numel() == 0:
            return
        if self.free_group is not None:
            self.full_free_group.append(self._copy_for_free_group(free_index))
            return
        self.full_attn_allocator.free(free_index.detach().to(torch.int64))
        self.full_attn_allocator.clear_inverse_history()

    def free_full_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        """Release a kv-row segment's owner side; its window side is already
        tombstoned."""
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
            self.full_attn_allocator.free(torch.cat(full_free_group))
            self.full_attn_allocator.clear_inverse_history()
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
        """Free page reps: `both` on every window and the owner, `full_only` on
        the owner."""
        if both:
            reps = both[0] if len(both) == 1 else torch.cat(tuple(both))
            for window in self._windows:
                self._free_window_page_reps(window, reps)
        pieces = (*both, *full_only)
        reps = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
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
