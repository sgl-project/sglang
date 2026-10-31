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
"""A band: one sub-pool's units in a `KVArena`.

A band places its sub-pool's units in its stretch of the arena's bytes: an end
grows from one end of the arena toward the other, a float holds a span between
two neighbours. It takes and frees physical units, records which virtual unit
each holds in its `UnitIds` (the owner mints the virtual units; a window binds
the owner's), and moves units to compact or make room, lazily behind the
forward that may still read them. Counts are in the sub-pool's tokens
(`page_size` per unit).

Requests are not a band's concern: the chain allocator turns a request's token
positions into new units and hands them to its bands.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import (
    Generic,
    Hashable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np
import torch
from torch.profiler import record_function

from sglang.kernels.ops.memory.virtual_slot import (
    alloc_bind_inplace,
    free_unbind_inplace,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv.alloc.id_space import UnitIds
from sglang.srt.mem_cache.kv.alloc.page_usage import PageUsage
from sglang.srt.mem_cache.kv.alloc.signals import (
    TransferRelease,
    signal_fired,
    wait_signals,
)
from sglang.srt.mem_cache.kv.memory.arena import KVArena
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


import atexit
import signal
import time as _time_mod  # local alias so tests can patch
import weakref

_LAZY_COMPACTION_STATS_ENABLED = envs.SGLANG_LOG_LAZY_COMPACTION_STATS.get()
_LAZY_COMPACTION_STATS_INTERVAL_SEC = float(
    envs.SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC.get()
)
# Signal handler emits each instance's final counters (atexit misses signal exits).
_STATS_INSTANCES: weakref.WeakSet[Band] = weakref.WeakSet()
_SIGNAL_HANDLERS_INSTALLED = False

# The key a HiCache load fixes its reserved pages under, from reservation to
# the submission of its H2D copy.
_LOAD_RESERVATION = "load"


def _page_runs(pages: Sequence[int]) -> List[Tuple[int, int]]:
    """``pages`` as ascending half-open runs ``[first, end)``."""
    runs: List[Tuple[int, int]] = []
    for page in sorted(pages):
        if runs and runs[-1][1] == page:
            runs[-1] = (runs[-1][0], page + 1)
        else:
            runs.append((page, page + 1))
    return runs


def _emit_all_final_stats(reason: str) -> None:
    for inst in list(_STATS_INSTANCES):
        try:
            inst._emit_stats_final(reason=reason)
        except Exception:
            pass


def _signal_handler(signum, frame):
    try:
        sig_name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        sig_name = str(signum)
    _emit_all_final_stats(reason=sig_name)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_signal_handlers_once() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    _SIGNAL_HANDLERS_INSTALLED = True
    # Only override the default handler (the scheduler subprocess installs none).
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
            if prev in (signal.SIG_DFL, signal.SIG_IGN, None):
                signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            # Raises off the main thread -- skip.
            pass


_T = TypeVar("_T")


class _CapacityField(Generic[_T]):
    """Data descriptor for a capacity-bearing allocator field.

    Every rebind bumps the owner's ``_capacity_epoch``, so the epoch-keyed
    capacity memos invalidate by construction. Contract: these fields are
    REBOUND, never mutated in place.
    """

    __slots__ = ("_name",)

    def __set_name__(self, owner, name: str) -> None:
        self._name = name

    def __get__(self, obj, objtype=None) -> _T:
        if obj is None:
            return self  # type: ignore[return-value]
        try:
            return obj.__dict__[self._name]
        except KeyError:
            raise AttributeError(self._name) from None

    def __set__(self, obj, value: _T) -> None:
        obj.__dict__[self._name] = value
        obj._capacity_epoch += 1


def _float_open_short_side(flt, demand) -> None:
    """Float relocation policy, driven by a demand vector: pages per band, zero
    for bands the operation does not touch. A grow-down end faces the float's
    HIGH side, a grow-up end its LOW side; `make_room`'s `min_bytes` is a TARGET
    for the whole band, so the ask is a total, never a delta.
    """
    if flt is None or flt._is_frontier_transparent():
        return  # no float involved / empty float never blocks
    if not any(pages > 0 for pages in demand.values()):
        return  # nothing demanded -- nothing to open (also keeps slack's max() total)
    for band_alloc, pages in demand.items():
        if pages <= 0:
            continue
        index_room = (
            band_alloc.num_pages
            - band_alloc.min_page_index
            - band_alloc._allocated_pages()
        )
        if pages > index_room:
            return  # index space binds; bytes cannot fix this
    sides = {"low": 0, "high": 0}
    for band_alloc, pages in demand.items():
        if band_alloc is flt or pages <= 0:
            continue
        holes = len(band_alloc._free_phys_pages) if band_alloc.lazy_compaction else 0
        ext = max(0, pages - holes)
        above = flt.high_peer
        while above is not None and above is not band_alloc:
            above = above.high_peer
        side = "high" if above is band_alloc else "low"
        sides[side] += ext * band_alloc.entry_bytes_per_page
    band = {
        "low": max(
            0, flt._byte_low_frontier() - flt._chain_high_frontier_below_bytes()
        ),
        "high": max(
            0, flt._chain_low_frontier_above_bytes() - flt._byte_high_frontier()
        ),
    }
    surplus = {side: band[side] - sides[side] for side in ("low", "high")}
    f_pages = demand.get(flt, 0)
    f_bytes = max(0, f_pages - flt._hole_pages()) * flt.entry_bytes_per_page

    def open_side(side, raw_gap):
        # The donor demand is measured from its raw frontier; make_room takes
        # a gap measured from the float's aligned region boundary.
        epp = flt.entry_bytes_per_page
        lo, hi = flt._region_bounds_pages()
        if side == "low":
            frontier = flt._chain_high_frontier_below_bytes()
            target = -(-(frontier + raw_gap) // epp)
            gap = (target - lo) * epp
        else:
            frontier = flt._chain_low_frontier_above_bytes()
            target = (frontier - raw_gap) // epp
            gap = (hi - target) * epp
        flt.make_room(side=side, min_bytes=max(0, gap))

    if surplus["low"] < 0 and surplus["high"] < 0:
        # One-sided relief does not jointly compact/reposition interior holes
        # when both peer gaps are deficient. This is not an impossibility proof.
        return
    if surplus["low"] < 0 or surplus["high"] < 0:
        short, far = ("low", "high") if surplus["low"] < 0 else ("high", "low")
        target = sides[short] + max(0, f_bytes - max(0, surplus[far]))
        if target > band[short]:
            open_side(short, target)
        return
    if f_bytes > max(surplus.values()):
        short = "low" if surplus["low"] >= surplus["high"] else "high"
        open_side(short, sides[short] + f_bytes)


def relieve_for_alloc(short_pool, need_tokens: int) -> bool:
    """THE shortfall ladder: every allocation shortfall in the unified pool runs
    exactly this, whether a single band's own alloc or the chain allocator's coupled
    multi-band alloc. The urgent flush is called unconditionally -- an eager END no-ops
    and a FLOAT always has boundary absorption to do -- so the ladder never
    branches on lazy mode, member kind, or layout.
    """
    for m in short_pool._flush_targets():
        m.flush_for_allocation()
    if need_tokens <= short_pool.available_size():
        return True
    short_pool._ask_float_for_room(need_tokens)
    return need_tokens <= short_pool.available_size()


def _drains_on_urgent_flush(neighbor) -> bool:
    """Whether an urgent flush of ``neighbor`` would release its holes; a blocked
    neighbor cannot, so its holes earn no credit."""
    return (
        neighbor is not None
        and neighbor.lazy_compaction
        and not neighbor.moves_blocked()
    )


def flush_deferred_free_group(
    allocator,
    pending_groups: Sequence[Optional[Sequence[torch.Tensor]]],
) -> None:
    """Apply queued frees and reopen the caller's free-group scope."""
    if allocator.free_group is None or not any(pending_groups):
        return
    allocator.free_group_end()
    allocator.free_group_begin()


def full_tokens_before_mamba_recheck(
    full_allocator: Band,
    mamba_allocator: Band,
    target_size: int,
) -> int:
    """Conservative Full-token lower bound for the next Mamba capacity check.

    The current Mamba slot count can hide at most one slot minus one byte of
    residual room. Subtract that possible residue so this estimate only skips
    checks that cannot succeed from Full bytes alone. Allocator capacity remains
    the stop condition after the bound is crossed.
    """
    missing_slots = max(0, target_size - mamba_allocator.schedulable_available_size())
    if missing_slots == 0:
        return 0
    mamba_page_bytes = mamba_allocator.entry_bytes_per_page
    minimum_missing_bytes = (missing_slots - 1) * mamba_page_bytes + 1
    dcp_size = get_parallel().attn_dcp_size if full_allocator.shards_under_dcp else 1
    return -(-minimum_missing_bytes * dcp_size // full_allocator.entry_bytes)


class Band:
    """One sub-pool's units in a `KVArena`: an end of the chain.

    ``need_sort`` applies to transfer-facing physical ids, not virtual ids.
    Physical free pages are sorted during compaction, and allocation hands out
    the free pages nearest the pool's own end first (`take_physical`).
    """

    # Capacity-bearing state: any rebind bumps `_capacity_epoch`, invalidating
    # the chain's capacity memos (see `_CapacityField`).
    _capacity_epoch: int = 0
    watermark_physical: _CapacityField[int] = _CapacityField()
    live_page_count: _CapacityField[int] = _CapacityField()
    # Behind the `_free_phys_pages` property, which also tracks its host copy.
    _free_pages: _CapacityField[torch.Tensor] = _CapacityField()
    # The owner's unminted virtual pages (None on a band that binds the owner's).
    free_virtual_ids: _CapacityField[Optional[torch.Tensor]] = _CapacityField()

    def __init__(
        self,
        *,
        arena: KVArena,
        sub_pool_name: str,
        device: str,
        is_id_owner: bool,
        virtual_num_pages: Optional[int] = None,
        page_size: int = 1,
        shards_under_dcp: bool = False,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):
        band = arena.band(sub_pool_name)
        max_slots = arena.num_entries(sub_pool_name)
        # DCP shards KV tokens only. Mamba state and the SWA rows are
        # replicated, so they stay slot-granular whatever the process width is.
        self.shards_under_dcp = shards_under_dcp
        dcp_size = get_parallel().attn_dcp_size if shards_under_dcp else 1
        self.size = max_slots * dcp_size
        self.device = device
        self.need_sort = need_sort
        # None: free right away. A list: hold frees until free_group_end().
        self.free_group: Optional[List[torch.Tensor]] = None
        self.arena = arena
        self.sub_pool_name = sub_pool_name
        self.max_slots = max_slots
        self.grow_direction = band.grow_direction
        self.entry_bytes = band.entry_bytes
        self.min_slot_index = arena.first_entry(sub_pool_name)
        self.is_id_owner = is_id_owner
        # Zero page envelopes on hand-out -- see _maybe_zero_pages.
        self._zero_pages_on_alloc = band.zero_on_handout
        # Overlap mode: `free` drops a wait_stream(forward_stream) barrier so its
        # v2p writes + move kernel serialize after the in-flight forward.
        self.forward_stream = forward_stream

        # --- Page-aware bookkeeping ---
        # Two page sizes, equal unless decode context parallelism is on:
        # `page_size` is VIRTUAL (the scheduler, the tree cache and the alloc/free
        # surface), `pool_page_size` is the PHYSICAL rows one page occupies here.
        # `RunnerIds.translate_dcp_read_ids` collapses `loc // dcp_size`
        # before reaching `UnitIds.translate`, so everything at or below the v2p
        # table -- byte budget, compaction moves, translate -- stays on
        # `pool_page_size`.
        self.pool_page_size = page_size
        assert band.unit_entries == page_size, (
            f"band {sub_pool_name!r} moves units of {band.unit_entries} entries, "
            f"but its allocator pages {page_size}"
        )
        self.page_size = page_size * dcp_size
        self.num_pages = max_slots // self.pool_page_size
        # `min_page_index` = ceil(min_slot_index / pool_page_size), keeping the
        # padding sink covered (see `KVStoragePlan.sink_bytes`).
        self.min_page_index = (
            self.min_slot_index + self.pool_page_size - 1
        ) // self.pool_page_size
        self.entry_bytes_per_page = self.entry_bytes * self.pool_page_size

        # v2p is indexed by VIRTUAL page id, p2v by PHYSICAL page id. A non-owner
        # consumes the owner's ids, so the two counts are unrelated.
        assert virtual_num_pages is None or not is_id_owner, (
            "only a non-owner allocator may use another pool's virtual-id space"
        )
        self.ids = UnitIds(
            num_virtual_ids=(
                self.num_pages if virtual_num_pages is None else virtual_num_pages
            ),
            num_physical_units=self.num_pages,
            unit_tokens=self.pool_page_size,
            shards_under_dcp=shards_under_dcp,
            device=device,
        )

        # Chain neighbours: `low_peer` toward byte 0, `high_peer` toward
        # `total_bytes`. Ends have one (`bind_peer`), float middles have both.
        self.low_peer: Optional[Band] = None
        self.high_peer: Optional[Band] = None

        # Inverse history of relocations (spec rollback), at PAGE granularity.
        self._inverse_history: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

        # --- Lazy compaction state (all unused when lazy_compaction=False) ---
        # Pending pages (`usage.pending_pages`): compaction srcs whose remap
        #   completed but whose readers haven't retired -- reusing one races a READ.
        # `live_page_count`: CPU slot-conservation counter, invariant under compaction.
        # KV copy and v2p/p2v remap both run on `schedule_stream`, so single-stream
        # ordering serializes them -- no separate copy-done event needed.
        self.lazy_compaction = lazy_compaction
        # The host copy of `_free_phys_pages`, or None when unknown; see the
        # property below.
        self._free_pages_host: Optional[List[int]] = None
        self._free_phys_pages: torch.Tensor = torch.empty(
            0, dtype=torch.int64, device=device
        )
        # The `_free_phys_pages` tensor last known to be ascending. Frees and
        # pending-reuse returns rebind the list unsorted, so identity tells
        # `_sorted_free_phys_pages` whether a sort is due.
        self._free_phys_sorted: Optional[torch.Tensor] = None
        # Who may still touch this band's pages: unfinished forwards and
        # transfers, and pages a move vacated while readers could still read
        # them (`PageUsage`).
        self.usage = PageUsage()
        # Cumulative observability counters (NOT reset at clear()).
        self._stats_n_free_lazy: int = 0
        self._stats_n_release_batch: int = 0
        self._stats_n_drain_calls: int = 0
        self._stats_n_drain_did_work: int = 0
        self._stats_n_drained_pages_total: int = 0
        self._stats_n_flush_calls: int = 0
        self._stats_n_flush_did_work: int = 0
        self._stats_n_flush_moves: int = 0
        self._stats_n_pages_absorbed: int = 0
        self._stats_peak_free_list_len: int = 0
        self._stats_peak_pending_pages: int = 0
        self._stats_n_emits: int = 0
        self._stats_last_emit_ts: float = _time_mod.monotonic()
        self._stats_final_emitted: bool = False
        if _LAZY_COMPACTION_STATS_ENABLED:
            atexit.register(self._emit_stats_final, reason="atexit")
            _STATS_INSTANCES.add(self)
            _install_signal_handlers_once()
        self.live_page_count = 0

        # Per-call move cap on NON-urgent `_flush`: bounds work per `on_idle()` so
        # a large backlog doesn't block ZMQ IPC. Urgent retries are uncapped.
        self._lazy_max_moves_per_call = (
            envs.SGLANG_LAZY_COMPACTION_MAX_MOVES_PER_CALL.get()
        )
        # A move-kernel launch copies at most this many pages; `_move_pages`
        # issues a larger move as consecutive launches within the same call
        # (see there). Bounded by the pool, which never has more pages.
        max_pages_per_move = envs.SGLANG_COMPACTION_MAX_PAGES_PER_MOVE.get()
        assert max_pages_per_move > 0, (
            "SGLANG_COMPACTION_MAX_PAGES_PER_MOVE must be positive, got "
            f"{max_pages_per_move}"
        )
        self._move_max_pages = min(max_pages_per_move, self.num_pages)

        # Epoch-keyed memos for the capacity views: pure between mutations, but
        # schedulers read them O(queue) times per step.
        self._avail_memo_epoch: Optional[int] = None
        self._avail_memo_tokens: int = 0
        self._sched_avail_memo_key: Optional[tuple] = None
        self._sched_avail_memo_tokens: int = 0
        self._credit_neighbors_memo: Optional[tuple] = None

        self.clear()

        logger.info(
            "[unified-memory-pool] Band(%r) ready: grow=%s, max_slots=%d, "
            "min_slot_index=%d, page_size=%d, num_pages=%d, min_page_index=%d, "
            "entry_bytes=%d, entry_bytes_per_page=%d, is_id_owner=%s, "
            "initial_watermark_page=%d, allocatable_pages=%d",
            self.sub_pool_name,
            self.grow_direction,
            self.max_slots,
            self.min_slot_index,
            self.page_size,
            self.num_pages,
            self.min_page_index,
            self.entry_bytes,
            self.entry_bytes_per_page,
            self.is_id_owner,
            self.watermark_physical,
            self.num_pages - self.min_page_index,
        )

    # -- chain-neighbor binding --

    def bind_peer(self, peer: Band) -> None:
        """Bind the OTHER end as this end's growth-side neighbor: a grow-up
        pool's neighbor sits above it, a grow-down pool's below.
        """
        assert self.grow_direction in ("up", "down") and peer.grow_direction in (
            "up",
            "down",
        ), (
            f"bind_peer is END-pool-only; got {self.sub_pool_name!r} "
            f"({self.grow_direction}) <-> {peer.sub_pool_name!r} "
            f"({peer.grow_direction}); wire floats via bind_low_peer/bind_high_peer"
        )
        if self.grow_direction == "up":
            self.high_peer = peer
        else:
            self.low_peer = peer
        self._capacity_epoch += 1  # rewiring changes what the chain walks see

    def bind_low_peer(self, peer: Band) -> None:
        self.low_peer = peer
        self._capacity_epoch += 1  # rewiring changes what the chain walks see

    def bind_high_peer(self, peer: Band) -> None:
        self.high_peer = peer
        self._capacity_epoch += 1  # rewiring changes what the chain walks see

    # -- state --

    def _reset_watermarks(self) -> None:
        """Reset frontier state to empty (float middles override)."""
        if self.grow_direction == "up":
            self.watermark_physical = self.min_page_index
        else:
            self.watermark_physical = self.num_pages - 1

    def clear(self) -> None:
        """Reset to initial state. Pages in `[0, min_page_index)` are reserved."""
        self._reset_watermarks()
        self.ids.reset()
        if self.is_id_owner:
            self.free_virtual_ids = torch.arange(
                self.min_page_index,
                self.num_pages,
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.free_virtual_ids = None
        self.free_group = None
        # Segment frees buffer page REPRESENTATIVES, not whole token ranges:
        # `torch.cat` of the ranges destroys the shape the stride derivation needs.
        self.free_page_reps_group: Optional[List[torch.Tensor]] = None
        self._inverse_history.clear()
        self._set_free_pages(
            torch.empty(0, dtype=torch.int64, device=self.device), host=[]
        )
        self.live_page_count = 0
        self.usage.reset()
        self._capacity_epoch += 1  # fixed pages are capacity-bearing

    def clear_inverse_history(self) -> None:
        self._inverse_history.clear()

    # -- the free list on the host, and ordering after forwards --

    @property
    def _free_phys_pages(self) -> torch.Tensor:
        """Free physical pages inside the band, the list allocation and the
        flush take from."""
        return self._free_pages

    @_free_phys_pages.setter
    def _free_phys_pages(self, pages: torch.Tensor) -> None:
        # A list rebuilt on the device is no longer known on the host.
        self._free_pages = pages
        self._free_pages_host = None

    def _set_free_pages(
        self, pages: torch.Tensor, *, host: Optional[List[int]]
    ) -> None:
        """Rebind the free list together with its host copy, in the same order."""
        self._free_pages = pages
        self._free_pages_host = host

    def _free_pages_snapshot(self) -> List[int]:
        """The free list on the host. Reads the device only when the host copy
        is unknown; callers must not mutate the returned list."""
        if self._free_pages_host is None:
            self._free_pages_host = self._free_pages.tolist()
        return self._free_pages_host

    def _to_device(self, host_ids: List[int]) -> torch.Tensor:
        """Page ids planned on the host, as a device tensor, through pinned
        memory and without blocking: a pageable copy first synchronizes the
        stream, which after a forward wait parks the host until that forward
        ends."""
        pinned = torch.device(self.device).type == "cuda"
        return torch.tensor(host_ids, dtype=torch.int64, pin_memory=pinned).to(
            self.device, non_blocking=True
        )

    def _chain_bands(self) -> List[Band]:
        """This band and every chain neighbour reachable through the peers."""
        bands = [self]
        for side in ("low_peer", "high_peer"):
            band = getattr(self, side)
            while band is not None and all(band is not b for b in bands):
                bands.append(band)
                band = getattr(band, side)
        return bands

    def _wait_on_stream(self, event) -> None:
        """Order the current stream's later work after ``event`` (an event, a
        completion flag, or a tuple of them), without blocking the host. A host
        read issued after such a wait parks the host until the event fires, so
        every chain band's free list is read first. Whether the event has fired
        decides only whether a wait is needed, never a layout."""
        if signal_fired(event):
            return
        for band in self._chain_bands():
            band._free_pages_snapshot()
        wait_signals(event)

    def _retired(self, event) -> bool:
        return self.usage.retired(event)

    def retire_forward(self, event) -> None:
        """The scheduler processed this forward's result: it, and every forward
        recorded before it, retires."""
        self.usage.retire_forward(event)

    def _retire_through(self, event) -> None:
        self.usage.retire_through(event)

    # -- size reporting --

    max_live_tokens: Optional[int] = None

    def cap_available(self, tokens: int) -> int:
        if self.max_live_tokens is None:
            return tokens
        return min(tokens, max(0, self.max_live_tokens - self.allocated_count()))

    def _allocated_pages(self) -> int:
        """Number of allocated PAGES (TOKEN callers use `allocated_count()`)."""
        if self.grow_direction == "up":
            return max(0, self.watermark_physical - self.min_page_index)
        return max(0, self.num_pages - 1 - self.watermark_physical)

    def allocated_count(self) -> int:
        """LIVE allocated TOKENS (excludes lazy holes / pending).

        Lazy mode uses `live_page_count`: the watermark span over-counts because
        holes and pending pages sit inside it but aren't live.
        """
        if self.lazy_compaction:
            return self.live_page_count * self.page_size
        return self._allocated_pages() * self.page_size

    def is_slot_allocated(self, slot: int) -> bool:
        """Whether the PAGE containing this virtual id is in use."""
        return self.ids.is_bound(slot // self.page_size)

    def allocator_state_str(self) -> str:
        return (
            f"sub_pool={self.sub_pool_name!r}, grow_direction={self.grow_direction}, "
            f"is_id_owner={self.is_id_owner}, page_size={self.page_size}, "
            f"min_page_index={self.min_page_index}, "
            f"num_pages={self.num_pages}, "
            f"num_virtual_ids={self.ids.num_virtual_ids}, "
            f"watermark_physical={self.watermark_physical}, "
            f"allocated_pages={self._allocated_pages()}"
        )

    def _byte_high_frontier(self) -> int:
        """Byte just past this side's last-allocated page (grow-up) / buffer top (grow-down)."""
        if self.grow_direction == "up":
            return self.watermark_physical * self.entry_bytes_per_page
        return self.num_pages * self.entry_bytes_per_page

    def _byte_accounting_violations(self) -> List[str]:
        """Per-sub-pool conservation strings; empty == healthy. Idle-time
        diagnostic -- pure host arithmetic."""
        out: List[str] = []
        total = self.arena.total_bytes
        lo_b, hi_b = self._byte_low_frontier(), self._byte_high_frontier()
        if not (0 <= lo_b <= hi_b <= total):
            out.append(
                f"[{self.sub_pool_name}] frontier out of bounds: "
                f"low={lo_b}, high={hi_b}, total={total}"
            )
        if self.lazy_compaction:
            # Lazy end: the watermark span contains live + holes + pending
            # (eager has no holes/pending -- span == live by construction).
            holes = int(self._free_phys_pages.numel())
            pending = len(self.usage.pending_pages)
            wm_span = self._allocated_pages()
            if wm_span != self.live_page_count + holes + pending:
                out.append(
                    f"[{self.sub_pool_name}] span {wm_span} != live "
                    f"{self.live_page_count} + holes {holes} + pending {pending}"
                )
        out.extend(self._capacity_memo_violations())
        return out

    def _capacity_memo_violations(self) -> List[str]:
        """Memo-coherence check: divergence from a fresh recompute means a
        mutation bypassed `_CapacityField` (an in-place write). Empty == healthy."""
        out: List[str] = []
        epoch = self._chain_capacity_epoch()
        if self._avail_memo_epoch == epoch:
            actual = self._available_tokens()
            if self._avail_memo_tokens != actual:
                out.append(
                    f"[{self.sub_pool_name}] stale available_size memo: "
                    f"cached={self._avail_memo_tokens}, actual={actual}"
                )
        if self._sched_avail_memo_key == self._schedulable_capacity_key():
            actual = self._available_tokens(
                extra_gap_bytes=self._peer_drainable_hole_bytes()
            )
            if self._sched_avail_memo_tokens != actual:
                out.append(
                    f"[{self.sub_pool_name}] stale schedulable_available_size "
                    f"memo: cached={self._sched_avail_memo_tokens}, "
                    f"actual={actual}"
                )
        return out

    def _byte_low_frontier(self) -> int:
        """Byte starting this side's allocatable range (grow-up) / just below its lowest live page (grow-down)."""
        if self.grow_direction == "up":
            return self.min_page_index * self.entry_bytes_per_page
        return (self.watermark_physical + 1) * self.entry_bytes_per_page

    # -- chain frontier walk --

    def _is_frontier_transparent(self) -> bool:
        """Whether neighbors' frontier walks may see THROUGH this pool. End pools
        are always opaque; an empty float middle overrides to transparent."""
        return False

    def _chain_low_frontier_above_bytes(self) -> int:
        """Byte low-frontier of the nearest NON-transparent chain member above
        this pool; the buffer top if none."""
        p = self.high_peer
        while p is not None and p._is_frontier_transparent():
            p = p.high_peer
        if p is None:
            return self.arena.total_bytes
        return p._byte_low_frontier()

    def _chain_high_frontier_below_bytes(self) -> int:
        """Byte high-frontier of the nearest NON-transparent chain member below
        this pool; 0 if none."""
        p = self.low_peer
        while p is not None and p._is_frontier_transparent():
            p = p.low_peer
        if p is None:
            return 0
        return p._byte_high_frontier()

    def _chain_capacity_epoch(self) -> int:
        """Sum of `_capacity_epoch` over the whole chain (self included). Capacity
        views read chain-neighbor frontiers, so a memo stays valid only while
        EVERY member is unmutated; the sum moves whenever any member does.
        """
        total = self._capacity_epoch
        p = self.low_peer
        while p is not None:
            total += p._capacity_epoch
            p = p.low_peer
        p = self.high_peer
        while p is not None:
            total += p._capacity_epoch
            p = p.high_peer
        return total

    def _growth_side_neighbor(self) -> Optional[Band]:
        """Nearest NON-transparent chain member on this pool's GROWTH side -- the
        one whose compaction releases bytes reachable at this pool's frontier."""
        p = self.high_peer if self.grow_direction == "up" else self.low_peer
        while p is not None and p._is_frontier_transparent():
            p = p.high_peer if self.grow_direction == "up" else p.low_peer
        return p

    def _current_gap_bytes(self) -> int:
        """Free byte band between this side's frontier and the nearest
        non-transparent chain frontier (2-pool: the peer's, byte-identical)."""
        if self.grow_direction == "up":
            return max(
                0, self._chain_low_frontier_above_bytes() - self._byte_high_frontier()
            )
        return max(
            0, self._byte_low_frontier() - self._chain_high_frontier_below_bytes()
        )

    def _available_tokens(self, extra_gap_bytes: int = 0) -> int:
        """Tokens allocatable given `extra_gap_bytes` of ADDED gap room
        (0 == current realizable; >0 == post-peer-compaction). Own index headroom
        is unaffected by `extra_gap_bytes`: peer bytes add no page indices here.
        """
        gap_bytes = self._current_gap_bytes() + extra_gap_bytes
        pages_by_bytes = gap_bytes // self.entry_bytes_per_page
        pages_by_index_space = (
            self.num_pages - self.min_page_index - self._allocated_pages()
        )
        pages_extend = min(pages_by_bytes, pages_by_index_space)
        # Lazy: drainable holes don't consume new bytes.
        pages_drain = len(self._free_phys_pages) if self.lazy_compaction else 0
        return (pages_extend + pages_drain) * self.page_size

    def available_size(self) -> int:
        """Tokens allocatable RIGHT NOW (no peer compaction).

        Alloc shortfall gates consult this, so it MUST NOT fold in peer holes; use
        `schedulable_available_size()` for that. Memoized on the chain epoch.
        """
        epoch = self._chain_capacity_epoch()
        if self._avail_memo_epoch != epoch:
            self._avail_memo_tokens = self._available_tokens()
            self._avail_memo_epoch = epoch
        return self.cap_available(self._avail_memo_tokens)

    def _peer_drainable_hole_bytes(self) -> int:
        """Gap bytes an urgent flush of the growth-side chain neighbor would
        release. Only `_free_phys_pages` counts -- NOT the pending pages, which
        awaits an event -- so the credit is realizable.
        """
        neighbor = self._growth_side_neighbor()
        if not _drains_on_urgent_flush(neighbor):
            return 0
        return len(neighbor._free_phys_pages) * neighbor.entry_bytes_per_page

    def moves_blocked(self) -> bool:
        """Whether a fixed page forbids relocating pages, for the paths that
        cannot move around one."""
        return bool(self.usage.fixed)

    def fix_pages(self, key: Hashable, pages: Sequence[int]) -> None:
        """Keep these physical pages where they are, under ``key`` (added to
        those already fixed under it). Moves go around them; a plan keeps them
        inside this band."""
        if not pages:
            return
        self.usage.fix(key, pages)
        self._capacity_epoch += 1

    def unfix_pages(self, key: Hashable, pages: Optional[Sequence[int]] = None) -> None:
        """Release the pages fixed under ``key``: every one, or ``pages``."""
        if key not in self.usage.fixed:
            return
        self.usage.unfix(key, pages)
        self._capacity_epoch += 1

    def fixed_pages(self) -> Set[int]:
        return self.usage.fixed_pages()

    def fixed_extent(self) -> Optional[Tuple[int, int]]:
        """Pages ``[first, end)`` spanning every fixed page, or None."""
        pages = self.fixed_pages()
        return (min(pages), max(pages) + 1) if pages else None

    def _credit_neighbors(self) -> tuple:
        """Chain members whose holes `schedulable_available_size` credits. Which
        members these are depends only on epoch-tracked state."""
        return (self._growth_side_neighbor(),)

    def _schedulable_capacity_key(self) -> tuple:
        """Chain epoch plus each credited neighbor's drain state: gates flip with
        no allocator write, so the epoch alone misses them."""
        epoch = self._chain_capacity_epoch()
        m = self._credit_neighbors_memo
        if m is None or m[0] != epoch:
            m = self._credit_neighbors_memo = (epoch, self._credit_neighbors())
        (neighbor,) = m[1]
        return epoch, _drains_on_urgent_flush(neighbor)

    def _schedulable_physical_tokens(self) -> int:
        """Physical room after neighbor flush, before the owner token quota."""
        key = self._schedulable_capacity_key()
        if self._sched_avail_memo_key != key:
            self._sched_avail_memo_tokens = self._available_tokens(
                extra_gap_bytes=self._peer_drainable_hole_bytes()
            )
            self._sched_avail_memo_key = key
        return self._sched_avail_memo_tokens

    def schedulable_available_size(self) -> int:
        """Tokens allocatable after flushing a neighbor, including reclaimable holes.

        Allocation checks use available_size(). Cache by capacity and gate state.
        """
        return self.cap_available(self._schedulable_physical_tokens())

    def _flush_targets(self):
        """A band short on its OWN alloc asks only its growth-side neighbour to
        flush. Never itself: own compaction trades one hole for one gap byte (net
        zero); only a NEIGHBOUR's compaction releases into the shared gap.
        """
        neighbor = self._growth_side_neighbor()
        return () if neighbor is None else (neighbor,)

    def _ask_float_for_room(self, need_tokens: int) -> None:
        """A band short on its OWN pages asks the growth-side member, if it is a
        float, to open the side facing it; the policy is `_float_open_short_side`."""
        blocker = self._growth_side_neighbor()
        if not isinstance(blocker, FloatBand):
            return
        _float_open_short_side(blocker, {self: -(-need_tokens // self.page_size)})

    # -- physical-slot / physical-page primitives --

    _physical_take_scope = None

    def take_physical(self, need_size: int) -> Optional[torch.Tensor]:
        """Reserve `need_size` TOKENS (multiple of page_size), returning backing
        physical PAGE ids, or `None` on shortfall.
        """
        with record_function("Band.take_physical"):
            if need_size > self.cap_available(need_size):
                return None
            if need_size <= 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)
            assert need_size % self.page_size == 0, (
                f"take_physical: need_size={need_size} must be a multiple of "
                f"page_size={self.page_size}"
            )
            num_pages = need_size // self.page_size

            if not self.lazy_compaction:
                return self._take_physical_eager(num_pages)

            # Lazy: slice the GPU free list (no D2H).
            n_drain = min(num_pages, int(self._free_phys_pages.shape[0]))
            need_more = num_pages - n_drain

            # Extend first (state untouched on failure), then drain holes.
            if need_more > 0:
                if not self._extend_watermark(need_more):
                    return None

            if n_drain > 0:
                drained_t = self._take_free_pages_from_end(
                    n_drain, "low" if self.grow_direction == "up" else "high"
                )
            else:
                drained_t = None

            self.live_page_count += num_pages

            if drained_t is None:
                return self._take_physical_arange(num_pages)

            # Pure drain -- clone off the free-list view so rebindings don't pin it.
            if need_more == 0:
                return drained_t.clone()

            # Mixed: drained holes ++ extended pages (`bind` is order-agnostic).
            if self.grow_direction == "up":
                new_wm = self.watermark_physical
                extended_t = torch.arange(
                    new_wm - need_more,
                    new_wm,
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                new_wm = self.watermark_physical
                extended_t = torch.arange(
                    new_wm + need_more,
                    new_wm,
                    -1,
                    dtype=torch.int64,
                    device=self.device,
                )
            return torch.cat([drained_t, extended_t])

    def _take_physical_eager(self, num_pages: int) -> Optional[torch.Tensor]:
        """Eager-mode take_physical — contiguous range."""
        if self.grow_direction == "up":
            start = self.watermark_physical
            end_exclusive = start + num_pages
            if end_exclusive > self.num_pages:
                return None
            self._wait_fences(start, end_exclusive)
            phys_pages = torch.arange(
                start, end_exclusive, dtype=torch.int64, device=self.device
            )
            self.watermark_physical = end_exclusive
            return phys_pages
        else:
            end = self.watermark_physical
            start = end - num_pages + 1
            if start < self.min_page_index:
                return None
            self._wait_fences(start, end + 1)
            phys_pages = torch.arange(
                start, end + 1, dtype=torch.int64, device=self.device
            )
            self.watermark_physical -= num_pages
            return phys_pages

    def _extend_watermark(self, num_pages: int) -> bool:
        """Advance the watermark by `num_pages`. Returns False on index-space
        overflow OR crossing the nearest non-transparent chain frontier.
        """
        if self.grow_direction == "up":
            new_wm = self.watermark_physical + num_pages
            if new_wm > self.num_pages:
                return False
            # The chain above; don't extend past its low frontier.
            chain_low_pages = (
                self._chain_low_frontier_above_bytes() // self.entry_bytes_per_page
            )
            if new_wm > chain_low_pages:
                return False
            self._wait_fences(self.watermark_physical, new_wm)
            self.watermark_physical = new_wm
        else:
            new_wm = self.watermark_physical - num_pages
            if new_wm < self.min_page_index - 1:
                return False
            # Backstop only: callers gate on `available_size()`, whose floor'd gap
            # already guarantees the extension fits.
            chain_high_pages = (
                self._chain_high_frontier_below_bytes() // self.entry_bytes_per_page
            )
            if new_wm + 1 < chain_high_pages:
                return False
            self._wait_fences(new_wm + 1, self.watermark_physical + 1)
            self.watermark_physical = new_wm
        return True

    def _take_physical_arange(self, num_pages: int) -> torch.Tensor:
        """Contiguous arange for an already-applied watermark extension."""
        if self.grow_direction == "up":
            return torch.arange(
                self.watermark_physical - num_pages,
                self.watermark_physical,
                dtype=torch.int64,
                device=self.device,
            )
        return torch.arange(
            self.watermark_physical + 1,
            self.watermark_physical + num_pages + 1,
            dtype=torch.int64,
            device=self.device,
        )

    def take_physical_pages(self, num_pages: int) -> Optional[torch.Tensor]:
        """Page-granular wrapper around ``take_physical``."""
        with record_function("Band.take_physical_pages"):
            scope = self._physical_take_scope
            if scope is not None and not scope.applying:
                return scope.take(self, num_pages)
            if num_pages * self.page_size > self.cap_available(
                num_pages * self.page_size
            ):
                return None
            return self.take_physical(num_pages * self.page_size)

    def take_planned_physical_pages(self, take) -> Optional[torch.Tensor]:
        """Apply a no-relief take only while its source geometry still holds."""
        from sglang.srt.mem_cache.kv.alloc.pricing import snapshot_band

        current = snapshot_band(self)
        if (
            current.low_byte != take.low_byte
            or current.high_byte != take.high_byte
            or current.holes != take.holes
            or current.live != take.live
        ):
            return None
        if take.fresh is not None:
            start, end = take.fresh
            if (
                start * self.entry_bytes_per_page
                < self._chain_high_frontier_below_bytes()
                or end * self.entry_bytes_per_page
                > self._chain_low_frontier_above_bytes()
            ):
                return None
        if self.grow_direction == "float":
            return self.take_physical_pages(take.pages, placement=take.fresh)
        return self.take_physical_pages(take.pages)

    def _sorted_free_phys_pages(self) -> torch.Tensor:
        """`_free_phys_pages` in ascending order. Sorts on the device, and only
        when the list was rebound since it was last known sorted; there is no host
        sync, because the length is tensor metadata.
        """
        free = self._free_phys_pages
        if free is not self._free_phys_sorted:
            if free.numel() > 1:
                free, _ = torch.sort(free)
                self._free_phys_pages = free
            self._free_phys_sorted = free
        return free

    def _take_free_pages_from_end(self, n: int, end: str) -> torch.Tensor:
        """Remove and return ``n`` free pages from the ``end`` ("low" or "high")
        of the ascending free list; the rest stays ascending. Taking the whole
        list needs no sort. ``0 < n <= len(_free_phys_pages)``.
        """
        n_free = int(self._free_phys_pages.shape[0])
        assert 0 < n <= n_free, f"take {n} of {n_free} free pages"
        host = self._free_pages_host
        if n == n_free:
            taken = self._free_phys_pages
            rest, rest_host = taken[:0], []
        else:
            free = self._sorted_free_phys_pages()
            host = None if host is None else sorted(host)
            if end == "low":
                taken, rest = free[:n], free[n:]
                rest_host = None if host is None else host[n:]
            else:
                taken, rest = free[n_free - n :], free[: n_free - n]
                rest_host = None if host is None else host[: n_free - n]
        # The host copy stays known, in the device list's order.
        self._set_free_pages(rest, host=rest_host)
        self._free_phys_sorted = rest
        return taken

    # -- fused take_physical_pages + bind --

    def place_pages(self, v_pages: torch.Tensor, N: int) -> Optional[torch.Tensor]:
        """Fuse `take_physical_pages` + `bind` into ONE Triton kernel when no
        holes need draining; fall through to the slow path (drains holes first)
        when holes exist. Returns physical page ids [N], or None on shortfall.
        """
        with record_function("Band.place_pages"):
            if self._physical_take_scope is not None:
                physical = self.take_physical_pages(N)
                if physical is not None:
                    self.ids.bind(v_pages, physical)
                    self._maybe_zero_pages(physical)
                return physical
            if N * self.page_size > self.cap_available(N * self.page_size):
                return None
            if N == 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)

            # FAST PATH: eager, or lazy with no current holes.
            if not self.lazy_compaction or self._free_phys_pages.numel() == 0:
                start_wm = self.watermark_physical  # kernel's `start_phys`

                # Lazy uses `_extend_watermark` (index + peer checks); eager
                # inlines the index-only check to match `_take_physical_eager`.
                if self.lazy_compaction:
                    if not self._extend_watermark(N):
                        return None
                else:
                    if self.grow_direction == "up":
                        new_wm = start_wm + N
                        if new_wm > self.num_pages:
                            return None
                        self._wait_fences(start_wm, new_wm)
                        self.watermark_physical = new_wm
                    else:
                        new_wm = start_wm - N
                        if new_wm < self.min_page_index - 1:
                            return None
                        self._wait_fences(new_wm + 1, start_wm + 1)
                        self.watermark_physical = new_wm

                # Lowest physical id of the new range (both directions yield
                # ascending `[start_phys, start_phys + N)`).
                if self.grow_direction == "up":
                    start_phys = start_wm
                else:
                    start_phys = start_wm - N + 1

                phys_pages = alloc_bind_inplace(
                    v_pages,
                    self.ids.virtual_to_physical,
                    self.ids.physical_to_virtual,
                    start_phys,
                )

                if self.lazy_compaction:  # live_page_count tracked only in lazy mode
                    self.live_page_count += N
                self._maybe_zero_pages(phys_pages)
                return phys_pages

            # SLOW PATH: holes exist -- drain them first, then bind.
            phys_pages = self.take_physical_pages(N)
            if phys_pages is None:
                return None
            self.ids.bind(v_pages, phys_pages)
            self._maybe_zero_pages(phys_pages)
            return phys_pages

    def _maybe_zero_pages(self, phys_pages: torch.Tensor) -> None:
        """Zero the page ENVELOPES on hand-out (MHA and MLA pools): the MLA
        kernels and trtllm_mha's fmha_v2 prefill (and fp8 XQA decode) load whole
        pages and give the rows past seq_len a zero weight, so a NaN there still
        reaches the output. Never-written bytes must read as finite values; a
        mamba neighbour's fp32 state, read as bf16, may not.
        """
        if not self._zero_pages_on_alloc or phys_pages.numel() == 0:
            return
        with record_function("Band._maybe_zero_pages"):
            self.arena.zero_units(self.sub_pool_name, phys_pages)

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate `need_size` virtual TOKEN ids (id-owner only). Returns
        token-granular, page-structured ids, or None on shortfall.

        All allocator GPU ops run on `schedule_stream`; `alloc` needs no
        `wait_stream` barrier because its v2p/p2v writes are picked up by the
        forward via `forward_stream.wait_stream(schedule_stream)` in `run_batch`.
        """
        with record_function("Band.alloc"):
            assert self.is_id_owner, (
                f"Band({self.sub_pool_name!r}).alloc called on a "
                "non-id-owner allocator; use alloc_with_virtual instead"
            )
            if need_size <= 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)
            assert need_size % self.page_size == 0, (
                f"Band({self.sub_pool_name!r}).alloc: need_size="
                f"{need_size} must be a multiple of page_size={self.page_size}"
            )
            if need_size > self.available_size():
                # Shortfall: flush the PEER, not own -- see `_flush_targets`.
                if not relieve_for_alloc(self, need_size):
                    return None
            num_pages = need_size // self.page_size
            if num_pages > len(self.free_virtual_ids):
                return None
            v_pages = self.free_virtual_ids[:num_pages]
            self.free_virtual_ids = self.free_virtual_ids[num_pages:]
            phys_pages = self.place_pages(v_pages, num_pages)
            if phys_pages is None:
                self.free_virtual_ids = torch.cat([v_pages, self.free_virtual_ids])
                return None
            return self._expand_pages_to_tokens(v_pages)

    def alloc_with_virtual(self, virtual_pages: torch.Tensor) -> None:
        """Take physical PAGES for caller-supplied virtual PAGE ids (not token
        ids), for a physical-holding non-owner such as the SWA `swa` sub-allocator.
        The chain allocator snapshots the virtual pages before the id-owner consumes them.
        """
        with record_function("Band.alloc_with_virtual"):
            if virtual_pages.numel() == 0:
                return
            phys_pages = self.place_pages(virtual_pages, int(virtual_pages.numel()))
            assert phys_pages is not None, (
                f"Band({self.sub_pool_name!r}).alloc_with_virtual: out of "
                "physical room (the chain allocator's byte-budget check should have "
                "caught this)"
            )

    # -- free with eager compaction --

    def free(
        self, free_index: torch.Tensor, *, _pages: Optional[torch.Tensor] = None
    ) -> None:
        """Free virtual TOKEN ids: recover virtual PAGE ids, un-map v2p/p2v,
        (if id-owner) recycle the page ids, trigger eager compaction.

        `_pages` carries virtual PAGE ids the caller already derived; when given,
        the data-dependent dedup is skipped. `free_index` is token-granular and
        need not be page-aligned. EAGER drops one `wait_stream(forward_stream)`
        barrier so the v2p/p2v writes and the compaction move serialize with the
        in-flight forward; LAZY needs none (a freed `v` has no live reader) and
        defers compaction to `_flush`.
        """
        with record_function("Band.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if self.free_group is not None:
                self.free_group.append(self._copy_for_free_group(free_index))
                return
            if self.lazy_compaction:
                self._free_lazy(free_index, pages=_pages)
                return
            # --- EAGER path ---
            self._wait_band_transfers()
            # Near-no-op in normal mode (sampling's CPU sync already drained
            # forward_stream); in overlap mode it does the serializing.
            if self.forward_stream is not None:
                with record_function("Band.free.wait_stream"):
                    torch.cuda.current_stream().wait_stream(self.forward_stream)
            with record_function("Band.free.v2p_lookup"):
                free_v_pages = (
                    _pages
                    if _pages is not None
                    else torch.unique(
                        free_index.detach().to(torch.int64) // self.page_size
                    )
                )
                freed_p_pages = self.ids.virtual_to_physical[free_v_pages]
            with record_function("Band.free.sync_check"):
                # `.item()` forces a CPU/GPU sync -- own trace region to measure it.
                if bool((freed_p_pages < 0).any().item()):
                    self._raise_stale_slot_assertion(
                        free_v=free_v_pages, freed_p=freed_p_pages
                    )
            self.ids.virtual_to_physical.index_fill_(0, free_v_pages, -1)
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._compact_pending(freed_p_pages)

    @staticmethod
    def _copy_for_free_group(free_index: torch.Tensor) -> torch.Tensor:
        """Take ownership before a caller can mutate a deferred tensor view."""
        return free_index.clone()

    def _page_reps(self, free_index: torch.Tensor, start_pos: int) -> torch.Tensor:
        """One token of every page touched by a page-aligned kv-row segment:
        the fixed-shape stand-in for `unique(free_index // page_size)`."""
        ps = self.page_size
        assert start_pos % ps == 0, f"segment start {start_pos} is not page-aligned"
        return free_index[::ps]

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        """Fixed-shape counterpart of `free()`; see `_page_reps`. Contract: see
        base; a page must be freed by only one call per group.
        """
        if free_index is None or free_index.numel() == 0:
            return
        if self.page_size == 1:
            # token == page: nothing to dedup, the plain path is already exact.
            self.free(free_index)
            return
        reps = self._page_reps(free_index.detach().to(torch.int64), start_pos)
        if self.free_page_reps_group is None:
            self.free(reps, _pages=reps // self.page_size)
        else:
            self.free_page_reps_group.append(reps)

    def _free_lazy(
        self, free_index: torch.Tensor, pages: Optional[torch.Tensor] = None
    ) -> None:
        """Lazy free path: disjoint-element scatters plus ONE `torch.cat` onto
        `_free_phys_pages`; boundary absorption is deferred to `_flush`. Callers
        must not double-free -- a tombstone (-1) here would join the free list.
        """
        self._stats_n_free_lazy += 1
        with record_function("Band._free_lazy"):
            free_v_pages_raw = free_index.detach().to(torch.int64)
            if pages is not None:
                # `free_segment` already derived these by stride slicing.
                free_v_pages = pages
            elif self.page_size == 1:
                # ps == 1: token == page, and callers pass unique ids, so no dedup.
                free_v_pages = free_v_pages_raw
            else:
                free_v_pages = torch.unique(free_v_pages_raw // self.page_size)
            # One kernel for the v2p read and both tombstones; disjoint-element
            # scatters need no barrier (a freed v has no live reader). Never the
            # scalar `t[idx] = -1` form: it materialises -1 on the CPU and blocks
            # the scheduler on a pageable H2D copy (~16 ms per 8192-token free).
            freed_p_pages = free_unbind_inplace(
                free_v_pages, self.ids.virtual_to_physical, self.ids.physical_to_virtual
            )
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._free_phys_pages = torch.cat([self._free_phys_pages, freed_p_pages])
            self.live_page_count -= int(freed_p_pages.shape[0])

    def _release_phys_pages_batch(
        self, pages: torch.Tensor, *, host: Optional[List[int]] = None
    ) -> None:
        """Cat `pages` onto `_free_phys_pages`. `_flush` calls it only AFTER its
        trailing dst-slice, so `_free_phys_pages == holes_cpu` for the whole walk.
        No watermark / `live_page_count` change: vacated srcs re-enter as storage.
        ``host``: the same pages on the host, keeping the free list's host copy.
        """
        if pages.numel() == 0:
            return
        self._stats_n_release_batch += 1
        with record_function("Band._release_phys_pages_batch"):
            known = self._free_pages_host
            self._set_free_pages(
                torch.cat([self._free_phys_pages, pages]),
                host=None if known is None or host is None else known + host,
            )

    def _compact_pending(self, freed_physical_pages: torch.Tensor) -> None:
        """Eager compaction: move survivors out of the vacated band into the holes
        in the kept band. `src`/`dst` are disjoint by construction, so the batched
        copy is order-independent; the caller's `wait_stream` already serialized us
        with the in-flight forward.
        """
        with record_function("Band._compact_pending"):
            self._compact_pending_impl(freed_physical_pages)

    def _compact_pending_impl(self, freed_physical_pages: torch.Tensor) -> None:
        freed_set = set(int(x) for x in freed_physical_pages.tolist())
        if not freed_set:
            return
        K = len(freed_set)
        if self.grow_direction == "up":
            # allocated == [min_page_index, old_wm); after the free == [min_page_index, new_wm)
            old_wm = self.watermark_physical
            new_wm = old_wm - K
            assert new_wm >= self.min_page_index, (
                f"_compact_pending({self.sub_pool_name!r}): freeing {K} pages "
                f"would push the watermark below min_page_index "
                f"({new_wm} < {self.min_page_index})"
            )
            assert all(self.min_page_index <= h < old_wm for h in freed_set), (
                f"_compact_pending({self.sub_pool_name!r}): freed physical pages "
                f"{sorted(freed_set)} not all within allocated range "
                f"[{self.min_page_index}, {old_wm})"
            )
            # vacated band = [new_wm, old_wm); kept band = [min_page_index, new_wm)
            src_list = [s for s in range(new_wm, old_wm) if s not in freed_set]
            dst_list = sorted(h for h in freed_set if h < new_wm)
            self.watermark_physical = new_wm
            vacated_lo, vacated_hi = new_wm, old_wm
        else:
            # allocated == (old_wm, num_pages); after the free == (new_wm, num_pages)
            old_wm = self.watermark_physical
            new_wm = old_wm + K
            assert new_wm <= self.num_pages - 1, (
                f"_compact_pending({self.sub_pool_name!r}): freeing {K} pages "
                f"would push the watermark above num_pages "
                f"({new_wm} > {self.num_pages - 1})"
            )
            assert all(old_wm < h < self.num_pages for h in freed_set), (
                f"_compact_pending({self.sub_pool_name!r}): freed physical pages "
                f"{sorted(freed_set)} not all within allocated range "
                f"({old_wm}, {self.num_pages})"
            )
            # vacated band = (old_wm, new_wm] = [old_wm+1, new_wm+1); kept band = (new_wm, num_pages)
            src_list = [s for s in range(old_wm + 1, new_wm + 1) if s not in freed_set]
            dst_list = sorted(h for h in freed_set if h > new_wm)
            self.watermark_physical = new_wm
            vacated_lo, vacated_hi = old_wm + 1, new_wm + 1

        assert len(src_list) == len(dst_list), (
            f"_compact_pending({self.sub_pool_name!r}): {len(src_list)} survivors vs "
            f"{len(dst_list)} holes — corrupt allocator state"
        )

        if src_list:
            src_pages = torch.tensor(src_list, dtype=torch.int64, device=self.device)
            dst_pages = torch.tensor(dst_list, dtype=torch.int64, device=self.device)
            # `dst` holes are outside the vacated band by construction, so
            # rebinding them before the band wipe is order-equivalent.
            self._move_pages_and_rebind(src_pages, dst_pages)
            self.ids.physical_to_virtual[vacated_lo:vacated_hi] = -1
        else:
            self.ids.physical_to_virtual[vacated_lo:vacated_hi] = -1

    def _move_pages_and_rebind(
        self, src_pages: torch.Tensor, dst_pages: torch.Tensor
    ) -> torch.Tensor:
        """Copy live pages src->dst (disjoint sets), rebind v2p/p2v for the moved
        virtuals, record inverse history. Does NOT clear p2v[src] -- callers own
        vacated-region clearing. Returns the moved virtual page ids.
        """
        # Callers keep fixed pages out of `src_pages`.
        v_moved = self.ids.physical_to_virtual[src_pages].clone()  # read pre-wipe
        self._move_pages(src_pages, dst_pages)
        self.ids.virtual_to_physical[v_moved] = dst_pages
        self.ids.physical_to_virtual[dst_pages] = v_moved
        self._inverse_history.append((src_pages, dst_pages, v_moved))
        return v_moved

    def _move_pages(self, src_pages: torch.Tensor, dst_pages: torch.Tensor) -> None:
        # Whole page envelopes by physical page id, at most `_move_max_pages`
        # per launch, every launch issued in order on the current stream before
        # this returns. Callers pass disjoint src/dst sets, so no launch reads
        # a page another launch writes.
        for start in range(0, src_pages.numel(), self._move_max_pages):
            self.arena.move_units(
                self.sub_pool_name,
                dst_pages[start : start + self._move_max_pages],
                src_pages[start : start + self._move_max_pages],
            )

    def prewarm_compaction(self) -> None:
        """Run the compaction commit's first-use work once at startup, on the
        reserved sink page 0, so that no serving-time compaction is the first
        to compile, load or launch it.

        A first use while serving stalls the scheduler on the page-copy kernel's
        compile and load and on the first `.all()` reduction and `_assert_async`
        of the commit. A kernel's first launch can also take device memory
        outside every pool (a spilling kernel grows the context's local-memory
        reservation), and an urgent compaction under a full pool then fails
        with `Triton Error [CUDA]: out of memory`. Paid here, before KV sizing,
        that memory is accounted like any other startup allocation.

        The page-copy kernel specializes on neither the move size (a grid
        dimension) nor the alignment of the page-id slices, so this launch
        selects the variant every serving-time move of this sub-pool uses.
        """
        if torch.device(self.device).type != "cuda":
            return
        assert self.min_page_index >= 1, (
            f"Band({self.sub_pool_name!r}).prewarm_compaction needs "
            "the reserved sink page 0"
        )
        # The sink page copied onto itself, listed twice: every byte written is
        # the byte read, so the coinciding src/dst are harmless, and two entries
        # make `.all()` launch the reduction kernel a real commit uses (it skips
        # the kernel for one element).
        sink = torch.zeros(2, dtype=torch.int64, device=self.device)
        self._move_pages(sink, sink)
        # The commit's p2v check; its page-table gather and writes are the ones
        # alloc and free run, so they are warm before any compaction can exist.
        torch._assert_async((sink >= 0).all(), "prewarm_compaction")
        torch.cuda.current_stream().synchronize()

    # -- lazy compaction primitives --

    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:
        """Record the most-recent forward's `forward_done` event; pending pages
        are reused only after its retirement. None = no in-flight forward.
        """
        with record_function("Band.set_latest_forward_done_event"):
            self.usage.register_forward(event)
            self.usage.latest_forward = event

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        written_virtual: Optional[torch.Tensor],
        *,
        writes_unknown: bool = False,
    ) -> None:
        """Record a just-launched forward's `forward_done` event and the virtual
        ids it writes in this band, for `_flush`'s write-race check; no GPU work,
        only references. Forwards still running stay recorded: a pipeline stage
        launches the next microbatch before the previous one ends. Pass
        `written_virtual=None` when the forward writes nothing here, and
        ``writes_unknown`` when its write ids cannot be read before it ends:
        nothing then moves until it retires.
        """
        with record_function("Band.set_inflight_forward"):
            self.usage.record_forward(
                forward_done, written_virtual, writes_unknown=writes_unknown
            )

    def _physical_pages_of(self, ids: torch.Tensor, *, widened: bool) -> Set[int]:
        """The physical PAGE ids that virtual ``ids`` name now. Pays a D2H sync;
        call it before any stream wait, or the read parks the host. One id set at
        a time, not concatenated: a kernel's first launch loads its module, which
        waits for the running forwards."""
        # Forward write ids are WIDENED virtual ids under DCP; collapse them. A
        # widened page covers the same page, so non-owned ids fold in.
        if widened and self.shards_under_dcp:
            dcp_size = get_parallel().attn_dcp_size
            if dcp_size > 1:
                ids = ids // dcp_size
        # Negative ids (an empty tracking slot) translate to the sink.
        phys_tokens = self.ids.translate(ids)
        if self.pool_page_size > 1:
            phys_tokens = (phys_tokens // self.pool_page_size).unique()
        return set(phys_tokens.tolist())  # syncs the stream

    def _inflight_write_sets(self) -> List[Tuple[torch.cuda.Event, Optional[Set[int]]]]:
        """Each unretired forward with the physical PAGE ids it writes in this
        band (None when unknown), oldest first. Pays a bs-sized D2H sync per
        forward; call it before any stream wait, or the read parks the host.
        """
        self.usage.prune()
        forwards = self.usage.forwards()
        if not forwards:
            return []
        with record_function("Band._inflight_write_sets"):
            return [
                (
                    use.signal,
                    None
                    if use.ids is None
                    else self._physical_pages_of(use.ids, widened=True),
                )
                for use in forwards
            ]

    def record_transfer(
        self,
        signal,
        ids,
        *,
        writes: bool,
        release: Optional[TransferRelease] = None,
        unit_tokens: Optional[int] = None,
    ) -> None:
        """Record a transfer that reads, or with ``writes`` writes, the pages
        this band's controller ids ``ids`` name: until ``signal`` completes, or
        with ``release`` until it is released. With ``unit_tokens``, ``ids`` are
        physical units of that many tokens, the rows an engine outside every
        stream addresses whatever moves later. A copy of a page it writes waits
        ``signal``; a page it reads that a move vacates is reused only after it."""
        if ids is None or len(ids) == 0:
            return
        self.usage.record_transfer(
            signal, ids, writes=writes, release=release, unit_tokens=unit_tokens
        )

    def _transfer_page_sets(
        self,
    ) -> List[Tuple[object, Set[int], bool, Optional[TransferRelease]]]:
        """Each live transfer with the physical PAGE ids it touches now, oldest
        first. Pays a D2H sync per transfer; call it before any stream wait."""
        self.usage.prune()
        transfers = self.usage.transfers()
        if not transfers:
            return []
        with record_function("Band._transfer_page_sets"):
            return [
                (
                    use.signal,
                    (
                        self._physical_pages_of(use.ids, widened=False)
                        if use.unit_tokens is None
                        else self._pages_of_units(use.ids, use.unit_tokens)
                    ),
                    use.writes,
                    use.release,
                )
                for use in transfers
            ]

    def _pages_of_units(self, units, unit_tokens: int) -> Set[int]:
        """The physical PAGE ids physical ``units`` of ``unit_tokens`` tokens
        cover. Reads a device tensor on the host (a stream sync)."""
        if isinstance(units, torch.Tensor):
            units = units.tolist()
        rows = self.pool_page_size
        return {
            page
            for unit in np.asarray(units).reshape(-1).tolist()
            for page in range(
                unit * unit_tokens // rows, ((unit + 1) * unit_tokens - 1) // rows + 1
            )
        }

    @staticmethod
    def _readers_of(srcs: Sequence[int], transfer_sets) -> tuple:
        """The release keys of the transfers that still read any of ``srcs``:
        a source they read is reused only after them. A transfer with a logical
        release keys on both: the release decides when reuse is due, alike on
        every rank, and the signal is what the GPU waits before reusing."""
        src_set = set(srcs)
        return tuple(
            signal if release is None else (signal, release)
            for signal, pages, writes, release in transfer_sets
            if not writes and not src_set.isdisjoint(pages)
        )

    def _wait_writers(
        self,
        srcs: Sequence[int],
        write_sets: List[Tuple[torch.cuda.Event, Optional[Set[int]]]],
        transfer_sets=(),
    ) -> None:
        """Order a copy of ``srcs`` after the newest unfinished forward that
        writes one of them (or whose writes are unknown), and after every
        transfer that writes one of them; after nothing when none does: reading
        a page that unfinished work only reads is safe. Forwards run in order
        on one stream, so the newest covers the older; transfers do not."""
        src_set = set(srcs)
        writer = None
        for event, written in write_sets:
            if written is None or not src_set.isdisjoint(written):
                writer = event
        if writer is not None:
            self._wait_on_stream(writer)
            self._retire_through(writer)
        for signal, pages, writes, _ in transfer_sets:
            if writes and not src_set.isdisjoint(pages):
                self._wait_on_stream(signal)

    def _fence_vacated(self, pages: Sequence[int], readers: tuple = ()) -> None:
        """``pages`` left this band while the newest recorded forward, and the
        ``readers`` transfers, could still read them; a later write into them
        waits those first (`_wait_fences`)."""
        key = tuple(k for k in (self.usage.latest_forward, *readers) if k is not None)
        if not key or not pages:
            return
        key = key[0] if len(key) == 1 else key
        self.usage.vacate(
            key, list(_page_runs(pages)), page_bytes=self.entry_bytes_per_page
        )

    def _wait_fences(self, first: int, end: int) -> None:
        """Before anything writes this band's pages ``[first, end)``: wait each
        fence of the chain they meet. A waited fence is dropped, as later work
        on the stream is ordered after it; so is one whose event has fired, as
        bookkeeping only."""
        if end <= first:
            return
        lo, hi = first * self.entry_bytes_per_page, end * self.entry_bytes_per_page
        for band in self._chain_bands():
            for signal in band.usage.take_fences_meeting(lo, hi):
                self._wait_on_stream(signal)

    def _wait_fences_on(self, pages: Sequence[int]) -> None:
        for first, end in _page_runs(pages):
            self._wait_fences(first, end)

    def _maybe_emit_stats(self) -> None:
        """Env-gated periodic stats emit (at most once per interval) at `_flush` end.
        Disabled unless `SGLANG_LOG_LAZY_COMPACTION_STATS=1`.
        """
        if not _LAZY_COMPACTION_STATS_ENABLED:
            return
        now = _time_mod.monotonic()
        if now - self._stats_last_emit_ts < _LAZY_COMPACTION_STATS_INTERVAL_SEC:
            return
        self._stats_last_emit_ts = now
        self._stats_n_emits += 1
        cur_holes = int(self._free_phys_pages.shape[0])
        cur_pending = len(self.usage.pending_pages)
        self._stats_peak_free_list_len = max(self._stats_peak_free_list_len, cur_holes)
        self._stats_peak_pending_pages = max(
            self._stats_peak_pending_pages, cur_pending
        )
        logger.info(
            f"[lazy-stats sub={self.sub_pool_name!r}] "
            f"free_lazy={self._stats_n_free_lazy} "
            f"flush={self._stats_n_flush_calls} "
            f"(work={self._stats_n_flush_did_work} "
            f"moves={self._stats_n_flush_moves} "
            f"abs={self._stats_n_pages_absorbed}) "
            f"drain={self._stats_n_drain_did_work}/{self._stats_n_drain_calls} "
            f"peak_holes={self._stats_peak_free_list_len} "
            f"peak_pending={self._stats_peak_pending_pages} "
            f"cur_holes={cur_holes} cur_pending={cur_pending} "
            f"live={self.live_page_count} wm={self.watermark_physical}"
        )

    def _emit_stats_final(self, reason: str = "exit") -> None:
        """Force-emit final counters at shutdown (bypasses the interval gate).
        Idempotent (signal handler + atexit may both fire); best-effort.
        """
        if not _LAZY_COMPACTION_STATS_ENABLED:
            return
        if self._stats_final_emitted:
            return
        try:
            cur_holes = int(self._free_phys_pages.shape[0])
            cur_pending = len(self.usage.pending_pages)
            self._stats_peak_free_list_len = max(
                self._stats_peak_free_list_len, cur_holes
            )
            self._stats_peak_pending_pages = max(
                self._stats_peak_pending_pages, cur_pending
            )
            self._stats_final_emitted = True
            logger.info(
                f"[lazy-stats FINAL sub={self.sub_pool_name!r} reason={reason}] "
                f"free_lazy={self._stats_n_free_lazy} "
                f"flush={self._stats_n_flush_calls} "
                f"(work={self._stats_n_flush_did_work} "
                f"moves={self._stats_n_flush_moves} "
                f"abs={self._stats_n_pages_absorbed}) "
                f"drain={self._stats_n_drain_did_work}/{self._stats_n_drain_calls} "
                f"peak_holes={self._stats_peak_free_list_len} "
                f"peak_pending={self._stats_peak_pending_pages} "
                f"cur_holes={cur_holes} cur_pending={cur_pending} "
                f"live={self.live_page_count} wm={self.watermark_physical} "
                f"n_emits={self._stats_n_emits}"
            )
        except Exception:
            pass

    def _drain_pending_reuse(self, *, urgent: bool) -> None:
        """Move ready pending pages back into `_free_phys_pages`: those whose
        readers retired, and every one when urgent. The stream waits each one's
        signal -- a stream-side dependency, not a host block (a retired forward
        can still be running its tail). No watermark / `live_page_count` change.
        """
        self._stats_n_drain_calls += 1
        if not self.usage.has_pending():
            return
        with record_function("Band._drain_pending_reuse"):
            ready = self.usage.take_ready_vacated(urgent=urgent)
            for use in ready:
                if use.signal is not None:
                    self._wait_on_stream(use.signal)
                    self._retire_through(use.signal)
            ready_tensors = [use.batch for use in ready]
            if ready_tensors:
                host = self._free_pages_host
                self._set_free_pages(
                    torch.cat([self._free_phys_pages] + ready_tensors),
                    host=(
                        None
                        if host is None
                        else host + [p for use in ready for p in use.pages]
                    ),
                )
                self._stats_n_drain_did_work += 1
                self._stats_n_drained_pages_total += sum(
                    t.numel() for t in ready_tensors
                )

    def _topmost_survivor(
        self,
        start_hint: Optional[int] = None,
        *,
        holes_cpu: Optional[List[int]] = None,
        j_in: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Topmost live PAGE in the allocated band, excluding `holes_cpu` (the
        sorted-ASCENDING snapshot) and the pending pages. Returns
        `(p, j)`, or `(None, j)` if none -- the hole cursor `j` is threaded back
        in so the two-pointer membership test stays O(1). `holes_cpu`/`j_in` are
        optional only for test fixtures; `_flush` always passes them. Uncommitted
        dsts already read p2v == -1, so no exclude set is needed.
        """
        if holes_cpu is None:
            holes_cpu = self._free_phys_pages.tolist()
        if self.grow_direction == "up":
            if start_hint is None or start_hint >= self.watermark_physical:
                p = self.watermark_physical - 1
            else:
                p = start_hint
            j = j_in if j_in is not None else len(holes_cpu) - 1
            while p >= self.min_page_index:
                while j >= 0 and holes_cpu[j] > p:
                    j -= 1
                is_hole = j >= 0 and holes_cpu[j] == p
                if is_hole or p in self.usage.pending_pages:
                    if is_hole:
                        j -= 1
                    p -= 1
                    continue
                return p, j
            return None, j
        else:
            if start_hint is None or start_hint <= self.watermark_physical:
                p = self.watermark_physical + 1
            else:
                p = start_hint
            j = j_in if j_in is not None else 0
            while p < self.num_pages:
                while j < len(holes_cpu) and holes_cpu[j] < p:
                    j += 1
                is_hole = j < len(holes_cpu) and holes_cpu[j] == p
                if is_hole or p in self.usage.pending_pages:
                    if is_hole:
                        j += 1
                    p += 1
                    continue
                return p, j
            return None, j

    def _absorb_boundary_holes(self, all_cpu: List[int]) -> Tuple[int, List[int]]:
        """Retreat the watermark past free pages ALREADY contiguous with it, slice
        them off `_free_phys_pages`, return ``(new_watermark, interior_holes_cpu)``.
        ``all_cpu`` is the sorted-ascending snapshot.
        """
        M = len(all_cpu)
        wm = self.watermark_physical
        n = 0
        if self.grow_direction == "up":
            while n < M and all_cpu[M - 1 - n] == wm - 1 - n:
                n += 1
            new_wm = wm - n
            holes_cpu = all_cpu[: M - n]
            self._set_free_pages(self._free_phys_pages[: M - n], host=holes_cpu)
        else:
            while n < M and all_cpu[n] == wm + 1 + n:
                n += 1
            new_wm = wm + n
            holes_cpu = all_cpu[n:]
            self._set_free_pages(self._free_phys_pages[n:], host=holes_cpu)
        self.watermark_physical = new_wm
        self._stats_n_pages_absorbed += n
        return new_wm, holes_cpu

    def _settle_inflight_forward(self) -> None:
        """Stream-wait the in-flight forward's done event so freed slots are safe
        to MOVE (write settled) and REUSE (read settled): the event is recorded
        after the WHOLE forward, so one wait covers both hazards. Every recorded
        forward retires: all ran before it on the forward stream.
        """
        ev = self.usage.latest_forward
        if ev is not None:
            self._wait_on_stream(ev)
            self.usage.retire_all()

    def _flush(self, *, urgent: bool) -> int:
        """One batched compaction pass; returns the number of survivor moves.

        Two hazards per survivor: a WRITE race (an unfinished forward overwrites
        KV[src], so a compaction read would corrupt KV[dst]) stops a non-urgent
        walk at that src, and an urgent one waits that forward before copying
        it; a READ race (a forward READS KV[src]) gates src REUSE, so such srcs
        stay pending, or, when an urgent pass retreats the frontier
        past them, to a fence that later writes into them wait.

        Fixed pages never move: a non-urgent walk stops at one, and an urgent
        pass packs the other live pages below the frontier its farthest fixed
        page allows, leaving the holes between free.

        `_topmost_survivor` excludes all p2v=-1 pages, so a negative virtual id in
        the batched mapping lookup is a corrupt-state bug and raises.
        """
        if not self.lazy_compaction:
            return 0
        self.usage.prune()
        # Read the transfers' pages on the host before any stream wait below.
        transfer_sets = self._transfer_page_sets()
        # Only a transfer with a logical release may stop a walk: whether one
        # without is still listed depends on its event having fired, which ranks
        # see at different times. A copy of a page such a transfer writes waits
        # it on the GPU (`_commit_move_batch`).
        transfer_written = set().union(
            *(
                pages
                for _, pages, writes, release in transfer_sets
                if writes and release is not None
            )
        )
        if self.usage.writes_unknown():
            # A running forward whose writes are unknown may write any survivor.
            if not urgent:
                self._drain_pending_reuse(urgent=False)
                return 0
            self._settle_inflight_forward()
        # Urgent: read the write sets on the host before any stream wait below.
        write_sets = self._inflight_write_sets() if urgent else None
        self._stats_n_flush_calls += 1
        with record_function("Band._flush"):
            self._drain_pending_reuse(urgent=urgent)

            # Sort ASCENDING, on the host: a device sort would drop the host copy,
            # and reading it back after a forward wait blocks.
            all_cpu = self._free_pages_snapshot()  # at most one batched D2H sync
            if any(a > b for a, b in zip(all_cpu, all_cpu[1:])):
                all_cpu = sorted(all_cpu)
                self._set_free_pages(self._to_device(all_cpu), host=all_cpu)

            # `holes_cpu` = interior holes; `_free_phys_pages == holes_cpu` after.
            new_wm, holes_cpu = self._absorb_boundary_holes(all_cpu)

            latest_event = self.usage.latest_forward

            # The frontier an urgent pass leaves: just past the live pages packed
            # from the base, or past the farthest fixed page if that is farther.
            # Every survivor beyond it moves into a hole below it; none is fixed.
            fixed = self.fixed_pages()
            if self.grow_direction == "up":
                live = new_wm - self.min_page_index - len(holes_cpu)
                packed_end = self.min_page_index + live
                if fixed:
                    packed_end = max(packed_end, max(fixed) + 1)
            else:
                live = self.num_pages - (new_wm + 1) - len(holes_cpu)
                packed_end = self.num_pages - live
                if fixed:
                    packed_end = min(packed_end, min(fixed))

            # Single-pass FULL-PACK (urgent only): every interior hole is filled
            # and the frontier retreats past the vacated survivors, which are
            # fenced rather than waited for (a forward may still read them).
            single_pass_absorb = urgent and len(holes_cpu) > 0

            # write_set: None = not yet materialized (do it inline on the first
            # survivor needing the check); set() = no write race; else materialized.
            write_set: Optional[Set[int]] = None
            if write_sets is not None:
                write_set = set().union(*(w for _, w in write_sets if w is not None))

            srcs: List[int] = []
            dsts: List[int] = []

            # Flush-scoped accumulator for event-FIRED srcs, merged AFTER the
            # trailing dst-slice so `_free_phys_pages` stays byte-identical to
            # `holes_cpu` for the whole walk; catting mid-flush would chop the
            # wrong end. Srcs whose readers run on stay pending.
            released_fired: List[Tuple[torch.Tensor, List[int]]] = []

            cursor: Optional[int] = None
            j_cursor: Optional[int] = None

            # Dst cursor reads `holes_cpu` directly (no per-dst sync): grow-up from
            # the front, grow-down from the back; consumed entries sliced at exit.
            if self.grow_direction == "up":
                dst_cursor = 0
            else:
                dst_cursor = len(holes_cpu) - 1
            n_dst_consumed = 0

            move_cap = self._lazy_max_moves_per_call if not urgent else None

            n_moves = 0
            while n_dst_consumed < len(holes_cpu):
                src, j_cursor = self._topmost_survivor(
                    start_hint=cursor,
                    holes_cpu=holes_cpu,
                    j_in=j_cursor,
                )
                if src is None:
                    break
                if urgent and (
                    src < packed_end
                    if self.grow_direction == "up"
                    else src >= packed_end
                ):
                    break  # packed: the rest stays where it is
                if src in fixed:
                    break  # non-urgent: a fixed page stops the walk
                if not urgent and src in transfer_written:
                    # A transfer still writes it: only a move an allocation
                    # needs waits for that, not an opportunistic one.
                    break

                # Case A: write race.
                if write_set is None:
                    write_set = set().union(
                        *(w for _, w in self._inflight_write_sets() if w is not None)
                    )
                if write_set and src in write_set:
                    if urgent:
                        # Commit accumulated moves, then wait the newest forward
                        # that writes this src; forwards after it may still write.
                        self._commit_move_batch(
                            srcs,
                            dsts,
                            latest_event,
                            released_fired,
                            fence=single_pass_absorb,
                            transfer_sets=transfer_sets,
                        )
                        n_moves += len(srcs)
                        srcs.clear()
                        dsts.clear()
                        self._wait_writers([src], write_sets)
                        write_sets = [
                            (event, written)
                            for event, written in write_sets
                            if not self._retired(event)
                        ]
                        write_set = set().union(
                            *(w for _, w in write_sets if w is not None)
                        )
                        if self._retired(latest_event):
                            latest_event = None
                        # DO NOT reset cursor/j_cursor: rewinding would re-pick
                        # the just-committed srcs (now p2v=-1, not in holes_cpu)
                        # and trip the p2v=-1 assertion.
                        continue
                    else:
                        break  # non-urgent: top blocker stops the walk

                # Case B/C: no write race. dst from holes_cpu by cursor (no sync).
                dst = holes_cpu[dst_cursor]
                # Two-pointer crossing check: past the crossing the band is packed,
                # and moving further would shuffle a hole back toward the frontier
                # and block the watermark retreat.
                if (self.grow_direction == "up" and src < dst) or (
                    self.grow_direction == "down" and src > dst
                ):
                    break
                if self.grow_direction == "up":
                    dst_cursor += 1
                else:
                    dst_cursor -= 1
                n_dst_consumed += 1

                srcs.append(src)
                dsts.append(dst)

                # Advance cursor strictly past the picked src.
                if self.grow_direction == "up":
                    cursor = src - 1
                else:
                    cursor = src + 1

                if move_cap is not None and len(srcs) >= move_cap:
                    break

            self._commit_move_batch(
                srcs,
                dsts,
                latest_event,
                released_fired,
                fence=single_pass_absorb,
                transfer_sets=transfer_sets,
            )
            n_moves += len(srcs)

            if single_pass_absorb:
                # FULL-PACK reclaim (urgent): the frontier retreats to the packed
                # end; holes beyond it are beyond-frontier space, and only holes
                # a fixed page keeps below it stay free. `released_fired` is
                # dropped: every source lay beyond the packed end.
                if self.grow_direction == "up":
                    kept = [h for h in holes_cpu[n_dst_consumed:] if h < packed_end]
                    n_reclaimed = new_wm - packed_end
                    self.watermark_physical = packed_end
                else:
                    kept = [
                        h
                        for h in holes_cpu[: len(holes_cpu) - n_dst_consumed]
                        if h >= packed_end
                    ]
                    n_reclaimed = packed_end - 1 - new_wm
                    self.watermark_physical = packed_end - 1
                self._stats_n_pages_absorbed += n_reclaimed
                self._set_free_pages(self._to_device(kept), host=kept)
            else:
                # Non-urgent partial pass: the watermark stays. `_free_phys_pages`
                # is still == holes_cpu, so the consumed dsts are exactly the front
                # (grow-up) / back (grow-down) `n_dst_consumed` entries.
                if n_dst_consumed > 0:
                    if self.grow_direction == "up":
                        self._set_free_pages(
                            self._free_phys_pages[n_dst_consumed:],
                            host=holes_cpu[n_dst_consumed:],
                        )
                    else:
                        self._set_free_pages(
                            self._free_phys_pages[:-n_dst_consumed],
                            host=holes_cpu[:-n_dst_consumed],
                        )
                if released_fired:
                    self._release_phys_pages_batch(
                        torch.cat([pages for pages, _ in released_fired]),
                        host=[p for _, host in released_fired for p in host],
                    )
            if n_moves > 0:
                self._stats_n_flush_did_work += 1
                self._stats_n_flush_moves += n_moves
            self._maybe_emit_stats()
            return n_moves

    def _commit_move_batch(
        self,
        srcs: List[int],
        dsts: List[int],
        latest_event: Optional[torch.cuda.Event],
        released_fired: List[Tuple[torch.Tensor, List[int]]],
        *,
        fence: bool = False,
        transfer_sets=(),
    ) -> None:
        """Issue the page copies (`_move_pages`) plus ONE bulk v2p/p2v remap for
        the accumulated `(src, dst)` pairs, after the transfers that write a src.
        With ``fence`` the srcs leave the band (the frontier retreats past them)
        and are fenced; otherwise srcs whose readers retired accumulate in
        `released_fired` (merged by `_flush` after its dst-slice) and the others
        stay pending for read-race gating. A src a transfer still
        reads is released only with that transfer too.
        """
        if not srcs:
            return
        assert not set(srcs) & self.fixed_pages(), (
            f"[{self.sub_pool_name}] page copy of a fixed page"
        )
        with record_function("Band._commit_move_batch"):
            readers = self._readers_of(srcs, transfer_sets)
            self._wait_writers(srcs, [], transfer_sets)
            src_pages_t = self._to_device(srcs)
            dst_pages_t = self._to_device(dsts)
            v_moveds_t = self.ids.physical_to_virtual[src_pages_t]
            torch._assert_async(
                (v_moveds_t >= 0).all(),
                "invalid p2v mapping in Band._flush",
            )
            self._move_pages(src_pages_t, dst_pages_t)
            # ONE bulk remap (single-writer on schedule_stream), issued after the
            # copies on the same stream: work ordered after it sees moved pages.
            self.ids.virtual_to_physical[v_moveds_t] = dst_pages_t
            self.ids.physical_to_virtual[dst_pages_t] = v_moveds_t
            self.ids.physical_to_virtual.index_fill_(0, src_pages_t, -1)
            self._inverse_history.append((src_pages_t, dst_pages_t, v_moveds_t))
            # Src disposition. `src_pages_t` is reused as the pending pages' device
            # tensor (no second H2D at drain).
            if fence:
                self._fence_vacated(srcs, readers)
                return
            release = tuple(k for k in (latest_event, *readers) if k is not None)
            release = (
                None if not release else release[0] if len(release) == 1 else release
            )
            if self._retired(release):
                if release is not None:
                    # A retired forward can still run its tail, and a transfer
                    # counts as retired once its event is waited: reuse after.
                    self._wait_on_stream(release)
                released_fired.append((src_pages_t, list(srcs)))
            else:
                self.usage.vacate(
                    release,
                    (),
                    page_bytes=self.entry_bytes_per_page,
                    pages=srcs,  # copied: the caller mutates `srcs`
                    batch=src_pages_t,
                )

    def flush_for_allocation(self) -> int:
        """Public urgent flush used by peer allocation-pressure recovery."""
        with record_function("Band.flush_for_allocation"):
            return self._flush(urgent=True)

    def flush_opportunistic(self) -> int:
        """Public, non-urgent flush at quiescent points; never blocks
        `schedule_stream`. Fast-path the empty state: the scheduler triggers this
        very often and ~99% of calls have no holes and no pending entries.
        """
        with record_function("Band.flush_opportunistic"):
            if not self.lazy_compaction:
                return 0
            if self._free_phys_pages.numel() == 0 and not self.usage.has_pending():
                return 0
            return self._flush(urgent=False)

    def _raise_stale_slot_assertion(self, *, free_v, freed_p) -> None:
        bad = free_v[freed_p < 0].tolist()
        frames = inspect.stack()[1:9]
        callers = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames)
        raise AssertionError(
            f"Band({self.sub_pool_name!r}).free: virtual id(s) {bad} have "
            f"virtual_to_physical == -1 (double-free or never-allocated). "
            f"State: {self.allocator_state_str()}. free_index unique={free_v.tolist()}. "
            f"recent _inverse_history (last 3): "
            f"{[(s.tolist(), d.tolist()) for s, d, _ in self._inverse_history[-3:]]}. "
            f"Caller: {callers}."
        )

    # -- free-group --

    def free_group_begin(self) -> None:
        assert self.free_group is None, "free groups cannot be nested"
        self.free_group = []
        self.free_page_reps_group = []

    def free_group_end(self) -> None:
        pending, self.free_page_reps_group = self.free_page_reps_group, None
        pending_frees, self.free_group = self.free_group, None
        if pending_frees:
            self.free(torch.cat(pending_frees))
        if pending:
            reps = torch.cat(pending)
            self.free(reps, _pages=reps // self.page_size)

    def _set_capacity(
        self, max_slots: int, *, virtual_num_pages: Optional[int] = None
    ) -> None:
        """Set active ranges while retaining the captured v2p/p2v storage."""
        num_pages = int(max_slots) // self.pool_page_size
        num_virtual_ids = (
            num_pages if virtual_num_pages is None else int(virtual_num_pages)
        )
        self.num_pages = num_pages
        self.ids.num_virtual_ids = num_virtual_ids
        self.max_slots = num_pages * self.pool_page_size
        self.size = num_pages * self.page_size
        self.clear()

    def set_hicache_transfer_done_event(self, transfer_key: Hashable, event) -> None:
        """A HiCache copy was submitted. Its pages are ordered by the transfer
        recorded on them (`record_transfer`); a load's submission ends its
        reservation, as the copy now resolves the reserved rows."""
        if transfer_key == "load" or (
            isinstance(transfer_key, tuple) and transfer_key[-1] == "load"
        ):
            self.unfix_pages(_LOAD_RESERVATION)

    def end_load_reservation(self) -> None:
        """The tree bound the reserved rows to owner ids and the queued load now
        names them by those ids, which it translates at submission: the rows
        move like any page from here on."""
        self.unfix_pages(_LOAD_RESERVATION)

    def _wait_band_transfers(self) -> None:
        """Order the current stream after the transfers recorded on this band,
        before a free that hands pages back at once (eager compaction, a float's
        free, a cancelled reservation): a copy may still read or write a freed
        page. Freed pages are device ids here; matching them to the transfers'
        pages would read the device, so every transfer of the band is waited."""
        self.usage.prune()
        for use in self.usage.transfers():
            self._wait_on_stream(use.signal)

    def _expand_pages_to_tokens(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        return (
            pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

    def alloc_physical(self, need_size: int) -> Optional[torch.Tensor]:
        """Reserve physical token slots without assigning virtual page ids."""
        if need_size <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        assert need_size % self.page_size == 0, (
            f"Band({self.sub_pool_name!r}).alloc_physical: need_size="
            f"{need_size} must be a multiple of page_size={self.page_size}"
        )
        if need_size > self.available_size():
            if not relieve_for_alloc(self, need_size):
                return None
        free_before = set(self._free_pages_snapshot())
        span_before = set(range(*self._span_range()))
        physical_pages = self.take_physical_pages(need_size // self.page_size)
        if physical_pages is None:
            return None
        free_after = self._free_pages_host
        if free_after is None:
            taken = physical_pages.tolist()
        else:
            # The pages that left the free list, and those the span grew by.
            taken = sorted(
                (free_before - set(free_after))
                | (set(range(*self._span_range())) - span_before)
            )
        # Fixed until the H2D is submitted or cancelled.
        self.fix_pages(_LOAD_RESERVATION, taken)
        return self._expand_pages_to_tokens(physical_pages)

    def _span_range(self) -> Tuple[int, int]:
        """Pages ``[first, end)`` this band spans."""
        if self.grow_direction == "up":
            return self.min_page_index, self.watermark_physical
        return self.watermark_physical + 1, self.num_pages

    def cancel_physical_reservation(self, free_index: torch.Tensor) -> None:
        """Roll back a HiCache physical allocation before its H2D is submitted."""
        if free_index is None or free_index.numel() == 0:
            return
        # Reads the device: a cancelled load is rare and off the step path.
        pages = (free_index.detach()[:: self.page_size] // self.page_size).tolist()
        reserved = self.usage.fixed.get(_LOAD_RESERVATION, ())
        assert all(page in reserved for page in pages), (
            f"Band({self.sub_pool_name!r}) released pages {pages} it did not "
            f"reserve for HiCache ({sorted(reserved)})"
        )
        self.unfix_pages(_LOAD_RESERVATION, pages)
        self.free_physical(free_index)

    def free_physical(self, free_index: torch.Tensor) -> None:
        """Release physical token slots reserved by :meth:`alloc_physical`."""
        if free_index is None or free_index.numel() == 0:
            return
        assert free_index.numel() % self.page_size == 0, (
            f"Band({self.sub_pool_name!r}).free_physical: "
            f"{free_index.numel()} tokens must contain whole pages of "
            f"size {self.page_size}"
        )
        physical_pages = (
            free_index.detach().to(torch.int64)[:: self.page_size] // self.page_size
        )
        self._wait_band_transfers()
        if self.forward_stream is not None and not self.lazy_compaction:
            torch.cuda.current_stream().wait_stream(self.forward_stream)
        virtual_pages = self.ids.physical_to_virtual[physical_pages]
        bound_mask = virtual_pages >= 0
        self.ids.virtual_to_physical.index_fill_(0, virtual_pages[bound_mask], -1)
        self.ids.physical_to_virtual.index_fill_(0, physical_pages, -1)
        if self.lazy_compaction:
            self._free_phys_pages = torch.cat([self._free_phys_pages, physical_pages])
            self.live_page_count -= int(physical_pages.shape[0])
            return
        self._compact_pending(physical_pages)


def chain_byte_accounting_violations(
    chain: List[Band],
) -> List[str]:
    """Conservation for an ordered low-to-high chain of band allocators: each
    member's own accounting, plus the frontier total order -- a member's low
    frontier must clear the previous member's high frontier, or the bands overlap
    in the shared byte buffer. Transparent members are skipped by the ordering
    walk only; their per-pool conservation still runs.
    """
    out: List[str] = []
    for a in chain:
        out.extend(a._byte_accounting_violations())
    frontier = 0
    for a in chain:
        if a._is_frontier_transparent():
            continue
        lo_b, hi_b = a._byte_low_frontier(), a._byte_high_frontier()
        if lo_b < frontier:
            out.append(
                f"[chain] {a.sub_pool_name} low frontier {lo_b} overlaps the "
                f"previous pool's high frontier {frontier}"
            )
        frontier = max(frontier, hi_b)
    return out


def _end_pair_chain(a: Band, b: Band) -> List[Band]:
    """Order an end pair low→high by grow direction (the factories and the
    unit fixtures orient the pair differently; the chain check must not care)."""
    return sorted((a, b), key=lambda x: x.grow_direction != "up")


class FloatBand(Band):
    """Float MIDDLE cache pool: a span ``[low_wm_page, high_wm_page)`` between
    two chain neighbors, with freed HOLES allowed inside the span.

    Holes-first, because a middle CACHE pool is not a band: ``free`` marks
    interior holes and absorbs boundary ones; alloc reuses holes before extending
    the boundary on the side with the LARGER free gap, and from empty it positions
    the span at the MIDPOINT of the inter-frontier region so free gap exists on
    both sides and neighbor growth does not immediately force a data move. Data
    moves happen only ON DEMAND, via ``make_room`` / ``compact_holes``. An EMPTY
    float resets its span and is frontier-transparent: it occupies no bytes and
    must never wall off free space.

    Floats skip the lazy event pipeline (`lazy_compaction` must be False):
    frees/allocs are zero-copy by design, so only the on-demand moves need
    write-set safety, which their scheduler-phase call sites provide.
    """

    # The span IS this pool's capacity state (it has no watermark): moving it
    # changes its own availability and, through transparency, both neighbours'.
    low_wm_page: _CapacityField[int] = _CapacityField()
    high_wm_page: _CapacityField[int] = _CapacityField()

    # Only `free` can make a boundary page a hole, so a clean flag proves both
    # boundaries are live and the deferred absorb can skip its D2H.
    _holes_dirty: bool = False

    # A middle pool has no origin of its own, so its requests take holes from the
    # two ends of the ascending hole list in turn: the whole request from one end,
    # the next request from the other, whatever their sizes. Only a request that
    # drains holes and succeeds flips it.
    _next_hole_end: str = "low"

    def __init__(self, **kwargs):
        assert not kwargs.get("lazy_compaction", False), (
            "FloatBand is holes-first; the lazy event pipeline "
            "is end-pool machinery and must stay off for float middles"
        )
        # Base __init__ ends with self.clear(), which reads these via our
        # _reset_watermarks override -- pre-seed so the override can run.
        self.low_wm_page = 0
        self.high_wm_page = 0
        super().__init__(**kwargs)
        assert self.grow_direction == "float", (
            f"FloatBand needs a 'float' sub-pool spec; got {self.grow_direction!r}"
        )

    # -- span / frontier state --

    def _reset_watermarks(self) -> None:
        # Park empty at the buffer top; empty-transparency makes the parked
        # position irrelevant to neighbors.
        self.low_wm_page = self.num_pages
        self.high_wm_page = self.num_pages
        self.watermark_physical = -1  # unused for float pools (logs only)

    def _span_pages(self) -> int:
        return self.high_wm_page - self.low_wm_page

    def _span_range(self) -> Tuple[int, int]:
        return self.low_wm_page, self.high_wm_page

    def _hole_pages(self) -> int:
        return int(self._free_phys_pages.numel())

    def _live_pages(self) -> int:
        return self._span_pages() - self._hole_pages()

    def _is_frontier_transparent(self) -> bool:
        return self._live_pages() == 0

    def _allocated_pages(self) -> int:
        return self._live_pages()

    def _byte_low_frontier(self) -> int:
        return self.low_wm_page * self.entry_bytes_per_page

    def _byte_high_frontier(self) -> int:
        return self.high_wm_page * self.entry_bytes_per_page

    def _region_bounds_pages(self) -> Tuple[int, int]:
        """Page bounds ``[lo, hi)`` of the inter-frontier region available to
        this float (chain-transparent walk; clamped to the slot-0 sink
        reservation). Rounded conservatively: ``lo`` up, ``hi`` down."""
        epp = self.entry_bytes_per_page
        lo = (self._chain_high_frontier_below_bytes() + epp - 1) // epp
        lo = max(lo, self.min_page_index)
        hi = self._chain_low_frontier_above_bytes() // epp
        hi = min(hi, self.num_pages)
        return lo, hi

    def pages_in_band(self, *, low_byte: int, high_byte: int) -> int:
        """Pages obtainable from ``[low_byte, high_byte)`` on this pool's OWN
        page grid. A raw ``(high - low) // entry_bytes_per_page`` over-counts by
        a page whenever ``low_byte`` is off the grid, which is the generic case:
        the bounding frontier is a multiple of the NEIGHBOUR's entry size.
        """
        epp = self.entry_bytes_per_page
        lo = max((low_byte + epp - 1) // epp, self.min_page_index)
        hi = min(high_byte // epp, self.num_pages)
        return max(0, hi - lo)

    def _gap_pages(self) -> Tuple[int, int]:
        """(gap_low, gap_high) in own page units; both == the whole region
        when the span is empty/parked."""
        lo, hi = self._region_bounds_pages()
        if self._is_frontier_transparent():
            room = max(0, hi - lo)
            return room, room
        return max(0, self.low_wm_page - lo), max(0, hi - self.high_wm_page)

    # -- availability --

    def _side_neighbor(self, side: str) -> Optional[Band]:
        """Nearest non-transparent member on ``side``."""
        p = self.low_peer if side == "low" else self.high_peer
        while p is not None and p._is_frontier_transparent():
            p = p.low_peer if side == "low" else p.high_peer
        return p

    def _side_drainable_hole_bytes(self, side: str) -> int:
        """Realizable gap bytes an urgent flush of the neighbour on ``side``
        would release, walking past transparent members like the frontier walk.
        """
        p = self._side_neighbor(side)
        if not _drains_on_urgent_flush(p):
            return 0
        return len(p._free_phys_pages) * p.entry_bytes_per_page

    def _peer_drainable_hole_bytes(self) -> int:
        """The better of the two sides. `_growth_side_neighbor()` is undefined
        for a float -- its `grow_direction` is "float", so the base answers
        `low_peer` and never sees the high neighbour.
        """
        return max(
            self._side_drainable_hole_bytes("low"),
            self._side_drainable_hole_bytes("high"),
        )

    def _credit_neighbors(self) -> tuple:
        return (self._side_neighbor("low"), self._side_neighbor("high"))

    def _schedulable_capacity_key(self) -> tuple:
        # One drain bit per side: `_available_tokens` credits each side on its own.
        epoch = self._chain_capacity_epoch()
        m = self._credit_neighbors_memo
        if m is None or m[0] != epoch:
            m = self._credit_neighbors_memo = (epoch, self._credit_neighbors())
        low, high = m[1]
        return epoch, _drains_on_urgent_flush(low), _drains_on_urgent_flush(high)

    def _available_tokens(self, extra_gap_bytes: int = 0) -> int:
        gap_low, gap_high = self._gap_pages()
        if extra_gap_bytes > 0:
            # Per side: the base hands down one undirected scalar because an
            # END pool grows one way, but a float grows both.
            epp = self.entry_bytes_per_page
            gap_low += self._side_drainable_hole_bytes("low") // epp
            gap_high += self._side_drainable_hole_bytes("high") // epp
        gap_pages = max(gap_low, gap_high)  # a single alloc extends ONE side
        pages_by_index_space = self.num_pages - self.min_page_index - self._live_pages()
        pages_extend = min(gap_pages, pages_by_index_space)
        return (pages_extend + self._hole_pages()) * self.page_size

    # -- physical page primitives (holes-first) --

    def take_physical_pages(
        self, num_pages: int, *, placement: Optional[Tuple[int, int]] = None
    ) -> Optional[torch.Tensor]:
        scope = self._physical_take_scope
        if scope is not None and not scope.applying:
            return scope.take(self, num_pages)
        if num_pages <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        n_drain = min(num_pages, self._hole_pages())
        need_more = num_pages - n_drain

        fresh: Optional[torch.Tensor] = None
        if need_more > 0:
            lo, hi = self._region_bounds_pages()
            if placement is not None:
                start, end = placement
                if end - start != need_more or start < lo or end > hi:
                    return None
                if self._is_frontier_transparent():
                    self.low_wm_page, self.high_wm_page = start, end
                elif end == self.low_wm_page:
                    self.low_wm_page = start
                elif start == self.high_wm_page:
                    self.high_wm_page = end
                else:
                    return None
                self._wait_fences(start, end)
                fresh = torch.arange(start, end, dtype=torch.int64, device=self.device)
            elif self._is_frontier_transparent():
                # Reposition-on-alloc-from-empty: collapse to the midpoint so
                # free gap remains on BOTH sides.
                if need_more > hi - lo:
                    return None
                start = lo + (hi - lo - need_more) // 2
                self.low_wm_page = start
                self.high_wm_page = start + need_more
                self._wait_fences(start, start + need_more)
                fresh = torch.arange(
                    start, start + need_more, dtype=torch.int64, device=self.device
                )
            else:
                gap_low = self.low_wm_page - lo
                gap_high = hi - self.high_wm_page
                # Extend toward the roomier gap; fall back to the other side.
                sides = ("high", "low") if gap_high >= gap_low else ("low", "high")
                for side in sides:
                    if side == "high" and need_more <= gap_high:
                        start = self.high_wm_page
                        self.high_wm_page += need_more
                        break
                    if side == "low" and need_more <= gap_low:
                        start = self.low_wm_page - need_more
                        self.low_wm_page = start
                        break
                else:
                    return None  # neither side fits; state untouched
                self._wait_fences(start, start + need_more)
                fresh = torch.arange(
                    start, start + need_more, dtype=torch.int64, device=self.device
                )

        if n_drain > 0:
            drained = self._take_free_pages_from_end(
                n_drain, self._next_hole_end
            ).clone()
            self._next_hole_end = "high" if self._next_hole_end == "low" else "low"
        else:
            drained = None

        if drained is None:
            return fresh
        if fresh is None:
            return drained
        return torch.cat([drained, fresh])

    def take_physical(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        assert need_size % self.page_size == 0, (
            f"take_physical: need_size={need_size} must be a multiple of "
            f"page_size={self.page_size}"
        )
        return self.take_physical_pages(need_size // self.page_size)

    def place_pages(self, v_pages: torch.Tensor, N: int) -> Optional[torch.Tensor]:
        # Holes-first always routes through take_physical_pages (no fused
        # watermark fast path -- float alloc cadence doesn't need it).
        if N == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        phys_pages = self.take_physical_pages(N)
        if phys_pages is None:
            return None
        self.ids.bind(v_pages, phys_pages)
        self._maybe_zero_pages(phys_pages)
        return phys_pages

    # -- free: hole-marking, boundary absorption, park-on-empty --

    def free(
        self, free_index: torch.Tensor, *, _pages: Optional[torch.Tensor] = None
    ) -> None:
        """Mark the freed pages as interior HOLES / absorb boundary ones.

        `_pages` carries virtual PAGE ids the caller already derived -- same
        contract as the base allocator, and honoured for the same reason: deriving
        them again via `torch.unique` is a data-dependent-shape op, i.e. a HOST
        SYNC on the per-step free path.
        """
        with record_function("FloatBand.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if self.free_group is not None:
                self.free_group.append(self._copy_for_free_group(free_index))
                return
            # Page-derivation ladder, as `_free_lazy`: caller ids, the ps==1
            # identity, then dedup. No stale-slot assert -- callers must not
            # double-free (a tombstoned page would join the hole list); the
            # chain allocator's filters uphold it and the byte verifier catches a miss.
            free_v_pages_raw = free_index.detach().to(torch.int64)
            if _pages is not None:
                free_v_pages = _pages
            elif self.page_size == 1:
                free_v_pages = free_v_pages_raw
            else:
                free_v_pages = torch.unique(free_v_pages_raw // self.page_size)
            # The same fused lookup + tombstones as the END free path avoids
            # three separate launches on each sliding-window retirement.
            freed_p_pages = free_unbind_inplace(
                free_v_pages.contiguous(),
                self.ids.virtual_to_physical,
                self.ids.physical_to_virtual,
            )
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._free_phys_pages = torch.cat([self._free_phys_pages, freed_p_pages])
            # Park is sync-free (span/hole COUNTS only); boundary absorption is
            # DEFERRED -- see `_absorb_span_boundary_holes`.
            self._holes_dirty = True
            self._park_if_empty()

    def free_physical(self, free_index: torch.Tensor) -> None:
        """Release physical reservations using the float's hole/span bookkeeping."""
        if free_index is None or free_index.numel() == 0:
            return
        assert free_index.numel() % self.page_size == 0
        physical_pages = (
            free_index.detach().to(torch.int64)[:: self.page_size] // self.page_size
        )
        self._wait_band_transfers()
        virtual_pages = self.ids.physical_to_virtual[physical_pages]
        bound_pages = virtual_pages[virtual_pages >= 0]
        self.ids.virtual_to_physical.index_fill_(0, bound_pages, -1)
        self.ids.physical_to_virtual.index_fill_(0, physical_pages, -1)
        self._free_phys_pages = torch.cat([self._free_phys_pages, physical_pages])
        self._holes_dirty = True
        self._park_if_empty()

    def _park_if_empty(self) -> bool:
        """Reset the span and go frontier-transparent once no live page
        remains. Sync-free: `_live_pages()` is span minus hole COUNT, both
        host-side (`numel()` is tensor metadata). Returns whether it parked."""
        if self._live_pages() != 0:
            return False
        self._reset_watermarks()
        self._free_phys_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        self._holes_dirty = False
        return True

    def _absorb_span_boundary_holes(self) -> int:
        """Shrink the span past any holes touching its boundaries (zero-copy),
        returning the number of pages handed back to the neighbours.

        DEFERRED, not per-free: deciding how far to walk needs the hole set on the
        HOST, so this is the float's one D2H, and doing it per free put a host sync
        on the per-decode-step path. Callers place it where a sync is already free
        (the per-step opportunistic flush) or already warranted (the head of the
        shortfall ladder). Skipping it is only ever CONSERVATIVE: the span reads
        wider than its live content, while `_live_pages()`, hence transparency and
        the byte-conservation identity, stay exact.
        """
        if self._park_if_empty():
            self._holes_dirty = False
            return 0
        if not self._holes_dirty or self._free_phys_pages.numel() == 0:
            # Nothing freed since the last absorb: both boundaries are still live,
            # so the walk provably finds nothing -- skip the D2H.
            self._holes_dirty = False
            return 0
        self._holes_dirty = False
        before = self._span_pages()
        holes = set(self._free_pages_snapshot())
        changed = False
        while self.low_wm_page in holes:
            holes.remove(self.low_wm_page)
            self.low_wm_page += 1
            changed = True
        while (self.high_wm_page - 1) in holes:
            holes.remove(self.high_wm_page - 1)
            self.high_wm_page -= 1
            changed = True
        if changed:
            remaining = sorted(holes)
            self._set_free_pages(self._to_device(remaining), host=remaining)
        return before - self._span_pages()

    # -- on-demand data movement --

    def make_room(self, *, side: str, min_bytes: int) -> int:
        """Open >= ``min_bytes`` of CONTIGUOUS free space between this pool's
        ``side`` boundary and the region bound on that side, relocating the
        minimum set of live boundary pages (holes-first destinations, then the far
        gap). Returns the bytes now open on ``side``; a result < ``min_bytes``
        means the ask is impossible now or movement is gated, and state is then
        unchanged.
        Scheduler-phase only; stream safety is owned HERE, not by the caller:
        the copy waits only a forward that writes a moved page, after the plan
        and the write sets are read on the host, and the vacated pages are
        fenced. Moves at most min(L_live, G) pages: every live page when the ask
        exceeds them.
        """
        assert side in ("low", "high"), f"side must be 'low'|'high'; got {side!r}"
        epp = self.entry_bytes_per_page
        gap_low, gap_high = self._gap_pages()
        gap_side_bytes = (gap_low if side == "low" else gap_high) * epp
        if self.moves_blocked():
            return gap_side_bytes
        lo, hi = self._region_bounds_pages()
        if gap_side_bytes >= min_bytes or self._is_frontier_transparent():
            return gap_side_bytes

        # Capacity: even packing every live page flush against the far side
        # cannot open more than (region - live) bytes.
        live = self._live_pages()
        if (hi - lo - live) * epp < min_bytes:
            return gap_side_bytes  # impossible now; untouched

        need_pages = (min_bytes - gap_side_bytes + epp - 1) // epp

        holes = set(self._free_pages_snapshot())
        span = range(self.low_wm_page, self.high_wm_page)
        live_pages_sorted = [p for p in span if p not in holes]

        if need_pages >= live:
            # Pack only far enough to satisfy this side. Packing against the
            # opposite edge would erase room reserved for that neighbour.
            gap_pages = -(-min_bytes // epp)
            if side == "high":
                end = hi - gap_pages
                final = list(range(end - live, end))
            else:
                start = lo + gap_pages
                final = list(range(start, start + live))
            self._relocate_to_positions(live_pages_sorted, final)
            gap_low2, gap_high2 = self._gap_pages()
            return (gap_low2 if side == "low" else gap_high2) * epp

        # Boundary relocation (G < L_live):
        # Sources: live pages nearest the demanded side, retreating inward.
        if side == "high":
            srcs = list(reversed(live_pages_sorted))[: min(need_pages, live)]
        else:
            srcs = live_pages_sorted[: min(need_pages, live)]
        src_set = set(srcs)

        # Strictly on the far side of EVERY source: keeps the batched move
        # src/dst-disjoint and actually retreats the edge.
        if side == "high":
            usable_holes = sorted(h for h in holes if h < min(srcs))
        else:
            usable_holes = sorted((h for h in holes if h > max(srcs)), reverse=True)
        dsts: List[int] = list(usable_holes[: len(srcs)])
        n_fresh = len(srcs) - len(dsts)
        if n_fresh > 0:
            # Far-gap feasibility for the fresh destinations.
            if side == "high":
                if n_fresh > gap_low:
                    return gap_side_bytes
                dsts += list(
                    range(self.low_wm_page - 1, self.low_wm_page - 1 - n_fresh, -1)
                )
            else:
                if n_fresh > gap_high:
                    return gap_side_bytes
                dsts += list(range(self.high_wm_page, self.high_wm_page + n_fresh))

        if srcs:
            write_sets = self._inflight_write_sets()
            transfer_sets = self._transfer_page_sets()
            src_t, dst_t = self._to_device(srcs), self._to_device(dsts)
            self._wait_writers(srcs, write_sets, transfer_sets)
            self._wait_fences_on([p for p in dsts if p not in holes])
            self._move_pages_and_rebind(src_t, dst_t)
            self.ids.physical_to_virtual.index_fill_(0, src_t, -1)
            # The sources end outside the new span: the edge retreats past them.
            self._fence_vacated(srcs, self._readers_of(srcs, transfer_sets))

        # Reconstruct the span from final live positions: everything between
        # the extremes is span; non-live pages inside are holes.
        final_live = sorted((set(live_pages_sorted) - src_set) | set(dsts))
        self.low_wm_page = final_live[0]
        self.high_wm_page = final_live[-1] + 1
        final_live_set = set(final_live)
        new_holes = [
            p
            for p in range(self.low_wm_page, self.high_wm_page)
            if p not in final_live_set
        ]
        self._set_free_pages(self._to_device(new_holes), host=new_holes)
        self._holes_dirty = True  # the span moved; re-check its boundaries
        self._absorb_span_boundary_holes()

        gap_low2, gap_high2 = self._gap_pages()
        return (gap_low2 if side == "low" else gap_high2) * epp

    def relocate_into(self, first: int, end: int) -> int:
        """Move this float into the page block ``[first, end)``: live pages
        already inside stay, the others are copied to free pages of the block,
        and the block's remaining pages become holes. The caller guarantees the
        block holds every live page, contains every fixed page and meets no
        other band. Returns pages moved."""
        holes = set(self._free_pages_snapshot())
        live = [p for p in range(self.low_wm_page, self.high_wm_page) if p not in holes]
        assert len(live) <= end - first, "the block cannot hold the live pages"
        inside = {p for p in live if first <= p < end}
        srcs = [p for p in live if p not in inside]
        assert not set(srcs) & self.fixed_pages(), (
            f"[{self.sub_pool_name}] the block [{first}, {end}) leaves a fixed "
            "page behind"
        )
        # Not live, so never a source: one batched, disjoint move.
        dsts = [p for p in range(first, end) if p not in inside][: len(srcs)]
        if srcs:
            write_sets = self._inflight_write_sets()
            transfer_sets = self._transfer_page_sets()
            src_t, dst_t = self._to_device(srcs), self._to_device(dsts)
            self._wait_writers(srcs, write_sets, transfer_sets)
            self._wait_fences_on([p for p in dsts if p not in holes])
            self._move_pages_and_rebind(src_t, dst_t)
            self.ids.physical_to_virtual.index_fill_(0, src_t, -1)
            # Every source lies outside the block, so outside the new span.
            self._fence_vacated(srcs, self._readers_of(srcs, transfer_sets))
        kept = inside | set(dsts)
        self.low_wm_page, self.high_wm_page = first, end
        new_holes = [p for p in range(first, end) if p not in kept]
        self._set_free_pages(self._to_device(new_holes), host=new_holes)
        return len(srcs)

    def _relocate_to_positions(self, live_sorted: List[int], final: List[int]) -> int:
        """Order-preserving relocation of the live pages onto the ``final``
        positions (an ascending hole-free block). Batched disjoint move when
        possible, else ORDERED singleton moves: by induction each destination is a
        hole or an already-vacated source. Returns pages moved.
        """
        assert len(live_sorted) == len(final)
        pairs = [(s, d) for s, d in zip(live_sorted, final) if s != d]
        if pairs:
            src_set = {s for s, _ in pairs}
            dst_set = {d for _, d in pairs}
            live_set = set(live_sorted)
            transfer_sets = self._transfer_page_sets()
            if src_set.isdisjoint(dst_set):
                write_sets = self._inflight_write_sets()
                src_t = self._to_device([s for s, _ in pairs])
                dst_t = self._to_device([d for _, d in pairs])
                self._wait_writers(src_set, write_sets, transfer_sets)
                self._wait_fences_on([d for d in dst_set if d not in live_set])
                self._move_pages_and_rebind(src_t, dst_t)
                self.ids.physical_to_virtual.index_fill_(0, src_t, -1)
                # Disjoint: every source lies outside the final block.
                self._fence_vacated(
                    sorted(src_set), self._readers_of(src_set, transfer_sets)
                )
            else:
                # Interior packing can move pages in both directions. For an
                # order-preserving mapping, leftward moves are safe in ascending
                # source order and rightward moves in descending source order;
                # the two groups cannot overwrite each other's unread sources.
                ordered = [pair for pair in pairs if pair[1] < pair[0]] + [
                    pair for pair in reversed(pairs) if pair[1] > pair[0]
                ]
                # Built ONCE: a copy per page in the loop is an H2D per page, and
                # a shift can span the whole pool.
                src_all = self._to_device([s for s, _ in ordered])
                dst_all = self._to_device([d for _, d in ordered])
                # Some destinations are sources vacated earlier in this pass, which
                # a forward or a transfer may still read: wait every recorded
                # forward, and every transfer touching a source.
                self._settle_inflight_forward()
                for signal, pages, _, _ in transfer_sets:
                    if not src_set.isdisjoint(pages):
                        self._wait_on_stream(signal)
                self._wait_fences_on([d for d in dst_set if d not in live_set])
                for i in range(len(ordered)):
                    s_t, d_t = src_all[i : i + 1], dst_all[i : i + 1]
                    self._move_pages_and_rebind(s_t, d_t)
                    self.ids.physical_to_virtual.index_fill_(0, s_t, -1)
        if final:
            self.low_wm_page = final[0]
            self.high_wm_page = final[-1] + 1
        else:
            self._reset_watermarks()
        self._set_free_pages(
            torch.empty(0, dtype=torch.int64, device=self.device), host=[]
        )
        return len(pairs)

    def _byte_accounting_violations(self) -> List[str]:
        out: List[str] = []
        total = self.arena.total_bytes
        lo_b, hi_b = self._byte_low_frontier(), self._byte_high_frontier()
        if not self._is_frontier_transparent() and not (0 <= lo_b <= hi_b <= total):
            out.append(
                f"[{self.sub_pool_name}] float span out of bounds: "
                f"low={lo_b}, high={hi_b}, total={total}"
            )
        # Independent live count from the p2v table -- `_live_pages()` is DERIVED
        # as span - holes, so checking against it would be circular.
        if self._span_pages() > 0:
            bound = int(
                (
                    self.ids.physical_to_virtual[self.low_wm_page : self.high_wm_page]
                    != -1
                )
                .sum()
                .item()
            )
            if self._span_pages() != bound + self._hole_pages():
                out.append(
                    f"[{self.sub_pool_name}] float span {self._span_pages()} != "
                    f"p2v-bound {bound} + holes {self._hole_pages()}"
                )
        out.extend(self._capacity_memo_violations())
        return out

    def _flush(self, *, urgent: bool) -> int:
        """Boundary absorption only -- never data movement. For a float the
        `_free_phys_pages` entries are INTERIOR HOLES, reusable assets by design,
        not a compaction backlog; relocation happens on demand via `make_room` /
        `compact_holes`. What a flush point buys is handing back unneeded span."""
        return self._absorb_span_boundary_holes()

    def flush_opportunistic(self) -> int:
        """Public gated wrapper around `_flush(urgent=False)`. The override exists
        only for the gate: the base keys its fast path on `lazy_compaction`, which
        a float never has, so a float keys on `_holes_dirty` instead."""
        with record_function("FloatBand.flush_opportunistic"):
            if not self._holes_dirty or self._free_phys_pages.numel() == 0:
                return 0
            return self._flush(urgent=False)

    def compact_holes(self, *, retreat_side: str) -> int:
        """Close ALL interior holes by packing live pages toward the side
        OPPOSITE ``retreat_side`` (order-preserving), shrinking the span on
        ``retreat_side`` by the hole count. Returns pages moved."""
        assert retreat_side in ("low", "high")
        if self.moves_blocked() or self._hole_pages() == 0:
            return 0
        # `_relocate_to_positions` settles before its first copy.
        holes = set(self._free_pages_snapshot())
        live_sorted = [
            p for p in range(self.low_wm_page, self.high_wm_page) if p not in holes
        ]
        if retreat_side == "high":
            final = list(range(self.low_wm_page, self.low_wm_page + len(live_sorted)))
        else:
            final = list(range(self.high_wm_page - len(live_sorted), self.high_wm_page))
        return self._relocate_to_positions(live_sorted, final)

    # -- band-incompatible base APIs --

    def bind_peer(self, peer: Band) -> None:  # pragma: no cover
        raise AssertionError(
            "float middles must be wired via bind_low_peer/bind_high_peer"
        )
