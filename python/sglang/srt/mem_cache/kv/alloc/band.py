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

import bisect
import contextlib
import enum
import functools
import inspect
import logging
import os
from typing import (
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
from torch.profiler import record_function

from sglang.kernels.ops.memory.virtual_slot import (
    alloc_bind_inplace,
    free_unbind_inplace,
)
from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.mem_cache.kv.alloc.forward_steps import (
    NO_PENDING_PAGES,
    PendingPages,
)
from sglang.srt.mem_cache.kv.alloc.host_free_list import HostFreeList
from sglang.srt.mem_cache.kv.alloc.id_space import UnitIds
from sglang.srt.mem_cache.kv.alloc.invariants import (
    FREE_OUTSIDE_RESERVATION,
    GEOMETRY_READY,
    HOST_READ_BEFORE_WAIT,
)
from sglang.srt.mem_cache.kv.alloc.page_usage import (
    Clearance,
    Consumer,
    PageUsage,
)
from sglang.srt.mem_cache.kv.alloc.signals import (
    NetworkWaiter,
    ReceiveHold,
    StreamTransfer,
    TransferRelease,
    keys_of,
    signal_fired,
    unreleased_transfers,
    wait_signals,
)
from sglang.srt.mem_cache.kv.memory.arena import KVArena
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.invariants import expect, expect_host, resolve_level

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


class Release(enum.Enum):
    """Why pages return to a band's free list (`Band.release`)."""

    # Freed by their owner: work launched before the free may still touch them.
    FREE = "free"
    # Vacated by a move, or made holes by a relocation, once their readers
    # retired or their fences are recorded.
    VACATED = "vacated"


def _page_hazard(spans: Sequence[Tuple[int, int]], page_bytes: int):
    """Whether a page meets one of the byte ``spans`` (sorted by start), by
    bisection over the spans: a page costs a lookup, not a walk of them."""
    starts = [lo for lo, _ in spans]
    # The farthest end among the spans starting at or before each one, so an
    # overlapping long span is not hidden by a later short one.
    reach = []
    farthest = None
    for _, hi in spans:
        farthest = hi if farthest is None else max(farthest, hi)
        reach.append(farthest)

    def hazarded(page: int) -> bool:
        lo, hi = page * page_bytes, (page + 1) * page_bytes
        i = bisect.bisect_left(starts, hi)
        return i > 0 and reach[i - 1] > lo

    return hazarded


def _page_runs(pages: Sequence[int]) -> List[Tuple[int, int]]:
    """``pages`` as ascending half-open runs ``[first, end)``."""
    runs: List[Tuple[int, int]] = []
    for page in sorted(pages):
        if runs and runs[-1][1] == page:
            runs[-1] = (runs[-1][0], page + 1)
        else:
            runs.append((page, page + 1))
    return runs


def _holds_stream_copy(key) -> bool:
    """Whether a release key holds a host copy's events."""
    return any(isinstance(k, StreamTransfer) for k in keys_of(key))


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


class _AllocatorOperation:
    """One allocator operation over some bands (`allocator_operation`): which
    bands read ahead on the host, everything (``read``) or their free and
    pending pages only (``geometry``), whether it moves pages, and whether a
    stream wait was queued."""

    __slots__ = ("bands", "read", "geometry", "moves", "waited")

    def __init__(self, bands, moves: bool) -> None:
        self.bands = list(bands)
        self.read: Set[int] = set()
        self.geometry: Set[int] = set()
        self.moves = moves
        self.waited = False


@contextlib.contextmanager
def allocator_operation(bands, *, moves: bool = False):
    """Scope one allocator operation (a take, a chain allocation, a flush, a
    recovery) over ``bands``. Every host read of device state must come before
    its first stream wait: a read after it parks the host until the waited
    work ends (`Band._check_host_read`). What is read ahead before that wait
    is decided by `Band._before_wait`. Nested scopes join the outer one, and a
    nested one that ``moves`` makes it move."""
    bands = [b for b in bands if b is not None]
    op = next((b._operation for b in bands if b._operation is not None), None)
    joined = [b for b in bands if b._operation is None]
    if op is None:
        op = _AllocatorOperation(bands, moves)
    else:
        op.bands.extend(joined)
        op.moves = op.moves or moves
    for b in joined:
        b._operation = op
    try:
        yield op
    finally:
        for b in joined:
            b._operation = None


def _band_operation(method):
    """Run a band's public take as one allocator operation over its chain
    (`allocator_operation`), or as part of the caller's."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with allocator_operation(self._chain_bands()):
            return method(self, *args, **kwargs)

    return wrapper


@contextlib.contextmanager
def settle_pending_for_recovery(bands):
    """The entry of every urgent recovery, before it decides anything.

    1. One host read, before any wait, of the holes and pending pages of every
       band in ``bands`` whose host copies are unknown; then each band reads
       ahead the rest its operations read (`Band._read_ahead`: the pages
       unfinished forwards write and live transfers touch).
    2. Each band with pending pages settles all of them (`Band.settle_pending`),
       installing the read as its free list's host copy.
    3. A float the settle left with no live page parks.
    Inside, the recovery reads no device state on the host (asserted). Nested
    entries over ready bands do nothing.
    """
    bands = tuple(b for b in bands if b is not None)
    # The operation spans the bands' chains: a wait of the recovery comes
    # after every chain band read ahead, as the next one may read it.
    chain = {id(c): c for b in bands for c in b._chain_bands()}
    with (
        allocator_operation(chain.values(), moves=True) as op,
        _settled_for_recovery(bands, op),
    ):
        yield


@contextlib.contextmanager
def _settled_for_recovery(bands, op):
    reading = [
        b
        for b in bands
        if b._host_free is None or (b._pending_until_retire and b._pending_host is None)
    ]
    if reading:
        parts = []
        for b in reading:
            b._check_host_read("the free list and pending pages")
            parts.append(b._free_phys_pages)
            parts.extend(b._pending_until_retire.pages())
        host = torch.cat(parts).tolist()
        at = 0
        for b in reading:
            n_free = int(b._free_phys_pages.shape[0])
            n_pending = b._pending_until_retire_pages
            b._set_free_pages(b._free_phys_pages, host=host[at : at + n_free])
            b._pending_host = host[at + n_free : at + n_free + n_pending]
            at += n_free + n_pending
    for b in bands:
        b._read_ahead()
        op.read.add(id(b))
    # The rest of the chain too, before any settle waits.
    for c in op.bands:
        if id(c) not in op.read:
            c._read_ahead()
            op.read.add(id(c))
    for b in bands:
        if b._pending_until_retire:
            b.settle_pending(prepared_holes=b._free_pages_snapshot() + b._pending_host)
            if b.grow_direction == "float":
                b._park_if_empty()
    for b in bands:
        b._recovery_depth += 1
    try:
        yield
    finally:
        for b in bands:
            b._recovery_depth -= 1


def flush_deferred_free_group(
    allocator,
    pending_groups: Sequence[Optional[Sequence[torch.Tensor]]],
) -> None:
    """Apply queued frees and reopen the caller's free-group scope."""
    if allocator.free_group is None or not any(pending_groups):
        return
    allocator.free_group_end()
    allocator.free_group_begin()


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
    # Freed pages pending until the forward in flight at their free is processed.
    _pending_until_retire: _CapacityField[PendingPages] = _CapacityField()
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
        lazy_compaction: bool = True,
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
        # The chain's step counter: pages freed while a forward is in flight
        # are pending until its step is processed.
        self.forward_steps = None
        # Oldest step first. They count as free; a take that needs them, and
        # every urgent recovery, first settles them (`settle_pending`).
        self._pending_until_retire = NO_PENDING_PAGES
        # Their pages on the host, in batch order, read ahead (`_read_ahead`)
        # before any wait; None when unknown.
        self._pending_host: Optional[List[int]] = None
        # Open recovery entries over this band (`settle_pending_for_recovery`).
        self._recovery_depth = 0
        # The allocator operation this band takes part in (`allocator_operation`).
        self._operation: Optional[_AllocatorOperation] = None
        self.arena = arena
        self.sub_pool_name = sub_pool_name
        self.max_slots = max_slots
        self.grow_direction = band.grow_direction
        self.entry_bytes = band.entry_bytes
        self.min_slot_index = arena.first_entry(sub_pool_name)
        self.is_id_owner = is_id_owner
        # Zero page envelopes on hand-out -- see _maybe_zero_pages.
        self._zero_pages_on_alloc = band.zero_on_handout
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

        # --- Lazy compaction state ---
        # An end compacts lazily, behind the forwards that may still read the
        # pages it moves; a float moves only when a planned layout relocates it.
        assert lazy_compaction or band.grow_direction == "float", (
            f"Band({sub_pool_name!r}): an end band compacts lazily"
        )
        # Pending pages (`usage.pending_pages`): compaction srcs whose remap
        #   completed but whose readers haven't retired -- reusing one races a READ.
        # `live_page_count`: CPU slot-conservation counter, invariant under compaction.
        # KV copy and v2p/p2v remap both run on `schedule_stream`, so single-stream
        # ordering serializes them -- no separate copy-done event needed.
        self.lazy_compaction = lazy_compaction
        # The host copy of `_free_phys_pages`, or None when unknown; see the
        # property below.
        self._host_free: Optional[HostFreeList] = None
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
        # Move batches that may have moved a held receive row (one issued
        # while a hold is open and not fixed in place), and whether a relief
        # has the held rows fixed now.
        self.held_row_moves = 0
        self.holds_fixed = False
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

        # Epoch-keyed memo for the capacity view: pure between mutations, but
        # schedulers read it O(queue) times per step.
        self._avail_memo_epoch: Optional[int] = None
        self._avail_memo_tokens: int = 0

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
        steps = self.forward_steps
        assert steps is None or steps.in_flight() or not self._pending_until_retire, (
            f"{self.sub_pool_name}: pages pending with no step in flight"
        )
        # A clear runs with no forward in flight: pending pages are free.
        self._pending_until_retire = NO_PENDING_PAGES
        self._pending_host = []
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
        # A list rebuilt on the device is no longer known on the host, unless
        # it is empty (its length is metadata).
        self._set_free_pages(pages, host=None)

    def _set_free_pages(
        self,
        pages: torch.Tensor,
        *,
        host: Optional[List[int]],
        ascending: bool = False,
    ) -> None:
        """Rebind the free list together with its host copy, in the same order
        (``ascending`` when it is known sorted); an empty list's copy is
        always known. The host list is owned from here on."""
        self._free_pages = pages
        if host is None and int(pages.shape[0]) == 0:
            host = []
        self._host_free = (
            None if host is None else HostFreeList(host, ascending=ascending)
        )

    @property
    def _free_pages_host(self) -> Optional[List[int]]:
        """A copy of the free list's host copy; None when it is unknown."""
        return None if self._host_free is None else self._host_free.to_list()

    def _known_free_pages_host(self, *, after_wait: bool = False) -> HostFreeList:
        """The free list's host copy, read from the device only when unknown,
        which must not happen ``after_wait`` (a read behind a stream wait parks
        the host until the forward ends: it is read ahead)."""
        if self._host_free is None:
            assert not after_wait, (
                f"{self.sub_pool_name}: the free list read on the host after a "
                "stream wait"
            )
            self._check_host_read("the free list")
            self._host_free = HostFreeList(self._free_pages.tolist())
        return self._host_free

    def _free_pages_snapshot(self, *, after_wait: bool = False) -> List[int]:
        """The free list on the host, as a copy (`_known_free_pages_host`)."""
        return self._known_free_pages_host(after_wait=after_wait).to_list()

    def _free_pages_host_known(self) -> bool:
        """Whether the free list's host copy matches it."""
        return self._host_free is not None

    def _assert_ready_for_geometry(self, where: str) -> None:
        """Layout-changing geometry decides on holes and live pages only: the
        band holds no pending page and, inside an urgent recovery, its free
        list's host copy matches it (the recovery's entry made both true).
        Signalled at invariant level WARN (logged) and STRICT (raises)."""
        level = resolve_level()
        if level < InvariantCheckLevel.WARN:
            return
        expect_host(
            GEOMETRY_READY,
            not self._pending_until_retire,
            msg=f"{self.sub_pool_name}: {where} with "
            f"{self._pending_until_retire_pages} pages pending until their "
            "step is processed",
            level=level,
        )
        expect_host(
            GEOMETRY_READY,
            not (self._recovery_depth and not self._free_pages_host_known()),
            msg=f"{self.sub_pool_name}: {where} with no host copy of its free list",
            level=level,
        )

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
        network transfer, or a tuple of them): an event on the GPU, an
        unreleased network transfer on the host (`network_waiter`). The
        read-ahead runs before the fired check, so every rank reads the device
        at the same points: whether an event fired decides only whether a wait
        is queued, never a layout."""
        self._before_wait()
        if signal_fired(event):
            return
        wait_signals(event, network_waiter=self.network_waiter)

    def _before_wait(self) -> None:
        """Before a stream wait: every band of the current allocator operation
        that has not yet read ahead does (`_read_ahead`), once per operation,
        and the operation counts as waited; without an operation, every chain
        band reads ahead. Called whether or not the wait is then needed, so
        every rank reads at the same points."""
        op = self._operation
        if op is None:
            for band in self._chain_bands():
                band._read_ahead()
            return
        for band in op.bands:
            if id(band) in op.read:
                continue
            if op.moves:
                band._read_ahead()
                op.read.add(id(band))
            elif id(band) not in op.geometry and band._fence_may_order_takes():
                band._read_ahead_geometry()
                op.geometry.add(id(band))
        op.waited = True

    def _check_host_read(self, what: str) -> None:
        """A device read on the host must not follow a stream wait of the
        same allocator operation: the read would park the host until the
        waited work ends. Signalled at invariant level WARN (logged) and
        STRICT (raises)."""
        op = getattr(self, "_operation", None)
        if op is not None and op.waited:
            expect_host(
                HOST_READ_BEFORE_WAIT,
                False,
                msg=f"{self.sub_pool_name}: {what} read on the host after a "
                "stream wait of the same allocator operation",
                level=resolve_level(),
            )

    def _keeps_host_copy(self) -> bool:
        """Whether a take keeps the free list's host copy: inside an urgent
        recovery, an operation in which this band read ahead, or while a fence
        may make a take choose holes on the host. Elsewhere a take drops it,
        so it does no host work in proportion to the holes. Keeping it costs
        a take only the pages it takes (`HostFreeList`)."""
        op = self._operation
        return (
            bool(self._recovery_depth)
            or (op is not None and (id(self) in op.read or id(self) in op.geometry))
            # A fenced band chooses holes on the host: a dropped copy would be
            # read back from the device, behind whatever the stream queued.
            or self._fence_may_order_takes()
        )

    def _reach_bytes(self) -> Tuple[int, int]:
        """The bytes ``[lo, hi)`` this band's span may cover before a move:
        its span, and the gap up to the nearest opaque frontier it grows
        into (both gaps for a float)."""
        epp = self.entry_bytes_per_page
        first, end = self._span_range()
        lo, hi = first * epp, end * epp
        if self.grow_direction != "down":
            hi = max(hi, self._chain_low_frontier_above_bytes())
        if self.grow_direction != "up":
            lo = min(lo, self._chain_high_frontier_below_bytes())
        return lo, hi

    def _fence_may_order_takes(self) -> bool:
        """Whether an outstanding fence of any chain band's ledger meets the
        bytes this band may span before a move, so that a take here chooses its
        holes on the host. A fence only ever becomes done, so a band this skips
        before a wait needs no host read after it."""
        chain = self._chain_bands()
        if not any(band.usage.has_fences() for band in chain):
            return False
        lo, hi = self._reach_bytes()
        return any(band.usage.fence_outstanding(lo, hi) for band in chain)

    def _read_ahead_geometry(self) -> None:
        """Read on the host, now, what a take that chooses its holes on the
        host reads: the free list, and the pending pages a settle adds to it."""
        self._known_free_pages_host()
        self._pending_host_snapshot()

    def _read_ahead(self) -> None:
        """Read on the host, now, everything a flush, a relocation or a take of
        this band reads from the device: the free list, the pages pending until
        their step is processed, and the pages unfinished forwards write and
        live transfers (held receive rows included) touch. A move keeps these
        copies current (`_note_moved`); a free without host ids drops the free
        list's."""
        self.usage.prune()
        self._inflight_write_sets()
        self._transfer_page_sets()
        # Held receive rows, which a relief reads (`network_rows`).
        for use in self.usage.transfers():
            if isinstance(use.release, ReceiveHold) and not use.release.released:
                self._use_physical_pages(use)
        self._known_free_pages_host()
        self._pending_host_snapshot()

    def _pending_host_snapshot(self, *, after_wait: bool = False) -> List[int]:
        """The pages pending until their step is processed, on the host, in batch
        order. Reads the device only when unknown, which must not happen
        ``after_wait``."""
        if self._pending_host is None:
            assert not after_wait, (
                f"{self.sub_pool_name}: pending pages read on the host after a "
                "stream wait"
            )
            pending = self._pending_until_retire.pages()
            if pending:
                self._check_host_read("pending pages")
            self._pending_host = torch.cat(pending).tolist() if pending else []
        return self._pending_host

    def _note_moved(self, pairs: Sequence[Tuple[int, int]]) -> None:
        """Pages moved ``src -> dst``, in this order: keep the pages the
        ledger's forwards and transfers name on the host as their ids now
        translate; a transfer recorded by physical units keeps its rows. Each
        copy is replaced, not changed in place: a caller still holding the
        sets it read before the move keeps them as they were."""
        if not pairs:
            return
        for use in self.usage.uses:
            # Physical units stay the rows an engine was handed.
            if use.host_pages is None or use.unit_tokens is not None:
                continue
            pages = set(use.host_pages)
            for src, dst in pairs:
                if src in pages:
                    pages.discard(src)
                    pages.add(dst)
            use.host_pages = pages

    def retire_forward(self, event) -> None:
        """The scheduler processed this forward's result: it, and every forward
        recorded before it, retires."""
        self.usage.retire_forward(event)

    def _retire_through(self, event) -> None:
        self.usage.retire_through(event)

    # -- size reporting --

    max_live_tokens: Optional[int] = None

    # Waits on the host for network transfers a move cannot avoid; shared by
    # every band of a chain (`ChainAllocator.set_network_waiter`).
    network_waiter: Optional[NetworkWaiter] = None

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
        # The watermark span holds the live pages, the holes and the pending ones.
        holes = int(self._free_phys_pages.numel())
        pending = len(self.usage.pending_pages) + self._pending_until_retire_pages
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

    def _available_tokens(self) -> int:
        """Tokens this band alone can take in place, its neighbours unmoved."""
        pages_by_bytes = self._current_gap_bytes() // self.entry_bytes_per_page
        pages_by_index_space = (
            self.num_pages - self.min_page_index - self._allocated_pages()
        )
        pages_extend = min(pages_by_bytes, pages_by_index_space)
        # Holes, and pages pending until their step is processed, take no new bytes.
        pages_drain = self._reusable_pages()
        return (pages_extend + pages_drain) * self.page_size

    def available_size(self) -> int:
        """Tokens this band alone can take in place now, nothing in the chain moved
        (the chain-wide read is `units_available`); memoized on the chain epoch."""
        epoch = self._chain_capacity_epoch()
        if self._avail_memo_epoch != epoch:
            self._avail_memo_tokens = self._available_tokens()
            self._avail_memo_epoch = epoch
        return self.cap_available(self._avail_memo_tokens)

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

    # The chain allocator this band is a member of, set when it joins one.
    chain_allocator = None

    @_band_operation
    def relieve(self, need_tokens: int) -> bool:
        """Have the chain lay itself out for a take of ``need_tokens`` from this
        band, by the planner's least rearrangement; whether it then fits in place."""
        chain = self.chain_allocator
        if chain is None:
            return False
        pages = -(-need_tokens // self.page_size)
        if not chain.lay_out_for({self.sub_pool_name: pages}):
            return False
        return need_tokens <= self.available_size()

    # -- the one path that hands out writable pages --

    _physical_take_scope = None
    # Host copies of the pages the latest take handed out and the holes among
    # them, kept only when `_record_acquired_host` asks; None when not known.
    _record_acquired_host = False
    _acquired_host: Optional[List[int]] = None
    _taken_holes_host: Optional[List[int]] = None

    def _consumers(self, consumer: Consumer) -> Tuple[Consumer, ...]:
        """Who writes a page handed out to ``consumer``: every hand-out but a
        host load's is zeroed first on a band that zeroes (`_maybe_zero_pages`)."""
        if self._zero_pages_on_alloc and consumer is not Consumer.HOST_LOAD:
            return (consumer, Consumer.ZERO)
        return (consumer,)

    def acquire(
        self,
        num_pages: int,
        consumer: Consumer,
        *,
        bind: Optional[torch.Tensor] = None,
        placement: Optional[Tuple[int, int]] = None,
        planned: bool = False,
    ) -> Optional[torch.Tensor]:
        """See `_acquire`; one allocator operation over the chain, or part of
        the caller's (`allocator_operation`)."""
        with allocator_operation(self._chain_bands()):
            return self._acquire(
                num_pages, consumer, bind=bind, placement=placement, planned=planned
            )

    def _acquire(
        self,
        num_pages: int,
        consumer: Consumer,
        *,
        bind: Optional[torch.Tensor] = None,
        placement: Optional[Tuple[int, int]] = None,
        planned: bool = False,
    ) -> Optional[torch.Tensor]:
        """Hand out ``num_pages`` physical pages for ``consumer`` to write: the
        only path that makes a page writable. Holes first, then pending pages
        (the oldest whole batches that cover the shortfall are settled), then
        the frontier; None on shortfall, with nothing changed. Before returning,
        the current stream waits what ``consumer`` must clear on the pages
        (`page_usage.CLEARANCE`). Counts decide every check, never whether an
        event fired.

        With ``bind`` (virtual pages) and no hole to take, the frontier pages
        are taken and bound in one kernel. ``placement``: a float's planned
        extension block; a ``planned`` take extends only into it."""
        with record_function("Band.acquire"):
            if num_pages <= 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)
            self._note_step_activity()
            consumers = self._consumers(consumer)
            free = int(self._free_phys_pages.shape[0])
            pending = self._pending_until_retire
            short = num_pages - min(num_pages, free)
            need_more = max(0, short - pending.count)
            settles = short > 0 and bool(pending)
            if need_more > 0 and (
                (planned and placement is None)
                or not self._frontier_fits(
                    need_more, placement, free + (pending.count if settles else 0)
                )
            ):
                return None
            if settles:
                # Whole oldest batches (the operation reads ahead before the
                # settle's wait, `settle_pending`); a float keeps its span (no park).
                self.settle_pending(through=pending.batches_for(short))
            n_drain = min(num_pages, int(self._free_phys_pages.shape[0]))
            # Extend, then take holes; every host read before the first wait.
            extension = None
            if need_more > 0:
                extension = self._advance_frontier(need_more, placement)
                assert extension is not None, "a checked extension failed"
            clearance = Clearance()
            if extension is not None:
                clearance.merge(self._range_clearance(*extension, consumers))
            holes = None
            # A take that asks (`_alloc_reserved` fixes exactly these) records
            # its pages on the host when known: its holes and its extension.
            record = self._record_acquired_host
            acquired: Optional[List[int]] = (
                None
                if not record
                else []
                if extension is None
                else list(range(*extension))
            )
            if n_drain > 0:
                self._taken_holes_host = None
                holes, hole_clearance = self._take_holes(n_drain, consumers)
                clearance.merge(hole_clearance)
                if self._taken_holes_host is None:
                    acquired = None
                elif record:
                    acquired += self._taken_holes_host
            self._acquired_host = acquired
            self._count_taken(num_pages)
            self._wait_clearance(clearance)
            if holes is None and bind is not None:
                first, _ = extension
                return alloc_bind_inplace(
                    bind,
                    self.ids.virtual_to_physical,
                    self.ids.physical_to_virtual,
                    first,
                )
            fresh = None
            if extension is not None:
                fresh = self._extension_pages(*extension, after_holes=holes is not None)
            pages = fresh if holes is None else holes
            if holes is not None and fresh is not None:
                pages = torch.cat([holes, fresh])
            if bind is not None:
                self.ids.bind(bind, pages)
            return pages

    def _count_taken(self, num_pages: int) -> None:
        self.live_page_count += num_pages

    def _extension_pages(
        self, first: int, end: int, *, after_holes: bool = False
    ) -> torch.Tensor:
        """The pages ``[first, end)`` an extension took: ascending, or after
        holes on a grow-down end nearest the end first (bind is
        order-agnostic)."""
        if after_holes and self.grow_direction == "down":
            return torch.arange(
                end - 1, first - 1, -1, dtype=torch.int64, device=self.device
            )
        return torch.arange(first, end, dtype=torch.int64, device=self.device)

    def _frontier_fits(
        self, num_pages: int, placement: Optional[Tuple[int, int]], holes: int
    ) -> bool:
        """Whether `_advance_frontier` would grow by ``num_pages`` with
        ``holes`` holes on the free list; changes nothing."""
        return self._frontier_extension(num_pages) is not None

    def _frontier_extension(self, num_pages: int) -> Optional[Tuple[int, int, int]]:
        """The pages ``[first, end)`` growing by ``num_pages`` would cover, and
        the watermark after; None on index-space overflow or on crossing the
        nearest non-transparent chain frontier."""
        if self.grow_direction == "up":
            first = self.watermark_physical
            new_wm = first + num_pages
            if new_wm > self.num_pages:
                return None
            # The chain above; don't extend past its low frontier.
            chain_low_pages = (
                self._chain_low_frontier_above_bytes() // self.entry_bytes_per_page
            )
            if new_wm > chain_low_pages:
                return None
            return first, new_wm, new_wm
        new_wm = self.watermark_physical - num_pages
        if new_wm < self.min_page_index - 1:
            return None
        # Backstop only: callers gate on `available_size()`, whose floor'd gap
        # already guarantees the extension fits.
        chain_high_pages = (
            self._chain_high_frontier_below_bytes() // self.entry_bytes_per_page
        )
        if new_wm + 1 < chain_high_pages:
            return None
        return new_wm + 1, self.watermark_physical + 1, new_wm

    def _advance_frontier(
        self, num_pages: int, placement: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int]]:
        """Advance the watermark by ``num_pages`` and return the pages
        ``[first, end)`` it covered; None on index-space overflow or on
        crossing the nearest non-transparent chain frontier."""
        extension = self._frontier_extension(num_pages)
        if extension is None:
            return None
        first, end, self.watermark_physical = extension
        return first, end

    def _hole_end(self) -> str:
        """The end of the ascending hole list a take draws from: an end band's
        own end, keeping holes far from the frontier for compaction."""
        return "low" if self.grow_direction == "up" else "high"

    def _take_holes(
        self, n: int, consumers: Tuple[Consumer, ...]
    ) -> Tuple[torch.Tensor, Clearance]:
        """Remove ``n`` holes from the free list for ``consumers`` and return
        them, in the order they bind, with what they must clear.

        The choice depends on retirement and releases only, never on a fired
        event, so every rank takes the same pages: holes no unretired fence
        makes ``consumers`` wait for come first, then the rest, each in order
        from `_hole_end`. An owner-freed page is never hazarded: it reaches the
        free list only after its forward was processed or waited. A fixed page
        on the free list holds nothing to keep, so it is a hole like any."""
        end = self._hole_end()
        epp = self.entry_bytes_per_page
        lo, hi = (page * epp for page in self._span_range())
        chain = self._chain_bands()
        # Byte ranges an unretired forward (or an unreleased transfer) makes
        # ``consumers`` wait to write: the only fences that order the choice.
        spans = [
            span
            for band in chain
            for span in band.usage.ordering_spans(lo, hi, consumers)
        ]
        on_host = bool(spans) or any(
            band.usage.fences_pending(lo, hi, consumers) for band in chain
        )
        if on_host:
            # The choice reads the free list on the host: read, and keep, it
            # for the rest of the operation.
            self._read_ahead_geometry()
            op = self._operation
            if op is not None:
                op.geometry.add(id(self))
        if spans:
            spans.sort()
            return self._take_holes_on_host(n, consumers, end, spans)
        if on_host:
            # A host copy's fence, or one of a retired forward whose event
            # this rank has not seen fire, orders nothing: the holes are those
            # the device takes, and only their clearance is read, on the host.
            self._sorted_free_phys_pages()
            chosen = self._host_free.peek(n, end)
            taken = self._take_free_pages_from_end(n, end)
            return taken, self._runs_clearance(_page_runs(chosen), consumers)
        return self._take_free_pages_from_end(n, end), Clearance()

    def _take_holes_on_host(
        self,
        n: int,
        consumers: Tuple[Consumer, ...],
        end: str,
        spans: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, Clearance]:
        """`_take_holes`'s choice on the host (one D2H at most): the holes the
        byte ``spans`` meet (sorted, disjoint or not) are hazarded. The holes
        are walked in order from ``end`` only until ``n`` quiet ones are found,
        so a take whose holes are quiet touches only those and the hazarded
        ones it passes; only when fewer than ``n`` are quiet does it walk them
        all and take hazarded ones too, nearest ``end`` first."""
        free = self._sorted_free_phys_pages()
        host = self._host_free
        hazarded = _page_hazard(spans, self.entry_bytes_per_page)
        quiet, passed, walked = [], [], 0
        for page in host.iter_from(end):
            walked += 1
            if hazarded(page):
                passed.append(page)
                continue
            quiet.append(page)
            if len(quiet) == n:
                break
        chosen = (quiet + passed)[:n]
        self._taken_holes_host = list(chosen)
        clearance = self._runs_clearance(_page_runs(chosen), consumers)
        # The walked holes leave the free list but those passed over and not
        # taken, which go back at ``end`` (ascending, as the list is).
        kept = passed[n - len(quiet) :] if len(quiet) < n else passed
        kept_ascending = kept if end == "low" else kept[::-1]
        host.take(walked, end, keep=kept_ascending)
        n_free = int(free.shape[0])
        if end == "low":
            rest = free[walked:]
            if kept:
                rest = torch.cat([self._to_device(kept_ascending), rest])
        else:
            rest = free[: n_free - walked]
            if kept:
                rest = torch.cat([rest, self._to_device(kept_ascending)])
        self._free_pages = rest
        self._free_phys_sorted = rest
        return self._to_device(chosen), clearance

    def _range_clearance(
        self, first: int, end: int, consumers: Tuple[Consumer, ...]
    ) -> Clearance:
        """What ``consumers`` must clear before writing this band's free pages
        ``[first, end)``: the fences of the chain they meet; fences they clear
        by waiting are removed."""
        return self._runs_clearance(((first, end),), consumers)

    def _runs_clearance(
        self, runs: Sequence[Tuple[int, int]], consumers: Tuple[Consumer, ...]
    ) -> Clearance:
        """`_range_clearance` over several disjoint page runs at once: each
        chain band's ledger is walked once for all of them."""
        epp = self.entry_bytes_per_page
        spans = [(first * epp, end * epp) for first, end in runs if first < end]
        clearance = Clearance()
        if not spans:
            return clearance
        for band in self._chain_bands():
            clearance.merge(band.usage.take_clearance_runs(spans, consumers))
        return clearance

    def _wait_clearance(self, clearance: Clearance) -> None:
        """Wait what a clearance asks on the current stream (network transfers
        on the host). A forward that must retire before a page's id reaches a
        network peer is waited on the stream too: a publish reads the id back
        after the allocation, so the peer sees it after that forward."""
        if clearance.excluded:
            raise RuntimeError(f"[{self.sub_pool_name}] a fixed page cannot be written")
        # Each signal once: zeroing and a network receive can name one forward.
        keys = tuple(
            {id(key): key for key in (*clearance.waits(), *clearance.after)}.values()
        )
        if keys:
            self._wait_on_stream(keys)

    def _clear_destinations(self, pages: Sequence[int]) -> None:
        """Before a schedule-stream copy writes ``pages`` (host ids): wait every
        hazard on them (the fences they meet)."""
        self._wait_clearance(
            self._runs_clearance(_page_runs(pages), (Consumer.SCHEDULE_COPY,))
        )

    @_band_operation
    def take_physical(self, need_size: int) -> Optional[torch.Tensor]:
        """Reserve `need_size` TOKENS (multiple of page_size) for a forward,
        returning backing physical PAGE ids, or `None` on shortfall."""
        with record_function("Band.take_physical"):
            if need_size > self.cap_available(need_size):
                return None
            if need_size <= 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)
            assert need_size % self.page_size == 0, (
                f"take_physical: need_size={need_size} must be a multiple of "
                f"page_size={self.page_size}"
            )
            return self.acquire(need_size // self.page_size, Consumer.FORWARD)

    @_band_operation
    def take_physical_pages(
        self,
        num_pages: int,
        consumer: Consumer = Consumer.FORWARD,
    ) -> Optional[torch.Tensor]:
        """``num_pages`` physical pages for ``consumer`` (`acquire`), within
        the band's live-token cap; a reservation scope in force plans them."""
        with record_function("Band.take_physical_pages"):
            scope = self._physical_take_scope
            if scope is not None and not scope.applying:
                return scope.take(self, num_pages, consumer)
            if num_pages * self.page_size > self.cap_available(
                num_pages * self.page_size
            ):
                return None
            return self.acquire(num_pages, consumer)

    def takes_at_frontier(self, take) -> bool:
        """Whether a plan's ``take`` only extends this end's frontier, with no hole
        to fill: `place_pages` then takes and binds the pages in one kernel."""
        return (
            self.grow_direction != "float"
            and not take.reuse
            and self._free_phys_pages.numel() == 0
        )

    @_band_operation
    def take_planned_pages(
        self, take, consumer: Consumer = Consumer.FORWARD
    ) -> Optional[torch.Tensor]:
        """Take a plan's pages (a `PlanTake` of the current snapshot): holes, then
        pending pages, then the planned extension; a float never grows unplanned.
        A float placed afresh first takes the holes a settle left in its old span."""
        if self.grow_direction == "float":
            if take.side == "place":
                self._park_if_empty()
            # Planned: with no fresh range the float never grows.
            return self.take_physical_pages(
                take.count(), consumer, placement=take.pages, planned=True
            )
        return self.take_physical_pages(take.count(), consumer)

    def _sorted_free_phys_pages(self) -> torch.Tensor:
        """`_free_phys_pages` in ascending order. Sorts on the device, and only
        when the list was rebound since it was last known sorted; there is no host
        sync, because the length is tensor metadata.
        """
        free = self._free_phys_pages
        if free is not self._free_phys_sorted:
            if free.numel() > 1:
                free, _ = torch.sort(free)
                self._free_pages = free
                # The host copy, when kept, stays known: sorted the same way.
                if self._host_free is not None and self._keeps_host_copy():
                    self._host_free.sort()
                else:
                    self._host_free = None
            self._free_phys_sorted = free
        return free

    def _take_free_pages_from_end(self, n: int, end: str) -> torch.Tensor:
        """Remove and return ``n`` free pages in order from the ``end`` ("low"
        or "high") of the ascending free list, as `_take_holes` orders them;
        the rest stays ascending. No host sync: the sort runs on the device.
        ``0 < n <= len(_free_phys_pages)``.
        """
        n_free = int(self._free_phys_pages.shape[0])
        assert 0 < n <= n_free, f"take {n} of {n_free} free pages"
        free = self._sorted_free_phys_pages()
        if end == "low":
            # Off the free-list view, so later rebindings don't pin it.
            taken, rest = free[:n].clone(), free[n:]
        else:
            taken, rest = free[n_free - n :].flip(0), free[: n_free - n]
        self._free_pages = rest
        # Kept only where a later step may read it (`_keeps_host_copy`); then
        # it is ascending with the list (`_sorted_free_phys_pages`), and the
        # take consumes its end in place.
        if self._host_free is not None and self._keeps_host_copy():
            self._taken_holes_host = self._host_free.peek(n, end)
            self._host_free.take(n, end)
        else:
            self._taken_holes_host = None
            self._host_free = None if int(rest.shape[0]) else HostFreeList([])
        self._free_phys_sorted = rest
        return taken

    # -- fused acquire + bind --

    @_band_operation
    def place_pages(
        self, v_pages: torch.Tensor, N: int, consumer: Consumer = Consumer.FORWARD
    ) -> Optional[torch.Tensor]:
        """Acquire ``N`` pages for ``consumer`` and bind them to virtual pages
        ``v_pages``; with no hole to take, one Triton kernel does both. Returns
        physical page ids [N], or None on shortfall."""
        with record_function("Band.place_pages"):
            if self._physical_take_scope is not None:
                physical = self.take_physical_pages(N, consumer)
                if physical is not None:
                    self.ids.bind(v_pages, physical)
                    self._maybe_zero_pages(physical)
                return physical
            if N * self.page_size > self.cap_available(N * self.page_size):
                return None
            if N == 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)
            phys_pages = self.acquire(N, consumer, bind=v_pages)
            if phys_pages is None:
                return None
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

    @_band_operation
    def alloc(
        self, need_size: int, consumer: Consumer = Consumer.FORWARD
    ) -> Optional[torch.Tensor]:
        """Allocate `need_size` virtual TOKEN ids (id-owner only), whose pages
        ``consumer`` writes first (`acquire`). Returns
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
            if need_size > self.available_size() and not self.relieve(need_size):
                return None
            num_pages = need_size // self.page_size
            if num_pages > len(self.free_virtual_ids):
                return None
            v_pages = self.free_virtual_ids[:num_pages]
            self.free_virtual_ids = self.free_virtual_ids[num_pages:]
            phys_pages = self.place_pages(v_pages, num_pages, consumer)
            if phys_pages is None:
                self.free_virtual_ids = torch.cat([v_pages, self.free_virtual_ids])
                return None
            return self._expand_pages_to_tokens(v_pages)

    @_band_operation
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

    def virtual_pages_of(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The virtual PAGE ids of virtual TOKEN ids in the band's alloc and
        free surface (DCP-widened under DCP, so a page spans `page_size` ids;
        duplicates kept). Frees and forward write sets name pages through this
        one conversion, so a freed page and the pages a forward writes compare
        in one coordinate."""
        return token_ids // self.page_size

    # -- free --

    def free(
        self, free_index: torch.Tensor, *, _pages: Optional[torch.Tensor] = None
    ) -> None:
        """Free virtual TOKEN ids: recover virtual PAGE ids, un-map v2p/p2v,
        (if id-owner) recycle the page ids, and return the physical pages to the
        free list (`release`); compaction is deferred to `_flush`.

        `_pages` carries virtual PAGE ids the caller already derived; when given,
        the data-dependent dedup is skipped. `free_index` is token-granular and
        need not be page-aligned. While a forward is in flight the physical
        pages are pending until its step is processed (`_pend_until_retire`).
        """
        with record_function("Band.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if self.free_group is not None:
                self.free_group.append(self._copy_for_free_group(free_index))
                return
            self._free_lazy(free_index, pages=_pages)

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
            self.free(reps, _pages=self.virtual_pages_of(reps))
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
                free_v_pages = torch.unique(self.virtual_pages_of(free_v_pages_raw))
            # The ids go back at once: the scheduler's write-after-read barrier
            # orders these schedule-stream writes after the forward in flight's
            # shared reads of them. One kernel for the v2p read and both
            # tombstones. Never the scalar `t[idx] = -1` form: it materialises -1
            # on the CPU and blocks the scheduler on a pageable H2D copy (~16 ms
            # per 8192-token free).
            freed_p_pages = free_unbind_inplace(
                free_v_pages, self.ids.virtual_to_physical, self.ids.physical_to_virtual
            )
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self.live_page_count -= int(freed_p_pages.shape[0])
            self._check_free_outside_reservation(freed_p_pages)
            if not self._pend_until_retire(freed_p_pages):
                self.release(freed_p_pages, Release.FREE)

    def _check_free_outside_reservation(self, freed_p_pages: torch.Tensor) -> None:
        """An owner free never overlaps a host load's reservation (a cancelled
        load unfixes first). Signal only, on the device, and only when
        invariant checks are on: the default free path reads nothing back."""
        if resolve_level() < InvariantCheckLevel.WARN:
            return
        reserved = self.usage.fixed.get(_LOAD_RESERVATION)
        if not reserved:
            return
        expect(
            FREE_OUTSIDE_RESERVATION,
            ~torch.isin(freed_p_pages, self._to_device(sorted(reserved))),
            msg=f"{self.sub_pool_name}: a free names pages reserved for a host load",
        )

    # -- pages pending until their step is processed --

    def _reusable_pages(self) -> int:
        """Pages a take reuses without growing: holes, and pages pending until
        their step is processed (after a stream wait). Every capacity count of
        a band's free pages inside its span reads this."""
        return int(self._free_phys_pages.shape[0]) + self._pending_until_retire_pages

    @property
    def _pending_until_retire_pages(self) -> int:
        """Pages pending until their step is processed; they count as free."""
        return self._pending_until_retire.count

    def _note_step_activity(self) -> None:
        """A free or an allocation: none may fall between a step's launch and
        its forward's event (`ForwardSteps.forward_recorded`)."""
        if self.forward_steps is not None:
            self.forward_steps.note_activity()

    def _pend_until_retire(self, freed_p_pages: torch.Tensor) -> bool:
        """While a forward is in flight, keep freed physical pages pending
        under the newest unprocessed step with a recorded forward: it may still
        read or write them. Return whether they were. The count is the tensor's
        shape: no host read."""
        steps = self.forward_steps
        if steps is None:
            return False
        steps.note_activity()
        key = steps.pending_key()
        if key is None:
            return False
        step, event = key
        self._pending_until_retire = self._pending_until_retire.add(
            step, event, freed_p_pages, int(freed_p_pages.shape[0])
        )
        self._pending_host = None
        return True

    def release_pending_through(self, processed_step: int) -> None:
        """Pages pending under a step at or before ``processed_step`` join the
        free list, once that step's result was processed (see `forward_steps`
        for why that point is safe)."""
        ready, rest = self._pending_until_retire.processed_through(processed_step)
        if not ready:
            return
        count = self._pending_until_retire_pages - rest.count
        host = self._pending_host
        self._pending_until_retire = rest
        self._pending_host = None if host is None else host[count:]
        self.release(ready, Release.FREE, host=None if host is None else host[:count])

    def settle_pending(
        self,
        through: Optional[int] = None,
        prepared_holes: Optional[List[int]] = None,
    ) -> None:
        """Make the ``through`` oldest pending batches (all by default) holes
        now, whole: the current stream waits each distinct event of those
        batches (on the GPU, without blocking the host), then their pages join
        the free list in one concatenation. ``prepared_holes`` is the free
        list's host copy after it, read before the wait by an urgent recovery;
        without it the host copy is kept only if it was known with the pending
        pages'. Reads nothing on the host. The forwards stay unprocessed: pages
        freed under them later still pend."""
        pending = self._pending_until_retire
        if not pending:
            return
        chunks, events, rest = pending.oldest(
            len(pending) if through is None else through
        )
        assert prepared_holes is None or not rest, "a partial settle with a copy"
        # What the rest of the operation reads, this band's pending pages
        # included, is read before the wait.
        self._before_wait()
        count = pending.count - rest.count
        host = self._pending_host
        self._pending_until_retire = rest
        self._pending_host = None if host is None else host[count:]
        keys = tuple(e for e in events if e is not None)
        if keys and not signal_fired(keys):
            # Whether they fired decides only whether a wait is queued.
            wait_signals(keys)
        if len(keys) < len(events) and (
            self.forward_stream is not None and torch.device(self.device).type == "cuda"
        ):
            # Recorded without an event: the forward is enqueued.
            torch.cuda.current_stream().wait_stream(self.forward_stream)
        if prepared_holes is not None:
            self._set_free_pages(
                torch.cat([self._free_phys_pages, *chunks]), host=prepared_holes
            )
            self._holes_dirty = True
        else:
            self.release(
                chunks, Release.FREE, host=None if host is None else host[:count]
            )

    # -- the one path that returns pages to the free list --

    _holes_dirty: bool = False

    def release(
        self,
        pages: Union[torch.Tensor, Sequence[torch.Tensor]],
        cause: Release,
        *,
        host: Optional[List[int]] = None,
        readers: Sequence[Tuple[object, bool]] = (),
    ) -> None:
        """Return physical ``pages`` (a tensor, or tensors in order) to the
        free list: the only path that does. An owner-freed page
        (``Release.FREE``) arrives quiet: freed with no forward in flight, or
        after its forward was processed or settled. ``readers`` (``(key,
        writes)``, with ``host`` naming the pages) are transfers still on the
        pages; they fence them for `acquire` to clear. The per-step `free`
        passes none: a host copy runs only over rows its owner keeps locked
        until it ends, and a network transfer's rows are freed only once it is
        released. No device read."""
        parts = [pages] if isinstance(pages, torch.Tensor) else list(pages)
        if sum(part.numel() for part in parts) == 0:
            return
        known = self._host_free
        self._free_pages = torch.cat([self._free_phys_pages, *parts])
        if known is None or host is None:
            self._host_free = None
        else:
            known.extend(host)
        self._holes_dirty = True
        if readers:
            assert host is not None, "transfer readers need the pages on the host"
            runs = list(_page_runs(host))
            for key, writes in readers:
                self.usage.vacate(
                    key, runs, page_bytes=self.entry_bytes_per_page, writes=writes
                )

    def _release_phys_pages_batch(
        self, pages: torch.Tensor, *, host: Optional[List[int]] = None
    ) -> None:
        """Release move sources whose readers retired. `_flush` calls it only
        AFTER its trailing dst-slice, so `_free_phys_pages == holes_cpu` for the
        whole walk. No watermark / `live_page_count` change: vacated srcs
        re-enter as storage. ``host``: the same pages on the host, keeping the
        free list's host copy."""
        if pages.numel() == 0:
            return
        self._stats_n_release_batch += 1
        with record_function("Band._release_phys_pages_batch"):
            self.release(pages, Release.VACATED, host=host)

    def _move_pages_and_rebind(
        self, src_pages: torch.Tensor, dst_pages: torch.Tensor
    ) -> torch.Tensor:
        """Copy live pages src->dst (disjoint sets), rebind v2p/p2v for the moved
        virtuals, record inverse history. Does NOT clear p2v[src] -- callers own
        vacated-region clearing. Returns the moved virtual page ids.
        """
        # Callers keep fixed pages out of `src_pages`.
        v_moved = self.ids.physical_to_virtual[src_pages].clone()  # read pre-wipe
        self._count_move_batch()
        self._move_pages(src_pages, dst_pages)
        self.ids.virtual_to_physical[v_moved] = dst_pages
        self.ids.physical_to_virtual[dst_pages] = v_moved
        self._inverse_history.append((src_pages, dst_pages, v_moved))
        return v_moved

    def _count_move_batch(self) -> None:
        """Count a page-move batch that may have moved a held receive row,
        on every path that rebinds moved pages."""
        if not self.holds_fixed and self.usage.holds_receive_rows():
            # A held receive row may be among the pages moved.
            self.held_row_moves += 1

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
        # A hole take's sort and reversal (`_take_free_pages_from_end`).
        torch.sort(sink)
        sink.flip(0)
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
        ids it writes in this band, for `_flush`'s write-race check. The ledger
        keeps its own copy of the ids (a device clone on the current stream), so
        call it on the stream that page moves run on, after the ids are written
        there. Forwards still running stay recorded: a pipeline stage
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
        # Negative ids (an empty tracking slot) translate to the sink. Ids a
        # host copy was handed may live on the host; translate on the device.
        phys_tokens = self.ids.translate(ids.to(self.ids.virtual_to_physical.device))
        if self.pool_page_size > 1:
            phys_tokens = (phys_tokens // self.pool_page_size).unique()
        Band._check_host_read(self, "pages named by ids")
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
                    None if use.ids is None else self._use_physical_pages(use),
                )
                for use in forwards
            ]

    def _use_physical_pages(self, use) -> Set[int]:
        """The physical pages ``use``'s ids name, in the coordinate they were
        recorded in (``use.widened``): read from the device the first time
        (`_physical_pages_of`, or `_pages_of_units` for physical units), then
        from the host copy, which `_note_moved` keeps current."""
        if use.host_pages is None:
            use.host_pages = (
                self._physical_pages_of(use.ids, widened=use.widened)
                if use.unit_tokens is None
                else self._pages_of_units(use.ids, use.unit_tokens)
            )
        return use.host_pages

    def record_transfer(
        self,
        signal,
        ids,
        *,
        writes: bool,
        release: Optional[TransferRelease] = None,
        unit_tokens: Optional[int] = None,
        widened: bool = False,
        copy: bool = True,
    ) -> None:
        """Record a transfer that reads, or with ``writes`` writes, the pages
        this band's controller ids ``ids`` name: until ``signal`` completes, or
        with ``release`` until it is released. ``unit_tokens``: ``ids`` are
        physical units of that many tokens; ``widened``: alloc-surface token
        ids, not rank-local rows. A copy of a page it writes waits ``signal``;
        a page it reads that a move vacates is reused only after it. Call it on
        the stream page moves run on: the ledger copies ``ids`` there, unless
        not ``copy``."""
        if ids is None or len(ids) == 0:
            return
        self.usage.record_transfer(
            signal,
            ids,
            writes=writes,
            release=release,
            unit_tokens=unit_tokens,
            widened=widened,
            copy=copy,
        )

    def _transfer_page_sets(
        self,
    ) -> List[Tuple[object, Set[int], bool, Optional[TransferRelease]]]:
        """Each live transfer with the physical PAGE ids it touches now, oldest
        first. Pays a D2H sync per transfer; call it before any stream wait."""
        self.usage.prune()
        # Held receive rows: nothing touches them yet, so no copy or free
        # waits on them; only a relief resolves them (`network_rows`).
        transfers = [
            use
            for use in self.usage.transfers()
            if not isinstance(use.release, ReceiveHold)
        ]
        if not transfers:
            return []
        with record_function("Band._transfer_page_sets"):
            return [
                (
                    use.signal,
                    self._use_physical_pages(use),
                    use.writes,
                    use.release,
                )
                for use in transfers
            ]

    def _pages_of_units(self, units, unit_tokens: int) -> Set[int]:
        """The physical PAGE ids physical ``units`` of ``unit_tokens`` tokens
        cover. Reads a device tensor on the host (a stream sync)."""
        if isinstance(units, torch.Tensor):
            self._check_host_read("pages of units")
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
        """The keys of the transfers that still read any of ``srcs``: a source
        they read is reused only after them. A network transfer keys on its
        release, which decides when reuse is due alike on every rank."""
        src_set = set(srcs)
        return tuple(
            signal if release is None else release
            for signal, pages, writes, release in transfer_sets
            if not writes and not src_set.isdisjoint(pages)
        )

    @classmethod
    def _reader_groups(
        cls, srcs: Sequence[int], transfer_sets
    ) -> List[Tuple[tuple, List[int]]]:
        """``srcs`` grouped by the release keys of the transfers that still
        read them (`_readers_of`), so a page no transfer reads is not held
        with one that a slow transfer reads."""
        groups: Dict[tuple, List[int]] = {}
        for src in srcs:
            groups.setdefault(cls._readers_of([src], transfer_sets), []).append(src)
        return list(groups.items())

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
        keys = tuple(
            signal
            for signal, pages, writes, _ in transfer_sets
            if writes and not src_set.isdisjoint(pages)
        )
        if writer is not None:
            keys = (writer, *keys)
        if keys:
            self._wait_on_stream(keys)
        if writer is not None:
            self._retire_through(writer)

    def _fence_vacated(self, pages: Sequence[int], readers: tuple = ()) -> None:
        """``pages`` were vacated while the newest recorded forward, and the
        ``readers`` transfers, could still read them; a later write into them
        waits those first (`_wait_fences`, `acquire`)."""
        key = tuple(k for k in (self.usage.latest_forward, *readers) if k is not None)
        if not key or not pages:
            return
        key = key[0] if len(key) == 1 else key
        self.usage.vacate(
            key, list(_page_runs(pages)), page_bytes=self.entry_bytes_per_page
        )

    def _wait_fences(self, first: int, end: int) -> None:
        """Before a schedule-stream copy writes this band's pages ``[first,
        end)``: wait each fence of the chain they meet. A waited fence is
        dropped, as later work on the stream is ordered after it; so is one
        whose event has fired, as bookkeeping only."""
        self._wait_clearance(
            self._range_clearance(first, end, (Consumer.SCHEDULE_COPY,))
        )

    def _wait_network_fences(self, first: int, end: int) -> None:
        """Wait on the host for every unreleased network transfer whose fence
        meets this band's pages ``[first, end)``."""
        lo, hi = first * self.entry_bytes_per_page, end * self.entry_bytes_per_page
        transfers = {
            id(transfer): transfer
            for band in self._chain_bands()
            for use in band.usage.network_fences(lo, hi)
            for transfer in unreleased_transfers(use.signal)
        }
        if transfers:
            self._wait_on_stream(tuple(transfers.values()))

    def _fenced(self, page: int) -> bool:
        """Whether an unfinished fence of the chain meets ``page``."""
        lo = page * self.entry_bytes_per_page
        hi = lo + self.entry_bytes_per_page
        return any(band.usage.fence_meets(lo, hi) for band in self._chain_bands())

    # -- one relief: every host read before the first stream wait --

    def held_pending_count(self) -> int:
        """Pending pages that are fixed: they stay where they are and return
        to the free list only later, so they count as occupied."""
        pending = self.usage.pending_pages
        if not pending or not self.usage.fixed:
            return 0
        return len(pending & self.fixed_pages())

    def occupied_pages(self) -> int:
        """Pages an urgent pass keeps: live pages, and pending pages that are
        fixed."""
        return self.live_page_count + self.held_pending_count()

    def network_rows(self, transfer_sets=None) -> Set[int]:
        """Non-hole pages of the span a network transfer touches until its logical
        release, or pending on one; decided from logical releases, so rank-uniform."""
        if transfer_sets is None:
            transfer_sets = self._transfer_page_sets()
        rows: Set[int] = set()
        for _, pages, _, release in transfer_sets:
            if release is not None and not release.released:
                rows.update(pages)
        rows.update(self.usage.network_held_pending())
        # Receive rows held for a publish still to come: by id until now, read
        # once and kept current on the host as pages move.
        for use in self.usage.transfers():
            if isinstance(use.release, ReceiveHold) and not use.release.released:
                rows.update(self._use_physical_pages(use))
        # Rows outside the span (a send's page moved past the frontier, a window
        # row outside its tail) are not this band's; the chain keeps layouts
        # out of their bytes (`network_held_bytes`).
        first, end = self._span_range()
        holes = set(self._free_pages_snapshot())
        return {page for page in rows if first <= page < end and page not in holes}

    def has_network_rows(self) -> bool:
        """Whether a network transfer is unreleased on this band, or a page
        pending on one: no device read."""
        if self.usage.network_held_pending():
            return True
        return any(
            use.release is not None and not use.release.released
            for use in self.usage.transfers()
        )

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
        events -- a stream-side dependency, not a host block (a retired forward
        can still be running its tail) -- except, when not urgent, a host copy
        still reading them: their bytes are fenced instead, so only a write
        into them waits it; an urgent drain waits for the network transfers
        still reading one on the host, once for all of them. No watermark /
        `live_page_count` change.
        """
        self._stats_n_drain_calls += 1
        if not self.usage.has_pending():
            return
        with record_function("Band._drain_pending_reuse"):
            ready = self.usage.take_ready_vacated(
                urgent=urgent, keep=self.fixed_pages()
            )
            waits = []
            for use in ready:
                if use.signal is None:
                    continue
                if not urgent and _holds_stream_copy(use.signal):
                    self.usage.vacate(
                        use.signal,
                        list(_page_runs(use.pages)),
                        page_bytes=self.entry_bytes_per_page,
                    )
                else:
                    waits.append(use.signal)
            keys = tuple(waits)
            if keys:
                self._wait_on_stream(keys)
                for key in keys:
                    self._retire_through(key)
            ready_tensors = [use.batch for use in ready]
            if ready_tensors:
                # Their readers retired, and the stream or their fences order
                # a write into them after their tails.
                self.release(
                    torch.cat(ready_tensors),
                    Release.VACATED,
                    host=[p for use in ready for p in use.pages],
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
        sorted-ASCENDING snapshot) and the vacated pending pages. Returns
        `(p, j)`, or `(None, j)` if none -- the hole cursor `j` is threaded back
        in so the two-pointer membership test stays O(1). `holes_cpu`/`j_in` are
        optional only for test fixtures; `_flush` always passes them. Uncommitted
        dsts already read p2v == -1, so no exclude set is needed. No page is
        pending until its step is processed during a walk (`_flush`).
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

    def _sorted_free_pages_host(self, *, after_wait: bool = False) -> List[int]:
        """The free list on the host, ascending, the device list sorted to
        match (on the host: a device sort would drop the host copy, and reading
        it back after a forward wait blocks)."""
        all_cpu = self._free_pages_snapshot(after_wait=after_wait)
        if any(a > b for a, b in zip(all_cpu, all_cpu[1:])):
            all_cpu = sorted(all_cpu)
            self._set_free_pages(
                self._to_device(all_cpu), host=list(all_cpu), ascending=True
            )
        return all_cpu

    def _absorb_boundary_holes(self, all_cpu: List[int]) -> Tuple[int, List[int]]:
        """Retreat the watermark past free pages ALREADY contiguous with it, slice
        them off `_free_phys_pages`, return ``(new_watermark, interior_holes_cpu)``.
        ``all_cpu`` is the sorted-ascending snapshot.

        The watermark never retreats past a fixed page, free or not: the band's
        extent keeps covering its fixed pages, as every plan assumes. An urgent
        pass then keeps the holes between the packed end and the farthest fixed
        page as holes, so none of them leaves both the span and the free list.
        """
        M = len(all_cpu)
        wm = self.watermark_physical
        fixed = self.fixed_pages()
        n = 0
        if self.grow_direction == "up":
            while (
                n < M
                and all_cpu[M - 1 - n] == wm - 1 - n
                and all_cpu[M - 1 - n] not in fixed
            ):
                n += 1
            new_wm = wm - n
            holes_cpu = all_cpu[: M - n]
            self._set_free_pages(self._free_phys_pages[: M - n], host=holes_cpu)
        else:
            while n < M and all_cpu[n] == wm + 1 + n and all_cpu[n] not in fixed:
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
        stay pending, or, when an urgent pass retreats the frontier past them,
        are fenced so that a later write into them waits.

        Fixed pages never move: a non-urgent walk stops at one, and an urgent
        pass packs the other live pages below the frontier its farthest fixed
        page allows, leaving the holes between free.

        `_topmost_survivor` excludes all p2v=-1 pages, so a negative virtual id in
        the batched mapping lookup is a corrupt-state bug and raises.
        """
        self.usage.prune()
        if not urgent and self.usage.holds_receive_rows():
            # Receive rows wait for their publish: leave them, and read
            # nothing of them on the device, for the step until it.
            self._drain_pending_reuse(urgent=False)
            return 0
        if urgent:
            # Its recovery entry settled the band and read its free list.
            self._assert_ready_for_geometry("an urgent compaction")
        elif self._pending_until_retire:
            # Pages pending until their step is processed stay in place: a
            # non-urgent pass only absorbs the holes at the frontier, which
            # never retreats over a pending page (it is not a hole).
            self._drain_pending_reuse(urgent=False)
            self._absorb_boundary_holes(self._sorted_free_pages_host())
            return 0
        settled = bool(self._recovery_depth)
        # Read the transfers' pages on the host before any stream wait below.
        transfer_sets = self._transfer_page_sets()
        # Only a transfer with a logical release may stop a walk: one without
        # is listed until its event fires, which ranks see at different times;
        # a copy of a page it writes waits it on the GPU (`_commit_move_batch`).
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
            # At most one batched D2H sync, before any wait of this pass.
            all_cpu = self._sorted_free_pages_host(after_wait=settled)

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
            # Srcs whose writers this walk waited: their copies are ordered after.
            waited: Set[int] = set()

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
                if write_set and src in write_set and src not in waited:
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
                            if not self.usage.forward_retired(event)
                        ]
                        write_set = set().union(
                            *(w for _, w in write_sets if w is not None)
                        )
                        if self.usage.forward_retired(latest_event):
                            latest_event = None
                        # Resume AT this src: the hole cursor already stands
                        # past the holes above it, so a resume above it would
                        # take those holes for survivors; the srcs just committed
                        # lie above it too. Its writers are now waited.
                        waited.add(src)
                        cursor = src
                        continue
                    else:
                        break  # non-urgent: top blocker stops the walk

                # Case B/C: no write race. dst from holes_cpu by cursor (no sync).
                dst = holes_cpu[dst_cursor]
                if not urgent and self._fenced(dst):
                    # A forward may still touch this hole: only a move an
                    # allocation needs waits for that.
                    break
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
            groups = self._reader_groups(srcs, transfer_sets)
            self._wait_writers(srcs, [], transfer_sets)
            self._clear_destinations(dsts)
            src_pages_t = self._to_device(srcs)
            dst_pages_t = self._to_device(dsts)
            v_moveds_t = self.ids.physical_to_virtual[src_pages_t]
            torch._assert_async(
                (v_moveds_t >= 0).all(),
                "invalid p2v mapping in Band._flush",
            )
            self._count_move_batch()
            self._move_pages(src_pages_t, dst_pages_t)
            # ONE bulk remap (single-writer on schedule_stream), issued after the
            # copies on the same stream: work ordered after it sees moved pages.
            self.ids.virtual_to_physical[v_moveds_t] = dst_pages_t
            self.ids.physical_to_virtual[dst_pages_t] = v_moveds_t
            self.ids.physical_to_virtual.index_fill_(0, src_pages_t, -1)
            self._inverse_history.append((src_pages_t, dst_pages_t, v_moveds_t))
            self._note_moved(list(zip(srcs, dsts)))
            # Src disposition, per group of srcs the same transfers read.
            # A single group reuses `src_pages_t` as the pending pages' device
            # tensor (no second H2D at drain).
            for readers, group in groups:
                if fence:
                    self._fence_vacated(group, readers)
                    continue
                group_t = src_pages_t if len(groups) == 1 else self._to_device(group)
                release = tuple(k for k in (latest_event, *readers) if k is not None)
                release = (
                    None
                    if not release
                    else release[0]
                    if len(release) == 1
                    else release
                )
                if self.usage.ended(release):
                    if readers:
                        # A host copy may still read the srcs (a backup):
                        # fence their bytes, so only a write into them waits
                        # it, not the stream now.
                        self._fence_vacated(group, readers)
                    elif release is not None:
                        # A retired forward can still run its tail: reuse after.
                        self._wait_on_stream(release)
                    released_fired.append((group_t, list(group)))
                else:
                    self.usage.vacate(
                        release,
                        (),
                        page_bytes=self.entry_bytes_per_page,
                        pages=list(group),  # copied: the caller mutates `srcs`
                        batch=group_t,
                    )

    def flush_for_allocation(self) -> int:
        """Public urgent flush used by peer allocation-pressure recovery. It
        settles the band's pending pages first (`settle_pending_for_recovery`)."""
        with record_function("Band.flush_for_allocation"):
            with settle_pending_for_recovery((self,)):
                return self._flush(urgent=True)

    def flush_opportunistic(self) -> int:
        """Public, non-urgent flush at quiescent points; never blocks
        `schedule_stream`. Fast-path the empty state: the scheduler triggers this
        very often and ~99% of calls have no holes and no pending entries.
        """
        with record_function("Band.flush_opportunistic"):
            if self._free_phys_pages.numel() == 0 and not self.usage.has_pending():
                return 0
            with allocator_operation(self._chain_bands(), moves=True):
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
            self.free(reps, _pages=self.virtual_pages_of(reps))

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

    def end_load_reservation(self) -> None:
        """The tree bound the reserved rows to owner ids and the queued load now
        names them by those ids, which it translates at submission: the rows
        move like any page from here on."""
        self.unfix_pages(_LOAD_RESERVATION)

    def _transfers_on(self, pages: Sequence[int]) -> List[Tuple[object, bool]]:
        """The transfers recorded on this band that touch ``pages`` (physical,
        on the host), as ``(key, writes)``: a copy may still read or write a
        page freed under it. A network transfer keys on its release. Reads
        the transfers' ids, before they are unbound."""
        freed = set(pages)
        if not freed:
            return []
        return [
            (signal if release is None else release, writes)
            for signal, touched, writes, release in self._transfer_page_sets()
            if not freed.isdisjoint(touched)
        ]

    def _unbind_physical(
        self, free_index: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int], List[Tuple[object, bool]]]:
        """Unbind the physical token slots ``free_index`` (whole pages) and
        return their pages on the device and the host, and the transfers still
        on them. Off the step path (a cancelled or released reservation): reads
        the device."""
        assert free_index.numel() % self.page_size == 0, (
            f"Band({self.sub_pool_name!r}).free_physical: "
            f"{free_index.numel()} tokens must contain whole pages of "
            f"size {self.page_size}"
        )
        physical_pages = (
            free_index.detach().to(torch.int64)[:: self.page_size] // self.page_size
        )
        host = physical_pages.tolist()
        readers = self._transfers_on(host)
        virtual_pages = self.ids.physical_to_virtual[physical_pages]
        bound_mask = virtual_pages >= 0
        self.ids.virtual_to_physical.index_fill_(0, virtual_pages[bound_mask], -1)
        self.ids.physical_to_virtual.index_fill_(0, physical_pages, -1)
        return physical_pages, host, readers

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
        num_pages = need_size // self.page_size
        with allocator_operation(self._chain_bands()) as op:
            if self._pending_until_retire and num_pages > int(
                self._free_phys_pages.shape[0]
            ):
                # Holes do not serve it: settle every pending page first, so
                # the take uses holes and growth only and the host copy it
                # names its pages from stays known. When even the pending pages
                # leave it short, a relief follows: the operation then moves,
                # so every chain band reads ahead what that relief reads.
                self._read_ahead_geometry()
                op.geometry.add(id(self))
                if need_size > self.available_size():
                    op.moves = True
                self._before_wait()
                self.settle_pending(
                    prepared_holes=self._free_pages_host + self._pending_host
                )
            else:
                # Read ahead: the reservation names its pages from the host.
                self._read_ahead_geometry()
                op.geometry.add(id(self))
            return self._alloc_reserved(need_size)

    def _alloc_reserved(self, need_size: int) -> Optional[torch.Tensor]:
        num_pages = need_size // self.page_size
        if need_size > self.available_size() and not self.relieve(need_size):
            return None
        self._acquired_host = None
        self._record_acquired_host = True
        try:
            physical_pages = self.take_physical_pages(num_pages, Consumer.HOST_LOAD)
        finally:
            self._record_acquired_host = False
        if physical_pages is None:
            return None
        # The take's pages as it took them on the host: no device read, and
        # none of the holes a fresh placement dropped.
        taken = self._acquired_host
        assert taken is not None and len(taken) == num_pages, (
            f"{self.sub_pool_name}: a reserved take's pages are unknown on the host"
        )
        taken = sorted(taken)
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
        pages, host, readers = self._unbind_physical(free_index)
        self.live_page_count -= int(pages.shape[0])
        self.release(pages, Release.FREE, host=host, readers=readers)


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
    moves happen only ON DEMAND, when the chain's planned layout relocates the
    float (`relocate_into`). An EMPTY
    float (no live page and none fixed) resets its span and is
    frontier-transparent: it occupies no bytes and must never wall off free
    space. A fixed page keeps the span over it, free or not.

    Floats never compact lazily (`lazy_compaction` is False): their frees and
    takes move no data, and their on-demand moves order themselves after the
    forwards and transfers that may touch the pages (`relocate_into`).
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
        assert not kwargs.setdefault("lazy_compaction", False), (
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

    def _occupied_pages(self) -> int:
        """Span pages that are not holes: live pages, and pages pending until
        their step is processed, whose bytes a forward may still touch."""
        return self._span_pages() - self._hole_pages()

    def _live_pages(self) -> int:
        return self._occupied_pages() - self._pending_until_retire_pages

    def _is_frontier_transparent(self) -> bool:
        # A fixed page keeps the span where it is, free or not, and so does a
        # page pending until its step is processed: the float stays opaque over
        # them, as every plan assumes, until they are settled.
        return self._occupied_pages() == 0 and not self.fixed_pages()

    def _allocated_pages(self) -> int:
        # Live pages: a take reuses the pending ones before it grows.
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

    def _available_tokens(self) -> int:
        gap_low, gap_high = self._gap_pages()
        gap_pages = max(gap_low, gap_high)  # a single alloc extends ONE side
        pages_by_index_space = (
            self.num_pages - self.min_page_index - self._occupied_pages()
        )
        pages_extend = min(gap_pages, pages_by_index_space)
        # Pages pending until their step is processed count as free: a take settles
        # them after a stream wait.
        return (pages_extend + self._reusable_pages()) * self.page_size

    # -- physical page primitives (holes-first) --

    @_band_operation
    def take_physical_pages(
        self,
        num_pages: int,
        consumer: Consumer = Consumer.FORWARD,
        *,
        placement: Optional[Tuple[int, int]] = None,
        planned: bool = False,
    ) -> Optional[torch.Tensor]:
        scope = self._physical_take_scope
        if scope is not None and not scope.applying:
            return scope.take(self, num_pages, consumer)
        return self.acquire(num_pages, consumer, placement=placement, planned=planned)

    def _count_taken(self, num_pages: int) -> None:
        # The span and the hole count say how many pages are live.
        pass

    def _hole_end(self) -> str:
        end = self._next_hole_end
        self._next_hole_end = "high" if end == "low" else "low"
        return end

    def _advance_frontier(
        self, num_pages: int, placement: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int]]:
        """Grow the span by ``num_pages`` and return the pages ``[first, end)``
        it gained; None with the span unchanged when they do not fit. From
        empty the span sits at the midpoint of its region, so free gap remains
        on both sides; otherwise it grows toward the roomier gap. A planned
        ``placement`` is the block to grow by."""
        if not self._frontier_fits(num_pages, placement, self._hole_pages()):
            return None
        lo, hi = self._region_bounds_pages()
        # Empty: no live page and no hole (settled pending pages are holes
        # inside the span until the float takes them or parks).
        empty = self._is_frontier_transparent() and self._hole_pages() == 0
        if placement is not None:
            start, end = placement
            if empty:
                self.low_wm_page, self.high_wm_page = start, end
            elif end == self.low_wm_page:
                self.low_wm_page = start
            elif start == self.high_wm_page:
                self.high_wm_page = end
            else:
                return None
            return start, end
        if empty:
            start = lo + (hi - lo - num_pages) // 2
            self.low_wm_page = start
            self.high_wm_page = start + num_pages
            return start, start + num_pages
        gap_low = self.low_wm_page - lo
        gap_high = hi - self.high_wm_page
        # Extend toward the roomier gap; fall back to the other side.
        if num_pages <= gap_high and (gap_high >= gap_low or num_pages > gap_low):
            start = self.high_wm_page
            self.high_wm_page += num_pages
            return start, start + num_pages
        start = self.low_wm_page - num_pages
        self.low_wm_page = start
        return start, start + num_pages

    def _frontier_fits(
        self, num_pages: int, placement: Optional[Tuple[int, int]], holes: int
    ) -> bool:
        """Whether `_advance_frontier` would grow the span by ``num_pages``
        with ``holes`` holes on the free list; changes nothing."""
        lo, hi = self._region_bounds_pages()
        empty = self._is_frontier_transparent() and holes == 0
        if placement is not None:
            start, end = placement
            if end - start != num_pages or start < lo or end > hi:
                return False
            return empty or end == self.low_wm_page or start == self.high_wm_page
        if empty:
            return num_pages <= hi - lo
        return num_pages <= max(self.low_wm_page - lo, hi - self.high_wm_page)

    @_band_operation
    def take_physical(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        assert need_size % self.page_size == 0, (
            f"take_physical: need_size={need_size} must be a multiple of "
            f"page_size={self.page_size}"
        )
        return self.take_physical_pages(need_size // self.page_size)

    @_band_operation
    def place_pages(
        self, v_pages: torch.Tensor, N: int, consumer: Consumer = Consumer.FORWARD
    ) -> Optional[torch.Tensor]:
        # Holes-first always routes through take_physical_pages (no fused
        # watermark fast path -- float alloc cadence doesn't need it).
        if N == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        phys_pages = self.take_physical_pages(N, consumer)
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
                free_v_pages = torch.unique(self.virtual_pages_of(free_v_pages_raw))
            # The same fused lookup + tombstones as the END free path avoids
            # three separate launches on each sliding-window retirement. The
            # ids go back at once (see `Band._free_lazy`).
            freed_p_pages = free_unbind_inplace(
                free_v_pages.contiguous(),
                self.ids.virtual_to_physical,
                self.ids.physical_to_virtual,
            )
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._check_free_outside_reservation(freed_p_pages)
            if not self._pend_until_retire(freed_p_pages):
                self.release(freed_p_pages, Release.FREE)
            # Park is sync-free (span/hole COUNTS only); boundary absorption is
            # DEFERRED -- see `_absorb_span_boundary_holes`.
            self._park_if_empty()

    def free_physical(self, free_index: torch.Tensor) -> None:
        """Release physical reservations using the float's hole/span bookkeeping."""
        if free_index is None or free_index.numel() == 0:
            return
        pages, host, readers = self._unbind_physical(free_index)
        self.release(pages, Release.FREE, host=host, readers=readers)
        self._park_if_empty()

    def _park_if_empty(self) -> bool:
        """Reset the span and go frontier-transparent once no live page
        remains and no page is fixed: a fixed page, free or not, keeps the span
        over it. Sync-free: `_live_pages()` is span minus hole COUNT, both
        host-side (`numel()` is tensor metadata). Returns whether it parked."""
        if not self._is_frontier_transparent():
            return False
        self._reset_watermarks()
        self._set_free_pages(
            torch.empty(0, dtype=torch.int64, device=self.device), host=[]
        )
        self._holes_dirty = False
        return True

    def settle_pending(
        self,
        through: Optional[int] = None,
        prepared_holes: Optional[List[int]] = None,
    ) -> None:
        """`Band.settle_pending`. The span stays put: settled pages are holes
        in place, even with no live page left (a take uses them at once; an
        urgent recovery parks such a float after its settle)."""
        if not self._pending_until_retire:
            return
        super().settle_pending(through, prepared_holes)
        self._holes_dirty = True

    def release_pending_through(self, processed_step: int) -> None:
        before = self._pending_until_retire_pages
        super().release_pending_through(processed_step)
        if self._pending_until_retire_pages != before:
            self._holes_dirty = True
            self._park_if_empty()

    def unfix_pages(self, key: Hashable, pages: Optional[Sequence[int]] = None) -> None:
        """As `Band.unfix_pages`; the pages it releases may have kept the span
        over boundary holes, or an empty float opaque: park now if empty, and
        let the next flush absorb the boundary."""
        super().unfix_pages(key, pages)
        self._holes_dirty = True
        self._park_if_empty()

    def _absorb_span_boundary_holes(self) -> int:
        """Shrink the span past any holes touching its boundaries (zero-copy),
        returning the number of pages handed back to the neighbours.

        DEFERRED, not per-free: deciding how far to walk needs the hole set on the
        HOST, so this is the float's one D2H, and doing it per free put a host sync
        on the per-decode-step path. Callers place it where a sync is already free
        (the per-step opportunistic flush) or already warranted (an urgent
        flush). Skipping it is only ever CONSERVATIVE: the span reads
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
        # The span never shrinks past a fixed page, free or not.
        fixed = self.fixed_pages()
        changed = False
        while self.low_wm_page in holes and self.low_wm_page not in fixed:
            holes.remove(self.low_wm_page)
            self.low_wm_page += 1
            changed = True
        while (self.high_wm_page - 1) in holes and (self.high_wm_page - 1) not in fixed:
            holes.remove(self.high_wm_page - 1)
            self.high_wm_page -= 1
            changed = True
        if changed:
            remaining = sorted(holes)
            self._set_free_pages(self._to_device(remaining), host=remaining)
        return before - self._span_pages()

    # -- on-demand data movement --

    def relocate_into(self, first: int, end: int) -> int:
        """Move this float into the page block ``[first, end)``: live pages
        already inside stay, the others are copied to free pages of the block,
        and the block's remaining pages become holes. The caller guarantees the
        block holds every live page, contains every fixed page and meets no
        other band. Returns pages moved."""
        # The relief's recovery entry settled the float and read its free
        # list (`ChainAllocator._relief`).
        self._assert_ready_for_geometry("relocate_into")
        holes = set(self._free_pages_snapshot(after_wait=bool(self._recovery_depth)))
        live = [p for p in range(self.low_wm_page, self.high_wm_page) if p not in holes]
        assert len(live) <= end - first, "the block cannot hold the live pages"
        inside = {p for p in live if first <= p < end}
        srcs = [p for p in live if p not in inside]
        assert not set(srcs) & self.fixed_pages(), (
            f"[{self.sub_pool_name}] the block [{first}, {end}) leaves a fixed "
            "page behind"
        )
        # Not live, so never a source: one batched, disjoint move.
        vacant = [p for p in range(first, end) if p not in inside]
        dsts = vacant[: len(srcs)]
        # Host reads before the first stream wait.
        if srcs:
            write_sets = self._inflight_write_sets()
            transfer_sets = self._transfer_page_sets()
        # Wait on the host for network sends still reading the block,
        # so that no hole of a span waits for one.
        self._wait_network_fences(first, end)
        # Bytes a neighbour vacated in ``dsts`` may still be read; the block's
        # other pages become holes whose fences `acquire` clears.
        if srcs:
            self._clear_destinations(dsts)
            src_t, dst_t = self._to_device(srcs), self._to_device(dsts)
            self._wait_writers(srcs, write_sets, transfer_sets)
            self._move_pages_and_rebind(src_t, dst_t)
            self._note_moved(list(zip(srcs, dsts)))
            self.ids.physical_to_virtual.index_fill_(0, src_t, -1)
            # Every source lies outside the block, so outside the new span.
            for readers, group in self._reader_groups(srcs, transfer_sets):
                self._fence_vacated(group, readers)
        kept = inside | set(dsts)
        self.low_wm_page, self.high_wm_page = first, end
        new_holes = [p for p in range(first, end) if p not in kept]
        self._set_free_pages(
            torch.empty(0, dtype=torch.int64, device=self.device), host=[]
        )
        self.release(self._to_device(new_holes), Release.VACATED, host=new_holes)
        return len(srcs)

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
            pending = self._pending_until_retire_pages
            if self._span_pages() != bound + self._hole_pages() + pending:
                out.append(
                    f"[{self.sub_pool_name}] float span {self._span_pages()} != "
                    f"p2v-bound {bound} + holes {self._hole_pages()} + pending "
                    f"{pending}"
                )
        out.extend(self._capacity_memo_violations())
        return out

    def _flush(self, *, urgent: bool) -> int:
        """Boundary absorption only -- never data movement. For a float the
        `_free_phys_pages` entries are INTERIOR HOLES, reusable assets by design,
        not a compaction backlog; relocation happens on demand, when a planned
        layout moves the float. What a flush point buys is handing back unneeded
        span."""
        if urgent:
            self._assert_ready_for_geometry("an urgent span retreat")
        return self._absorb_span_boundary_holes()

    def flush_opportunistic(self) -> int:
        """Public gated wrapper around `_flush(urgent=False)`. The override exists
        only for the gate: a float's holes are reusable assets, not a compaction
        backlog, so it keys on `_holes_dirty` instead."""
        with record_function("FloatBand.flush_opportunistic"):
            if not self._holes_dirty or self._free_phys_pages.numel() == 0:
                return 0
            with allocator_operation(self._chain_bands(), moves=True):
                return self._flush(urgent=False)

    # -- band-incompatible base APIs --

    def bind_peer(self, peer: Band) -> None:  # pragma: no cover
        raise AssertionError(
            "float middles must be wired via bind_low_peer/bind_high_peer"
        )
