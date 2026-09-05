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
"""MultiEndedAllocator: one allocator per sub-pool over a `UnifiedKVPool`.

`alloc*` run the upstream kernels ONCE in virtual space using `free_virtual_ids`
as the free-page pointer, then bind consumed virtual pages to physical pages so
`translate_kv_loc` resolves. Public methods take/return TOKEN-granular tensors;
`free_virtual_ids` and the v2p/p2v tables are page-granular. For `page_size == 1`
page math collapses to slot math byte-identically.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import torch
from torch.profiler import record_function

from sglang.kernels.ops.memory.virtual_slot import (
    alloc_bind_inplace,
    bind_inplace,
    free_unbind_inplace,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    UnifiedKVPool,
    UnifiedMLATokenToKVPool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import get_num_new_pages, next_power_of_2

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
_STATS_INSTANCES: weakref.WeakSet[MultiEndedAllocator] = weakref.WeakSet()
_SIGNAL_HANDLERS_INSTALLED = False


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
        side = "high" if band_alloc.grow_direction == "down" else "low"
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
    slack = max(b.entry_bytes_per_page for b, pages in demand.items() if pages > 0)
    if surplus["low"] < 0 and surplus["high"] < 0:
        return  # zero-sum: opening one side closes the other
    if surplus["low"] < 0 or surplus["high"] < 0:
        short, far = ("low", "high") if surplus["low"] < 0 else ("high", "low")
        target = sides[short] + max(0, f_bytes - max(0, surplus[far])) + slack
        if target > band[short]:
            flt.make_room(side=short, min_bytes=target)
        return
    if f_bytes > max(surplus.values()):
        short = "low" if surplus["low"] >= surplus["high"] else "high"
        flt.make_room(side=short, min_bytes=sides[short] + f_bytes + slack)


def _relieve_for_alloc(short_pool, need_tokens: int) -> bool:
    """THE shortfall ladder: every allocation shortfall in the unified pool runs
    exactly this, whether a single band's own alloc or a composite's coupled
    multi-band alloc. `_flush` is called unconditionally -- an eager END no-ops
    and a FLOAT always has boundary absorption to do -- so the ladder never
    branches on lazy mode, member kind, or layout.
    """
    for m in short_pool._flush_targets():
        m._flush(urgent=True)
    if need_tokens <= short_pool.available_size():
        return True
    short_pool._ask_float_for_room(need_tokens)
    return need_tokens <= short_pool.available_size()


class MultiEndedAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for one sub-pool over a `UnifiedKVPool`."""

    # Capacity-bearing state: any rebind bumps `_capacity_epoch`, invalidating
    # the chain's capacity memos (see `_CapacityField`).
    _capacity_epoch: int = 0
    watermark_physical: _CapacityField[int] = _CapacityField()
    live_page_count: _CapacityField[int] = _CapacityField()
    _free_phys_pages: _CapacityField[torch.Tensor] = _CapacityField()

    def __init__(
        self,
        *,
        kvcache,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        device: str,
        is_id_owner: bool,
        virtual_num_pages: Optional[int] = None,
        page_size: int = 1,
        shards_under_dcp: bool = False,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
        kernel_page_multiplier: Optional[int] = None,
    ):
        spec = unified_buffer.spec(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)
        # DCP shards KV tokens only. Mamba state and the SWA rows are
        # replicated, so they stay slot-granular whatever the process width is.
        self.shards_under_dcp = shards_under_dcp
        dcp_size = get_parallel().attn_dcp_size if shards_under_dcp else 1
        super().__init__(
            size=max_slots * dcp_size,
            page_size=page_size * dcp_size,
            dtype=spec.get_dtype(),
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.unified_buffer = unified_buffer
        self.sub_pool_name = sub_pool_name
        self.spec = spec
        self.max_slots = max_slots
        self.grow_direction = spec.grow_direction
        self.entry_bytes = spec.entry_bytes()
        self.min_slot_index = unified_buffer.min_slot_index(sub_pool_name)
        self.is_id_owner = is_id_owner
        # Kernel-facing page-stride scale, from the spec that owns the layout;
        # `kernel_page_multiplier=` overrides it only for tests.
        self.kernel_page_multiplier = (
            spec.blocks_per_page()
            if kernel_page_multiplier is None
            else kernel_page_multiplier
        )
        # Zero page envelopes on hand-out -- see _maybe_zero_pages.
        self._zero_pages_on_alloc = isinstance(kvcache, UnifiedMLATokenToKVPool)
        # Overlap mode: `free` drops a wait_stream(forward_stream) barrier so its
        # v2p writes + move kernel serialize after the in-flight forward.
        self.forward_stream = forward_stream

        # --- Page-aware bookkeeping ---
        # Two page sizes, equal unless decode context parallelism is on:
        # `page_size` is VIRTUAL (the scheduler, the tree cache and the alloc/free
        # surface), `pool_page_size` is the PHYSICAL rows one page occupies here.
        # `KVIndexTranslator.translate_dcp_read_ids` collapses `loc // dcp_size`
        # before reaching `translate_kv_loc*`, so everything at or below the v2p
        # table -- byte budget, compaction moves, translate -- stays on
        # `pool_page_size`.
        self.pool_page_size = page_size
        self.page_size = page_size * dcp_size
        self.num_pages = max_slots // self.pool_page_size
        # `min_page_index` = ceil(min_slot_index / pool_page_size), keeping the
        # reserved sink floor covered (see `_reserved_floor_bytes`).
        self.min_page_index = (
            self.min_slot_index + self.pool_page_size - 1
        ) // self.pool_page_size
        self.entry_bytes_per_page = self.entry_bytes * self.pool_page_size

        # v2p is indexed by VIRTUAL page id, p2v by PHYSICAL page id. A non-owner
        # consumes the owner's ids, so the two counts are unrelated.
        self.num_virtual_ids = (
            self.num_pages if virtual_num_pages is None else virtual_num_pages
        )
        # Page 0 is the padding anchor; the trailing row is the -1 sentinel.
        self.virtual_to_physical = torch.full(
            (self.num_virtual_ids + 1,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.physical_to_virtual = torch.full(
            (self.num_pages + 1,),
            -1,
            dtype=torch.int64,
            device=device,
        )

        # Chain neighbours: `low_peer` toward byte 0, `high_peer` toward
        # `total_bytes`. Ends have one (`bind_peer`), float middles have both.
        self.low_peer: Optional[MultiEndedAllocator] = None
        self.high_peer: Optional[MultiEndedAllocator] = None

        # Inverse history of relocations (spec rollback), at PAGE granularity.
        self._inverse_history: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

        # --- Lazy compaction state (all unused when lazy_compaction=False) ---
        # `_pending_reuse`: compaction-src pages whose remap completed but whose
        #   reader event hasn't fired -- reusing one races a live READ.
        # `live_page_count`: CPU slot-conservation counter, invariant under compaction.
        # KV copy and v2p/p2v remap both run on `schedule_stream`, so single-stream
        # ordering serializes them -- no separate copy-done event needed.
        self.lazy_compaction = lazy_compaction
        self._free_phys_pages: torch.Tensor = torch.empty(
            0, dtype=torch.int64, device=device
        )
        # ONE entry per BATCH, keyed by Event: `cpu_list` drives the Set update
        # (no sync); `gpu_tensor` is kept alive so drain cats it without an H2D.
        self._pending_reuse: Dict[
            torch.cuda.Event,
            Tuple[List[int], torch.Tensor],
        ] = {}
        # CPU mirror of `_pending_reuse` for O(1) membership in the survivor walk.
        self._pending_reuse_pages_cpu: Set[int] = set()
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
        # While this returns False, `_flush` must not relocate any page.
        self.disagg_move_gate: Optional[Callable[[], bool]] = None
        self._latest_forward_done_event: Optional[torch.cuda.Event] = None
        # Most-recent forward's (done_event, out_cache_loc_virtual) for `_flush`'s
        # write-race check. Single slot: at most ONE forward in flight per call
        # site; `_flush` materializes the write-set lazily, avoiding a sync here.
        self._inflight_forward: Optional[Tuple[torch.cuda.Event, torch.Tensor]] = None

        # Per-call move cap on NON-urgent `_flush`: bounds work per `on_idle()` so
        # a large backlog doesn't block ZMQ IPC. Urgent retries are uncapped.
        self._lazy_max_moves_per_call = (
            envs.SGLANG_LAZY_COMPACTION_MAX_MOVES_PER_CALL.get()
        )

        # Epoch-keyed memos for the capacity views: pure between mutations, but
        # schedulers read them O(queue) times per step.
        self._avail_memo_epoch: Optional[int] = None
        self._avail_memo_tokens: int = 0
        self._sched_avail_memo_epoch: Optional[int] = None
        self._sched_avail_memo_tokens: int = 0

        self.clear()

        logger.info(
            "[unified-memory-pool] MultiEndedAllocator(%r) ready: grow=%s, max_slots=%d, "
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

    def bind_peer(self, peer: MultiEndedAllocator) -> None:
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

    def bind_low_peer(self, peer: MultiEndedAllocator) -> None:
        self.low_peer = peer
        self._capacity_epoch += 1  # rewiring changes what the chain walks see

    def bind_high_peer(self, peer: MultiEndedAllocator) -> None:
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
        self.virtual_to_physical.fill_(-1)
        # Virtual page 0 <-> physical page 0 (padding sink).
        self.virtual_to_physical[0] = 0
        self.virtual_to_physical[-1] = -1  # trailing sentinel
        self.physical_to_virtual.fill_(-1)
        self.physical_to_virtual[0] = 0
        self.physical_to_virtual[-1] = -1
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
        self._free_phys_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        self._pending_reuse.clear()
        self._pending_reuse_pages_cpu.clear()
        self.live_page_count = 0
        self._inflight_forward = None
        self._latest_forward_done_event = None

    def clear_inverse_history(self) -> None:
        self._inverse_history.clear()

    # -- size reporting --

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
        virt_page = slot // self.page_size
        if virt_page < 0 or virt_page >= self.num_virtual_ids:
            return False
        return int(self.virtual_to_physical[virt_page].item()) != -1

    def allocator_state_str(self) -> str:
        return (
            f"sub_pool={self.sub_pool_name!r}, grow_direction={self.grow_direction}, "
            f"is_id_owner={self.is_id_owner}, page_size={self.page_size}, "
            f"min_page_index={self.min_page_index}, "
            f"num_pages={self.num_pages}, "
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
        total = self.unified_buffer.total_bytes
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
            pending = len(self._pending_reuse_pages_cpu)
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
        if self._sched_avail_memo_epoch == epoch:
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
            return self.unified_buffer.total_bytes
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

    def _growth_side_neighbor(self) -> Optional[MultiEndedAllocator]:
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
        return self._avail_memo_tokens

    def _peer_drainable_hole_bytes(self) -> int:
        """Gap bytes an urgent flush of the growth-side chain neighbor would
        release. Only `_free_phys_pages` counts -- NOT `_pending_reuse`, which
        awaits an event -- so the credit is realizable.
        """
        neighbor = self._growth_side_neighbor()
        if neighbor is None or not neighbor.lazy_compaction:
            return 0
        if neighbor.disagg_move_gate is not None and not neighbor.disagg_move_gate():
            # Not realizable: a PD transfer blocks the neighbour's compaction, so
            # crediting these bytes would admit work no flush can satisfy.
            return 0
        return len(neighbor._free_phys_pages) * neighbor.entry_bytes_per_page

    def schedulable_available_size(self) -> int:
        """Tokens allocatable AFTER a neighbor urgent-flush; alloc gates use
        `available_size()` instead. Memoized on the chain capacity epoch.
        """
        epoch = self._chain_capacity_epoch()
        if self._sched_avail_memo_epoch != epoch:
            self._sched_avail_memo_tokens = self._available_tokens(
                extra_gap_bytes=self._peer_drainable_hole_bytes()
            )
            self._sched_avail_memo_epoch = epoch
        return self._sched_avail_memo_tokens

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
        if not isinstance(blocker, FloatMultiEndedAllocator):
            return
        _float_open_short_side(blocker, {self: -(-need_tokens // self.page_size)})

    # -- physical-slot / physical-page primitives --

    def take_physical(self, need_size: int) -> Optional[torch.Tensor]:
        """Reserve `need_size` TOKENS (multiple of page_size), returning backing
        physical PAGE ids, or `None` on shortfall.
        """
        with record_function("MultiEndedAlloc.take_physical"):
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
                drained_t = self._free_phys_pages[:n_drain]
                self._free_phys_pages = self._free_phys_pages[n_drain:]
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
        with record_function("MultiEndedAlloc.take_physical_pages"):
            return self.take_physical(num_pages * self.page_size)

    def bind(self, virtual_ids: torch.Tensor, physical_ids: torch.Tensor) -> None:
        """Bind page-granular virtual ids to physical ids."""
        with record_function("MultiEndedAlloc.bind"):
            bind_inplace(
                virtual_ids,
                physical_ids,
                self.virtual_to_physical,
                self.physical_to_virtual,
            )

    def bind_pages(
        self, virtual_pages: torch.Tensor, physical_pages: torch.Tensor
    ) -> None:
        """Page-granular alias of ``bind``."""
        with record_function("MultiEndedAlloc.bind_pages"):
            self.bind(virtual_pages, physical_pages)

    # -- fused take_physical_pages + bind_pages --

    def _alloc_bind_fast_or_slow(
        self, v_pages: torch.Tensor, N: int
    ) -> Optional[torch.Tensor]:
        """Fuse `take_physical_pages` + `bind` into ONE Triton kernel when no
        holes need draining; fall through to the slow path (drains holes first)
        when holes exist. Returns physical page ids [N], or None on shortfall.
        """
        with record_function("MultiEndedAlloc._alloc_bind_fast_or_slow"):
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
                        self.watermark_physical = new_wm
                    else:
                        new_wm = start_wm - N
                        if new_wm < self.min_page_index - 1:
                            return None
                        self.watermark_physical = new_wm

                # Lowest physical id of the new range (both directions yield
                # ascending `[start_phys, start_phys + N)`).
                if self.grow_direction == "up":
                    start_phys = start_wm
                else:
                    start_phys = start_wm - N + 1

                phys_pages = alloc_bind_inplace(
                    v_pages,
                    self.virtual_to_physical,
                    self.physical_to_virtual,
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
            self.bind(v_pages, phys_pages)
            self._maybe_zero_pages(phys_pages)
            return phys_pages

    def _maybe_zero_pages(self, phys_pages: torch.Tensor) -> None:
        """Zero the page ENVELOPES on hand-out (MLA full pool only): the MLA
        kernels arithmetically mask the rows beyond seq_len, so never-written page
        bytes must read as finite values.
        """
        if not self._zero_pages_on_alloc or phys_pages.numel() == 0:
            return
        with record_function("MultiEndedAlloc._maybe_zero_pages"):
            self._kvcache.zero_physical_pages(phys_pages)

    # -- translate (virtual TOKEN ids -> physical TOKEN ids) --

    def translate_kv_loc(
        self,
        virt_tokens: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Translate token-granular virtual ids to physical ids.

        Under DCP the input is the DCP-collapsed id (`widened // dcp_size`, what
        `KVIndexTranslator.translate_dcp_read_ids` hands down), so this works on
        `pool_page_size`. ``out=`` writes in-place into a caller-owned buffer,
        required under cuda-graph capture: the captured graph records the gather
        against a fixed ``data_ptr``.
        """
        if out is not None:
            assert out.dtype == torch.int64, (
                f"translate_kv_loc: out= dtype must be int64 (matches v2p), "
                f"got {out.dtype}"
            )
            assert out.shape == virt_tokens.shape, (
                f"translate_kv_loc: out= shape {tuple(out.shape)} must match "
                f"virt_tokens shape {tuple(virt_tokens.shape)}"
            )
        with record_function("MultiEndedAlloc.translate_kv_loc"):
            return self._translate_kv_loc_impl(virt_tokens, out)

    def _translate_kv_loc_impl(
        self,
        virt_tokens: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Tombstone-safety clamp: a tombstoned v2p entry (-1) must not reach
        # `k_buffer[-1]` (illegal access under captured graph replay). Clamping to
        # 0 routes it to physical slot 0, reserved sink space holding no real data.
        ps = self.pool_page_size
        if ps == 1:
            if out is not None:
                # `index_select(out=out)` forbids index/out aliasing, but the
                # canonical caller passes `out=kv_indices` in place.
                tmp = torch.index_select(self.virtual_to_physical, 0, virt_tokens)
                tmp = torch.clamp_min(tmp, 0)
                out.copy_(tmp)
                return out
            result = torch.index_select(self.virtual_to_physical, 0, virt_tokens)
            return torch.clamp_min(result, 0)
        # ps > 1: page math. `virt_pages`/`offsets` are fresh, so they
        # cannot alias `out` -- `index_select(out=out)` is safe.
        virt_pages = virt_tokens // ps
        offsets = virt_tokens % ps
        if out is not None:
            torch.index_select(self.virtual_to_physical, 0, virt_pages, out=out)
            out.mul_(ps)
            out.add_(offsets)
            out.clamp_(min=0)  # tombstoned page: -1*ps + offset in [-ps, -1]
            return out
        phys_pages = self.virtual_to_physical[virt_pages]
        result = phys_pages * ps + offsets
        return torch.clamp_min(result, 0)

    def translate_kv_loc_for_kernel(
        self,
        virt_tokens: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Virtual token ids -> kernel-facing ids:

            kernel_id(t) = (t // ps) * (ps * kernel_page_multiplier) + t % ps

        Internal machinery (compaction, in-flight write sets) MUST keep using
        `translate_kv_loc`: kernel-facing ids are for kernels only. Tombstones (-1)
        clamp to kernel-facing id 0, the page-0 sink. int64 out; a consumer whose
        kernel ABI wants int32 narrows where it fills that buffer.
        """
        ps = self.pool_page_size
        stride = ps * self.kernel_page_multiplier
        with record_function("MultiEndedAlloc.translate_kv_loc_for_kernel"):
            pages = virt_tokens if ps == 1 else virt_tokens // ps
            offsets = None if ps == 1 else virt_tokens % ps
            if out is None:
                phys = self.virtual_to_physical[pages]
                ids = phys * stride if offsets is None else phys * stride + offsets
                return ids.clamp_(min=0)
            assert out.dtype == torch.int64, (
                f"translate_kv_loc_for_kernel: out= dtype must be int64 (matches v2p), "
                f"got {out.dtype}"
            )
            assert out.shape == virt_tokens.shape, (
                f"translate_kv_loc_for_kernel: out= shape {tuple(out.shape)} must "
                f"match virt_tokens shape {tuple(virt_tokens.shape)}"
            )
            if pages.dtype != torch.int64:
                pages = pages.to(torch.int64)
            if pages is virt_tokens:
                out.copy_(torch.take(self.virtual_to_physical, pages))
            else:
                torch.take(self.virtual_to_physical, pages, out=out)
            out.mul_(stride)
            if offsets is not None:
                out.add_(offsets)
            return out.clamp_(min=0)

    def translate_write_loc_for_kernel(
        self,
        widened_loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Widened virtual WRITE loc (`out_cache_loc`) -> kernel-facing id.

        Reads arrive already DCP-collapsed, but `out_cache_loc` does not: it still
        carries the owner rule in `loc % dcp_size`. Ids this rank does not own go
        to kernel id 0, the padding sink every write kernel skips.
        """
        parallel = get_parallel()
        dcp_size = parallel.attn_dcp_size if self.shards_under_dcp else 1
        if dcp_size == 1:
            return self.translate_kv_loc_for_kernel(widened_loc, out=out)
        with record_function("MultiEndedAlloc.translate_write_loc_for_kernel"):
            owned = (widened_loc % dcp_size) == parallel.attn_dcp_rank
            dense = self.translate_kv_loc_for_kernel(widened_loc // dcp_size)
            dense = torch.where(owned, dense, torch.zeros_like(dense))
            if out is not None:
                out.copy_(dense)
                return out
            return dense

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate `need_size` virtual TOKEN ids (id-owner only). Returns
        token-granular, page-structured ids, or None on shortfall.

        All allocator GPU ops run on `schedule_stream`; `alloc` needs no
        `wait_stream` barrier because its v2p/p2v writes are picked up by the
        forward via `forward_stream.wait_stream(schedule_stream)` in `run_batch`.
        """
        with record_function("MultiEndedAlloc.alloc"):
            assert self.is_id_owner, (
                f"MultiEndedAllocator({self.sub_pool_name!r}).alloc called on a "
                "non-id-owner allocator; use alloc_with_virtual instead"
            )
            if need_size <= 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)
            assert need_size % self.page_size == 0, (
                f"MultiEndedAllocator({self.sub_pool_name!r}).alloc: need_size="
                f"{need_size} must be a multiple of page_size={self.page_size}"
            )
            if need_size > self.available_size():
                # Shortfall: flush the PEER, not own -- see `_flush_targets`.
                if not _relieve_for_alloc(self, need_size):
                    return None
            num_pages = need_size // self.page_size
            v_pages = self.free_virtual_ids[:num_pages]
            self.free_virtual_ids = self.free_virtual_ids[num_pages:]
            phys_pages = self._alloc_bind_fast_or_slow(v_pages, num_pages)
            if phys_pages is None:
                self.free_virtual_ids = torch.cat([v_pages, self.free_virtual_ids])
                return None
            if self.page_size == 1:
                return v_pages  # v_pages already IS the token id list
            # Expand page ids to token ids: (P, 1) * S + (S,) -> (P, S) -> (P*S,).
            return (
                v_pages[:, None] * self.page_size
                + torch.arange(self.page_size, device=self.device)
            ).reshape(-1)

    def alloc_with_virtual(self, virtual_pages: torch.Tensor) -> None:
        """Take physical PAGES for caller-supplied virtual PAGE ids (not token
        ids), for a physical-holding non-owner such as the SWA `swa` sub-allocator.
        The composite snapshots the virtual pages before the id-owner consumes them.
        """
        with record_function("MultiEndedAlloc.alloc_with_virtual"):
            if virtual_pages.numel() == 0:
                return
            phys_pages = self._alloc_bind_fast_or_slow(
                virtual_pages, int(virtual_pages.numel())
            )
            assert phys_pages is not None, (
                f"MultiEndedAllocator({self.sub_pool_name!r}).alloc_with_virtual: out of "
                "physical room (the composite's byte-budget check should have caught this)"
            )

    # -- paged alloc surface --

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Allocate ``extend_num_tokens`` new tokens across ``bs`` requests,
        preserving the tail-page-reuse contract.

        Runs the kernel in VIRTUAL space (``free_page_ptr == free_virtual_ids``),
        so ``out_indices`` are virtual token ids; each consumed virtual page is
        then bound to a physical page here, else v2p stays -1 and translation
        yields negative ids (CUDA OOB).
        """
        with record_function("MultiEndedAlloc.alloc_extend"):
            assert self.is_id_owner, (
                f"alloc_extend on a non-id-owner allocator ({self.sub_pool_name!r})"
            )
            if num_new_pages is None:
                num_new_pages = get_num_new_pages(
                    seq_lens=seq_lens_cpu,
                    page_size=self.page_size,
                    prefix_lens=prefix_lens_cpu,
                )
            if num_new_pages > len(self.free_virtual_ids):
                return None
            # Lazy: physical-capacity pre-check; on shortfall run the ladder.
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not _relieve_for_alloc(self, need_tokens):
                    return None
            bs = len(prefix_lens)
            if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
                self.free_virtual_ids
            ):
                self.merge_and_sort_free()

            # Snapshot the virtual pages the kernel will consume, to bind them
            # to physical pages afterward.
            if num_new_pages > 0:
                new_virtual_pages = self.free_virtual_ids[:num_new_pages].clone()
            else:
                new_virtual_pages = None

            out_indices = torch.empty(
                (extend_num_tokens,), dtype=torch.int64, device=self.device
            )
            # `free_virtual_ids` passed as `free_page_ptr`: the kernel does
            # `page_id * page_size + offset` regardless of virtual vs physical.
            with record_function("MultiEndedAlloc.alloc_extend.kernel"):
                alloc_extend_kernel[(bs,)](
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.free_virtual_ids,
                    out_indices,
                    next_power_of_2(bs),
                    self.page_size,
                )

            # Bind the consumed virtual pages to fresh physical pages here. The
            # peer (swa side) binds the same pages via `alloc_with_virtual`.
            if new_virtual_pages is not None:
                phys_pages = self._alloc_bind_fast_or_slow(
                    new_virtual_pages, num_new_pages
                )
                if phys_pages is None:
                    return None  # defensive; pre-check should have prevented it

            self.free_virtual_ids = self.free_virtual_ids[num_new_pages:]
            return out_indices  # virtual token ids

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Allocate one new token per request (decode), preserving the
        tail-page-reuse contract. Runs in virtual space, binding each consumed
        virtual page here (else v2p stays -1 and translation goes OOB).
        """
        with record_function("MultiEndedAlloc.alloc_decode"):
            assert self.is_id_owner, (
                f"alloc_decode on a non-id-owner allocator ({self.sub_pool_name!r})"
            )
            bs = len(seq_lens)
            # CPU-only count BEFORE the kernel, to snapshot the exact slice the
            # kernel will consume.
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )
            if num_new_pages > len(self.free_virtual_ids):
                return None
            # Lazy: physical-capacity pre-check; on shortfall flush PEER.
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not _relieve_for_alloc(self, need_tokens):
                    return None
            if self.need_sort and bs > len(self.free_virtual_ids):
                self.merge_and_sort_free()

            # Most decode steps reuse the prefix's tail page -> num_new_pages == 0.
            if num_new_pages > 0:
                new_virtual_pages = self.free_virtual_ids[:num_new_pages].clone()
            else:
                new_virtual_pages = None

            out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
            with record_function("MultiEndedAlloc.alloc_decode.kernel"):
                alloc_decode_kernel[(bs,)](
                    seq_lens,
                    last_loc,
                    self.free_virtual_ids,
                    out_indices,
                    next_power_of_2(bs),
                    self.page_size,
                )

            if new_virtual_pages is not None:
                phys_pages = self._alloc_bind_fast_or_slow(
                    new_virtual_pages, num_new_pages
                )
                if phys_pages is None:
                    return None

            self.free_virtual_ids = self.free_virtual_ids[num_new_pages:]
            return out_indices  # virtual token ids

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
        with record_function("MultiEndedAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if self.free_group is not None:
                self.free_group.append(self._copy_for_free_group(free_index))
                return
            if self.lazy_compaction:
                self._free_lazy(free_index, pages=_pages)
                return
            # --- EAGER path ---
            # Near-no-op in normal mode (sampling's CPU sync already drained
            # forward_stream); in overlap mode it does the serializing.
            if self.forward_stream is not None:
                with record_function("MultiEndedAlloc.free.wait_stream"):
                    torch.cuda.current_stream().wait_stream(self.forward_stream)
            with record_function("MultiEndedAlloc.free.v2p_lookup"):
                free_v_pages = (
                    _pages
                    if _pages is not None
                    else torch.unique(
                        free_index.detach().to(torch.int64) // self.page_size
                    )
                )
                freed_p_pages = self.virtual_to_physical[free_v_pages]
            with record_function("MultiEndedAlloc.free.sync_check"):
                # `.item()` forces a CPU/GPU sync -- own trace region to measure it.
                if bool((freed_p_pages < 0).any().item()):
                    self._raise_stale_slot_assertion(
                        free_v=free_v_pages, freed_p=freed_p_pages
                    )
            self.virtual_to_physical.index_fill_(0, free_v_pages, -1)
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._compact_pending(freed_p_pages)

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
        with record_function("MultiEndedAlloc._free_lazy"):
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
                free_v_pages, self.virtual_to_physical, self.physical_to_virtual
            )
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._free_phys_pages = torch.cat([self._free_phys_pages, freed_p_pages])
            self.live_page_count -= int(freed_p_pages.shape[0])

    def _release_phys_pages_batch(self, pages: torch.Tensor) -> None:
        """Cat `pages` onto `_free_phys_pages`. `_flush` calls it only AFTER its
        trailing dst-slice, so `_free_phys_pages == holes_cpu` for the whole walk.
        No watermark / `live_page_count` change: vacated srcs re-enter as storage.
        """
        if pages.numel() == 0:
            return
        self._stats_n_release_batch += 1
        with record_function("MultiEndedAlloc._release_phys_pages_batch"):
            self._free_phys_pages = torch.cat([self._free_phys_pages, pages])

    def _compact_pending(self, freed_physical_pages: torch.Tensor) -> None:
        """Eager compaction: move survivors out of the vacated band into the holes
        in the kept band. `src`/`dst` are disjoint by construction, so the batched
        copy is order-independent; the caller's `wait_stream` already serialized us
        with the in-flight forward.
        """
        with record_function("MultiEndedAlloc._compact_pending"):
            self._compact_pending_impl(freed_physical_pages)

    def _compact_pending_impl(self, freed_physical_pages: torch.Tensor) -> None:
        assert self.disagg_move_gate is None, (
            f"_compact_pending({self.sub_pool_name!r}): eager compaction ran with "
            "a PD-disaggregation move gate installed; PD requires lazy_compaction."
        )
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
            self.physical_to_virtual[vacated_lo:vacated_hi] = -1
        else:
            self.physical_to_virtual[vacated_lo:vacated_hi] = -1

    def _move_pages_and_rebind(
        self, src_pages: torch.Tensor, dst_pages: torch.Tensor
    ) -> torch.Tensor:
        """Copy live pages src->dst (disjoint sets), rebind v2p/p2v for the moved
        virtuals, record inverse history. Does NOT clear p2v[src] -- callers own
        vacated-region clearing. Returns the moved virtual page ids.
        """
        v_moved = self.physical_to_virtual[src_pages].clone()  # read pre-wipe

        # Expand to PHYSICAL token granularity (the move kernel is
        # token-granular over pool rows).
        if self.pool_page_size == 1:
            src_t, dst_t = src_pages, dst_pages
        else:
            ps = self.pool_page_size
            offsets = torch.arange(ps, dtype=torch.int64, device=self.device)
            src_t = (src_pages[:, None] * ps + offsets).reshape(-1)
            dst_t = (dst_pages[:, None] * ps + offsets).reshape(-1)

        # Un-translated copy: the public copy_from translates virtual ids,
        # which we must NOT do here.
        self._kvcache.move_kv_cache(dst_t, src_t)
        self.virtual_to_physical[v_moved] = dst_pages
        self.physical_to_virtual[dst_pages] = v_moved
        self._inverse_history.append((src_pages, dst_pages, v_moved))
        return v_moved

    # -- lazy compaction primitives --

    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:
        """Stash the most-recent forward's `forward_done` event; `_pending_reuse`
        uses it to gate src reuse on read-path settling. None = no in-flight forward.
        """
        with record_function("MultiEndedAlloc.set_latest_forward_done_event"):
            self._latest_forward_done_event = event

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        """Stash the just-launched forward's `forward_done` event plus the virtual
        `out_cache_loc` for `_flush`'s write-race check; no GPU work, only
        references. Pass `out_cache_loc_virtual=None` when the forward does not
        write this pool (Mamba state goes through mamba kernels, not
        `set_kv_buffer`). No-op in eager mode.
        """
        with record_function("MultiEndedAlloc.set_inflight_forward"):
            if not self.lazy_compaction:
                return
            if out_cache_loc_virtual is None or out_cache_loc_virtual.numel() == 0:
                # No write race on this pool -- clear the slot so `_flush`
                # short-circuits and the prior tensor reference can be GC'd.
                self._inflight_forward = None
                return
            self._inflight_forward = (forward_done, out_cache_loc_virtual)

    def _materialize_inflight_write_set(self) -> Optional[Set[int]]:
        """The in-flight forward's write-set (physical PAGE ids it is about to
        write), or `None` if there is none / it already completed. Pays a bs-sized
        D2H sync, once per call and only when a survivor needs classifying.
        """
        inflight = self._inflight_forward
        if inflight is None:
            return None
        event, oclv = inflight
        # Forward completed -> no write race. Clear so later flushes in the same
        # tick don't re-check the fired event.
        if event.query():
            self._inflight_forward = None
            return None
        # `oclv` is non-None here (set_inflight_forward clears the slot otherwise).
        with record_function("MultiEndedAlloc._materialize_inflight_write_set"):
            # `oclv` is a WIDENED virtual id under DCP; collapse it. A widened page
            # covers the same page, so the non-owned ids fold in harmlessly.
            dcp_size = get_parallel().attn_dcp_size if self.shards_under_dcp else 1
            if dcp_size > 1:
                oclv = oclv // dcp_size
            phys_tokens = self.translate_kv_loc(oclv)
            if self.pool_page_size > 1:
                phys_pages = (phys_tokens // self.pool_page_size).unique()
            else:
                phys_pages = phys_tokens
            return set(phys_pages.tolist())  # .tolist() syncs schedule_stream

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
        cur_pending = len(self._pending_reuse_pages_cpu)
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
            cur_pending = len(self._pending_reuse_pages_cpu)
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
        """Move ready `_pending_reuse` entries back into `_free_phys_pages`.
        Urgent uses `stream.wait_event` on unfired events -- a stream-side
        dependency, not a host block. ONE dict entry per BATCH, keyed by Event;
        no watermark / `live_page_count` change.
        """
        self._stats_n_drain_calls += 1
        if not self._pending_reuse:
            return
        with record_function("MultiEndedAlloc._drain_pending_reuse"):
            ready_tensors: List[torch.Tensor] = []
            ready_entries: List[Tuple[torch.cuda.Event, List[int]]] = []
            for event, (cpu_list, gpu_tensor) in self._pending_reuse.items():
                if event is None or event.query():
                    ready_tensors.append(gpu_tensor)
                    ready_entries.append((event, cpu_list))
                elif urgent:
                    torch.cuda.current_stream().wait_event(event)
                    ready_tensors.append(gpu_tensor)
                    ready_entries.append((event, cpu_list))

            for event, cpu_list in ready_entries:
                del self._pending_reuse[event]
                self._pending_reuse_pages_cpu.difference_update(cpu_list)

            if ready_tensors:
                self._free_phys_pages = torch.cat(
                    [self._free_phys_pages] + ready_tensors
                )
                self._stats_n_drain_did_work += 1
                self._stats_n_drained_pages_total += sum(
                    t.numel() for t in ready_tensors
                )

    def maybe_drain_pending_reuse(self) -> None:
        """Public scheduler hook (once per step): flow fired compaction-src pages
        back into `_free_phys_pages` for immediate reuse without waiting for `_flush`.
        """
        if not self.lazy_compaction:
            return
        if not self._pending_reuse:
            return
        self._drain_pending_reuse(urgent=False)

    def _topmost_survivor(
        self,
        start_hint: Optional[int] = None,
        *,
        holes_cpu: Optional[List[int]] = None,
        j_in: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Topmost live PAGE in the allocated band, excluding `holes_cpu` (the
        sorted-ASCENDING snapshot) and `_pending_reuse_pages_cpu`. Returns
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
                if is_hole or p in self._pending_reuse_pages_cpu:
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
                if is_hole or p in self._pending_reuse_pages_cpu:
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
            self._free_phys_pages = self._free_phys_pages[: M - n]
        else:
            while n < M and all_cpu[n] == wm + 1 + n:
                n += 1
            new_wm = wm + n
            holes_cpu = all_cpu[n:]
            self._free_phys_pages = self._free_phys_pages[n:]
        self.watermark_physical = new_wm
        self._stats_n_pages_absorbed += n
        return new_wm, holes_cpu

    def _settle_inflight_forward(self) -> None:
        """Stream-wait the in-flight forward's done event so freed slots are safe
        to MOVE (write settled) and REUSE (read settled): the event is recorded
        after the WHOLE forward, so one wait covers both hazards.
        """
        ev = self._latest_forward_done_event
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
            self._inflight_forward = None

    def _flush(self, *, urgent: bool) -> int:
        """One batched compaction pass; returns the number of survivor moves.

        Two hazards per survivor, both keyed on the single `forward_done` event:
        a WRITE race (the forward overwrites KV[src], so a compaction read would
        corrupt KV[dst]) stops a non-urgent walk at that src and is settled up
        front when urgent; a READ race (the forward READS KV[src]) gates src
        REUSE, so such srcs route to `_pending_reuse`.

        `_topmost_survivor` excludes all p2v=-1 pages, so a negative virtual id in
        the batched mapping lookup is a corrupt-state bug and raises.
        """
        if not self.lazy_compaction:
            return 0
        if self.disagg_move_gate is not None and not self.disagg_move_gate():
            # Holes stay in the free list; the next flush picks them up.
            return 0
        self._stats_n_flush_calls += 1
        with record_function("MultiEndedAlloc._flush"):
            self._drain_pending_reuse(urgent=urgent)

            # Sort ASCENDING.
            if self._free_phys_pages.numel() > 1:
                self._free_phys_pages, _ = torch.sort(self._free_phys_pages)

            all_cpu = self._free_phys_pages.tolist()  # one batched D2H sync

            # `holes_cpu` = interior holes; `_free_phys_pages == holes_cpu` after.
            new_wm, holes_cpu = self._absorb_boundary_holes(all_cpu)

            latest_event = self._latest_forward_done_event

            # Single-pass FULL-PACK (urgent only): `_latest_forward_done_event` is
            # recorded after the WHOLE forward, so waiting it once settles BOTH
            # hazards and the walk then runs race-free (empty write_set).
            single_pass_absorb = urgent and len(holes_cpu) > 0
            if single_pass_absorb:
                self._settle_inflight_forward()
                latest_event = None  # reads/writes settled -> srcs are fired

            # write_set: None = not yet materialized (do it inline on the first
            # survivor needing the check); set() = no write race; else materialized.
            write_set: Optional[Set[int]] = set() if single_pass_absorb else None

            srcs: List[int] = []
            dsts: List[int] = []

            # Flush-scoped accumulator for event-FIRED srcs, merged AFTER the
            # trailing dst-slice so `_free_phys_pages` stays byte-identical to
            # `holes_cpu` for the whole walk; catting mid-flush would chop the
            # wrong end. Event-PENDING srcs still route to `_pending_reuse`.
            released_fired: List[torch.Tensor] = []

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

                # Case A: write race.
                if write_set is None:
                    materialized = self._materialize_inflight_write_set()
                    write_set = materialized if materialized is not None else set()
                if write_set and src in write_set:
                    if urgent:
                        # Commit accumulated moves, then wait the forward so the
                        # rest of the walk is race-free.
                        self._commit_move_batch(
                            srcs, dsts, latest_event, released_fired
                        )
                        n_moves += len(srcs)
                        srcs.clear()
                        dsts.clear()
                        inflight = self._inflight_forward
                        if inflight is not None:
                            torch.cuda.current_stream().wait_event(inflight[0])
                            self._inflight_forward = None
                        write_set = set()  # forward drained -> no race
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

            self._commit_move_batch(srcs, dsts, latest_event, released_fired)
            n_moves += len(srcs)

            if single_pass_absorb:
                # FULL-PACK reclaim (urgent): all interior holes now sit above the
                # frontier, so retreat past the lot and EMPTY the free list; those
                # pages are beyond-frontier space, so `released_fired` is dropped.
                n_reclaimed = len(holes_cpu)
                if self.grow_direction == "up":
                    self.watermark_physical = new_wm - n_reclaimed
                else:
                    self.watermark_physical = new_wm + n_reclaimed
                self._stats_n_pages_absorbed += n_reclaimed
                self._free_phys_pages = self._free_phys_pages[:0]
            else:
                # Non-urgent partial pass: the watermark stays. `_free_phys_pages`
                # is still == holes_cpu, so the consumed dsts are exactly the front
                # (grow-up) / back (grow-down) `n_dst_consumed` entries.
                if n_dst_consumed > 0:
                    if self.grow_direction == "up":
                        self._free_phys_pages = self._free_phys_pages[n_dst_consumed:]
                    else:
                        self._free_phys_pages = self._free_phys_pages[:-n_dst_consumed]
                if released_fired:
                    self._release_phys_pages_batch(
                        released_fired[0]
                        if len(released_fired) == 1
                        else torch.cat(released_fired)
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
        released_fired: List[torch.Tensor],
    ) -> None:
        """Issue ONE `move_kv_cache` plus ONE bulk v2p/p2v remap for the
        accumulated `(src, dst)` pairs. Fired srcs accumulate in `released_fired`
        (merged by `_flush` after its dst-slice); event-pending srcs route to
        `_pending_reuse` for read-race gating.
        """
        if not srcs:
            return
        with record_function("MultiEndedAlloc._commit_move_batch"):
            src_pages_t = torch.tensor(srcs, dtype=torch.int64, device=self.device)
            dst_pages_t = torch.tensor(dsts, dtype=torch.int64, device=self.device)
            v_moveds_t = self.physical_to_virtual[src_pages_t]
            torch._assert_async(
                (v_moveds_t >= 0).all(),
                "invalid p2v mapping in MultiEndedAllocator._flush",
            )
            # Expand to PHYSICAL token granularity (the move kernel is
            # token-granular over pool rows).
            if self.pool_page_size == 1:
                src_t, dst_t = src_pages_t, dst_pages_t
            else:
                ps = self.pool_page_size
                offsets = torch.arange(ps, dtype=torch.int64, device=self.device)
                src_t = (src_pages_t[:, None] * ps + offsets).reshape(-1)
                dst_t = (dst_pages_t[:, None] * ps + offsets).reshape(-1)
            self._kvcache.move_kv_cache(dst_t, src_t)
            # ONE bulk remap (single-writer on schedule_stream).
            self.virtual_to_physical[v_moveds_t] = dst_pages_t
            self.physical_to_virtual[dst_pages_t] = v_moveds_t
            self.physical_to_virtual.index_fill_(0, src_pages_t, -1)
            self._inverse_history.append((src_pages_t, dst_pages_t, v_moveds_t))
            # Src disposition -- ONE entry per batch. `src_pages_t` is reused as the
            # `_pending_reuse` GPU tensor (no second H2D at drain).
            event_fired = latest_event is None or latest_event.query()
            if event_fired:
                released_fired.append(src_pages_t)
            else:
                srcs_copy: List[int] = list(srcs)  # caller mutates `srcs`
                self._pending_reuse[latest_event] = (srcs_copy, src_pages_t)
                self._pending_reuse_pages_cpu.update(srcs_copy)

    def flush_opportunistic(self) -> int:
        """Public, non-urgent flush at quiescent points; never blocks
        `schedule_stream`. Fast-path the empty state: the scheduler triggers this
        very often and ~99% of calls have no holes and no pending entries.
        """
        with record_function("MultiEndedAlloc.flush_opportunistic"):
            if not self.lazy_compaction:
                return 0
            if self._free_phys_pages.numel() == 0 and not self._pending_reuse:
                return 0
            return self._flush(urgent=False)

    def _raise_stale_slot_assertion(self, *, free_v, freed_p) -> None:
        bad = free_v[freed_p < 0].tolist()
        frames = inspect.stack()[1:9]
        callers = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames)
        raise AssertionError(
            f"MultiEndedAllocator({self.sub_pool_name!r}).free: virtual id(s) {bad} have "
            f"virtual_to_physical == -1 (double-free or never-allocated). "
            f"State: {self.allocator_state_str()}. free_index unique={free_v.tolist()}. "
            f"recent _inverse_history (last 3): "
            f"{[(s.tolist(), d.tolist()) for s, d, _ in self._inverse_history[-3:]]}. "
            f"Caller: {callers}."
        )

    # -- free-group --

    def free_group_begin(self) -> None:
        super().free_group_begin()
        self.free_page_reps_group = []

    def free_group_end(self) -> None:
        pending, self.free_page_reps_group = self.free_page_reps_group, None
        super().free_group_end()
        if pending:
            reps = torch.cat(pending)
            self.free(reps, _pages=reps // self.page_size)


def _chain_byte_accounting_violations(
    chain: List[MultiEndedAllocator],
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


def _end_pair_chain(
    a: MultiEndedAllocator, b: MultiEndedAllocator
) -> List[MultiEndedAllocator]:
    """Order an end pair low→high by grow direction (the factories and the
    unit fixtures orient the pair differently; the chain check must not care)."""
    return sorted((a, b), key=lambda x: x.grow_direction != "up")


class FloatMultiEndedAllocator(MultiEndedAllocator):
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

    def __init__(self, **kwargs):
        assert not kwargs.get("lazy_compaction", False), (
            "FloatMultiEndedAllocator is holes-first; the lazy event pipeline "
            "is end-pool machinery and must stay off for float middles"
        )
        # Base __init__ ends with self.clear(), which reads these via our
        # _reset_watermarks override -- pre-seed so the override can run.
        self.low_wm_page = 0
        self.high_wm_page = 0
        super().__init__(**kwargs)
        assert self.grow_direction == "float", (
            f"FloatMultiEndedAllocator needs a 'float' sub-pool spec; got "
            f"{self.grow_direction!r}"
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

    def _side_drainable_hole_bytes(self, side: str) -> int:
        """Realizable gap bytes an urgent flush of the neighbour on ``side``
        would release, walking past transparent members like the frontier walk.
        """
        p = self.low_peer if side == "low" else self.high_peer
        while p is not None and p._is_frontier_transparent():
            p = p.low_peer if side == "low" else p.high_peer
        if p is None or not p.lazy_compaction:
            return 0
        if p.disagg_move_gate is not None and not p.disagg_move_gate():
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

    def take_physical_pages(self, num_pages: int) -> Optional[torch.Tensor]:
        if num_pages <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        n_drain = min(num_pages, self._hole_pages())
        need_more = num_pages - n_drain

        fresh: Optional[torch.Tensor] = None
        if need_more > 0:
            lo, hi = self._region_bounds_pages()
            if self._is_frontier_transparent():
                # Reposition-on-alloc-from-empty: collapse to the midpoint so
                # free gap remains on BOTH sides.
                if need_more > hi - lo:
                    return None
                start = lo + (hi - lo - need_more) // 2
                self.low_wm_page = start
                self.high_wm_page = start + need_more
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
                fresh = torch.arange(
                    start, start + need_more, dtype=torch.int64, device=self.device
                )

        if n_drain > 0:
            drained = self._free_phys_pages[:n_drain].clone()
            self._free_phys_pages = self._free_phys_pages[n_drain:]
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

    def _alloc_bind_fast_or_slow(
        self, v_pages: torch.Tensor, N: int
    ) -> Optional[torch.Tensor]:
        # Holes-first always routes through take_physical_pages (no fused
        # watermark fast path -- float alloc cadence doesn't need it).
        if N == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        phys_pages = self.take_physical_pages(N)
        if phys_pages is None:
            return None
        self.bind(v_pages, phys_pages)
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
        with record_function("FloatMultiEndedAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if self.free_group is not None:
                self.free_group.append(self._copy_for_free_group(free_index))
                return
            # Page-derivation ladder, as `_free_lazy`: caller ids, the ps==1
            # identity, then dedup. No stale-slot assert -- callers must not
            # double-free (a tombstoned page would join the hole list); the
            # composite's filters uphold it and the byte verifier catches a miss.
            free_v_pages_raw = free_index.detach().to(torch.int64)
            if _pages is not None:
                free_v_pages = _pages
            elif self.page_size == 1:
                free_v_pages = free_v_pages_raw
            else:
                free_v_pages = torch.unique(free_v_pages_raw // self.page_size)
            freed_p_pages = self.virtual_to_physical[free_v_pages]
            # `index_fill_`, never `t[idx] = -1`: see the END free path.
            self.virtual_to_physical.index_fill_(0, free_v_pages, -1)
            self.physical_to_virtual.index_fill_(0, freed_p_pages, -1)
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._free_phys_pages = torch.cat([self._free_phys_pages, freed_p_pages])
            # Park is sync-free (span/hole COUNTS only); boundary absorption is
            # DEFERRED -- see `_absorb_span_boundary_holes`.
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
        holes = set(int(x) for x in self._free_phys_pages.tolist())
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
            self._free_phys_pages = torch.tensor(
                sorted(holes), dtype=torch.int64, device=self.device
            )
        return before - self._span_pages()

    # -- on-demand data movement --

    def make_room(self, *, side: str, min_bytes: int) -> int:
        """Open >= ``min_bytes`` of CONTIGUOUS free space between this pool's
        ``side`` boundary and the region bound on that side, relocating the
        minimum set of live boundary pages (holes-first destinations, then the far
        gap). Returns the bytes now open on ``side``; a result < ``min_bytes``
        means the ask is impossible now, and state is then unchanged.
        Scheduler-phase only; stream safety is owned HERE, not by the caller --
        the entry settles the in-flight forward before the first copy. Moves at
        most min(L_live, G) pages: every live page when the ask exceeds them.
        """
        assert side in ("low", "high"), f"side must be 'low'|'high'; got {side!r}"
        # Order the copies after the in-flight forward, or they carry pre-write
        # bytes; one wait covers read AND write (the event is post-forward).
        self._settle_inflight_forward()
        epp = self.entry_bytes_per_page
        lo, hi = self._region_bounds_pages()
        gap_low, gap_high = self._gap_pages()
        gap_side_bytes = (gap_low if side == "low" else gap_high) * epp
        if gap_side_bytes >= min_bytes or self._is_frontier_transparent():
            return gap_side_bytes

        # Capacity: even packing every live page flush against the far side
        # cannot open more than (region - live) bytes.
        live = self._live_pages()
        if (hi - lo - live) * epp < min_bytes:
            return gap_side_bytes  # impossible now; untouched

        need_pages = (min_bytes - gap_side_bytes + epp - 1) // epp

        holes = set(int(x) for x in self._free_phys_pages.tolist())
        span = range(self.low_wm_page, self.high_wm_page)
        live_pages_sorted = [p for p in span if p not in holes]

        if need_pages >= live:
            # Whole-pool LEAPFROG: pack every live page flush against the far
            # region edge; the capacity check above guarantees the resulting gap
            # satisfies the ask.
            if side == "high":
                final = list(range(lo, lo + live))
            else:
                final = list(range(hi - live, hi))
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
            self._move_pages_and_rebind(
                torch.tensor(srcs, dtype=torch.int64, device=self.device),
                torch.tensor(dsts, dtype=torch.int64, device=self.device),
            )
            self.physical_to_virtual.index_fill_(
                0, torch.tensor(srcs, dtype=torch.int64, device=self.device), -1
            )

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
        self._free_phys_pages = torch.tensor(
            new_holes, dtype=torch.int64, device=self.device
        )
        self._holes_dirty = True  # the span moved; re-check its boundaries
        self._absorb_span_boundary_holes()

        gap_low2, gap_high2 = self._gap_pages()
        return (gap_low2 if side == "low" else gap_high2) * epp

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
            if src_set.isdisjoint(dst_set):
                src_t = torch.tensor(
                    [s for s, _ in pairs], dtype=torch.int64, device=self.device
                )
                dst_t = torch.tensor(
                    [d for _, d in pairs], dtype=torch.int64, device=self.device
                )
                self._move_pages_and_rebind(src_t, dst_t)
                self.physical_to_virtual.index_fill_(0, src_t, -1)
            else:
                # Overlapping shift: process toward the move direction so each
                # destination is free by the time it is written.
                ordered = pairs if final[0] <= live_sorted[0] else list(reversed(pairs))
                # Built ONCE: `torch.tensor(..., device=cuda)` in the loop is a
                # pageable H2D per page, and a shift can span the whole pool.
                src_all = torch.tensor(
                    [s for s, _ in ordered], dtype=torch.int64, device=self.device
                )
                dst_all = torch.tensor(
                    [d for _, d in ordered], dtype=torch.int64, device=self.device
                )
                for i in range(len(ordered)):
                    s_t, d_t = src_all[i : i + 1], dst_all[i : i + 1]
                    self._move_pages_and_rebind(s_t, d_t)
                    self.physical_to_virtual.index_fill_(0, s_t, -1)
        if final:
            self.low_wm_page = final[0]
            self.high_wm_page = final[-1] + 1
        else:
            self._reset_watermarks()
        self._free_phys_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        return len(pairs)

    def _byte_accounting_violations(self) -> List[str]:
        out: List[str] = []
        total = self.unified_buffer.total_bytes
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
                (self.physical_to_virtual[self.low_wm_page : self.high_wm_page] != -1)
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
        with record_function("FloatMultiEndedAlloc.flush_opportunistic"):
            if not self._holes_dirty or self._free_phys_pages.numel() == 0:
                return 0
            return self._flush(urgent=False)

    def backup_state(self):
        # Span-aware snapshot (the base backs up `watermark_physical`, meaningless
        # here). Spec decode is asserted off under unified today.
        return (
            self.low_wm_page,
            self.high_wm_page,
            self._free_phys_pages.clone(),
            (len(self.free_virtual_ids) if self.is_id_owner else None),
            len(self._inverse_history),
        )

    def restore_state(self, state):
        low_wm, high_wm, holes, _n_free_virtual, n_inverse = state
        self.low_wm_page = low_wm
        self.high_wm_page = high_wm
        self._free_phys_pages = holes
        new_entries = self._inverse_history[n_inverse:]
        if new_entries:
            logger.warning(
                "FloatMultiEndedAllocator.restore_state: %d relocation(s) inside "
                "a backup window (sub_pool=%s) — float moves are not reversible.",
                len(new_entries),
                self.sub_pool_name,
            )
        del self._inverse_history[n_inverse:]
        return new_entries

    def compact_holes(self, *, retreat_side: str) -> int:
        """Close ALL interior holes by packing live pages toward the side
        OPPOSITE ``retreat_side`` (order-preserving), shrinking the span on
        ``retreat_side`` by the hole count. Returns pages moved."""
        assert retreat_side in ("low", "high")
        if self._hole_pages() == 0:
            return 0
        # Settle before the first copy -- see `make_room`.
        self._settle_inflight_forward()
        holes = set(int(x) for x in self._free_phys_pages.tolist())
        live_sorted = [
            p for p in range(self.low_wm_page, self.high_wm_page) if p not in holes
        ]
        if retreat_side == "high":
            final = list(range(self.low_wm_page, self.low_wm_page + len(live_sorted)))
        else:
            final = list(range(self.high_wm_page - len(live_sorted), self.high_wm_page))
        return self._relocate_to_positions(live_sorted, final)

    # -- band-incompatible base APIs --

    def bind_peer(self, peer: MultiEndedAllocator) -> None:  # pragma: no cover
        raise AssertionError(
            "float middles must be wired via bind_low_peer/bind_high_peer"
        )
