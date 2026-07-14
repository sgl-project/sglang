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
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.profiler import record_function

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.triton_ops.virtual_slot import alloc_bind_inplace
from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool
from sglang.srt.utils.common import get_num_new_pages, next_power_of_2

logger = logging.getLogger(__name__)


# OFF (default): cat unsorted, `_flush` sorts once. ON: sort after each cat.
_SORT_FREE_LIST_AFTER_MERGE = envs.SGLANG_SORT_FREE_LIST_AFTER_MERGE.get()


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
            # Raises off the main thread — skip.
            pass


class MultiEndedAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for one sub-pool over a `UnifiedKVPool`."""

    supports_page_aligned_alloc: bool = True

    def __init__(
        self,
        *,
        kvcache,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        device: str,
        is_id_owner: bool,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):
        spec = unified_buffer.spec(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)
        super().__init__(
            size=max_slots,
            page_size=page_size,
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
        # Overlap mode: `free` drops a wait_stream(forward_stream) barrier so its
        # v2p writes + move kernel serialize after the in-flight forward.
        self.forward_stream = forward_stream

        # --- Page-aware bookkeeping ---
        # `min_page_index` = ceil(min_slot_index / page_size), keeping the
        # reserved-sink invariant (min_page_index * entry_bytes_per_page >= entry_max).
        self.page_size = page_size
        self.num_pages = max_slots // page_size
        self.min_page_index = (self.min_slot_index + page_size - 1) // page_size
        self.entry_bytes_per_page = self.entry_bytes * page_size

        # v2p / p2v sized by PAGES. Page 0 is the padding anchor; trailing row is
        # the -1 sentinel.
        self.virtual_to_physical = torch.full(
            (self.num_pages + 1,),
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
        # Back-compat alias (count of virtual PAGES) consulted by is_slot_allocated.
        self.num_virtual_ids = self.num_pages

        self._peer: Optional[MultiEndedAllocator] = None

        # Inverse history of relocations (spec rollback), at PAGE granularity.
        self._inverse_history: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            []
        )

        # --- Lazy compaction state (all unused when lazy_compaction=False) ---
        # `_free_phys_pages`: GPU free list of physical PAGE ids, sorted at `_flush`.
        # `_pending_reuse`: compaction-src pages whose remap completed but whose
        #   reader event hasn't fired — can't re-enter the free list until the read
        #   settles (else a future alloc's WRITE races the READ).
        # `live_page_count`: CPU slot-conservation counter, invariant under compaction.
        # KV copy and v2p/p2v remap both run on `schedule_stream`, so single-stream
        # ordering serializes them — no separate copy-done event needed.
        self.lazy_compaction = lazy_compaction
        self._free_phys_pages: torch.Tensor = torch.empty(
            0, dtype=torch.int64, device=device
        )
        # Keyed by Event, ONE entry per BATCH. `(cpu_list, gpu_tensor)`: cpu_list
        # drives the Set update (no sync); gpu_tensor is the SAME tensor
        # `_commit_move_batch` remapped, kept alive so drain cats it without an H2D.
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
        self._latest_forward_done_event: Optional[torch.cuda.Event] = None
        # Most-recent forward's (done_event, out_cache_loc_virtual) for `_flush`'s
        # write-race check. Single slot: at most ONE forward in flight per call site.
        # Only the tensor reference is stored; `_flush` materializes the write-set
        # lazily, avoiding a launch-time sync.
        self._inflight_forward: Optional[Tuple[torch.cuda.Event, torch.Tensor]] = None

        # Per-call move cap on NON-urgent `_flush`: bounds work per `on_idle()` so a
        # large backlog doesn't block ZMQ IPC; the next flush picks up the rest.
        # Urgent (alloc-shortfall retry) is uncapped — must drain everything.
        self._lazy_max_moves_per_call = int(
            os.environ.get("SGLANG_LAZY_COMPACTION_MAX_MOVES_PER_CALL", "4096")
        )

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

    # -- peer binding --

    def bind_peer(self, peer: MultiEndedAllocator) -> None:
        self._peer = peer

    @property
    def peer(self) -> Optional[MultiEndedAllocator]:
        return self._peer

    # -- state --

    def clear(self) -> None:
        """Reset to initial state. Pages in `[0, min_page_index)` are reserved."""
        if self.grow_direction == "up":
            self.watermark_physical = self.min_page_index
        else:
            self.watermark_physical = self.num_pages - 1
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
        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []
        self._inverse_history.clear()
        self._free_phys_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        self._pending_reuse.clear()
        self._pending_reuse_pages_cpu.clear()
        self.live_page_count = 0
        self._inflight_forward = None
        self._latest_forward_done_event = None

    def backup_state(self):
        # Spec-decode allocates only inside a backup window (no free), so
        # `_inverse_history` doesn't grow under correct usage.
        return (
            self.watermark_physical,
            (len(self.free_virtual_ids) if self.is_id_owner else None),
            len(self._inverse_history),
        )

    def restore_state(self, state):
        watermark, n_free_virtual, n_inverse = state
        self.watermark_physical = watermark
        if self.is_id_owner and n_free_virtual is not None:
            pass  # spec asserted off; no free-list rollback.
        new_entries = self._inverse_history[n_inverse:]
        if new_entries:
            logger.warning(
                "MultiEndedAllocator.restore_state: %d relocation(s) recorded inside "
                "a backup window (sub_pool=%s). Eager compaction is not fully "
                "reversible; SGLang's spec path should not produce a free() inside a "
                "backup window.",
                len(new_entries),
                self.sub_pool_name,
            )
        del self._inverse_history[n_inverse:]
        return new_entries

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

        TOKENS, not pages — the leak checker's invariant is in tokens. Lazy mode
        uses `live_page_count` (invariant under compaction); the watermark span
        over-counts because holes/pending sit inside it but aren't live.
        """
        if self.lazy_compaction:
            return self.live_page_count * self.page_size
        return self._allocated_pages() * self.page_size

    def is_slot_allocated(self, slot: int) -> bool:
        """Whether the PAGE containing this virtual id is in use."""
        virt_page = slot // self.page_size
        if virt_page < 0 or virt_page >= self.num_pages:
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

    def _byte_low_frontier(self) -> int:
        """Byte starting this side's allocatable range (grow-up) / just below its lowest live page (grow-down)."""
        if self.grow_direction == "up":
            return self.min_page_index * self.entry_bytes_per_page
        return (self.watermark_physical + 1) * self.entry_bytes_per_page

    def _current_gap_bytes(self) -> int:
        """Free byte band between this side's frontier and the peer's CURRENT frontier."""
        if self.grow_direction == "up":
            my_high = self._byte_high_frontier()
            peer_low = (
                self._peer._byte_low_frontier()
                if self._peer is not None
                else self.unified_buffer.total_bytes
            )
            return max(0, peer_low - my_high)
        my_low = self._byte_low_frontier()
        peer_high = self._peer._byte_high_frontier() if self._peer is not None else 0
        return max(0, my_low - peer_high)

    def _available_tokens(self, extra_gap_bytes: int = 0) -> int:
        """Tokens allocatable given `extra_gap_bytes` of ADDED gap room
        (0 == current realizable; >0 == post-peer-compaction).

        `pages_by_index_space` is OWN index headroom, unaffected by
        `extra_gap_bytes`: peer bytes can't add page indices to our own table.
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

        Alloc shortfall gates consult this to decide whether to peer-flush, so it
        MUST NOT fold in peer holes (use `schedulable_available_size()` for that).
        """
        return self._available_tokens()

    def _peer_drainable_hole_bytes(self) -> int:
        """Gap bytes a peer urgent flush would release. Only `_free_phys_pages`
        count — NOT `_pending_reuse` (awaiting an event) — so the credit is realizable.
        """
        peer = self._peer
        if peer is None or not peer.lazy_compaction:
            return 0
        return len(peer._free_phys_pages) * peer.entry_bytes_per_page

    def schedulable_available_size(self) -> int:
        """Tokens allocatable AFTER a peer urgent-flush (realizable-with-compaction).
        Used by composite views; alloc gates use `available_size()`.
        """
        return self._available_tokens(extra_gap_bytes=self._peer_drainable_hole_bytes())

    def _flush_peer_for_alloc(self, need_tokens: int) -> bool:
        """One urgent peer-flush on alloc shortfall; returns whether THIS side now
        has enough. Only PEER compaction releases gap bytes (own compaction is net 0).
        """
        if not (self.lazy_compaction and self._peer is not None):
            return False
        self._peer._flush(urgent=True)
        return need_tokens <= self.available_size()

    # -- physical-slot / physical-page primitives --

    def take_physical(self, need_size: int) -> Optional[torch.Tensor]:
        """Reserve `need_size` TOKENS (multiple of page_size), returning backing
        physical PAGE ids, or `None` on shortfall.

        Eager: pure watermark advance. Lazy: drain `_free_phys_pages` holes first,
        then extend the watermark (extend first so state is untouched on failure).
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

            # Lazy: slice the GPU free list (no D2H). sort ON: take deepest-in-band
            # per direction (greedy clustering). sort OFF: take from front.
            n_drain = min(num_pages, int(self._free_phys_pages.shape[0]))
            need_more = num_pages - n_drain

            # Extend first (state untouched on failure), then drain holes.
            if need_more > 0:
                if not self._extend_watermark(need_more):
                    return None

            if n_drain > 0:
                if _SORT_FREE_LIST_AFTER_MERGE:
                    if self.grow_direction == "up":
                        drained_t = self._free_phys_pages[:n_drain]
                        self._free_phys_pages = self._free_phys_pages[n_drain:]
                    else:
                        drained_t = self._free_phys_pages[-n_drain:].flip(0)
                        self._free_phys_pages = self._free_phys_pages[:-n_drain]
                else:
                    drained_t = self._free_phys_pages[:n_drain]
                    self._free_phys_pages = self._free_phys_pages[n_drain:]
            else:
                drained_t = None

            self.live_page_count += num_pages

            if drained_t is None:
                return self._take_physical_arange(num_pages)

            # Pure drain — clone off the free-list view so rebindings don't pin it.
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
        """Advance the watermark by `num_pages` (lazy-path helper). Returns False
        on index-space overflow OR crossing the PEER's byte frontier.
        """
        if self.grow_direction == "up":
            new_wm = self.watermark_physical + num_pages
            if new_wm > self.num_pages:
                return False
            # Peer (grow-down) sits ABOVE; don't extend past its low frontier.
            if self._peer is not None:
                peer_low_pages = (
                    self._peer._byte_low_frontier() // self.entry_bytes_per_page
                )
                if new_wm > peer_low_pages:
                    return False
            self.watermark_physical = new_wm
        else:
            new_wm = self.watermark_physical - num_pages
            if new_wm < self.min_page_index - 1:
                return False
            # Peer (grow-up) sits BELOW; `new_wm + 1` (our new lowest live page)
            # must stay strictly above the peer's high frontier.
            if self._peer is not None:
                peer_high_pages = (
                    self._peer._byte_high_frontier() // self.entry_bytes_per_page
                )
                if new_wm + 1 < peer_high_pages:
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
            self.virtual_to_physical[virtual_ids] = physical_ids
            self.physical_to_virtual[physical_ids] = virtual_ids

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
                return phys_pages

            # SLOW PATH: holes exist — drain them first, then bind.
            phys_pages = self.take_physical_pages(N)
            if phys_pages is None:
                return None
            self.bind(v_pages, phys_pages)
            return phys_pages

    # -- translate (virtual TOKEN ids -> physical TOKEN ids) --

    def translate_kv_loc(
        self,
        virt_tokens: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Translate token-granular virtual ids to physical ids.

        ``out=`` writes in-place into a caller-owned buffer — required under
        cuda-graph capture for buffer-stability (the captured graph records the
        gather against a fixed ``data_ptr``).
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
        # Tombstone-safety clamp: tombstoned v2p entries (-1) must not reach
        # `k_buffer[-1]` (illegal access under captured graph replay). Clamp to 0
        # routes any tombstoned read/write to physical slot 0 — reserved
        # padding-sink space by the `min_slot_index` invariant (bytes [0, entry_max)
        # across all sub-pools hold no real data).
        if self.page_size == 1:
            if out is not None:
                # `index_select(out=out)` forbids index/out aliasing, but the
                # canonical caller does in-place `translate(kv_indices, out=kv_indices)`.
                # Route through a transient gather + `copy_` to satisfy that contract.
                tmp = torch.index_select(self.virtual_to_physical, 0, virt_tokens)
                tmp = torch.clamp_min(tmp, 0)
                out.copy_(tmp)
                return out
            result = torch.index_select(self.virtual_to_physical, 0, virt_tokens)
            return torch.clamp_min(result, 0)
        # page_size > 1: page math. `virt_pages`/`offsets` are fresh, so they
        # cannot alias `out` — `index_select(out=out)` is safe.
        virt_pages = virt_tokens // self.page_size
        offsets = virt_tokens % self.page_size
        if out is not None:
            torch.index_select(self.virtual_to_physical, 0, virt_pages, out=out)
            out.mul_(self.page_size)
            out.add_(offsets)
            out.clamp_(min=0)  # tombstoned page: -1*ps + offset in [-ps, -1]
            return out
        phys_pages = self.virtual_to_physical[virt_pages]
        result = phys_pages * self.page_size + offsets
        return torch.clamp_min(result, 0)

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate `need_size` virtual TOKEN ids (id-owner only). Returns
        token-granular, page-structured ids, or None on shortfall.

        `need_size` MUST be a multiple of `page_size`. All allocator GPU ops run
        on `schedule_stream`; `alloc` needs no `wait_stream` barrier because its
        v2p/p2v writes are picked up by the forward via the existing
        `forward_stream.wait_stream(schedule_stream)` at the top of `run_batch`.
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
                # Shortfall: flush the PEER, not own. Own compaction is net 0
                # (each move trades 1 hole for +1 gap byte); only peer compaction
                # releases bytes into the shared gap that own extension consumes.
                if not self._flush_peer_for_alloc(need_size):
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
            # Expand page ids to token ids: (P, 1) * S + (S,) → (P, S) → (P*S,).
            return (
                v_pages[:, None] * self.page_size
                + torch.arange(self.page_size, device=self.device)
            ).reshape(-1)

    def alloc_with_virtual(self, virtual_pages: torch.Tensor) -> None:
        """Take physical PAGES for caller-supplied virtual PAGE ids
        (physical-holding non-owner; the SWA `swa` sub-allocator).

        Input is virtual PAGE ids (not token ids): the composite snapshots the
        virtual pages before the id-owner consumes them from its free-list.
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
        so ``out_indices`` are virtual token ids. Each consumed virtual page is
        then bound to a physical page on THIS sub-allocator; without that binding
        v2p stays -1 and translation yields negative ids → CUDA OOB.
        """
        with record_function("MultiEndedAlloc.alloc_extend"):
            assert (
                self.is_id_owner
            ), f"alloc_extend on a non-id-owner allocator ({self.sub_pool_name!r})"
            if num_new_pages is None:
                num_new_pages = get_num_new_pages(
                    seq_lens=seq_lens_cpu,
                    page_size=self.page_size,
                    prefix_lens=prefix_lens_cpu,
                )
            if num_new_pages > len(self.free_virtual_ids):
                return None
            # Lazy: physical-capacity pre-check; on shortfall flush the PEER (own
            # compaction is internal — see `alloc`).
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not self._flush_peer_for_alloc(need_tokens):
                    return None
            bs = len(prefix_lens)
            if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
                self.free_virtual_ids
            ):
                self.merge_and_sort_free()

            # Snapshot the virtual pages the kernel will consume, to bind them to
            # physical pages afterward (else v2p stays -1 → CUDA OOB).
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
        tail-page-reuse contract. Runs in virtual space; binds each consumed
        virtual page on THIS sub-allocator (else v2p stays -1 → CUDA OOB).
        """
        with record_function("MultiEndedAlloc.alloc_decode"):
            assert (
                self.is_id_owner
            ), f"alloc_decode on a non-id-owner allocator ({self.sub_pool_name!r})"
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
                if not self._flush_peer_for_alloc(need_tokens):
                    return None
            if self.need_sort and bs > len(self.free_virtual_ids):
                self.merge_and_sort_free()

            # Most decode steps reuse the prefix's tail page → num_new_pages == 0.
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

    def free(self, free_index: torch.Tensor) -> None:
        """Free virtual TOKEN ids: recover virtual PAGE ids, un-map v2p/p2v,
        (if id-owner) recycle the page ids, trigger eager compaction.

        `free_index` is token-granular and need not be page-aligned. EAGER mode
        drops one `wait_stream(forward_stream)` barrier so v2p/p2v writes and the
        compaction move serialize with the in-flight forward. LAZY mode needs no
        barrier (a freed `v` has no live reader, so the scatters are
        disjoint-element from any forward read, atomic on Ampere+/Hopper) and
        defers compaction to `_flush`.
        """
        with record_function("MultiEndedAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if not self.is_not_in_free_group:
                self.free_group.append(free_index)
                return
            if self.lazy_compaction:
                self._free_lazy(free_index)
                return
            # --- EAGER path ---
            # Near-no-op in normal mode (sampling's CPU sync already drained
            # forward_stream); in overlap mode it serializes free+compaction with
            # the in-flight forward.
            if self.forward_stream is not None:
                with record_function("MultiEndedAlloc.free.wait_stream"):
                    torch.cuda.current_stream().wait_stream(self.forward_stream)
            with record_function("MultiEndedAlloc.free.v2p_lookup"):
                free_v_pages = torch.unique(
                    free_index.detach().to(torch.int64) // self.page_size
                )
                freed_p_pages = self.virtual_to_physical[free_v_pages]
            with record_function("MultiEndedAlloc.free.sync_check"):
                # `.item()` forces a CPU/GPU sync — own trace region to measure it.
                if bool((freed_p_pages < 0).any().item()):
                    self._raise_stale_slot_assertion(
                        free_v=free_v_pages, freed_p=freed_p_pages
                    )
            self.virtual_to_physical[free_v_pages] = -1
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._compact_pending(freed_p_pages)

    def _free_lazy(self, free_index: torch.Tensor) -> None:
        """Lazy free path: disjoint-element scatters + ONE `torch.cat` onto
        `_free_phys_pages`. No sort, no boundary absorb, no watermark mutation,
        no D2H sync. Boundary absorption is deferred to `_flush`.

        ps==1 skips `torch.unique` (token == page and `free_index` is already
        unique per caller contract); ps>1 needs it to dedup same-page tokens.
        Callers must not double-free: a tombstone (-1) here would be cat'd onto
        the free list.
        """
        self._stats_n_free_lazy += 1
        with record_function("MultiEndedAlloc._free_lazy"):
            with record_function("MultiEndedAlloc._free_lazy.v2p_lookup"):
                free_v_pages_raw = free_index.detach().to(torch.int64)
                if self.page_size == 1:
                    free_v_pages = free_v_pages_raw
                else:
                    free_v_pages = torch.unique(free_v_pages_raw // self.page_size)
                freed_p_pages = self.virtual_to_physical[free_v_pages]
            # Disjoint-element scatters — no barrier (a freed v has no live reader;
            # per-element scatter writes are atomic).
            self.virtual_to_physical[free_v_pages] = -1
            self.physical_to_virtual[freed_p_pages] = -1
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._free_phys_pages = torch.cat([self._free_phys_pages, freed_p_pages])
            if _SORT_FREE_LIST_AFTER_MERGE:
                self._free_phys_pages, _ = torch.sort(self._free_phys_pages)
            self.live_page_count -= int(freed_p_pages.shape[0])

    def _release_phys_pages_batch(self, pages: torch.Tensor) -> None:
        """Cat `pages` onto `_free_phys_pages` (+ optional sort). Called by `_flush`
        at END to merge event-fired compaction-srcs (`released_fired`) AFTER the
        trailing dst-slice, keeping `_free_phys_pages == holes_cpu` during the walk.

        No watermark / `live_page_count` change — these are vacated src positions
        re-entering as PURE storage, not freshly-freed live pages.
        """
        if pages.numel() == 0:
            return
        self._stats_n_release_batch += 1
        with record_function("MultiEndedAlloc._release_phys_pages_batch"):
            self._free_phys_pages = torch.cat([self._free_phys_pages, pages])
            if _SORT_FREE_LIST_AFTER_MERGE:
                self._free_phys_pages, _ = torch.sort(self._free_phys_pages)

    def _compact_pending(self, freed_physical_pages: torch.Tensor) -> None:
        """Eager compaction over the freed PHYSICAL pages: move survivors from the
        vacated band (K pages adjacent to the watermark) into the holes in the kept
        band, advance the watermark, remap the tables. `src`/`dst` are disjoint by
        construction, so the batched copy is order-independent. The caller's
        `wait_stream` barrier already serialized us with the in-flight forward.
        """
        with record_function("MultiEndedAlloc._compact_pending"):
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
            v_moved = self.physical_to_virtual[
                src_pages
            ].clone()  # read before clearing

            # Expand page ids to token ids for the token-granular move kernel.
            if self.page_size == 1:
                src_t, dst_t = src_pages, dst_pages
            else:
                offsets = torch.arange(
                    self.page_size, dtype=torch.int64, device=self.device
                )
                src_t = (src_pages[:, None] * self.page_size + offsets).reshape(-1)
                dst_t = (dst_pages[:, None] * self.page_size + offsets).reshape(-1)

            # Un-translated copy: the public copy_from translates virtual ids,
            # which we must NOT do here.
            move_fn = getattr(self._kvcache, "move_kv_cache", None)
            if move_fn is not None:
                move_fn(dst_t, src_t)
            else:
                copy_phys = getattr(self._kvcache, "_copy_from_physical", None)
                assert copy_phys is not None, (
                    f"sub-pool {self.sub_pool_name!r} supports neither move_kv_cache "
                    "nor _copy_from_physical"
                )
                copy_phys(src_t, dst_t)
            # Clear the vacated band, then re-bind the relocated dst pages.
            self.physical_to_virtual[vacated_lo:vacated_hi] = -1
            self.virtual_to_physical[v_moved] = dst_pages
            self.physical_to_virtual[dst_pages] = v_moved
            self._inverse_history.append((src_pages, dst_pages, v_moved))
        else:
            self.physical_to_virtual[vacated_lo:vacated_hi] = -1

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
        """Stash the just-launched forward's `forward_done` event + virtual
        `out_cache_loc` for `_flush`'s write-race check.

        No GPU work — only references; `_flush` materializes the write-set lazily
        on `schedule_stream`, avoiding a launch-time sync. Pass
        `out_cache_loc_virtual=None` when the forward doesn't write this pool
        (e.g. Mamba state, written by mamba kernels not `set_kv_buffer`). No-op
        in eager mode.
        """
        with record_function("MultiEndedAlloc.set_inflight_forward"):
            if not self.lazy_compaction:
                return
            if out_cache_loc_virtual is None or out_cache_loc_virtual.numel() == 0:
                # No write race on this pool — clear the slot so `_flush`
                # short-circuits and the prior tensor reference can be GC'd.
                self._inflight_forward = None
                return
            self._inflight_forward = (forward_done, out_cache_loc_virtual)

    def _materialize_inflight_write_set(self) -> Optional[Set[int]]:
        """Materialize the in-flight forward's write-set (physical PAGE ids it is
        about to write), or `None` if no in-flight forward / already completed.
        Called inside `_flush` on `schedule_stream`. Pays a bs-sized D2H sync, but
        only once per call and only when a survivor needs classifying.
        """
        inflight = self._inflight_forward
        if inflight is None:
            return None
        event, oclv = inflight
        # Forward completed → no write race. Clear so later flushes in the same
        # tick don't re-check the fired event.
        if event.query():
            self._inflight_forward = None
            return None
        # `oclv` is non-None here (set_inflight_forward clears the slot otherwise).
        with record_function("MultiEndedAlloc._materialize_inflight_write_set"):
            phys_tokens = self.translate_kv_loc(oclv)
            if self.page_size > 1:
                phys_pages = (phys_tokens // self.page_size).unique()
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
        sort_tag = "ON" if _SORT_FREE_LIST_AFTER_MERGE else "OFF"
        logger.info(
            f"[lazy-stats sub={self.sub_pool_name!r}] "
            f"free_lazy={self._stats_n_free_lazy} "
            f"flush={self._stats_n_flush_calls} "
            f"(work={self._stats_n_flush_did_work} "
            f"moves={self._stats_n_flush_moves} "
            f"abs={self._stats_n_pages_absorbed}) "
            f"drain={self._stats_n_drain_did_work}/{self._stats_n_drain_calls} "
            f"sort={sort_tag} "
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
            sort_tag = "ON" if _SORT_FREE_LIST_AFTER_MERGE else "OFF"
            self._stats_final_emitted = True
            logger.info(
                f"[lazy-stats FINAL sub={self.sub_pool_name!r} reason={reason}] "
                f"free_lazy={self._stats_n_free_lazy} "
                f"flush={self._stats_n_flush_calls} "
                f"(work={self._stats_n_flush_did_work} "
                f"moves={self._stats_n_flush_moves} "
                f"abs={self._stats_n_pages_absorbed}) "
                f"drain={self._stats_n_drain_did_work}/{self._stats_n_drain_calls} "
                f"sort={sort_tag} "
                f"peak_holes={self._stats_peak_free_list_len} "
                f"peak_pending={self._stats_peak_pending_pages} "
                f"cur_holes={cur_holes} cur_pending={cur_pending} "
                f"live={self.live_page_count} wm={self.watermark_physical} "
                f"n_emits={self._stats_n_emits}"
            )
        except Exception:
            pass

    def _drain_pending_reuse(self, *, urgent: bool) -> None:
        """Move ready `_pending_reuse` entries back into `_free_phys_pages` via
        pure-GPU `torch.cat`.

          * non-urgent: release only entries whose event is None or has fired.
          * urgent: `stream.wait_event` (stream-side dep, not host block) on
            unfired events, then release.

        ONE dict entry per BATCH (keyed by Event); cpu_list drives the Set update,
        gpu_tensor is cat'd directly. No watermark / `live_page_count` change.
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
                if _SORT_FREE_LIST_AFTER_MERGE:
                    self._free_phys_pages, _ = torch.sort(self._free_phys_pages)

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
        """Topmost live PAGE in the allocated band (largest `p < watermark` for
        grow-up / smallest `p > watermark` for grow-down), excluding holes
        (`holes_cpu`, the sorted-ASCENDING snapshot) and `_pending_reuse_pages_cpu`.

        Two-pointer: `p` is monotonic and `holes_cpu` is sorted, so the hole cursor
        `j` (threaded back via the returns) advances alongside for O(1) membership;
        no exclude-set needed because uncommitted dsts have p2v=-1 and are correctly
        reported by the snapshot. Returns `(p, j)`, or `(None, j)` if none.

        `holes_cpu`/`j_in` are optional only for test fixtures (else a `.tolist()`
        sync); `_flush` always passes them.
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
        """Retreat the watermark past free slots ALREADY contiguous with it, slice
        them off `_free_phys_pages`, return ``(new_watermark, interior_holes_cpu)``.
        `all_cpu` is the sorted-ascending snapshot; interior holes feed the survivor
        walk.
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
        to MOVE (write settled) and REUSE (read settled). The event is recorded
        after the WHOLE forward, so one wait covers both hazards; drop the write-set.
        """
        ev = self._latest_forward_done_event
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
            self._inflight_forward = None

    def _flush(self, *, urgent: bool) -> int:
        """One batched compaction pass; returns the number of survivor moves.

        Pipeline (one D2H total, at step 3):
          1. `_drain_pending_reuse` — return read-settled prior srcs.
          2. sort the free list (or skip via env knob; either way ascending after).
          3. `.tolist()` snapshot → `all_cpu`  *(the one sync)*.
          4-5. `_absorb_boundary_holes` — retreat past boundary-contiguous holes;
               `holes_cpu` = interior holes. After this `_free_phys_pages==holes_cpu`.
          6. (urgent) `_settle_inflight_forward` — wait once so the walk is race-free.
          7. survivor walk — TWO-POINTER: move topmost live slot into the next hole,
             STOPPING when the pointers cross (band packed); batch into one
             `move_kv_cache` + one v2p/p2v scatter at `_commit_move_batch`.
          8-9. exit: urgent → FULL-PACK reclaim (retreat past ALL holes, empty list);
               non-urgent → slice consumed dsts, merge freed srcs back.

        Two hazards per survivor (both keyed on the single `forward_done` event):
          * WRITE race — forward overwrites KV[src]; a compaction read corrupts
            KV[dst]. Non-urgent STOPS at such a src; urgent settles up front (step 6).
          * READ race — forward READS KV[src]; src REUSE must wait the reader event.
            `_commit_move_batch` routes such srcs to `_pending_reuse`; urgent's
            settle makes them immediately reusable.

        `_topmost_survivor` excludes all p2v=-1 pages, so a `v_moved < 0` in the
        loop is a corrupt-state bug and raises.
        """
        if not self.lazy_compaction:
            return 0
        self._stats_n_flush_calls += 1
        with record_function("MultiEndedAlloc._flush"):
            self._drain_pending_reuse(urgent=urgent)

            # Sort ASCENDING (skip if the env knob keeps the list always-sorted).
            if not _SORT_FREE_LIST_AFTER_MERGE and self._free_phys_pages.numel() > 1:
                self._free_phys_pages, _ = torch.sort(self._free_phys_pages)

            all_cpu = self._free_phys_pages.tolist()  # the ONE D2H sync per flush

            # `holes_cpu` = interior holes; `_free_phys_pages == holes_cpu` after.
            new_wm, holes_cpu = self._absorb_boundary_holes(all_cpu)

            latest_event = self._latest_forward_done_event

            # Single-pass FULL-PACK (urgent only): the crossing-checked walk packs
            # all live below the frontier so the exit can retreat past every
            # interior hole at once — but only if each freed src is reuse-safe.
            # `_latest_forward_done_event` is recorded after the WHOLE forward, so
            # waiting it once settles BOTH hazards; then every src is event-fired
            # and the walk runs race-free (empty write_set, no `_pending_reuse`).
            single_pass_absorb = urgent and len(holes_cpu) > 0
            if single_pass_absorb:
                self._settle_inflight_forward()
                latest_event = None  # reads/writes settled → srcs are fired

            # write_set: None = not yet materialized (do it inline on the first
            # survivor needing the check); set() = no write race; else materialized.
            write_set: Optional[Set[int]] = set() if single_pass_absorb else None

            srcs: List[int] = []
            dsts: List[int] = []
            v_moveds: List[int] = []

            # Flush-scoped accumulator for event-FIRED srcs. `_commit_move_batch`
            # appends here instead of catting onto `_free_phys_pages`; the merge is
            # deferred to AFTER the trailing dst-slice, keeping `_free_phys_pages`
            # byte-identical to `holes_cpu` for the whole walk. That invariant is
            # what makes the directional dst-slice correct in both directions
            # (catting srcs mid-flush would chop the wrong end / scramble under
            # sort=ON, leaving ghost p2v=-1 pages + double-bound dsts). Event-
            # PENDING srcs still route to `_pending_reuse` (read-race gating).
            released_fired: List[torch.Tensor] = []

            cursor: Optional[int] = None
            j_cursor: Optional[int] = None

            # Dst cursor reads `holes_cpu` directly (no per-dst sync): grow-up from
            # the front, grow-down from the back. Consumed prefix/suffix is sliced
            # off in one GPU op at exit.
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
                            srcs, dsts, v_moveds, latest_event, released_fired
                        )
                        n_moves += len(srcs)
                        srcs.clear()
                        dsts.clear()
                        v_moveds.clear()
                        inflight = self._inflight_forward
                        if inflight is not None:
                            torch.cuda.current_stream().wait_event(inflight[0])
                            self._inflight_forward = None
                        write_set = set()  # forward drained → no race
                        latest_event = None
                        # DO NOT reset cursor/j_cursor: rewinding would re-pick the
                        # just-committed srcs (now p2v=-1, not in holes_cpu) and
                        # trip the p2v=-1 assertion. Preserving cursor resumes at
                        # the blocker itself, which now passes under empty write_set.
                        continue
                    else:
                        break  # non-urgent: top blocker stops the walk

                # Case B/C: no write race. dst from holes_cpu by cursor (no sync).
                dst = holes_cpu[dst_cursor]
                # Two-pointer crossing check: once src and dst cross, the band is
                # packed. Moving further would shuffle a hole back toward the
                # frontier and block the watermark retreat, so stop — this is what
                # lets one urgent pass reclaim ALL holes (not just a contiguous run).
                if (self.grow_direction == "up" and src < dst) or (
                    self.grow_direction == "down" and src > dst
                ):
                    break
                if self.grow_direction == "up":
                    dst_cursor += 1
                else:
                    dst_cursor -= 1
                n_dst_consumed += 1

                v_moved = int(self.physical_to_virtual[src].item())
                if v_moved < 0:
                    # `_topmost_survivor` excludes all p2v=-1 pages — corrupt state.
                    raise AssertionError(
                        f"MultiEndedAllocator({self.sub_pool_name!r})."
                        f"_flush: topmost survivor p={src} has p2v=-1; "
                        "this should be impossible (`_topmost_survivor` "
                        "excludes `holes_cpu` and `_pending_reuse_pages_cpu`)."
                        f" State: {self.allocator_state_str()}, "
                        f"#holes={len(holes_cpu)}, "
                        f"#pending_reuse={len(self._pending_reuse_pages_cpu)}"
                    )

                srcs.append(src)
                dsts.append(dst)
                v_moveds.append(v_moved)

                # Advance cursor strictly past the picked src.
                if self.grow_direction == "up":
                    cursor = src - 1
                else:
                    cursor = src + 1

                if move_cap is not None and len(srcs) >= move_cap:
                    break

            self._commit_move_batch(srcs, dsts, v_moveds, latest_event, released_fired)
            n_moves += len(srcs)

            if single_pass_absorb:
                # FULL-PACK reclaim (urgent): all interior holes now sit above the
                # frontier, so retreat past the whole lot and EMPTY the free list —
                # those pages are beyond-frontier free space (reclaimed by the next
                # extension), so `released_fired` is simply dropped too.
                n_reclaimed = len(holes_cpu)
                if self.grow_direction == "up":
                    self.watermark_physical = new_wm - n_reclaimed
                else:
                    self.watermark_physical = new_wm + n_reclaimed
                self._stats_n_pages_absorbed += n_reclaimed
                self._free_phys_pages = self._free_phys_pages[:0]
            else:
                # Non-urgent partial pass: watermark stays; a later flush absorbs the
                # now-top holes. `_free_phys_pages` is still == holes_cpu, so the
                # consumed dsts are exactly the front (grow-up) / back (grow-down)
                # `n_dst_consumed` entries; slice them, then merge freed srcs in one cat.
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
        v_moveds: List[int],
        latest_event: Optional[torch.cuda.Event],
        released_fired: List[torch.Tensor],
    ) -> None:
        """Issue ONE `move_kv_cache` + ONE bulk v2p/p2v remap for the accumulated
        `(src, dst, v_moved)` triples. Fired srcs accumulate in `released_fired`
        (merged by `_flush` AFTER its dst-slice, keeping the free list == holes_cpu);
        event-pending srcs route to `_pending_reuse` (read-race gating).
        """
        if not srcs:
            return
        with record_function("MultiEndedAlloc._commit_move_batch"):
            src_pages_t = torch.tensor(srcs, dtype=torch.int64, device=self.device)
            dst_pages_t = torch.tensor(dsts, dtype=torch.int64, device=self.device)
            v_moveds_t = torch.tensor(v_moveds, dtype=torch.int64, device=self.device)
            # Expand to token granularity (the move kernel is token-granular).
            if self.page_size == 1:
                src_t, dst_t = src_pages_t, dst_pages_t
            else:
                offsets = torch.arange(
                    self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                )
                src_t = (src_pages_t[:, None] * self.page_size + offsets).reshape(-1)
                dst_t = (dst_pages_t[:, None] * self.page_size + offsets).reshape(-1)
            move_fn = getattr(self._kvcache, "move_kv_cache", None)
            if move_fn is not None:
                move_fn(dst_t, src_t)
            else:
                copy_phys = getattr(self._kvcache, "_copy_from_physical", None)
                assert copy_phys is not None, (
                    f"sub-pool {self.sub_pool_name!r} supports neither "
                    "move_kv_cache nor _copy_from_physical"
                )
                copy_phys(src_t, dst_t)
            # ONE bulk remap (single-writer on schedule_stream).
            self.virtual_to_physical[v_moveds_t] = dst_pages_t
            self.physical_to_virtual[dst_pages_t] = v_moveds_t
            self.physical_to_virtual[src_pages_t] = -1
            self._inverse_history.append((src_pages_t, dst_pages_t, v_moveds_t))
            # Src disposition — ONE entry per batch. `src_pages_t` is reused as the
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
        `schedule_stream`. No-op if `lazy_compaction=False`.

        Empty-set fast-path: the scheduler triggers this very often and ~99% hit
        the empty state. Skip whenever there is no possible work — no holes AND no
        pending entries (the in-flight write-set only matters when compacting).
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
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self.free_group = []
            self.free(merged)


class UnifiedMambaTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Composite allocator for the MHA (full-attn) + Mamba hybrid pair.

    The token-slot surface delegates to the full-attn side (`alloc(N)` →
    MHA token slots). The Mamba sub-pool's per-request `alloc(1)` is driven
    separately by `UnifiedHybridReqToTokenPool`. Both sub-allocators are id-owners
    of their own (independent) virtual-id spaces.
    """

    supports_page_aligned_alloc: bool = True

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        kvcache,  # HybridLinearKVPool
        device: str,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):
        full_max = unified_buffer.max_slots("full")
        super().__init__(
            size=full_max - 1,
            page_size=page_size,
            dtype=unified_buffer.mha_spec("full").store_dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.unified_buffer = unified_buffer
        self._kvcache = kvcache
        self.page_size = page_size
        self.lazy_compaction = lazy_compaction

        # FULL is page-aware; MAMBA stays page_size=1 (state is per-request,
        # orthogonal to the full side's per-token paging).
        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="full",
            device=device,
            is_id_owner=True,
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        self.mamba_allocator = MultiEndedAllocator(
            kvcache=kvcache.mamba_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="mamba",
            device=device,
            is_id_owner=True,
            page_size=1,  # Mamba state stays slot-granular (1-per-req)
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        self.full_attn_allocator.bind_peer(self.mamba_allocator)
        self.mamba_allocator.bind_peer(self.full_attn_allocator)

        # The mamba slot allocator (PHYSICAL view) is built later by
        # `init_unified_mamba_pools`, which wraps `self.mamba_allocator` in a
        # `UnifiedMambaSlotAllocator` owning the v2p translate; the mamba pool is a
        # pure PHYSICAL store. The full-attn KV pool needs no allocator either —
        # write locations are resolved in the attention metadata.

        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []
        # Base init left these None; we use watermark math, not free-lists.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[unified-memory-pool] UnifiedMambaTokenToKVPoolAllocator ready: "
            "full max_slots=%d (min_slot_index=%d, page_size=%d, "
            "num_pages=%d), mamba max_slots=%d (min_slot_index=%d), "
            "full_available=%d, mamba_available=%d",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.full_attn_allocator.page_size,
            self.full_attn_allocator.num_pages,
            self.mamba_allocator.max_slots,
            self.mamba_allocator.min_slot_index,
            self.full_attn_allocator.available_size(),
            self.mamba_allocator.available_size(),
        )

    # -- size: dynamic --
    @property
    def size(self) -> int:
        # TOKENS. MUST use the SAME available view as `available_size()` so the
        # leak invariant self-cancels (available term cancels → check reduces to
        # `evictable + ... == allocated`, independent of peer-hole credit).
        return (
            self.full_attn_allocator.schedulable_available_size()
            + self.full_attn_allocator.allocated_count()
        )

    @size.setter
    def size(self, value) -> None:
        pass  # base init writes here; computed dynamically

    # -- token-slot surface: MHA side --

    # Realizable-with-compaction view so the retract gate / evict / schedule_policy
    # don't over-retract when the mamba peer holds drainable holes an urgent flush
    # would convert into shared-gap room. Per-side alloc gates still use the
    # un-credited `available_size()` so they flush before extending.
    def available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def full_available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def mamba_slot_full_token_cost(self) -> int:
        """Full-token-equivalents of shared-gap bytes ONE mamba state consumes.

        full and mamba share one byte buffer, so a mamba slot removes that many
        full-KV tokens from the gap; the prefill planner reserves this so admission
        stays inside the JOINT budget. = mamba bytes/slot ÷ full bytes/token, rounded
        UP (conservative). Only on the shared composite (non-shared pools are separate,
        so the planner sources this via `getattr(..., None)`).
        """
        return -(
            -self.mamba_allocator.entry_bytes_per_page
            // self.full_attn_allocator.entry_bytes
        )

    @property
    def size_full(self) -> int:
        return self.full_attn_allocator.max_slots - 1

    @property
    def size_mamba(self) -> int:
        return self.mamba_allocator.max_slots - 1

    def debug_print(self) -> str:
        return (
            f"#full-available={self.full_attn_allocator.available_size()}, "
            f"#mamba-available={self.mamba_allocator.available_size()}"
        )

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with record_function("UnifiedMambaAlloc.alloc"):
            return self.full_attn_allocator.alloc(need_size)

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
        """Paged extend. Mamba state is per-request (doesn't advance per-token),
        so forward only to the full sub-allocator."""
        with record_function("UnifiedMambaAlloc.alloc_extend"):
            return self.full_attn_allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages=num_new_pages,
            )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Paged decode. Mamba side stays untouched per-decode."""
        with record_function("UnifiedMambaAlloc.alloc_decode"):
            return self.full_attn_allocator.alloc_decode(
                seq_lens, seq_lens_cpu, last_loc
            )

    def translate_kv_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-pool virtual TOKEN ids -> physical TOKEN ids. Delegates to the
        full-side sub-allocator. Supports ``out=`` for cuda-graph buffer stability.
        `-1` inputs map to `-1` (treated as padding downstream).
        """
        result = self.full_attn_allocator.translate_kv_loc(loc, out=out)
        return result

    def is_slot_allocated(self, slot: int) -> bool:
        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()

    def free(self, free_index: torch.Tensor) -> None:
        with record_function("UnifiedMambaAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if not self.is_not_in_free_group:
                self.free_group.append(free_index)
                return
            self.full_attn_allocator.free(free_index)
            self.full_attn_allocator.clear_inverse_history()
            self.mamba_allocator.clear_inverse_history()

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self.free_group = []
            self.full_attn_allocator.free(merged)
            self.full_attn_allocator.clear_inverse_history()
            self.mamba_allocator.clear_inverse_history()

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.mamba_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        full_rollback = self.full_attn_allocator.restore_state(state[0])
        mamba_rollback = self.mamba_allocator.restore_state(state[1])
        return full_rollback + mamba_rollback

    def clear(self) -> None:
        self.full_attn_allocator.clear()
        self.mamba_allocator.clear()
        self.is_not_in_free_group = True
        self.free_group = []

    # -- Lazy compaction hooks --

    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:
        """Forward the per-batch `forward_done` event to BOTH sub-allocators."""
        with record_function("UnifiedMambaAlloc.set_latest_forward_done_event"):
            self.full_attn_allocator.set_latest_forward_done_event(event)
            self.mamba_allocator.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        """Hand the forward's metadata to BOTH sub-pools. Full derives its write-set
        from `out_cache_loc`; the Mamba state pool isn't written via `out_cache_loc`
        (mamba kernels, not `set_kv_buffer`), so it gets `None`.
        """
        with record_function("UnifiedMambaAlloc.set_inflight_forward"):
            self.full_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )
            self.mamba_allocator.set_inflight_forward(forward_done, None)

    def flush_opportunistic(self) -> int:
        """Non-urgent flush of BOTH sub-allocators; sync-free. Composite empty-set
        fast-path skips both calls when neither side has work.
        """
        with record_function("UnifiedMambaAlloc.flush_opportunistic"):
            fa = self.full_attn_allocator
            ma = self.mamba_allocator
            if (
                fa._free_phys_pages.numel() == 0
                and not fa._pending_reuse
                and ma._free_phys_pages.numel() == 0
                and not ma._pending_reuse
            ):
                return 0
            return fa.flush_opportunistic() + ma.flush_opportunistic()


class UnifiedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Composite allocator for the hybrid SWA pair (full + swa MHA sub-pools).

    Inherits from `SWATokenToKVPoolAllocator` only for the isinstance contract;
    we call grand-parent `BaseTokenToKVPoolAllocator.__init__` directly to skip
    the parent's static-partition sub-pool allocation (which unified-memory-pool
    replaces).

    Capacity views:
    - `available_size()`: joint byte-budget, the only safe `alloc(N)` pre-check
      (N slots cost N*(entry_full + entry_swa) shared-gap bytes).
    - `_conserve_*`: slot-conservation, for the LEAK invariant only.
    - `schedulable_*`: byte-coordinated, realizable-with-compaction.
    - `full_available_size()` / `swa_available_size()`: per-side scheduler view
      = min(conserve, schedulable).
    """

    supports_page_aligned_alloc: bool = True

    # Parent's `size` property has no setter but base init does `self.size = size`;
    # override with a no-op setter. Reading returns `min(_size_full, _size_swa)`.
    @property
    def size(self) -> int:
        return min(self._size_full, self._size_swa)

    @size.setter
    def size(self, value) -> None:
        pass

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        kvcache,  # UnifiedSWAKVPool
        device: str,
        full_max_total_num_tokens: int,
        swa_max_total_num_tokens: int,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):
        # Set _size_full / _size_swa BEFORE base init (read during it). STATIC
        # partition caps — the slot-conservation value the leak invariant expects.
        self._size_full = full_max_total_num_tokens
        self._size_swa = swa_max_total_num_tokens
        self._full_max_total_num_tokens = full_max_total_num_tokens
        self._swa_max_total_num_tokens = swa_max_total_num_tokens
        self.page_size = page_size

        # Skip SWATokenToKVPoolAllocator.__init__; call grand-parent base init
        # directly (its `self.size = size` is absorbed by our no-op setter).
        BaseTokenToKVPoolAllocator.__init__(
            self,
            size=full_max_total_num_tokens,
            page_size=page_size,
            dtype=unified_buffer.mha_spec("full").store_dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.unified_buffer = unified_buffer
        self._kvcache = kvcache
        self.lazy_compaction = lazy_compaction

        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="full",
            device=device,
            is_id_owner=True,
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        self.swa_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.swa_kv_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="swa",
            device=device,
            is_id_owner=False,  # non-owner; consumes virtuals minted by full
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        self.full_attn_allocator.bind_peer(self.swa_attn_allocator)
        self.swa_attn_allocator.bind_peer(self.full_attn_allocator)

        # The full/SWA KV pools need no allocator wiring (write locations resolved
        # in attention metadata); the composite keeps allocators for read-path translates.
        kvcache.attach_allocators(
            full_allocator=self.full_attn_allocator,
            swa_allocator=self.swa_attn_allocator,
        )

        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []
        # Empty (not None) for the leak checker.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[unified-memory-pool] UnifiedSWATokenToKVPoolAllocator ready: "
            "full max_slots=%d (min_slot_index=%d, entry_bytes=%d), "
            "swa max_slots=%d (min_slot_index=%d, entry_bytes=%d), "
            "static caps full=%d swa=%d, joint available=%d",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.full_attn_allocator.entry_bytes,
            self.swa_attn_allocator.max_slots,
            self.swa_attn_allocator.min_slot_index,
            self.swa_attn_allocator.entry_bytes,
            self._full_max_total_num_tokens,
            self._swa_max_total_num_tokens,
            self.available_size(),
        )

    # -- capacity reporting (three-way split) --

    def available_size(self) -> int:
        """Tokens available for `alloc(N)` / `alloc_extend(N)` (TOKENS).

        Joint byte-budget: each composite alloc(1) consumes one full-side AND one
        swa-side page (same virtual id). The 3-phase lazy formula consumes both
        sides' holes maximally before extending toward the gap (H_f/H_s = holes,
        e_f/e_s = bytes/page, R_f/R_s = extension room, G = byte gap):
            Phase 1 (both drain, free):     K1 = min(H_f, H_s)
            Phase 2 (fewer-holes side extends): K2 limited by remaining holes & G
            Phase 3 (both extend):          K3 = G // (e_f + e_s)
        Total capped by index-space rooms (H_f + R_f, H_s + R_s). ps==1 collapses
        to slot math. Eager has no holes → original joint formula.
        """
        fa, sa = self.full_attn_allocator, self.swa_attn_allocator
        e_f = fa.entry_bytes_per_page
        e_s = sa.entry_bytes_per_page
        # Direction-agnostic shared gap: the free byte band between the two pools.
        if fa.grow_direction == "up":
            gap_bytes = max(0, sa._byte_low_frontier() - fa._byte_high_frontier())
        else:
            gap_bytes = max(0, fa._byte_low_frontier() - sa._byte_high_frontier())
        R_f = fa.num_pages - fa.min_page_index - fa._allocated_pages()
        R_s = sa.num_pages - sa.min_page_index - sa._allocated_pages()

        if not self.lazy_compaction:
            pages_by_bytes = gap_bytes // (e_f + e_s)
            return min(pages_by_bytes, R_f, R_s) * self.page_size

        H_f = len(fa._free_phys_pages)
        H_s = len(sa._free_phys_pages)

        K1 = min(H_f, H_s)  # Phase 1: both drain

        # Phase 2: fewer-holes side extends; more-holes side keeps draining.
        if H_f <= H_s:
            e_phase2 = e_f
            K_phase2_max = H_s
        else:
            e_phase2 = e_s
            K_phase2_max = H_f
        K2_room = K_phase2_max - K1
        K2 = min(K2_room, gap_bytes // e_phase2) if e_phase2 > 0 else K2_room
        gap_bytes -= K2 * e_phase2

        K3 = gap_bytes // (e_f + e_s)  # Phase 3: both extend

        K_total = K1 + K2 + K3
        K_total = min(K_total, H_f + R_f, H_s + R_s)  # index-space caps
        return K_total * self.page_size

    # Slot-conservation views — the ONLY views the leak invariant should see
    # (returning the byte-coordinated value would flag spurious leaks).
    # `allocated_count()` is in TOKENS (the unit the leak check expects).
    def _conserve_full_available_size(self) -> int:
        return (
            self._full_max_total_num_tokens - self.full_attn_allocator.allocated_count()
        )

    def _conserve_swa_available_size(self) -> int:
        return (
            self._swa_max_total_num_tokens - self.swa_attn_allocator.allocated_count()
        )

    # PHYSICAL per-side views read by scheduling / eviction consumers. The
    # `min(...)` is sound under dynamic borrowing: the static-conserve cap bounds
    # the lending side, the byte-coordinated `schedulable_*` bounds the side that
    # has grown into the shared gap; whichever is tighter wins.
    def full_available_size(self) -> int:
        return min(
            self._conserve_full_available_size(),
            self.schedulable_full_available_size(),
        )

    def swa_available_size(self) -> int:
        return min(
            self._conserve_swa_available_size(),
            self.schedulable_swa_available_size(),
        )

    # Byte-coordinated, realizable-with-compaction views (peer drainable holes
    # credited — see `MultiEndedAllocator.schedulable_available_size`).
    def schedulable_full_available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def schedulable_swa_available_size(self) -> int:
        return self.swa_attn_allocator.schedulable_available_size()

    def _flush_both_for_alloc(self, need_tokens: int) -> bool:
        """SWA analogue of `_flush_peer_for_alloc`. Each composite alloc consumes a
        full AND a swa page and either side's compaction opens gap for the other,
        so flush BOTH (one urgent pass each).
        """
        if not self.lazy_compaction:
            return need_tokens <= self.available_size()
        self.full_attn_allocator._flush(urgent=True)
        self.swa_attn_allocator._flush(urgent=True)
        return need_tokens <= self.available_size()

    # `size_full` / `size_swa` are inherited; they read `_size_full`/`_size_swa`
    # (set to the static caps). We do NOT report `max_slots - 1`: under unified
    # memory pool that ~= full_max + swa_max and would over-promise.

    def debug_print(self) -> str:
        return (
            f"#full-available={self.full_attn_allocator.available_size()}, "
            f"#swa-available={self.swa_attn_allocator.available_size()}, "
            f"#joint-available={self.available_size()}"
        )

    def get_kvcache(self):
        return self._kvcache

    def translate_kv_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-layer read path: virtual TOKEN ids -> full-physical TOKEN ids.
        Delegates to the full-side sub-allocator. Supports ``out=`` for cuda-graph.
        """
        result = self.full_attn_allocator.translate_kv_loc(loc, out=out)
        return result

    def translate_loc_from_full_to_swa(
        self,
        kv_indices: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SWA-layer read path: virtual TOKEN ids -> swa-physical TOKEN ids (int32,
        matching the non-shared API). Page math against the swa side's v2p table.
        Supports ``out=`` (int32, same shape) for cuda-graph buffer stability.
        """
        if out is not None:
            assert out.dtype == torch.int32, (
                f"translate_loc_from_full_to_swa: out= dtype must be int32 "
                f"(matches SWA Triton kernel contract), got {out.dtype}"
            )
            assert out.shape == kv_indices.shape, (
                f"translate_loc_from_full_to_swa: out= shape "
                f"{tuple(out.shape)} must match kv_indices shape "
                f"{tuple(kv_indices.shape)}"
            )
        # Tombstone-safety clamp (mirrors the full-side clamp): tombstoned (-1)
        # v2p_swa entries must not reach `swa_k_buffer[-1]` (illegal under replay).
        # Clamp to 0 routes them to the reserved padding sink (slot 0).
        if self.swa_attn_allocator.page_size == 1:
            if out is not None:
                # Gather into a transient int64, then cast into out (`out.copy_`).
                tmp = torch.index_select(
                    self.swa_attn_allocator.virtual_to_physical, 0, kv_indices
                )
                tmp = torch.clamp_min(tmp, 0)
                out.copy_(tmp.to(torch.int32))
                return out
            result = self.swa_attn_allocator.virtual_to_physical[kv_indices]
            result = torch.clamp_min(result, 0)
            return result.to(torch.int32)
        ps = self.swa_attn_allocator.page_size
        virt_pages = kv_indices // ps
        offsets = kv_indices % ps
        swa_phys_pages = self.swa_attn_allocator.virtual_to_physical[virt_pages]
        result = (swa_phys_pages * ps + offsets).to(torch.int32)
        result = torch.clamp_min(result, 0)
        if out is not None:
            out.copy_(result)
            return out
        return result

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with record_function("UnifiedSWAAlloc.alloc"):
            # Joint pre-check. Both sides are mutual peers (each side's compaction
            # opens gap for the other), so flush BOTH on shortfall.
            if need_size > self.available_size():
                if not self._flush_both_for_alloc(need_size):
                    return None
            # Snapshot the virtual PAGES full will consume, to bind them on swa too.
            num_pages = need_size // self.page_size
            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_pages].clone()

            v_tokens = fa.alloc(need_size)
            # Post-pre-check failure can only be internal-state inconsistency.
            assert v_tokens is not None, (
                "UnifiedSWA.alloc: full.alloc returned None after joint "
                "pre-check passed — internal-state inconsistency"
            )
            self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)
            return v_tokens

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ) -> Optional[torch.Tensor]:
        """Paged extend. Runs the kernel ONCE in virtual space, then binds the
        consumed virtual PAGES on the swa side via `alloc_with_virtual`. Returns
        virtual TOKEN ids respecting the tail-page-reuse contract and the
        cross-sub-pool identity (same virtual page maps to full- and swa-physical).
        """
        with record_function("UnifiedSWAAlloc.alloc_extend"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not self._flush_both_for_alloc(need_tokens):
                    return None

            # Snapshot the virtual PAGES the kernel will consume; clone so swa keeps
            # its view after the slice is consumed.
            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

            out_indices = fa.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages=num_new_pages,
            )
            assert out_indices is not None, (
                "UnifiedSWA.alloc_extend: full.alloc_extend returned None "
                "after joint pre-check passed — internal-state inconsistency"
            )
            self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)
            return out_indices  # virtual TOKEN ids

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Paged decode. One new token per request (a page is consumed iff the
        decode wraps). Same one-kernel-in-virtual-space discipline as ``alloc_extend``.
        """
        with record_function("UnifiedSWAAlloc.alloc_decode"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not self._flush_both_for_alloc(need_tokens):
                    return None

            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

            out_indices = fa.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
            assert out_indices is not None, (
                "UnifiedSWA.alloc_decode: full.alloc_decode returned None "
                "after joint pre-check passed — internal-state inconsistency"
            )

            if new_virtual_pages.numel() > 0:
                self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)

            return out_indices  # virtual TOKEN ids

    def is_slot_allocated(self, slot: int) -> bool:
        """Token-slot surface = the full side (which owns the virtual ids)."""
        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()

    # -- free --

    def free(self, free_index: torch.Tensor) -> None:
        with record_function("UnifiedSWAAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if not self.is_not_in_free_group:
                self.free_group.append(free_index)
                return
            # Free both peers; the per-sub-pool v2p IS the mapping, so order isn't
            # load-bearing. Filter the swa side to skip already-tombstoned virtuals
            # (`swa.v2p_page == -1` from an earlier `free_swa`); the full side needs
            # no filter (it's the lifecycle owner, so every value is still bound).
            v = free_index.detach().to(torch.int64)
            v_pages = v // self.page_size
            swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]
            # `> 0` strict: -1 = tombstoned, 0 = padding-sink page; both skipped.
            live_token_mask = swa_v2p_pages > 0
            live_tokens = v[live_token_mask]
            if live_tokens.numel() > 0:
                self.swa_attn_allocator.free(live_tokens)
            self.full_attn_allocator.free(v)
            self.full_attn_allocator.clear_inverse_history()
            self.swa_attn_allocator.clear_inverse_history()

    def free_swa(self, free_index: torch.Tensor) -> None:
        """SWA tombstone path: release swa-physical, leave virtual id and
        full-physical live. Called by `SWARadixCache._evict_swa_only` when a node
        ages past the sliding-window horizon. `swa.v2p_page[v_page] = -1` IS the
        tombstone.
        """
        if free_index is None or free_index.numel() == 0:
            return
        # Keep only tokens whose virtual PAGE is still bound on swa (calling
        # `swa.free` on an already-tombstoned one would assert).
        v = free_index.detach().to(torch.int64)
        v_pages = v // self.page_size
        # `> 0` strict: -1 = tombstoned, page 0 = padding sink (never freeable).
        swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]
        live = v[swa_v2p_pages > 0]
        if live.numel() == 0:
            return
        self.swa_attn_allocator.free(live)
        self.swa_attn_allocator.clear_inverse_history()

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        """No-op stub for HiCache load-back compatibility. In shared mode there is
        no mapping tensor (the swa v2p IS the mapping); HiCache for shared SWA is
        out of scope.
        """
        return

    # -- free-group --

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self.free_group = []
            self.free(merged)

    # -- spec-decode hooks (asserted off; preserved for future use) --

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        full_rollback = self.full_attn_allocator.restore_state(state[0])
        swa_rollback = self.swa_attn_allocator.restore_state(state[1])
        return full_rollback + swa_rollback

    def clear(self) -> None:
        self.full_attn_allocator.clear()
        self.swa_attn_allocator.clear()
        self.is_not_in_free_group = True
        self.free_group = []

    # -- Lazy compaction hooks --

    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:
        """Forward the per-batch `forward_done` event to BOTH sub-allocators."""
        with record_function("UnifiedSWAAlloc.set_latest_forward_done_event"):
            self.full_attn_allocator.set_latest_forward_done_event(event)
            self.swa_attn_allocator.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        """Hand the forward's metadata to BOTH sub-pools. Each materializes its
        write-set via its OWN v2p; the forward writes both sides per new token,
        so both get a non-empty in-flight tensor.
        """
        with record_function("UnifiedSWAAlloc.set_inflight_forward"):
            self.full_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )
            self.swa_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )

    def flush_opportunistic(self) -> int:
        """Non-urgent flush of BOTH sub-allocators; sync-free. Composite empty-set
        fast-path skips both calls when neither side has work.
        """
        with record_function("UnifiedSWAAlloc.flush_opportunistic"):
            fa = self.full_attn_allocator
            sa = self.swa_attn_allocator
            if (
                fa._free_phys_pages.numel() == 0
                and not fa._pending_reuse
                and sa._free_phys_pages.numel() == 0
                and not sa._pending_reuse
            ):
                return 0
            return fa.flush_opportunistic() + sa.flush_opportunistic()
