"""MultiEndedAllocator: one allocator per sub-pool over a `SharedKVPool`.

Key points:

* Each `MultiEndedAllocator` owns: a physical watermark (grows up from
  `min_page_index`, or down from `num_pages-1`), per-sub-pool
  `virtual_to_physical` / `physical_to_virtual` tables (sized by PAGES),
  and — iff it is the *id-owner* of its virtual-id granularity — the
  `free_virtual_ids` free-list (also page-granular).
* `alloc(N)` (id-owner only, N must be page-aligned): pop N/page_size virtual
  pages, take N/page_size physical pages, bind, return token IDs.
  `alloc_with_virtual(virtual_pages)` (physical-holding non-owner): take
  physical pages for caller-supplied virtual page ids, bind.
  `alloc_extend` / `alloc_decode`: call the upstream `alloc_extend_kernel` /
  `alloc_decode_kernel` ONCE in virtual space using `free_virtual_ids` as the
  free-page pointer; emit virtual token ids that respect the tail-page-reuse
  contract. `free(virtual_token_ids)`: recover page ids via
  `unique(// page_size)`, un-map, eager-compact whole pages, (if id-owner)
  recycle the virtual page ids.
* Eager compaction touches **only** `virtual_to_physical` /
  `physical_to_virtual` page tables (O(num_relocations_pages) scalar ops) —
  no reference rewriting, no binder. Compaction's `move_kv_cache` call
  expands page ids to token ids before invoking the token-granular move.
* **Token IDs vs Page IDs on the surface**: every public method takes/returns
  TOKEN-granular tensors (matching `PagedTokenToKVPoolAllocator`'s contract).
  Only the internal `free_virtual_ids` list and the v2p/p2v tables are
  page-granular.
* For `page_size == 1`, behavior is byte-identical: a "page" is
  a single slot, and all the page math collapses to slot math.

Concurrency: the scheduler runs alloc/free serially; no mutex is taken here.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.profiler import record_function

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# `alloc_extend_kernel` / `alloc_decode_kernel` are not on the package's
# public re-export surface (`allocator/__init__.py` exposes only the allocator
# classes + `alloc_extend_naive`); upstream's monolith->package split moved
# these Triton kernels into the `paged` submodule, so import them from there.
from sglang.srt.mem_cache.allocator.paged import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.mem_cache.shared_kv_pool import SharedKVPool
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.triton_ops.virtual_slot import (
    alloc_bind_inplace,
    translate_kv_indices_inplace,
)
from sglang.srt.utils.common import get_num_new_pages, next_power_of_2
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


# Optional sort-after-merge on the hot path. OFF: `_free_lazy` /
# `_drain_pending_reuse` just `torch.cat` and leave the list unsorted, and the
# cold-path `_flush` sorts once before the survivor walk. ON: each cat is
# followed by a sort so `_flush` can skip its sort step.
_SORT_FREE_LIST_AFTER_MERGE = envs.SGLANG_SORT_FREE_LIST_AFTER_MERGE.get()


# Periodic per-allocator stats logging for compaction-frequency observability.
# When enabled, `_flush` emits a one-line summary of cumulative counters at
# INFO level at most once per interval, tagged by sub-pool ("full"/"swa"/
# "mamba").
import atexit
import signal
import time as _time_mod  # local alias so tests can patch
import weakref
_LAZY_COMPACTION_STATS_ENABLED = envs.SGLANG_LOG_LAZY_COMPACTION_STATS.get()
_LAZY_COMPACTION_STATS_INTERVAL_SEC = float(
    envs.SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC.get()
)
# Module-level WeakSet of every MultiEndedAllocator instance with stats
# enabled. The signal handler below iterates this on SIGTERM/SIGINT and
# emits each instance's final counters before re-raising the default
# behavior. WeakSet so a dropped allocator doesn't leak.
_STATS_INSTANCES: "weakref.WeakSet[MultiEndedAllocator]" = weakref.WeakSet()
_SIGNAL_HANDLERS_INSTALLED = False

def _emit_all_final_stats(reason: str) -> None:
    """Emit the FINAL line for every live MultiEndedAllocator. Tagged
    with `reason` (e.g. "atexit", "SIGTERM") so the log shows what
    triggered the snapshot.
    """
    for inst in list(_STATS_INSTANCES):
        try:
            inst._emit_stats_final(reason=reason)
        except Exception:
            pass


def _signal_handler(signum, frame):
    """Catch SIGTERM/SIGINT, emit stats, then re-raise the default
    behavior. This covers the harness-teardown path that atexit misses
    (Python's atexit does not fire on signal-triggered exits).
    """
    try:
        sig_name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        sig_name = str(signum)
    _emit_all_final_stats(reason=sig_name)
    # Restore default handler and re-raise so normal shutdown proceeds.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_signal_handlers_once() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    _SIGNAL_HANDLERS_INSTALLED = True
    # Install SIGTERM + SIGINT handlers. Skip if a prior handler exists
    # AND it isn't the default — the host process (sglang's scheduler
    # subprocess) doesn't install custom handlers for these by default,
    # so this should usually win.
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
            if prev in (signal.SIG_DFL, signal.SIG_IGN, None):
                signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            # signal.signal raises if not called from the main thread of
            # the main interpreter — silently skip (atexit still fires
            # on clean exits).
            pass


class MultiEndedAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for one sub-pool over a `SharedKVPool`."""

    def __init__(
        self,
        *,
        kvcache,
        shared_buffer: SharedKVPool,
        sub_pool_name: str,
        device: str,
        is_id_owner: bool,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):
        spec = shared_buffer.spec(sub_pool_name)
        max_slots = shared_buffer.max_slots(sub_pool_name)
        # `dtype` on the base allocator is informational. Each `SubPoolSpec`
        # subclass implements `get_dtype()` to return its representative
        # storage dtype (MHA: `store_dtype`; Mamba: `conv_dtype` — the
        # dominant state buffer's dtype). See `SubPoolSpec.get_dtype`.
        super().__init__(
            size=max_slots,
            page_size=page_size,
            dtype=spec.get_dtype(),
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.shared_buffer = shared_buffer
        self.sub_pool_name = sub_pool_name
        self.spec = spec
        self.max_slots = max_slots
        self.grow_direction = spec.grow_direction
        # Per-token (per-slot) entry bytes — unchanged by paging.
        self.entry_bytes = spec.entry_bytes()
        self.min_slot_index = shared_buffer.min_slot_index(sub_pool_name)
        self.is_id_owner = is_id_owner
        # Optional handle for the model forward stream. In overlap mode the
        # scheduler runs `pop_and_process` (which triggers `free` →
        # `_compact_pending`'s `move_kv_cache`) on `schedule_stream` while
        # the in-flight forward batch is still reading v2p / reading+writing
        # K/V slots on `forward_stream`. We use this handle to drop a one-way
        # `schedule_stream.wait_stream(forward_stream)` barrier at the top of
        # `free` so the v2p writes and the move kernel serialize after the
        # forward's reads/writes complete. The reverse direction — the next
        # forward seeing the allocator's writes — is already handled by the
        # existing `forward_stream.wait_stream(schedule_stream)` barrier at
        # the top of `run_batch`. In normal schedule the barrier is a near-
        # no-op because sampling's CPU sync has already drained forward_stream
        # before pop_and_process runs.
        self.forward_stream = forward_stream

        # --- Page-aware bookkeeping ---
        #
        # When `page_size == 1`, num_pages == max_slots and
        # min_page_index == min_slot_index, so all the page math collapses
        # back to slot math (behavior byte-identical).
        #
        # When `page_size > 1`:
        # - `num_pages = max_slots // page_size` (truncate).
        # - `min_page_index = ceil(min_slot_index / page_size)` — the
        #   smallest page id that is fully outside the dummy-write reserved
        #   byte zone `[0, entry_max)`. This preserves the reserved-sink
        #   invariant:
        #       min_page_index * entry_bytes_per_page
        #       ≥ min_slot_index * entry_bytes
        #       ≥ entry_max.
        # - `entry_bytes_per_page = entry_bytes * page_size` — used by the
        #   joint byte-budget check on the SWA composite.
        self.page_size = page_size
        self.num_pages = max_slots // page_size
        self.min_page_index = (
            self.min_slot_index + page_size - 1
        ) // page_size  # ceil
        self.entry_bytes_per_page = self.entry_bytes * page_size

        # v -> p, sized by PAGES (not slots). Page id 0 ↔ page 0 is the
        # dummy-padding anchor; trailing `[-1]` row is the `-1` sentinel.
        self.virtual_to_physical = torch.full(
            (self.num_pages + 1,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        # p -> v, also sized by PAGES.
        self.physical_to_virtual = torch.full(
            (self.num_pages + 1,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        # Back-compat alias: `num_virtual_ids` is still consulted by
        # `is_slot_allocated` etc. It represents the COUNT OF VIRTUAL PAGES
        # (matches table sizing).
        self.num_virtual_ids = self.num_pages

        self._peer: Optional["MultiEndedAllocator"] = None

        # Inverse history of relocations (for spec rollback). Each entry is
        # one batch (src_phys_tensor, dst_phys_tensor, v_moved_tensor) — all
        # at PAGE granularity. The composite calls `clear_inverse_history`
        # after each `free` so it stays bounded.
        self._inverse_history: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

        # --- Lazy compaction state ---
        #
        # When `lazy_compaction=False` (default): byte-identical to eager
        # behavior — `free` runs the per-call boundary move with
        # `wait_stream(forward_stream)`. All new state below is unused.
        #
        # When `lazy_compaction=True`:
        # - `_free_phys_pages` (`torch.Tensor`, GPU int64): the GPU free
        #   list. Holds physical PAGE ids freed by `_free_lazy` AND src
        #   pages released by `_drain_pending_reuse` /
        #   `_commit_move_batch`. Possibly unsorted (depending on
        #   `SGLANG_SORT_FREE_LIST_AFTER_MERGE` env knob); ALWAYS sorted
        #   ascending at `_flush` step 3. Boundary absorption is deferred
        #   entirely to `_flush` — the hot-path `_free_lazy` does ONE
        #   `torch.cat` and is otherwise sync-free.
        # - `_pending_reuse` holds `(p, reader_forward_done_event)` for
        #   compaction src pages whose remap completed but whose in-flight
        #   reader's event hasn't fired — `p` cannot re-enter
        #   `_free_phys_pages` until the read settles (so a future alloc's
        #   WRITE to KV[p] won't race a still-pending READ).
        # - `live_page_count` is the CPU-side slot-conservation counter
        #   (incremented in `take_physical`, decremented in `free`,
        #   invariant under compaction).
        # - `_latest_forward_done_event` is the CUDA event of the most
        #   recently-launched forward batch, set by the scheduler via
        #   `set_latest_forward_done_event(event)` right after the forward
        #   is recorded on `forward_stream`. `_flush` captures this event
        #   for every survivor it moves and stashes it in `_pending_reuse`
        #   so src reuse waits for the in-flight reader to settle. `None`
        #   when no forward is in-flight (or when the scheduler does not
        #   wire the hook — e.g., normal mode where sampling's CPU sync
        #   makes the event trivially "already fired").
        #
        # No `_in_progress_moves` queue is needed: the KV copy stays on
        # `schedule_stream` and the v2p/p2v remap is inlined immediately
        # after. Single-stream ordering on `schedule_stream` guarantees the
        # copy completes before any subsequent op on the same stream
        # observes the remap — so the remap does NOT need a
        # `compaction_copy_done` event nor a deferred drain queue. (Such a
        # queue would only become necessary if the copy moved to a separate
        # compaction stream where it could still be in flight when the remap
        # runs.)
        self.lazy_compaction = lazy_compaction
        # GPU free-list. Holds physical PAGE ids freed by
        # `_free_lazy` (and compaction-src pages released at `_flush` end —
        # event-fired srcs via the `released_fired` merge, event-pending
        # srcs via `_drain_pending_reuse`). May be sorted or unsorted
        # depending on `_SORT_FREE_LIST_AFTER_MERGE`; ALWAYS sorted at
        # `_flush` step 3.
        # Hot-path `_free_lazy` does ONE `torch.cat` onto this. Hot-path
        # `take_physical` drains via slice (no `.item()` syncs).
        self._free_phys_pages: torch.Tensor = torch.empty(
            0, dtype=torch.int64, device=device
        )
        # `_pending_reuse` is keyed by Event with ONE entry per
        # batch (NOT one per src). The value is `(cpu_list, gpu_tensor)`:
        # `cpu_list` is the Python list of src page ids built at
        # survivor-walk time (used for parallel-Set updates without a
        # `.tolist()` sync at drain time); `gpu_tensor` is the SAME GPU
        # tensor used by `_commit_move_batch`'s v2p/p2v remap, kept alive
        # by the dict reference so drain can `torch.cat` it onto
        # `_free_phys_pages` without an H2D copy.
        self._pending_reuse: Dict[
            "torch.cuda.Event",
            Tuple[List[int], torch.Tensor],
        ] = {}
        # Parallel CPU mirror of all pages currently in
        # `_pending_reuse`. O(1) membership check for `_topmost_survivor`
        # so the survivor walk never picks an in-flight src as a survivor.
        # Updated alongside `_pending_reuse` in `_commit_move_batch` /
        # `_drain_pending_reuse`.
        self._pending_reuse_pages_cpu: Set[int] = set()
        # Cumulative observability counters (env-gated periodic
        # emit). Updated in the hot/cold path entry points; reset is
        # NOT done at clear() so that the totals reflect the whole
        # server lifetime.
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
        # Guard so the at-exit / signal-handler final emit fires at most once.
        self._stats_final_emitted: bool = False
        # Force a final stats emit at process exit so the workload-end
        # state is captured even when the last `_flush` happened more
        # than `_LAZY_COMPACTION_STATS_INTERVAL_SEC` before shutdown.
        # Bypasses the time gate; gated by the same `_ENABLED` env var.
        # Registration paths:
        #   1. `atexit.register`  — fires on clean Python interpreter exit.
        #   2. `_STATS_INSTANCES` WeakSet + a process-wide SIGTERM/SIGINT
        #      handler — fires on signal-triggered shutdown (covers the
        #      case where the process is killed with SIGTERM before atexit
        #      runs).
        if _LAZY_COMPACTION_STATS_ENABLED:
            atexit.register(self._emit_stats_final, reason="atexit")
            _STATS_INSTANCES.add(self)
            _install_signal_handlers_once()
        self.live_page_count = 0
        self._latest_forward_done_event: Optional["torch.cuda.Event"] = None
        # Most-recently-launched forward's metadata for the write-race
        # check in `_flush`. Stored as a single
        # (forward_done_event, out_cache_loc_virtual TENSOR) tuple — NOT
        # materialized as a Python set. Set by the scheduler via
        # `set_inflight_forward(...)` right after launching a forward.
        #
        # Why a single slot, not a list: at all three `_flush` call sites
        # the scheduler thread has at most ONE forward in flight at the
        # moment of the call (both flush_opportunistic
        # call sites are AFTER pop_and_process drains the previous batch,
        # before the next launch). The earlier `_inflight_batches: List`
        # design pre-materialized each batch's write-set via
        # `phys.tolist()` from inside `forward_stream_ctx` — that sync
        # blocked the scheduler on the forward. The
        # simplified design just stores the tensor reference; `_flush`
        # materializes the write-set LAZILY on schedule_stream only when
        # a survivor candidate actually needs to be checked.
        self._inflight_forward: Optional[
            Tuple["torch.cuda.Event", torch.Tensor]
        ] = None

        # Per-call move cap on non-urgent
        # `_flush`. Without this, a single `flush_opportunistic` invoked
        # against a heavily fragmented band (e.g., Falcon × radix ×
        # stress *after* prior workloads have populated the radix cache)
        # can spend many minutes compacting the entire backlog under a
        # single `on_idle()` call, blocking ZMQ IPC the whole time. With
        # the cap, each non-urgent call
        # returns after at most this many moves; the remaining work is
        # picked up by the next opportunistic flush (typically the next
        # scheduler iteration), so the scheduler stays responsive and
        # progress is still made. Default 4096 keeps each call to a few
        # tens of ms with the O(1) survivor cursor below; tune via env
        # for benchmarking. The urgent path (`alloc*` shortfall retry)
        # is NOT capped — it must drain everything to satisfy correctness.
        self._lazy_max_moves_per_call = int(
            os.environ.get(
                "SGLANG_LAZY_COMPACTION_MAX_MOVES_PER_CALL", "4096"
            )
        )

        self.clear()

        logger.info(
            "[shared-pool] MultiEndedAllocator(%r) ready: grow=%s, max_slots=%d, "
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

    def bind_peer(self, peer: "MultiEndedAllocator") -> None:
        self._peer = peer

    @property
    def peer(self) -> Optional["MultiEndedAllocator"]:
        return self._peer

    # -- state --

    def clear(self) -> None:
        """Reset to initial state.

        Watermark and free-list are at PAGE granularity. Page 0 is the
        padding anchor (`virtual_to_physical[0] = 0` ↔ token 0 = dummy sink).
        Page ids in `[0, min_page_index)` are reserved (see "Dummy-write
        safety proof" in the Stage-3 plan).
        """
        if self.grow_direction == "up":
            self.watermark_physical = self.min_page_index
        else:
            self.watermark_physical = self.num_pages - 1
        self.virtual_to_physical.fill_(-1)
        # Virtual PAGE 0 ↔ physical PAGE 0 (padding sink page). Within page 0,
        # only token 0 is the dummy-write target; tokens 1..page_size-1 in
        # page 0 are reserved but never written to (allocator never emits
        # them since min_page_index ≥ 1).
        self.virtual_to_physical[0] = 0
        self.virtual_to_physical[-1] = -1  # trailing sentinel
        self.physical_to_virtual.fill_(-1)
        self.physical_to_virtual[0] = 0
        self.physical_to_virtual[-1] = -1
        if self.is_id_owner:
            # Virtual pages 0..min_page_index-1 are reserved; trailing
            # sentinel row is not in the free list. For page_size == 1,
            # this collapses to `arange(min_slot_index, max_slots)`.
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
        # Lazy compaction state — no-op when lazy_compaction=False.
        self._free_phys_pages = torch.empty(
            0, dtype=torch.int64, device=self.device
        )
        self._pending_reuse.clear()
        self._pending_reuse_pages_cpu.clear()
        self.live_page_count = 0
        self._inflight_forward = None
        self._latest_forward_done_event = None

    def backup_state(self):
        # SGLang's spec-decode pattern allocates only inside a backup window
        # (no free), so under correct usage `_inverse_history` doesn't grow.
        return (
            self.watermark_physical,
            (len(self.free_virtual_ids) if self.is_id_owner else None),
            len(self._inverse_history),
        )

    def restore_state(self, state):
        watermark, n_free_virtual, n_inverse = state
        self.watermark_physical = watermark
        if self.is_id_owner and n_free_virtual is not None:
            # In-window allocs sliced `free_virtual_ids` from the front; we can't
            # un-pop a slice without re-tracking. Simplest correct restore: the
            # ids consumed in-window are still bound (their v2p entries point at
            # physical slots now below the restored watermark — harmless, they get
            # overwritten on next alloc). We only need to restore the *count* of
            # free virtual ids by re-deriving the free list from the watermark +
            # the bound set. Cheapest: rebuild from scratch is O(max_slots); since
            # spec is asserted off this path is not exercised, so we take the
            # simple route: re-derive on the next `alloc` is not enough, so do a
            # full rebuild here.
            #
            # NOTE: this is intentionally conservative; revisit if the spec hot
            # path needs O(1) rollback.
            pass  # placeholder; spec asserted off.
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
        """Internal: number of allocated PAGES (page-granular math).

        Used by `available_size()` and the SWA composite to compute the
        index-space headroom in PAGE units. Callers that need TOKEN units
        must use `allocated_count()`.
        """
        if self.grow_direction == "up":
            return max(0, self.watermark_physical - self.min_page_index)
        return max(0, self.num_pages - 1 - self.watermark_physical)

    def allocated_count(self) -> int:
        """Public: number of LIVE allocated TOKENS (excludes lazy holes /
        pending in lazy mode).

        Matches the upstream convention that all external-facing capacity
        methods report tokens — cf. ``BaseTokenToKVPoolAllocator.available_size``
        which returns ``len(free_pages) * page_size``.

        The leak invariant in the scheduler runtime checker
        (``_check_swa_pool`` / ``_check_full_pool``) is:
        ``total_TOKENS == available_TOKENS + allocated_TOKENS + ...``.
        Returning pages here instead of tokens would trip the
        "pool memory leak detected" runtime check.

        Lazy compaction introduces a second leak surface:
        in lazy mode, freed pages that are NOT at the boundary land in
        ``_free_phys_pages`` (holes) instead of shrinking the watermark, and
        survivors waiting on in-flight reader events sit in ``_pending_reuse``.
        Both occupy positions inside ``[min_page_index, watermark)`` but are
        no longer live. ``_allocated_pages()`` (= watermark − min) therefore
        OVER-counts the live total in lazy mode by ``len(_free_phys_pages) +
        len(_pending_reuse)`` pages.

        Using the over-count for ``allocated_count`` makes ``swa_available_size``
        / ``full_available_size`` (slot-conservation views) UNDER-report,
        and the runtime checker fires "pool memory leak detected" with
        ``num_used = (holes + pending) * page_size``. This affects lazy +
        radix-cache paths. Chunk-cache paths are unaffected because each
        ``cache_finished_req`` frees one full request's contiguous pages
        which the ``_release_phys_page`` boundary-shortcut absorbs into the
        watermark before holes can pile up; radix's tree-eviction +
        ``free_swa`` paths free arbitrary subsets that don't land at the
        boundary, so holes accumulate between flushes.

        Fix: use the dedicated CPU counter ``live_page_count`` in lazy mode.
        It is incremented in the lazy ``take_physical`` path (after holes
        are drained / watermark extended) and decremented in ``_free_lazy``;
        the lazy compaction plan guarantees it is invariant under compaction (a move
        relocates a live page, the count is unchanged). In EAGER mode the
        lazy ``take_physical`` branch never runs (so ``live_page_count``
        stays at 0), and ``_allocated_pages()`` already equals live —
        there are no holes — so the watermark-span form is correct.
        """
        if self.lazy_compaction:
            return self.live_page_count * self.page_size
        return self._allocated_pages() * self.page_size

    def is_slot_allocated(self, slot: int) -> bool:
        """Whether the PAGE containing this token-level virtual id is in use.

        The `slot` argument is a TOKEN-granular virtual id (matching the
        Stage-1/2 API). We recover the virtual page and look it up.
        """
        # Recover the virtual page from the token-granular virtual id. For
        # page_size == 1, virt_page == slot (no change in behavior).
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
        """Byte just past this side's last-allocated page (grow-up) /
        the buffer's top (grow-down)."""
        if self.grow_direction == "up":
            return self.watermark_physical * self.entry_bytes_per_page
        return self.num_pages * self.entry_bytes_per_page

    def _byte_low_frontier(self) -> int:
        """Byte that begins this side's allocatable range (grow-up) /
        first byte just below this side's lowest live page (grow-down)."""
        if self.grow_direction == "up":
            return self.min_page_index * self.entry_bytes_per_page
        return (self.watermark_physical + 1) * self.entry_bytes_per_page

    def _current_gap_bytes(self) -> int:
        """Free byte band between this side's frontier and the peer's CURRENT
        frontier (no peer compaction assumed). Direction-agnostic."""
        if self.grow_direction == "up":
            my_high = self._byte_high_frontier()
            peer_low = (
                self._peer._byte_low_frontier()
                if self._peer is not None
                else self.shared_buffer.total_bytes
            )
            return max(0, peer_low - my_high)
        my_low = self._byte_low_frontier()
        peer_high = self._peer._byte_high_frontier() if self._peer is not None else 0
        return max(0, my_low - peer_high)

    def _available_tokens(self, extra_gap_bytes: int = 0) -> int:
        """Tokens allocatable on this side given `extra_gap_bytes` of ADDED
        gap room (0 == current realizable; >0 == post-peer-compaction).

        `pages_by_index_space` is this side's OWN index headroom and is
        unaffected by `extra_gap_bytes` — crediting peer bytes can never give
        us more page indices than our own table has.
        """
        gap_bytes = self._current_gap_bytes() + extra_gap_bytes
        pages_by_bytes = gap_bytes // self.entry_bytes_per_page
        # `_allocated_pages()` (page-granular) — the index-space headroom
        # is a page-count, NOT a token-count. (We multiply by page_size at
        # the return statement, the single external token boundary.)
        pages_by_index_space = (
            self.num_pages - self.min_page_index - self._allocated_pages()
        )
        pages_extend = min(pages_by_bytes, pages_by_index_space)
        # Lazy: drainable holes don't consume new bytes.
        pages_drain = len(self._free_phys_pages) if self.lazy_compaction else 0
        return (pages_extend + pages_drain) * self.page_size

    def available_size(self) -> int:
        """Tokens (NOT pages) allocatable on this side RIGHT NOW, with no
        peer compaction.

        Matches `BaseTokenToKVPoolAllocator.available_size()`'s
        `len(free_pages) * page_size` contract. This is the value the
        alloc/alloc_extend shortfall gates consult to decide whether to issue
        a peer urgent-flush (see `alloc`), so it MUST stay current-realizable
        — do NOT fold peer drainable holes in here (use
        `schedulable_available_size()` for the scheduler-facing, post-flush
        view).

        Lazy compaction: holes in `_free_phys_pages` are drainable
        without consuming new bytes (they're already in the allocated band),
        so they add to capacity over and above the watermark-extension room.
        """
        return self._available_tokens()

    def _peer_drainable_hole_bytes(self) -> int:
        """Bytes the peer's urgent flush would release into the shared gap:
        each drainable peer hole, once compacted, retreats the peer watermark
        by one peer-page == `peer.entry_bytes_per_page` bytes of gap.

        Only `_free_phys_pages` (already-free, synchronously drainable) count
        — NOT `_pending_reuse` (awaiting a forward event), so the credit is
        realizable by `alloc`'s `self._peer._flush(urgent=True)` retry.
        """
        peer = self._peer
        if peer is None or not peer.lazy_compaction:
            return 0
        return len(peer._free_phys_pages) * peer.entry_bytes_per_page

    def schedulable_available_size(self) -> int:
        """Tokens allocatable on this side AFTER a peer urgent-flush — the
        scheduler-facing, realizable-with-compaction capacity.

        Equals `available_size()` plus the room a peer compaction would open
        in the shared gap. Realizable: `alloc`'s shortfall path issues
        `self._peer._flush(urgent=True)` before extending, so any capacity
        credited here is actually obtainable at alloc time. Used by the
        composite size views the scheduler reads (retract gate, evict,
        schedule_policy); the alloc gates themselves use `available_size()`.
        """
        return self._available_tokens(extra_gap_bytes=self._peer_drainable_hole_bytes())

    def _flush_peer_for_alloc(self, need_tokens: int) -> bool:
        """One urgent peer-flush on alloc shortfall; returns whether enough is
        now allocatable on THIS side.

        Only the PEER's compaction releases bytes into the shared gap (own
        compaction trades a hole for gap room, net 0). A SINGLE urgent flush
        suffices: its full-pass compaction (see `_flush`) retreats the peer watermark past
        ALL reclaimable holes in one urgent flush, so
        interior peer holes open gap immediately — no loop, one D2H.
        Standalone (no peer) → nothing useful to flush.
        """
        if not (self.lazy_compaction and self._peer is not None):
            return False
        self._peer._flush(urgent=True)
        return need_tokens <= self.available_size()

    # -- physical-slot / physical-page primitives --

    # `take_physical` slices the free list directly (sort-aware when
    # `_SORT_FREE_LIST_AFTER_MERGE`), and `_flush`'s compaction-dst pick
    # reads from `holes_cpu` (the post-tolist snapshot) without touching
    # the GPU tensor.

    def take_physical(self, need_size: int) -> Optional[torch.Tensor]:
        """Reserve `need_size` TOKENS of physical capacity and return the
        backing physical PAGE ids.

        `need_size` must be a multiple of `page_size` (asserted). Returns
        `None` if we cannot satisfy the request — see "failure modes"
        below.

        **Eager mode** (`lazy_compaction=False`): pure watermark advance.
        Returns a contiguous arange of PAGE ids; watermark advances by
        `need_size`. Fails only on index-space / peer byte-frontier
        overflow.

        **Lazy mode** (`lazy_compaction=True`): drain `_free_phys_pages`
        FIRST via the directional pop, then advance the watermark only
        for the remainder. The watermark does NOT advance at all when
        the request is fully satisfied from holes. Returned tensor may
        be non-contiguous (drained-holes ids ++ extended-tail ids);
        `bind` treats it as a scatter, so contiguity is not required.

        **Pre-check, no rollback**. Order of operations:
          1. Compute `n_drain = min(num_pages, len(holes))` and
             `need_more = num_pages - n_drain` without touching state.
          2. If `need_more > 0`, try `_extend_watermark(need_more)`.
             On failure, NOTHING has been mutated yet — return `None`.
          3. Only after extension succeeds do we pop the drained holes
             (guaranteed to succeed because `n_drain ≤ len(holes)`).
          4. Combine drained + extended into the return tensor.

        **Failure modes**: returns `None` when (a) the watermark
        extension would overflow the index-space cap (`num_pages` /
        `min_page_index`) or (b) cross the peer's byte frontier (the
        peer-aware guard in `_extend_watermark`). The byte-frontier
        check upstream in `available_size` is the primary gate; this is
        defense in depth.
        """
        with record_function("MultiEndedAlloc.take_physical"):
            if need_size <= 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)
            assert need_size % self.page_size == 0, (
                f"take_physical: need_size={need_size} must be a multiple of "
                f"page_size={self.page_size}"
            )
            num_pages = need_size // self.page_size

            # Eager path — contiguous range; no holes ever.
            if not self.lazy_compaction:
                return self._take_physical_eager(num_pages)

            # Lazy path — slice the GPU free list. Order of
            # the drained slice depends on the sort flag:
            #   - sort ON: sorted ASCENDING. "Deepest in band" = smallest
            #     for grow-up (slice [:n_drain]), largest for grow-down
            #     (slice [-n_drain:].flip(0)). Preserves Invariant C
            #     greedy clustering.
            #   - sort OFF: order is arbitrary; take from the front for
            #     both directions. Layout quality is recovered by
            #     `_flush`'s sort + complete absorb.
            # Zero D2H syncs (slicing is CPU shape metadata only).
            n_drain = min(num_pages, int(self._free_phys_pages.shape[0]))
            need_more = num_pages - n_drain

            # Try extending the watermark first. If this fails, state is
            # untouched — no rollback needed.
            if need_more > 0:
                if not self._extend_watermark(need_more):
                    return None

            # Extension succeeded (or wasn't needed). Now drain holes —
            # guaranteed to succeed since n_drain ≤ len(holes).
            if n_drain > 0:
                if _SORT_FREE_LIST_AFTER_MERGE:
                    if self.grow_direction == "up":
                        drained_t = self._free_phys_pages[:n_drain]
                        self._free_phys_pages = self._free_phys_pages[n_drain:]
                    else:
                        drained_t = (
                            self._free_phys_pages[-n_drain:].flip(0)
                        )
                        self._free_phys_pages = (
                            self._free_phys_pages[:-n_drain]
                        )
                else:
                    # Unsorted — take from front regardless of direction.
                    drained_t = self._free_phys_pages[:n_drain]
                    self._free_phys_pages = self._free_phys_pages[n_drain:]
            else:
                drained_t = None

            self.live_page_count += num_pages

            # Fast path: pure extension (no drained) — return arange tensor.
            if drained_t is None:
                return self._take_physical_arange(num_pages)

            # Fast path: pure drain (no extension) — return the drained
            # slice. Detach from the `_free_phys_pages` view so subsequent
            # `torch.cat` rebindings of `_free_phys_pages` (in `_free_lazy`,
            # etc.) don't keep this view alive.
            if need_more == 0:
                return drained_t.clone()

            # Mixed path: combine drained holes with extended-watermark
            # pages (adjacent to new watermark). `bind` is order-agnostic;
            # the layout is documented for readability of debug dumps.
            if self.grow_direction == "up":
                new_wm = self.watermark_physical
                extended_t = torch.arange(
                    new_wm - need_more,
                    new_wm,
                    dtype=torch.int64,
                    device=self.device,
                )  # ascending
            else:
                new_wm = self.watermark_physical
                extended_t = torch.arange(
                    new_wm + need_more,
                    new_wm,
                    -1,
                    dtype=torch.int64,
                    device=self.device,
                )  # descending
            return torch.cat([drained_t, extended_t])

    def _take_physical_eager(self, num_pages: int) -> Optional[torch.Tensor]:
        """Original eager-mode take_physical — contiguous range."""
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
        """Advance the watermark by `num_pages`. Returns False if the
        extension would overflow either:
          * the index-space cap (`num_pages` / `min_page_index`), or
          * the PEER's byte-frontier (the extension would walk past the
            peer's allocated band, corrupting peer state).

        User point 3 (defense-in-depth): the byte-frontier check used to
        live exclusively in `available_size` at the caller; pushing it
        into `_extend_watermark` makes the primitive safe even under a
        stale `available_size` cache (e.g., between an opportunistic
        flush and the caller's next read).

        Lazy-path helper.
        """
        if self.grow_direction == "up":
            new_wm = self.watermark_physical + num_pages
            # Index-space cap.
            if new_wm > self.num_pages:
                return False
            # Peer byte-frontier cap. For grow-up, the peer (grow-down,
            # if present) sits ABOVE us; extending past peer's low
            # frontier in PAGE units corrupts peer state.
            if self._peer is not None:
                peer_low_pages = (
                    self._peer._byte_low_frontier() // self.entry_bytes_per_page
                )
                if new_wm > peer_low_pages:
                    return False
            self.watermark_physical = new_wm
        else:
            new_wm = self.watermark_physical - num_pages
            # Index-space cap.
            if new_wm < self.min_page_index - 1:
                return False
            # Peer byte-frontier cap. For grow-down, the peer (grow-up)
            # sits BELOW us.
            if self._peer is not None:
                # Page index just above the peer's last live page.
                peer_high_pages = (
                    self._peer._byte_high_frontier() // self.entry_bytes_per_page
                )
                # Our watermark sits just BELOW our lowest live page, so
                # `new_wm + 1` is our new lowest live page; it must be
                # strictly above the peer's high frontier.
                if new_wm + 1 < peer_high_pages:
                    return False
            self.watermark_physical = new_wm
        return True

    def _take_physical_arange(self, num_pages: int) -> torch.Tensor:
        """Build a contiguous arange tensor for a watermark extension that
        has already happened (i.e., `watermark_physical` already reflects
        the new value). Returns the just-allocated range.
        """
        if self.grow_direction == "up":
            return torch.arange(
                self.watermark_physical - num_pages,
                self.watermark_physical,
                dtype=torch.int64,
                device=self.device,
            )
        # Grow-down: ascending range [new_wm + 1, old_wm].
        return torch.arange(
            self.watermark_physical + 1,
            self.watermark_physical + num_pages + 1,
            dtype=torch.int64,
            device=self.device,
        )

    def take_physical_pages(self, num_pages: int) -> Optional[torch.Tensor]:
        """Page-granular wrapper around ``take_physical``. Used by the SWA
        composite's ``alloc_extend`` to bind the non-owner side."""
        with record_function("MultiEndedAlloc.take_physical_pages"):
            return self.take_physical(num_pages * self.page_size)

    def bind(
        self, virtual_ids: torch.Tensor, physical_ids: torch.Tensor
    ) -> None:
        """Bind page-granular virtual ids to page-granular physical ids.

        For page_size == 1, virtual_ids and physical_ids are slot ids.
        For page_size > 1, both are PAGE ids — the v2p / p2v tables are
        page-granular.
        """
        with record_function("MultiEndedAlloc.bind"):
            self.virtual_to_physical[virtual_ids] = physical_ids
            self.physical_to_virtual[physical_ids] = virtual_ids

    def bind_pages(
        self, virtual_pages: torch.Tensor, physical_pages: torch.Tensor
    ) -> None:
        """Explicit page-granular binder. Alias of ``bind`` — kept distinct
        for readability at the SWA composite call site."""
        with record_function("MultiEndedAlloc.bind_pages"):
            self.bind(virtual_pages, physical_pages)

    # -- fused take_physical_pages + bind_pages --

    def _alloc_bind_fast_or_slow(
        self, v_pages: torch.Tensor, N: int
    ) -> Optional[torch.Tensor]:
        """O3 fast path: fuse the `take_physical_pages` + `bind` pair into
        ONE Triton kernel when no holes need draining. Falls through to
        the existing slow path when holes exist.

        Replaces the 3-GPU-op pattern (`torch.arange` + 2x `index_put_`,
        ~30-60 µs dispatch overhead) with one fused Triton kernel
        launch (~15-25 µs) — saving ~20-40 µs per alloc. Eager mode
        (no holes ever accumulate) and lazy-mode allocs with empty
        holeset (>95% of standard-matrix allocs) take the fast path.

        Invariant B (greedy hole reuse) is preserved by the GATE on
        `not self._free_phys_pages`: when ANY hole exists, control
        flows to the slow path which calls `take_physical_pages` —
        that path's `_pop_hole_directional` loop drains holes FIRST,
        then extends the watermark for any remainder. Holes are
        NEVER bypassed in favor of watermark extension.

        Invariant D (`take_physical_pages` ↔ `bind` atomicity) is
        preserved because the helper performs both operations
        internally on EVERY call: the fast path's kernel fuses them
        into one launch; the slow path runs them as the existing
        adjacent pair. Callers see a single atomic primitive.

        Args:
            v_pages: virtual PAGE ids to bind, int64 [N].
            N: page count (matches `v_pages.numel()`).

        Returns:
            Physical page ids tensor [N] on success; None on shortfall
            (either the watermark extension would overflow the
            index-space cap or the peer byte-frontier in lazy mode).
            On None, allocator state is unchanged — callers can roll
            back their own bookkeeping safely.
        """
        with record_function("MultiEndedAlloc._alloc_bind_fast_or_slow"):
            if N == 0:
                return torch.empty(
                    0, dtype=torch.int64, device=self.device
                )

            # FAST PATH: eager mode (no holes ever accumulate) OR
            # lazy mode with no current holes. One fused kernel.
            if (
                not self.lazy_compaction
                or self._free_phys_pages.numel() == 0
            ):
                # Capture watermark BEFORE advancing so we can compute
                # the kernel's `start_phys` from the pre-extension value.
                start_wm = self.watermark_physical

                # Pre-check + advance watermark. Lazy mode uses
                # `_extend_watermark` (does both the index-space and
                # peer byte-frontier checks). Eager mode inlines the
                # index-space-only check to exactly match the existing
                # `_take_physical_eager` semantics.
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

                # Compute the lowest physical id in the new range. Both
                # directions yield ascending `[start_phys, start_phys + N)`
                # — byte-identical to `_take_physical_eager`'s
                # `torch.arange(start, end)` output.
                if self.grow_direction == "up":
                    start_phys = start_wm
                else:
                    start_phys = start_wm - N + 1

                # Launch the fused kernel: arange + v2p scatter +
                # p2v scatter + write phys_pages output.
                phys_pages = alloc_bind_inplace(
                    v_pages,
                    self.virtual_to_physical,
                    self.physical_to_virtual,
                    start_phys,
                )

                # `live_page_count` is tracked ONLY in lazy mode (the
                # eager-mode `_take_physical_eager` does not maintain
                # it — see `take_physical` for the gate). Match that
                # invariant here to keep the leak-checker view of
                # `allocated_count()` accurate in both modes.
                if self.lazy_compaction:
                    self.live_page_count += N
                return phys_pages

            # SLOW PATH: holes exist — preserve greedy reuse via the
            # existing path. `take_physical_pages` drains holes via
            # the directional pop loop FIRST, then extends only for
            # the remainder. We then bind via the unfused 2-scatter
            # pair (same as today).
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
        """Translate token-granular virtual ids to token-granular physical ids.

        The ``out=`` parameter writes results in-place into a
        caller-owned buffer. Required under cuda-graph capture for
        buffer-stability — the captured graph records the gather/multiply/add
        sequence against a fixed ``data_ptr``, replay re-runs against the
        same buffer.

        Args:
            virt_tokens: int64[N] virtual token ids (page-structured).
            out: optional int64[N] output tensor. When ``None`` (default),
                returns a fresh tensor. When provided, writes in-place and
                returns ``out``.

        Returns:
            int64[N] physical token ids. If ``out`` was given, returns ``out``.
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
        # Tombstone-safety clamp: tombstoned `v2p` entries (-1) must not reach
        # `k_buffer[-1]` when read by the captured graph. Under cuda-graph
        # capture, this method is called eagerly each replay-prep to populate
        # capture-stable buffers (`out_cache_loc_full_physical`,
        # `cuda_graph_kv_indices`); the captured kernels then index k/v
        # buffers with those values. A captured `k_buffer[-1]` is an illegal
        # memory access and crashes `graph.replay()`.
        #
        # Negative outputs can arise from:
        #   - padded-tail positions in the captured buffer whose stale virtual
        #     ids point at pages that have since been tombstoned by free /
        #     `_compact_pending`;
        #   - the zero-clear sentinel positions in `bs != raw_bs` replays
        #     (these are 0 -> v2p[0] = 0 -> 0; clamp is a no-op for those).
        # Clamping to 0 routes any tombstoned read/write to physical slot 0,
        # which is reserved padding-sink space by the `min_slot_index`
        # invariant: bytes `[0, entry_max)` across all sub-pools hold no real
        # data. Cost: one elementwise op per call; safe.
        if self.page_size == 1:
            if out is not None:
                # CRITICAL: `torch.index_select(src, dim, index, out=out)`
                # does NOT support aliasing between `index` and `out`. The
                # canonical caller from `triton_backend.py` is
                # `self._translate_kv_loc(kv_indices, out=kv_indices)`, where
                # `virt_tokens` and `out` are the SAME buffer (in-place
                # translate). Route through a transient gather + in-place
                # `copy_` to satisfy index_select's no-aliasing contract.
                # The transient `tmp` is fresh per call but caching-allocator-
                # cached under cuda-graph capture; the observable mutation is
                # `out.copy_(tmp)` into the stable buffer.
                tmp = torch.index_select(
                    self.virtual_to_physical, 0, virt_tokens
                )
                tmp = torch.clamp_min(tmp, 0)
                out.copy_(tmp)
                return out
            result = torch.index_select(
                self.virtual_to_physical, 0, virt_tokens
            )
            return torch.clamp_min(result, 0)
        # page_size > 1: page math.
        # Note: `virt_pages` and `offsets` are fresh tensors (results of
        # `// page_size` and `% page_size`), so they cannot alias `out`. The
        # `index_select(out=out)` below is therefore safe even when `out`
        # aliases `virt_tokens`.
        virt_pages = virt_tokens // self.page_size  # fresh int64[N]
        offsets = virt_tokens % self.page_size  # fresh int64[N]
        if out is not None:
            torch.index_select(
                self.virtual_to_physical, 0, virt_pages, out=out
            )
            out.mul_(self.page_size)
            out.add_(offsets)
            # Tombstoned page: -1 * ps + offset is in [-ps, -1]; clamp to 0.
            out.clamp_(min=0)
            return out
        phys_pages = self.virtual_to_physical[virt_pages]
        result = phys_pages * self.page_size + offsets
        return torch.clamp_min(result, 0)

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate `need_size` virtual TOKEN ids (id-owner only). Returns
        the virtual token ids (token-granular, page-structured), or None if
        there is not enough byte room / page headroom.

        Contract (matches upstream ``PagedTokenToKVPoolAllocator.alloc``
        at ``allocator.py:386–407``): ``need_size`` MUST be a multiple of
        ``page_size``. Token ids are emitted as
        ``(virtual_pages[:, None] * page_size + arange(page_size)).reshape(-1)``.

        Stream model (minimal):
          - All allocator GPU ops run on the scheduler thread's current stream
            (== `schedule_stream` inside the scheduler's `StreamContext`).
            This matches the upstream non-shared allocator behavior, so there
            are zero cross-stream interactions between cat/slice/v2p writes
            and `write_cache_indices` (which is also on schedule_stream).
          - For correctness in overlap mode (where the model forward runs on
            `forward_stream` and may still be in flight when `free` is
            called), we issue one `current_stream.wait_stream(forward_stream)`
            barrier at the very top of `free` — see `free` for the rationale.
            `alloc` doesn't need that barrier: its writes to v2p / p2v are
            picked up by the forward via the existing
            `forward_stream.wait_stream(schedule_stream)` at the top of
            `run_batch`.
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
                # Lazy compaction shortfall: flush the PEER's pool (not own).
                # Own compaction is an internal slot relocation — it does NOT
                # change own_available_size (each move trades 1 hole for +1
                # gap byte room, net 0). Only the PEER's compaction releases
                # bytes into the shared gap that own extension can consume.
                # One urgent peer-flush: a single flush retreats the peer watermark
                # in one pass (its crossing-checked full-pack compaction), so interior peer
                # holes open gap immediately. Standalone → helper returns False.
                if not self._flush_peer_for_alloc(need_size):
                    return None
            num_pages = need_size // self.page_size
            v_pages = self.free_virtual_ids[:num_pages]
            self.free_virtual_ids = self.free_virtual_ids[num_pages:]
            # O3: fused take_physical_pages + bind_pages on the fast path
            # (no holes); falls through to the slow path with greedy
            # hole drain when holes exist.
            phys_pages = self._alloc_bind_fast_or_slow(v_pages, num_pages)
            if phys_pages is None:
                # Undo the virtual pop.
                self.free_virtual_ids = torch.cat(
                    [v_pages, self.free_virtual_ids]
                )
                return None
            if self.page_size == 1:
                # Avoid the extra reshape — v_pages already IS the token id list.
                return v_pages
            # Expand page ids to token ids: (P, 1) * S + (S,) → (P, S) → (P*S,).
            return (
                v_pages[:, None] * self.page_size
                + torch.arange(self.page_size, device=self.device)
            ).reshape(-1)

    def alloc_with_virtual(self, virtual_pages: torch.Tensor) -> None:
        """Take physical PAGES for caller-supplied virtual PAGE ids
        (physical-holding non-owner). Used by the SWA `swa` sub-allocator.

        Note: the input is **virtual page ids** (not token ids),
        matching the composite's ``alloc_extend`` design where the kernel
        produces virtual token ids and the composite snapshots the
        corresponding virtual page ids before consuming them from the
        id-owner's free-list.
        """
        with record_function("MultiEndedAlloc.alloc_with_virtual"):
            if virtual_pages.numel() == 0:
                return
            # O3: fused take_physical_pages + bind_pages.
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

        Mirrors ``PagedTokenToKVPoolAllocator.alloc_extend``
        (``allocator.py:409–457``) but operates in **virtual space**: the
        kernel's ``free_page_ptr`` is this allocator's ``free_virtual_ids``
        (virtual pages, not physical), so the emitted ``out_indices`` are
        **virtual token ids** that respect ``(last_loc + 1) % page_size ==
        prefix_lens % page_size`` in virtual space.

        Each consumed virtual page is also bound to a physical page on THIS
        sub-allocator (via ``take_physical_pages`` + ``bind_pages``) so that
        downstream ``translate_kv_loc(virt_token)`` resolves to a valid
        physical token id. Without this binding, ``v2p[virt_page]`` would
        stay ``-1`` and translation would produce negative token ids that
        crash the Triton attention kernel with a CUDA OOB.

        Peers (e.g., the swa side of the SWA composite) call
        ``alloc_with_virtual(new_virtual_pages)`` to bind their own physical
        pages to the same virtual pages (the SWA composite handles this).
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
            # Lazy: physical-capacity pre-check. available_size() already
            # includes hole drainage; on shortfall, flush the PEER (own
            # compaction is internal — see `alloc` for the rationale).
            # Standalone (no peer) → nothing useful to flush. `_flush`
            # does not change `len(free_virtual_ids)`, so don't re-check
            # it after the retry.
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                # One urgent peer-flush — retreats the peer watermark in one pass
                # via full-pack compaction in `_flush` (see `_flush_peer_for_alloc`).
                if not self._flush_peer_for_alloc(need_tokens):
                    return None
            bs = len(prefix_lens)
            if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
                self.free_virtual_ids
            ):
                self.merge_and_sort_free()

            # Snapshot the virtual pages the kernel is about to consume, so we
            # can bind them to physical pages on THIS sub-allocator afterward.
            # Without this, v2p stays -1 and `translate_kv_loc` returns
            # negative token ids → CUDA OOB.
            if num_new_pages > 0:
                new_virtual_pages = self.free_virtual_ids[:num_new_pages].clone()
            else:
                new_virtual_pages = None

            out_indices = torch.empty(
                (extend_num_tokens,), dtype=torch.int64, device=self.device
            )
            # Pass `free_virtual_ids` (virtual pages) as `free_page_ptr` — the
            # kernel just does `page_id * page_size + offset` math and doesn't
            # care whether page ids are virtual or physical.
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

            # Bind the consumed virtual pages to fresh physical pages on this
            # sub-allocator. Advances the watermark + sets v2p / p2v. The peer
            # (swa side, if any) does its own binding via `alloc_with_virtual`.
            # O3: fused take_physical_pages + bind_pages.
            if new_virtual_pages is not None:
                phys_pages = self._alloc_bind_fast_or_slow(
                    new_virtual_pages, num_new_pages
                )
                if phys_pages is None:
                    # Defensive — the pre-check should have prevented this. Return
                    # None so the composite can decide whether to roll back.
                    return None

            # Consume the new virtual pages from the free-list.
            self.free_virtual_ids = self.free_virtual_ids[num_new_pages:]
            return out_indices  # virtual token ids

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Allocate one new token per request (decode step), preserving the
        tail-page-reuse contract.

        Mirrors ``PagedTokenToKVPoolAllocator.alloc_decode``
        (``allocator.py:459–496``) in virtual space. Each consumed virtual
        page is bound to a physical page on THIS sub-allocator afterward
        (same correctness requirement as ``alloc_extend``: without binding,
        v2p stays -1 and downstream translation produces negative token ids
        → CUDA OOB).
        """
        with record_function("MultiEndedAlloc.alloc_decode"):
            assert self.is_id_owner, (
                f"alloc_decode on a non-id-owner allocator ({self.sub_pool_name!r})"
            )
            bs = len(seq_lens)
            # Compute num_new_pages BEFORE the kernel so we can snapshot the
            # exact slice of `free_virtual_ids` the kernel will consume.
            # `get_num_new_pages` is CPU-only and matches the kernel's count.
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )
            if num_new_pages > len(self.free_virtual_ids):
                return None
            # Lazy: physical-capacity pre-check. On shortfall, flush PEER
            # (own compaction does not increase own_available_size).
            # Standalone (no peer) → no useful flush.
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                # One urgent peer-flush — retreats the peer watermark in one pass
                # via full-pack compaction in `_flush` (see `_flush_peer_for_alloc`).
                if not self._flush_peer_for_alloc(need_tokens):
                    return None
            if self.need_sort and bs > len(self.free_virtual_ids):
                self.merge_and_sort_free()

            # Snapshot the virtual pages the kernel will consume (if any).
            # Most decode steps reuse the prefix's tail page → num_new_pages == 0.
            if num_new_pages > 0:
                new_virtual_pages = self.free_virtual_ids[:num_new_pages].clone()
            else:
                new_virtual_pages = None

            out_indices = torch.empty(
                (bs,), dtype=torch.int64, device=self.device
            )
            with record_function("MultiEndedAlloc.alloc_decode.kernel"):
                alloc_decode_kernel[(bs,)](
                    seq_lens,
                    last_loc,
                    self.free_virtual_ids,
                    out_indices,
                    next_power_of_2(bs),
                    self.page_size,
                )

            # Bind the consumed virtual pages to fresh physical pages on this
            # sub-allocator. Advances the watermark + sets v2p / p2v.
            # O3: fused take_physical_pages + bind_pages.
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
        """Free virtual TOKEN ids; recover virtual PAGE ids via
        ``unique(// page_size)``; un-map v2p / p2v at page granularity;
        (if id-owner) recycle the virtual page ids; trigger eager compaction.

        Mirrors ``PagedTokenToKVPoolAllocator.free``
        (``allocator.py:498–512``) — ``free_index`` is token-granular and
        does NOT need to be page-aligned; the caller-side invariant is that
        a token in a page is freed iff the page is no longer referenced.

        Stream model — EAGER (`lazy_compaction=False`): one
        `current.wait_stream(forward_stream)` barrier at the top serializes
        the v2p/p2v writes and the compaction move with the in-flight
        forward.

        Stream model — LAZY (`lazy_compaction=True`: NO
        `wait_stream` barrier. A freed `v` has no live references (its req
        finished), so the v2p/p2v scatters are disjoint-element from any
        in-flight forward's reads — safe by per-element atomicity on
        Ampere+/Hopper. Compaction is deferred to `_flush(urgent=...)`.
        For each freed `p`, the boundary-shortcut absorbs it into the
        watermark if at the boundary (walking inward through contiguous
        holes); otherwise `p` enters the sorted `_free_phys_pages` for
        alloc to drain on the next request.
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
            # --- EAGER path (unchanged from Stages 1–3.6.2) ---
            # Overlap-mode barrier (single, at the start). In normal mode this is
            # a near-no-op because sampling's CPU sync has already drained
            # forward_stream. In overlap mode it serializes free+compaction with
            # the in-flight forward, which is what we need for correctness.
            if self.forward_stream is not None:
                with record_function("MultiEndedAlloc.free.wait_stream"):
                    torch.cuda.current_stream().wait_stream(self.forward_stream)
            # Recover virtual PAGE ids from token-granular `free_index`. For
            # page_size == 1, `// page_size` is identity, so this collapses to
            # plain slot math byte-identically.
            with record_function("MultiEndedAlloc.free.v2p_lookup"):
                free_v_pages = torch.unique(
                    free_index.detach().to(torch.int64) // self.page_size
                )
                freed_p_pages = self.virtual_to_physical[free_v_pages]
            with record_function("MultiEndedAlloc.free.sync_check"):
                # `.item()` here forces CPU/GPU sync — surface it as its own
                # region in the trace so we can see how much wall-clock time
                # it actually costs per free call.
                if bool((freed_p_pages < 0).any().item()):
                    self._raise_stale_slot_assertion(
                        free_v=free_v_pages, freed_p=freed_p_pages
                    )
            # Un-map; (if id-owner) recycle the virtual page ids.
            self.virtual_to_physical[free_v_pages] = -1
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat(
                    [self.free_virtual_ids, free_v_pages]
                )
            self._compact_pending(freed_p_pages)

    def _free_lazy(self, free_index: torch.Tensor) -> None:
        """Lazy free path.

        Immediate disjoint-element scatters on schedule_stream, NO
        `wait_stream(forward_stream)`. The freed physical pages are
        appended to `_free_phys_pages` via ONE `torch.cat` — NO sort, NO
        boundary absorb, NO watermark mutation, ZERO D2H syncs (the
        debug-gated tombstone check pays a sync only when the env var is
        on). Boundary absorption is deferred entirely to `_flush`. This
        matches the baseline allocator's `free_pages = torch.cat((...))`
        cost profile.

        Vectorization: for `page_size == 1`, skip the `torch.unique`
        call — token-granular `free_index` is already page-unique by
        construction. For `page_size > 1`, `torch.unique` is needed to
        dedup same-page tokens; its output-size sync is unavoidable.

        `live_page_count` is updated from CPU shape metadata
        (`freed_p_pages.shape[0]`) — no sync. Tombstones in `freed_p_pages`
        (which mean a virtual was already freed — a caller bug caught by
        the debug check) would be cat'd onto the free list as `-1`; the
        existing `SGLANG_DEBUG_CHECK_V2P_TOMBSTONES` debug gate catches
        this loud rather than silently corrupting the free list.
        """
        self._stats_n_free_lazy += 1
        with record_function("MultiEndedAlloc._free_lazy"):
            with record_function("MultiEndedAlloc._free_lazy.v2p_lookup"):
                free_v_pages_raw = free_index.detach().to(torch.int64)
                if self.page_size == 1:
                    # Token-granular = page-granular at ps=1, AND
                    # `free_index` is already unique per caller contract
                    # (req_to_token rows are non-overlapping). Skip the
                    # `torch.unique` to avoid its output-size sync.
                    free_v_pages = free_v_pages_raw
                else:
                    free_v_pages = torch.unique(
                        free_v_pages_raw // self.page_size
                    )
                freed_p_pages = self.virtual_to_physical[free_v_pages]
            # Tombstone safety check (debug-gated for perf — the .item() sync
            # would otherwise dominate the lazy path's CPU cost).
            if envs.SGLANG_DEBUG_CHECK_V2P_TOMBSTONES.get():
                if bool((freed_p_pages < 0).any().item()):
                    self._raise_stale_slot_assertion(
                        free_v=free_v_pages, freed_p=freed_p_pages
                    )
            # Disjoint-element scatters on schedule_stream — no event/barrier
            # (a freed v has no live reader; v2p/p2v writes are disjoint from
            # any in-flight forward's reads, and per-element scatter writes
            # are atomic).
            self.virtual_to_physical[free_v_pages] = -1
            self.physical_to_virtual[freed_p_pages] = -1
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat(
                    [self.free_virtual_ids, free_v_pages]
                )
            # ONE torch.cat. No sort, no absorb, no watermark mutation.
            # Pure GPU op.
            self._free_phys_pages = torch.cat(
                [self._free_phys_pages, freed_p_pages]
            )
            if _SORT_FREE_LIST_AFTER_MERGE:
                # Optional sort — keeps the list always-sorted so `_flush`
                # can skip its sort step, at the cost of one extra GPU
                # sort per `_free_lazy` call. Default OFF.
                self._free_phys_pages, _ = torch.sort(self._free_phys_pages)
            # CPU-only bookkeeping (shape metadata — no sync).
            self.live_page_count -= int(freed_p_pages.shape[0])

    def _release_phys_pages_batch(self, pages: torch.Tensor) -> None:
        """Thin wrapper for the compaction-src release path. Just cats
        `pages` (a GPU int64 tensor) onto `_free_phys_pages` and applies the
        optional sort. Called by `_flush` at flush END to merge the
        event-fired compaction-src pages accumulated in `released_fired`,
        AFTER the trailing dst-slice (this ordering keeps `_free_phys_pages`
        == `holes_cpu` during the survivor walk, so the slice targets the
        consumed dsts correctly in both grow directions).

        No watermark mutation, no `live_page_count` update — these pages
        are vacated compaction-src positions (or pages whose owning
        request just finished and `_free_lazy` already debited
        `live_page_count`); they re-enter the free list as PURE storage,
        not as freshly-freed live pages. Boundary absorption happens at
        `_flush` time.
        """
        if pages.numel() == 0:
            return
        self._stats_n_release_batch += 1
        with record_function("MultiEndedAlloc._release_phys_pages_batch"):
            self._free_phys_pages = torch.cat(
                [self._free_phys_pages, pages]
            )
            if _SORT_FREE_LIST_AFTER_MERGE:
                self._free_phys_pages, _ = torch.sort(self._free_phys_pages)

    def _compact_pending(self, freed_physical_pages: torch.Tensor) -> None:
        """Eager compaction over the freed PHYSICAL pages: move the survivor
        pages that fall in the *vacated band* (the K pages adjacent to the
        watermark, where K = #freed pages) into the *holes in the kept band*,
        then advance the watermark and remap the two tables. The `src` set
        (⊆ vacated band) and the `dst` set (⊆ kept band) are disjoint by
        construction, so the batched data copy is order-independent.

        `move_kv_cache` is called with TOKEN-granular indices (per-page
        tokens flattened) since `move_kv_cache_native` already operates at
        token granularity. For page_size == 1, src_pages/dst_pages ARE the
        token ids (no expansion).

        Separable so a future flag can defer it (lazy mode); in Stages 1/2/3
        `free` calls it inline (eager).

        All GPU ops here run on the scheduler thread's current stream
        (schedule_stream). The wait_stream barrier in the caller (`free`)
        already serialized us with any in-flight forward kernels."""
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
            src_pages = torch.tensor(
                src_list, dtype=torch.int64, device=self.device
            )
            dst_pages = torch.tensor(
                dst_list, dtype=torch.int64, device=self.device
            )
            v_moved = self.physical_to_virtual[src_pages].clone()  # read before clearing

            # Expand page ids to token ids for the move kernel (which is
            # token-granular, see memory_pool.py:2204 `move_kv_cache_native`).
            # For page_size == 1, src_pages/dst_pages == src_t/dst_t.
            if self.page_size == 1:
                src_t, dst_t = src_pages, dst_pages
            else:
                offsets = torch.arange(
                    self.page_size, dtype=torch.int64, device=self.device
                )
                src_t = (
                    src_pages[:, None] * self.page_size + offsets
                ).reshape(-1)
                dst_t = (
                    dst_pages[:, None] * self.page_size + offsets
                ).reshape(-1)

            # Data copy. MHA (full) -> SharedMHATokenToKVPool.move_kv_cache(dst, src);
            # Mamba -> SharedMambaPool._copy_from_physical(src, dst) (un-translated —
            # the public copy_from translates virtual ids, which we must NOT do here).
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
            # Clear the whole vacated band, then re-bind the relocated dst pages
            # (dst_pages ⊆ kept band, disjoint from the vacated band). All
            # remapping is at PAGE granularity (the tables are page-granular).
            self.physical_to_virtual[vacated_lo:vacated_hi] = -1
            self.virtual_to_physical[v_moved] = dst_pages
            self.physical_to_virtual[dst_pages] = v_moved
            self._inverse_history.append((src_pages, dst_pages, v_moved))
        else:
            self.physical_to_virtual[vacated_lo:vacated_hi] = -1

    # -- lazy compaction primitives --

    def set_latest_forward_done_event(
        self, event: Optional["torch.cuda.Event"]
    ) -> None:
        """Scheduler-facing hook: stash the `forward_done` event of the
        most-recently-launched forward batch (recorded on `forward_stream`
        right after `forward_batch_generation`). `_pending_reuse` uses
        this to gate src reuse on read-path settling. Pass `None` to
        clear (no in-flight forward).
        """
        with record_function("MultiEndedAlloc.set_latest_forward_done_event"):
            self._latest_forward_done_event = event

    def set_inflight_forward(
        self,
        forward_done: "torch.cuda.Event",
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        """Scheduler-facing hook: stash the just-launched forward's
        `forward_done` event AND its virtual `out_cache_loc` tensor so
        `_flush`'s classification can check
        whether a survivor candidate is about to be written.

        IMPORTANT — no GPU work here. We only store references. The
        write-set is materialized LAZILY inside `_flush` on
        `schedule_stream`, and only when a survivor actually needs to
        be checked. This avoids the bottleneck of a design that
        materialized via `phys.tolist()` from inside `forward_stream_ctx`
        — that sync waited for the just-launched forward.

        Pass `out_cache_loc_virtual=None` when this allocator's pool is
        not written by the forward (e.g., the Mamba state pool — the
        forward only READS mamba state; mamba writes happen at alloc
        time). A None tensor means "no in-flight write race possible
        on this pool."

        No-op for eager mode.
        """
        with record_function("MultiEndedAlloc.set_inflight_forward"):
            if not self.lazy_compaction:
                return
            if out_cache_loc_virtual is None or out_cache_loc_virtual.numel() == 0:
                # "No in-flight write race on this pool" — clear the slot so
                # any prior stale entry is dropped (lets the prior tensor
                # reference be GC'd) and `_flush` short-circuits without
                # materializing. Used for the Mamba state pool, where the
                # forward writes via mamba kernels not `set_kv_buffer`.
                self._inflight_forward = None
                return
            # Just store references. No GPU work, no D2H, no sync.
            self._inflight_forward = (forward_done, out_cache_loc_virtual)

    def _materialize_inflight_write_set(self) -> Optional[Set[int]]:
        """Lazily materialize the write-set of the currently-tracked
        in-flight forward, called from inside `_flush` on the scheduler
        thread (current stream = `schedule_stream`).

        Returns the set of physical PAGE ids the in-flight forward is
        about to write, or `None` if there's no in-flight forward OR
        the forward has already completed. (Pools with no in-flight
        write race — e.g., Mamba state pool — have `_inflight_forward
        is None` directly, via `set_inflight_forward(_, None)`.)

        Cost: ~50 µs in typical case (small `schedule_stream` queue;
        bs-sized int64 D2H). The fundamental `.tolist()` sync goes on
        `schedule_stream` which is light at flush-call sites. Materializes ONCE per `_flush` call, only when
        the call has at least one survivor candidate to classify; most
        calls early-exit at the empty-set fast-path and never reach
        here.
        """
        inflight = self._inflight_forward
        if inflight is None:
            return None
        event, oclv = inflight
        # If the forward has already completed, no write race is possible.
        # Clear the slot so subsequent flushes in the same scheduler
        # tick don't re-check the same fired event.
        if event.query():
            self._inflight_forward = None
            return None
        # `oclv` cannot be None here: `set_inflight_forward(_, None)`
        # explicitly clears `_inflight_forward` instead of storing a
        # tuple, so a non-None `_inflight_forward` always has a valid
        # virtual `out_cache_loc` tensor.
        with record_function("MultiEndedAlloc._materialize_inflight_write_set"):
            phys_tokens = self.translate_kv_loc(oclv)
            if self.page_size > 1:
                phys_pages = (phys_tokens // self.page_size).unique()
            else:
                phys_pages = phys_tokens
            # .tolist() syncs schedule_stream (current stream). At all
            # `_flush` call sites schedule_stream has only small allocator
            # state writes queued (v2p scatters, sub-µs); the big memcpy
            # (Phase A) or compaction copy (Phase B) is either drained or
            # not on this stream. Typical wait: tens of µs.
            return set(phys_pages.tolist())

    # All release paths route through `_release_phys_pages_batch`, a thin
    # `torch.cat` wrapper; boundary absorption is deferred entirely to
    # `_flush`.

    def _maybe_emit_stats(self) -> None:
        """Env-gated periodic stats emit. Called at the end of
        `_flush`. Logs a one-line cumulative summary of memory-pressure
        and lazy-compaction activity at most once every
        `SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC` seconds (default
        30s). Disabled unless `SGLANG_LOG_LAZY_COMPACTION_STATS=1`.

        Output (single line, INFO):
          [lazy-stats sub=full] free_lazy=N flush=M (work=K moves=P abs=Q)
          drain=R/S sort=ON peak_holes=H peak_pending=I live=L wm=W
        """
        if not _LAZY_COMPACTION_STATS_ENABLED:
            return
        now = _time_mod.monotonic()
        if now - self._stats_last_emit_ts < _LAZY_COMPACTION_STATS_INTERVAL_SEC:
            return
        self._stats_last_emit_ts = now
        self._stats_n_emits += 1
        # Read current snapshot of pressure indicators (cheap CPU metadata).
        cur_holes = int(self._free_phys_pages.shape[0])
        cur_pending = len(self._pending_reuse_pages_cpu)
        self._stats_peak_free_list_len = max(
            self._stats_peak_free_list_len, cur_holes
        )
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
        """Force-emit the final counters at process shutdown,
        bypassing the time-interval gate of `_maybe_emit_stats`. Tagged
        `FINAL` in the log line so it's easy to grep out the workload-end
        snapshot; `reason` distinguishes atexit vs SIGTERM vs SIGINT.
        Idempotent (guards against double-emit when both the signal
        handler and atexit fire). Best-effort: any error is swallowed
        (we're at shutdown; raising would be worse).
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
        """Move ready `_pending_reuse` entries back into
        `_free_phys_pages` via pure-GPU `torch.cat`.

        Each entry is `Event → (cpu_list, gpu_tensor)`:
          * non-urgent: release only entries whose event is `None` or has
            already fired (`event.query()` True). Sync-free.
          * urgent: `stream.wait_event` on any unfired event (stream-side
            dependency, not host block), then release.

        Used at `_flush` entry, by the per-step `maybe_drain_pending_reuse`
        scheduler hook, and inside the alloc-retry path. Pages released
        this way go back to `_free_phys_pages` via ONE batched cat (plus
        an optional sort if the env knob is on); NO watermark mutation,
        NO `live_page_count` change (these are vacated compaction-src
        positions, not freshly freed pages).

        Batching: there is ONE dict entry per BATCH (keyed by Event), not
        one per src. The CPU list is used
        for the parallel-Set `difference_update`; the GPU tensor is
        cat'd directly without a second H2D copy.
        """
        self._stats_n_drain_calls += 1
        if not self._pending_reuse:
            return
        with record_function("MultiEndedAlloc._drain_pending_reuse"):
            ready_tensors: List[torch.Tensor] = []
            ready_entries: List[
                Tuple["torch.cuda.Event", List[int]]
            ] = []
            for event, (cpu_list, gpu_tensor) in self._pending_reuse.items():
                if event is None or event.query():
                    # Sync-free — event.query() is non-blocking.
                    ready_tensors.append(gpu_tensor)
                    ready_entries.append((event, cpu_list))
                elif urgent:
                    # Stream-side dependency, not host block.
                    torch.cuda.current_stream().wait_event(event)
                    ready_tensors.append(gpu_tensor)
                    ready_entries.append((event, cpu_list))

            for event, cpu_list in ready_entries:
                del self._pending_reuse[event]
                self._pending_reuse_pages_cpu.difference_update(cpu_list)

            if ready_tensors:
                # ONE torch.cat on GPU; no sync.
                self._free_phys_pages = torch.cat(
                    [self._free_phys_pages] + ready_tensors
                )
                # Stats: count batches drained and total pages returned to
                # the free list.
                self._stats_n_drain_did_work += 1
                self._stats_n_drained_pages_total += sum(
                    t.numel() for t in ready_tensors
                )
                if _SORT_FREE_LIST_AFTER_MERGE:
                    # Keep the sort-invariant on the post-cat tensor so
                    # subsequent `take_physical` calls (and the next
                    # `_flush`) can skip sorting.
                    self._free_phys_pages, _ = torch.sort(
                        self._free_phys_pages
                    )

    def maybe_drain_pending_reuse(self) -> None:
        """Public scheduler hook. Called once per scheduler step
        (gated on non-empty `_pending_reuse`). Keeps the pending-reuse
        queue short: as soon as a prior compaction batch's event fires,
        its src pages flow back into `_free_phys_pages` for immediate
        alloc reuse — without waiting for the next `_flush`.

        Cost when queue empty: a dict-empty check (~ns). When draining:
        a few `event.query()` calls + one `torch.cat` (~10-20µs). No
        D2H sync either way.
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
        """Two-pointer survivor walk.

        Find the topmost live PAGE in the allocated band (direction-
        aware), excluding holes (positions in `holes_cpu`, the post-sort
        CPU snapshot of `_free_phys_pages` taken at `_flush` entry) and
        pages in `_pending_reuse_pages_cpu` (whose `p2v` has already
        been cleared by a prior compaction remap).

        For grow-up: largest `p < watermark_physical` satisfying the
        exclusions. For grow-down: smallest `p > watermark_physical`
        satisfying them.

        Two-pointer membership check (no set, no mutation). The survivor
        cursor `p` is monotonic (decreasing for grow-up, increasing for
        grow-down). `holes_cpu` is a sorted-ASCENDING Python list — a
        READ-ONLY snapshot. We thread a cursor `j` alongside `p`; both
        move monotonically through their domains, so membership is O(1)
        per check.

        No `exclude` set needed: once `p` walks below the last picked dst,
        either the outer
        `_free_phys_pages.numel() > 0` gate has already exited (no more
        holes to dst into) OR `p`'s position was originally a hole (the
        snapshot correctly reports it). Either way, popped-but-
        uncommitted dsts have `p2v == -1` and treating them as "skip"
        via the original sorted snapshot is exactly correct — picking
        one as a survivor would corrupt the in-flight copy.

        Returns `(p, j)` so the caller can thread `j` into the next
        call (alongside `start_hint = p ± 1`). Returns `(None, j)`
        when no candidate exists.

        Callers SHOULD pass `holes_cpu` and `j_in` from `_flush`.
        They're optional so test fixtures can call without snapshot
        (the snapshot is lazily materialized via `.tolist()` — pays a
        sync, only OK for tests).
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
                # Advance `j` past holes strictly above `p` (already
                # behind the cursor; we'll never revisit them).
                while j >= 0 and holes_cpu[j] > p:
                    j -= 1
                is_hole = (j >= 0 and holes_cpu[j] == p)
                if is_hole or p in self._pending_reuse_pages_cpu:
                    if is_hole:
                        j -= 1  # consume the hole entry
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
                is_hole = (j < len(holes_cpu) and holes_cpu[j] == p)
                if is_hole or p in self._pending_reuse_pages_cpu:
                    if is_hole:
                        j += 1
                    p += 1
                    continue
                return p, j
            return None, j

    def _absorb_boundary_holes(self, all_cpu: List[int]) -> Tuple[int, List[int]]:
        """Retreat the watermark past free slots ALREADY contiguous with it,
        slice them off `_free_phys_pages`, and return
        ``(new_watermark, interior_holes_cpu)``. `all_cpu` is the sorted-ascending
        free-list snapshot. Pure CPU + one directional GPU slice; no sync.

        The returned `interior_holes_cpu` are the holes the watermark could NOT
        reach directly — the survivor walk compacts those.
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
        """Stream-wait the in-flight forward's done event so the slots a
        compaction is about to free are safe to MOVE (its write settled) and
        to REUSE (its read settled). `_latest_forward_done_event` is recorded
        after the WHOLE forward (the scheduler stamps the same event into
        `_inflight_forward`), so one wait covers both hazards; drop the now-moot
        write-set. No host block; a no-op when the event has already fired.
        """
        ev = self._latest_forward_done_event
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
            self._inflight_forward = None

    def _flush(self, *, urgent: bool) -> int:
        """One batched compaction pass; returns the number of survivor moves.

        Pipeline (one D2H total, at step 3):
          1. `_drain_pending_reuse` — return read-settled prior srcs to the list.
          2. sort the free list (or skip via env knob).
          3. `.tolist()` snapshot → `all_cpu`  *(the one sync)*.
          4-5. `_absorb_boundary_holes` — retreat past free slots already
               contiguous with the watermark; `holes_cpu` = interior holes.
          6. (urgent + interior holes) `_settle_inflight_forward` — wait the
             forward once so freed srcs are move- AND reuse-safe → race-free walk.
          7. survivor walk — TWO-POINTER compaction: move the topmost live slot
             (`_topmost_survivor`) into the next hole, STOPPING when the two
             pointers cross (the band is packed); batches `(src,dst,v_moved)`
             into ONE `move_kv_cache` + ONE v2p/p2v scatter at
             `_commit_move_batch`.
          8-9. flush exit:
             - urgent → FULL-PACK reclaim: the crossing-checked walk packed all
               live below the frontier, so retreat the watermark past ALL
               `len(holes_cpu)` interior holes and empty the free list.
             - non-urgent → slice consumed dsts off the free list and merge the
               freed srcs back (watermark unchanged; a later flush absorbs the
               now-top holes).

        Two hazards govern each survivor (both keyed on the single
        `forward_done` event — see `_settle_inflight_forward`):
          * WRITE race — the forward is about to overwrite `KV[src]`; a
            compaction read of `KV[src]` would corrupt `KV[dst]`. Non-urgent
            STOPS the walk at such a src (lazy write-set check); urgent settles
            up front (step 6) so the walk is race-free.
          * READ race — the forward READS `KV[src]` (attention); src REUSE must
            wait the reader event. `_commit_move_batch` routes such srcs to
            `_pending_reuse`; urgent's step-6 settle makes them immediately
            reusable instead.

        `_topmost_survivor` skips holes, already-moved srcs, and `_pending_reuse`
        pages (all have `p2v=-1`), so a `v_moved < 0` in the loop body is a
        corrupt-state bug and raises.
        """
        if not self.lazy_compaction:
            return 0
        self._stats_n_flush_calls += 1
        with record_function("MultiEndedAlloc._flush"):
            # Step 1: drain pending_reuse — pure GPU cat, no sync.
            self._drain_pending_reuse(urgent=urgent)

            # Step 2: sort on GPU — SKIP if env knob is set (then
            # the list is already sorted because `_free_lazy` and
            # `_drain_pending_reuse` sort after every cat). Either way,
            # `_free_phys_pages` is sorted ASCENDING after this step.
            if (
                not _SORT_FREE_LIST_AFTER_MERGE
                and self._free_phys_pages.numel() > 1
            ):
                self._free_phys_pages, _ = torch.sort(self._free_phys_pages)

            # Step 3: ONE D2H sync — `.tolist()` of the sorted
            # tensor. This is the ONE sync per `_flush` call.
            all_cpu = self._free_phys_pages.tolist()

            # Step 4-5: retreat the watermark past free slots already
            # contiguous with it (boundary absorb) and slice them off the free
            # list. `holes_cpu` = the remaining INTERIOR holes the survivor
            # walk below compacts. After this, `_free_phys_pages == holes_cpu`.
            new_wm, holes_cpu = self._absorb_boundary_holes(all_cpu)

            latest_event = self._latest_forward_done_event

            # --- Single-pass FULL-PACK compaction (urgent only) ---
            #
            # Step 4-5 above absorbs ONLY holes already boundary-contiguous.
            # Under urgent we want ONE flush to reclaim ALL interior holes so
            # the alloc-shortfall retry succeeds (else it OOMs despite
            # reclaimable holes. The crossing-checked
            # survivor walk below packs every live page below the frontier, so
            # all interior holes end up above the watermark; the flush-exit
            # block then retreats past the whole `len(holes_cpu)` in one shot.
            #
            # For that to be safe in a single pass, a freed src must be
            # immediately reuse-safe: the in-flight forward must have settled on
            # KV[src] — its READ (attn; else the next alloc's write races the
            # read) and, for pools the forward writes, its WRITE (else the
            # survivor MOVE reads mid-write). `_latest_forward_done_event` IS
            # the single `forward_done` event recorded after the WHOLE forward
            # (the scheduler stamps the SAME event into `_inflight_forward`; the
            # two fields differ only in metadata, not the event), so waiting it
            # ONCE settles both hazards. We wait it up front, drop the now-moot
            # write-set, and treat every src as event-FIRED — making the walk
            # race-free (empty `write_set`) with no `_pending_reuse` srcs, so
            # the full-pack reclaim is exact.
            single_pass_absorb = urgent and len(holes_cpu) > 0
            if single_pass_absorb:
                self._settle_inflight_forward()
                latest_event = None  # reads/writes settled → srcs are fired

            # Lazy write-set materialization.
            # `write_set is None` means "not yet
            # materialized — do it inline on first survivor that needs
            # the check." An empty `set()` means "no in-flight write
            # race possible." Otherwise it's the materialized set.
            #
            # Most flushes never reach the survivor-pick loop (the
            # empty-list fast path in `flush_opportunistic` short-
            # circuits), so the bs-sized D2H inside
            # `_materialize_inflight_write_set` is paid ONLY when needed.
            # When we settled the forward above, there is no race → empty set.
            write_set: Optional[Set[int]] = set() if single_pass_absorb else None

            # Accumulated move batch — committed in one shot below.
            srcs: List[int] = []
            dsts: List[int] = []
            v_moveds: List[int] = []

            # Flush-scoped accumulator for the event-FIRED compaction-src
            # pages. `_commit_move_batch` appends
            # each fired batch's `src_pages_t` here INSTEAD of catting it
            # straight onto `_free_phys_pages`. The merge is deferred to
            # AFTER the trailing dst-slice below, so `_free_phys_pages` stays
            # byte-identical to `holes_cpu` (the post-absorb snapshot) for
            # the entire survivor walk. That pristine-snapshot invariant is
            # what makes the directional dst-slice correct in BOTH grow
            # directions: with srcs appended mid-flush (the old behavior),
            # the grow-down `[:-n_dst_consumed]` slice would chop the
            # just-appended srcs instead of the consumed dsts, and a sort=ON
            # re-sort (inside `_release_phys_pages_batch`) would scramble the
            # grow-up slice too — both leaving ghost (`p2v=-1`, untracked)
            # pages that corrupt the free-list bookkeeping on a later flush,
            # plus double-bound dsts that silently corrupt KV. Event-PENDING
            # srcs still route to `_pending_reuse` (read-race gating MUST be
            # preserved — a src whose reader event has not fired cannot
            # re-enter the alloc-able free list yet).
            released_fired: List[torch.Tensor] = []

            # Survivor cursor — passed to `_topmost_survivor` as
            # `start_hint`. The two-pointer hole cursor `j_cursor` is
            # threaded alongside.
            cursor: Optional[int] = None
            j_cursor: Optional[int] = None

            # Dst cursor — reads dst values from `holes_cpu`
            # directly (NO per-dst `.item()` sync). For grow-up: dsts
            # come from the front (smallest first), `dst_cursor`
            # advances forward; for grow-down: dsts come from the back
            # (largest first), `dst_cursor` advances backward.
            #
            # At flush exit we slice the consumed prefix/suffix off
            # `_free_phys_pages` in ONE GPU op (no per-iter slicing).
            if self.grow_direction == "up":
                dst_cursor = 0
            else:
                dst_cursor = len(holes_cpu) - 1
            n_dst_consumed = 0

            # Per-call move cap on non-urgent flushes.
            move_cap = (
                self._lazy_max_moves_per_call
                if not urgent
                else None
            )

            n_moves = 0
            # Outer loop: stop when no more dsts available.
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
                    write_set = (
                        materialized if materialized is not None else set()
                    )
                if write_set and src in write_set:
                    if urgent:
                        # Commit any moves accumulated so far so the
                        # copy overlaps with what schedule_stream was
                        # doing.
                        self._commit_move_batch(
                            srcs, dsts, v_moveds, latest_event, released_fired
                        )
                        n_moves += len(srcs)
                        srcs.clear()
                        dsts.clear()
                        v_moveds.clear()
                        # Wait on the in-flight forward's event so
                        # subsequent compaction is clear of the write
                        # race. With the simplified design there's at
                        # most ONE in-flight forward.
                        inflight = self._inflight_forward
                        if inflight is not None:
                            torch.cuda.current_stream().wait_event(
                                inflight[0]
                            )
                            self._inflight_forward = None
                        write_set = set()  # forward drained → no race
                        latest_event = None
                        # DO NOT reset `cursor` / `j_cursor`. Resetting
                        # them would rewind the walk to wm-1, where the
                        # just-committed src positions live. Those
                        # positions now have p2v=-1 (cleared inside
                        # `_commit_move_batch`) but are NOT in
                        # `holes_cpu` (stale snapshot from flush entry)
                        # and not in `_pending_reuse_pages_cpu` when the
                        # commit took the immediate-release path
                        # (event_fired=True). Re-walking would return
                        # one of them as a survivor and trip the
                        # `p2v=-1` assertion. By preserving `cursor` —
                        # which already points strictly past the
                        # previously-picked survivors — the next iter
                        # resumes at the write-race blocker itself
                        # (which now passes the race check under the
                        # empty `write_set`).
                        continue  # retry this src under empty write-set
                    else:
                        # Non-urgent: top blocker → STOP the walk.
                        break

                # Case B/C: no write race. Pick dst from holes_cpu by
                # cursor (NO `.item()` sync).
                dst = holes_cpu[dst_cursor]
                # Two-pointer crossing check. `src` (topmost survivor) descends
                # for grow-up / ascends for grow-down; `dst` (next hole) moves
                # the opposite way. Once they CROSS — `src` is already on the
                # packed side of the smallest remaining hole — every remaining
                # live page is below (grow-up) / above (grow-down) every
                # remaining hole: the band is packed. Moving now would relocate
                # a live page the WRONG direction (into a hole beyond it),
                # shuffling a hole back toward the frontier and BLOCKING the
                # watermark retreat. Stop here so the walk fully compacts in one
                # pass (without this stop, urgent flushes would reclaim only a
                # small contiguous run instead of all holes).
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
                    # `_topmost_survivor` is supposed to exclude every
                    # page whose `p2v` is `-1` (holes + pending reuse).
                    # Reaching this branch means the allocator is in a
                    # corrupt state — fail loudly.
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

                # Per-call move cap on non-urgent flushes.
                if move_cap is not None and len(srcs) >= move_cap:
                    break

            # Commit any final accumulated batch.
            self._commit_move_batch(
                srcs, dsts, v_moveds, latest_event, released_fired
            )
            n_moves += len(srcs)

            if single_pass_absorb:
                # FULL-PACK reclaim (urgent). The crossing-checked walk packed
                # every live page below the frontier, so ALL `len(holes_cpu)`
                # interior holes — whether filled (their freed src is now a top
                # hole) or left unfilled at the top — sit above the watermark.
                # Retreat past the whole lot in one shot and EMPTY the free
                # list: those pages are beyond-frontier free space (p2v=-1),
                # reclaimed by the next watermark extension, so they neither go
                # on the free list nor need free-registration. The
                # `released_fired` tensors are simply dropped for the same
                # reason. One urgent flush == complete compaction, one D2H.
                n_reclaimed = len(holes_cpu)
                if self.grow_direction == "up":
                    self.watermark_physical = new_wm - n_reclaimed
                else:
                    self.watermark_physical = new_wm + n_reclaimed
                self._stats_n_pages_absorbed += n_reclaimed
                self._free_phys_pages = self._free_phys_pages[:0]
            else:
                # Lazy partial pass (non-urgent): the watermark stays at the
                # boundary-absorb value; a later flush's step 4 absorbs the
                # now-top holes. Slice consumed dsts off `_free_phys_pages`
                # (solution-(4) invariant: it is still byte-identical to
                # `holes_cpu`, so the consumed dsts occupy exactly the front
                # `n_dst_consumed` (grow-up) / back `n_dst_consumed` (grow-down)
                # entries), then merge the freed srcs back in ONE cat.
                if n_dst_consumed > 0:
                    if self.grow_direction == "up":
                        self._free_phys_pages = (
                            self._free_phys_pages[n_dst_consumed:]
                        )
                    else:
                        self._free_phys_pages = (
                            self._free_phys_pages[:-n_dst_consumed]
                        )
                if released_fired:
                    self._release_phys_pages_batch(
                        released_fired[0]
                        if len(released_fired) == 1
                        else torch.cat(released_fired)
                    )
            # Stats: count flushes that did real survivor work and the total
            # moves, then maybe emit the periodic summary.
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
        latest_event: Optional["torch.cuda.Event"],
        released_fired: List[torch.Tensor],
    ) -> None:
        """Issue ONE `move_kv_cache` + ONE bulk v2p/p2v remap for the
        accumulated `(src, dst, v_moved)` triples. Disposes each `src`
        either by APPENDING to the flush-scoped `released_fired` list (event
        fired) or via `_pending_reuse` (event pending). User point 4 —
        batched issuance keeps GPU kernel-launch cost amortized.

        solution-(4): fired srcs are NOT catted onto `_free_phys_pages` here.
        They accumulate in `released_fired` and `_flush` merges them AFTER
        its trailing dst-slice, so the free list stays equal to the
        `holes_cpu` snapshot throughout the survivor walk (see the slice
        comment in `_flush`). Event-PENDING srcs still route to
        `_pending_reuse` — read-race gating is unchanged: a src whose reader
        event has not fired must not become alloc-able yet.
        """
        if not srcs:
            return
        with record_function("MultiEndedAlloc._commit_move_batch"):
            src_pages_t = torch.tensor(
                srcs, dtype=torch.int64, device=self.device
            )
            dst_pages_t = torch.tensor(
                dsts, dtype=torch.int64, device=self.device
            )
            v_moveds_t = torch.tensor(
                v_moveds, dtype=torch.int64, device=self.device
            )
            # Expand to token granularity if page_size > 1 (the move
            # kernel is token-granular — see memory_pool.py).
            if self.page_size == 1:
                src_t, dst_t = src_pages_t, dst_pages_t
            else:
                offsets = torch.arange(
                    self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                )
                src_t = (
                    src_pages_t[:, None] * self.page_size + offsets
                ).reshape(-1)
                dst_t = (
                    dst_pages_t[:, None] * self.page_size + offsets
                ).reshape(-1)
            # ONE KV copy on schedule_stream covering every (src→dst).
            move_fn = getattr(self._kvcache, "move_kv_cache", None)
            if move_fn is not None:
                move_fn(dst_t, src_t)
            else:
                copy_phys = getattr(
                    self._kvcache, "_copy_from_physical", None
                )
                assert copy_phys is not None, (
                    f"sub-pool {self.sub_pool_name!r} supports neither "
                    "move_kv_cache nor _copy_from_physical"
                )
                copy_phys(src_t, dst_t)
            # ONE bulk remap (commit) — single-writer on schedule_stream.
            self.virtual_to_physical[v_moveds_t] = dst_pages_t
            self.physical_to_virtual[dst_pages_t] = v_moveds_t
            self.physical_to_virtual[src_pages_t] = -1
            self._inverse_history.append(
                (src_pages_t, dst_pages_t, v_moveds_t)
            )
            # Src disposition — ONE entry per batch (NOT one per src).
            # When the event has fired, accumulate the src pages in the
            # flush-scoped `released_fired` list (merged into
            # `_free_phys_pages` by `_flush` AFTER its dst-slice). When the
            # event is still pending,
            # store ONE `_pending_reuse` entry keyed by the event; the
            # value is the (cpu_list, gpu_tensor) tuple, and the GPU tensor
            # is the SAME `src_pages_t` we already built above (no second
            # H2D copy at drain time).
            event_fired = (
                latest_event is None or latest_event.query()
            )
            if event_fired:
                released_fired.append(src_pages_t)
            else:
                # Use a copy of the srcs list (the caller mutates it).
                srcs_copy: List[int] = list(srcs)
                self._pending_reuse[latest_event] = (srcs_copy, src_pages_t)
                self._pending_reuse_pages_cpu.update(srcs_copy)

    def flush_opportunistic(self) -> int:
        """Public, non-urgent flush — called by the scheduler at quiescent
        points (`on_idle` / `disable_overlap_for_batch` boundary). Drains
        `_pending_reuse` for fired events and compacts where holes exist;
        never blocks `schedule_stream`. No-op if `lazy_compaction=False`.

        Empty-set fast-path: the scheduler triggers this
        ~1200×/forward (profile counted 243K calls in 200 iters),
        and 99% hit the empty state where there is no work to do. Without
        this early exit, every call pays the cost of entering
        `_flush`'s `record_function` context, calling `_drain_pending_reuse`,
        and walking `_inflight_batches` via `_active_write_set`.

        Gate refinement: the original gate also
        required `_inflight_batches` to be empty, but under overlap mode
        that list is almost always non-empty (1–3 pipelined forwards), so
        the gate never fired and the residual `sched=overlap × cg=on`
        ~5.8% regression persisted. The refinement: skip whenever there
        is no compaction work that could possibly be done — i.e., no
        holes to compact AND no pending entries to drain. `_inflight_batches`
        is only consulted INSIDE `_flush` (via `_active_write_set`) to
        classify survivors against the in-flight write-set during a
        compaction; if we have no compaction to do, the in-flight list is
        irrelevant. Proofs 1–5 still hold because the gate is
        a strict subset of `_flush`'s internal predicates: skipping when
        there is no work to do can never introduce a correctness bug.
        """
        with record_function("MultiEndedAlloc.flush_opportunistic"):
            if not self.lazy_compaction:
                return 0
            if (
                self._free_phys_pages.numel() == 0
                and not self._pending_reuse
            ):
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


class SharedMambaTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Composite allocator for the MHA (full-attn) + Mamba hybrid pair over a
    `SharedKVPool`.

    The token-slot surface (the slot allocator the scheduler uses for the
    `out_cache_loc` path) delegates to the full-attn side — `alloc(N)` allocates
    MHA token slots (virtual per-token ids). The Mamba sub-pool's per-request
    `alloc(1)` is driven by `SharedHybridReqToTokenPool.alloc` ->
    `mamba_allocator.alloc(1)` (the `SharedMambaSlotAllocator`, which owns slots;
    the `mamba_pool` is storage-only). Both sub-allocators are id-owners of their own
    granularity's virtual-id space (the spaces are independent).
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedKVPool,
        kvcache,  # HybridLinearKVPool
        device: str,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):
        full_max = shared_buffer.max_slots("full")
        super().__init__(
            size=full_max - 1,
            page_size=page_size,
            dtype=shared_buffer.mha_spec("full").store_dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.shared_buffer = shared_buffer
        self._kvcache = kvcache
        self.page_size = page_size
        self.lazy_compaction = lazy_compaction

        # FULL sub-allocator is page-aware. MAMBA sub-allocator
        # stays page_size=1 because the Mamba state is per-request (one slot
        # per req), orthogonal to the per-token paging on the full side.
        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            shared_buffer=shared_buffer,
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
            shared_buffer=shared_buffer,
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

        # Wire the full-attn pool to translate slot ids via its allocator (the
        # MEA exposes the v2p table the KV pool reads directly). The mamba pool is
        # NOT wired here: it borrows `translate` from a `SharedMambaSlotAllocator`
        # (the physical view) that does not exist yet — it wraps this very
        # `self.mamba_allocator`. `init_shared_mamba_pools` builds that wrapper and
        # calls `mamba_pool.attach_allocator(slot_allocator)` once this composite
        # returns. (Attaching the raw MEA here would be dead wiring — overwritten
        # immediately, and the MEA has no `.translate` the pool could use.)
        kvcache.full_kv_pool.attach_allocator(self.full_attn_allocator)

        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []
        # The base init left these as None; we use watermark math, not free-lists.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[shared-pool] SharedMambaTokenToKVPoolAllocator ready: "
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

    # -- size: dynamic (so the leak checker's num_used = size - available - evictable
    #    reduces to allocated - evictable) --
    @property
    def size(self) -> int:
        # Both terms are in TOKENS:
        #   - `schedulable_available_size()` already converts pages → tokens.
        #   - `allocated_count()` returns tokens; a page/token unit mismatch
        #     here would trip the leak checker.
        # MUST use the SAME available view as `available_size()` below so the
        # leak invariant self-cancels (`total = size`, `available = available_size()`
        # → the available term cancels and the check reduces to
        # `evictable + protected + ... == allocated`, independent of the
        # peer-hole credit). See pool_stats_observer._get_mamba_token_info.
        return (
            self.full_attn_allocator.schedulable_available_size()
            + self.full_attn_allocator.allocated_count()
        )

    @size.setter
    def size(self, value) -> None:
        # Base.__init__ sets self.size = size; ignore — we compute it dynamically.
        pass

    # -- token-slot surface: MHA side --

    # Scheduler-facing capacity: the realizable-with-compaction view so the
    # retract gate / evict / schedule_policy do NOT over-retract when the mamba
    # peer holds drainable holes (139 MB/slot) that an urgent flush would
    # convert into shared-gap room. This is the fix for the Falcon "wedge"
    # (available_size collapsing to ~0 under lazy compaction's high-water
    # mamba watermark). The per-side alloc gates still use the un-credited
    # `MultiEndedAllocator.available_size()` so they flush before extending.
    def available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def full_available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def mamba_slot_full_token_cost(self) -> int:
        """Full-token-equivalents of shared-gap bytes ONE mamba state consumes.

        full and mamba share one byte buffer (the gap), so allocating a mamba
        state — even a single one (~139 MB) — removes that many full-KV tokens
        from the gap. The prefill planner reserves this per new mamba slot so
        admission stays inside the JOINT budget (`available_size()` reports the
        gap in full tokens but does NOT subtract the mamba slots the admitted
        requests will also need). Defined ONLY on the shared composite — the
        non-shared allocator has separate pools, so a mamba slot costs zero
        full-KV bytes and the planner sources this via `getattr(..., None)`.

        = mamba bytes/slot (mamba page_size==1 → `entry_bytes_per_page`)
        ÷ full bytes/token (`entry_bytes`).
        """
        return (
            self.mamba_allocator.entry_bytes_per_page
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
        with record_function("SharedMambaAlloc.alloc"):
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
        """Paged extend allocation. Mamba state is per-request and
        does NOT advance on per-token alloc, so the composite only forwards
        to the full sub-allocator."""
        with record_function("SharedMambaAlloc.alloc_extend"):
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
        """Paged decode allocation. Same dispatch logic as
        ``alloc_extend`` — the mamba side stays untouched per-decode."""
        with record_function("SharedMambaAlloc.alloc_decode"):
            return self.full_attn_allocator.alloc_decode(
                seq_lens, seq_lens_cpu, last_loc
            )

    def translate_kv_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-pool virtual TOKEN ids -> physical TOKEN ids (for the
        write/read paths). Delegates to the full-side sub-allocator's
        ``translate_kv_loc`` (page-math is in the base class).

        Supports ``out=`` for cuda-graph buffer stability — passes through
        to the base-class implementation.

        `-1` inputs map to `-1` via the trailing sentinel (page=1) or via
        the page math `(-1 // ps == -1)`, `v2p[-1] == -1`, `(-1)*ps + offset
        ≤ 0` — Triton's `select_index` semantics still treat this as padding.
        """
        result = self.full_attn_allocator.translate_kv_loc(loc, out=out)
        return result

    def translate_kv_loc_bounded(
        self,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        bs: int,
        *,
        src: Optional[torch.Tensor] = None,
    ) -> None:
        """GPU-bounded virtual->physical translate of
        ``kv_indices[0:kv_indptr[bs]]``, for CAPTURE into the decode cuda-graph.

        Unlike ``translate_kv_loc`` (which translates a Python-sliced prefix
        via `.item()` and allocates a transient), this reads the valid extent
        on-device from ``kv_indptr[bs]`` — no `.item()` sync, no transient,
        capturable. Reads the full sub-pool's live ``virtual_to_physical`` at
        replay (late-read invariant).

        ``src`` (default None -> ``kv_indices``, legacy in-place). When the
        out-of-place fix is on, ``src`` is the dedicated VIRTUAL buffer and
        ``kv_indices`` is the PHYSICAL graph buffer the attention reads
        (idempotent under replay: the src is never overwritten here).
        """
        fa = self.full_attn_allocator
        translate_kv_indices_inplace(
            kv_indices, fa.virtual_to_physical, kv_indptr, bs, fa.page_size, src=src
        )

    def is_slot_allocated(self, slot: int) -> bool:
        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()

    def free(self, free_index: torch.Tensor) -> None:
        with record_function("SharedMambaAlloc.free"):
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
        return [self.full_attn_allocator.backup_state(), self.mamba_allocator.backup_state()]

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

    def set_latest_forward_done_event(
        self, event: Optional["torch.cuda.Event"]
    ) -> None:
        """Scheduler-facing: forwards the per-batch `forward_done` event to
        BOTH sub-allocators so each side's `_flush` can gate src reuse on
        the in-flight reader settling. No-op when `lazy_compaction=False`."""
        with record_function("SharedMambaAlloc.set_latest_forward_done_event"):
            self.full_attn_allocator.set_latest_forward_done_event(event)
            self.mamba_allocator.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: "torch.cuda.Event",
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        """Hand the just-launched forward's metadata to BOTH sub-pools.

        Replaces the earlier `register_inflight_batch` design — see
        `MultiEndedAllocator.set_inflight_forward` for why
        we now store the tensor reference instead of materializing the
        write-set at launch time.

        The full-attention sub-pool's write-set is derived from
        `out_cache_loc` (virtual token ids for new-token KV writes).
        The Mamba state sub-pool is NOT written by the forward via
        `out_cache_loc` (mamba state lives in conv/temporal buffers
        and is updated by mamba kernels, not `set_kv_buffer`), so its
        in-flight tensor is `None` — `_flush` on the Mamba side will
        see "no in-flight write race" and skip the check.
        """
        with record_function("SharedMambaAlloc.set_inflight_forward"):
            self.full_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )
            self.mamba_allocator.set_inflight_forward(forward_done, None)

    def flush_opportunistic(self) -> int:
        """Non-urgent flush of BOTH sub-allocators. Called by the
        scheduler at quiescent points (`on_idle` /
        `disable_overlap_for_batch` boundary). Sync-free on
        `schedule_stream` — only polls events. No-op when
        `lazy_compaction=False`.

        Composite-level empty-set fast-path: when neither
        sub-allocator has work to do, skip both function calls entirely.
        The sub-allocator-level fast-path (per `flush_opportunistic`
        above) already exits cheaply on the empty state; this composite
        gate just avoids the two function-call frames + Python integer
        addition when both are empty — a small but per-scheduler-tick
        saving that compounds with the per-sub-pool gate.
        """
        with record_function("SharedMambaAlloc.flush_opportunistic"):
            # Gate refinement matches the per-sub-pool gate:
            # `_inflight_batches` is irrelevant when there is no compaction
            # or drain work to do. Under overlap
            # mode it is almost always non-empty, so including it in the
            # gate made the fast-path almost never fire.
            fa = self.full_attn_allocator
            ma = self.mamba_allocator
            if (
                fa._free_phys_pages.numel() == 0 and not fa._pending_reuse
                and ma._free_phys_pages.numel() == 0 and not ma._pending_reuse
            ):
                return 0
            return fa.flush_opportunistic() + ma.flush_opportunistic()


class SharedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Composite allocator for the hybrid SWA pair (full + swa MHA sub-pools)
    over a `SharedKVPool`.

    Inherits from `SWATokenToKVPoolAllocator` purely for the typing/contract
    relationship — `isinstance(allocator, SWATokenToKVPoolAllocator)` is
    asserted across SWARadixCache, schedule_batch, chunk_cache, and disagg.
    We do NOT call `SWATokenToKVPoolAllocator.__init__`: it would allocate two
    static-partition sub-pools (`TokenToKVPoolAllocator` over freshly created
    `MHATokenToKVPool` buffers), which is exactly what shared-pool replaces.
    Grand-parent `BaseTokenToKVPoolAllocator.__init__` is called directly.

    Three views on capacity:

    - `available_size()`            : joint byte-budget, the only safe pre-check
                                      for `alloc(N)` because N slots cost N*
                                      (entry_full + entry_swa) bytes out of the
                                      shared gap.
    - `_conserve_full_available_size()` /
      `_conserve_swa_available_size()` : slot-conservation (static cap −
                                         allocated_count), for the LEAK invariant
                                         only (via pool_stats_observer).
    - `schedulable_full_available_size()` /
      `schedulable_swa_available_size()` : byte-coordinated, realizable-with-
                                           compaction (peer drainable holes
                                           credited).
    - `full_available_size()` /
      `swa_available_size()`        : PHYSICAL per-side view for the scheduler /
                                      evict / radix = min(conserve, schedulable).
    """

    # The parent declares `size` as a `@property` without a setter, but
    # `BaseTokenToKVPoolAllocator.__init__` does `self.size = size`. Override
    # the property here with a no-op setter so the base init's assignment
    # doesn't raise; reading still returns `min(_size_full, _size_swa)` as
    # the parent intends.
    @property
    def size(self) -> int:
        return min(self._size_full, self._size_swa)

    @size.setter
    def size(self, value) -> None:
        # No-op: `size` is computed from `_size_full` / `_size_swa`. Base
        # class init writes here; we ignore.
        pass

    def __init__(
        self,
        *,
        shared_buffer: SharedKVPool,
        kvcache,  # SharedSWAKVPool
        device: str,
        full_max_total_num_tokens: int,
        swa_max_total_num_tokens: int,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):
        # Set _size_full / _size_swa BEFORE base init so anything that reads
        # `self.size` / `self.size_full` / `self.size_swa` during base init
        # sees a sane value. Stored as the STATIC partition caps — this is
        # the value the leak invariant expects (slot-conservation, not
        # dynamic / byte-coordinated). See v1 lines 1158–1166 for why.
        self._size_full = full_max_total_num_tokens
        self._size_swa = swa_max_total_num_tokens
        self._full_max_total_num_tokens = full_max_total_num_tokens
        self._swa_max_total_num_tokens = swa_max_total_num_tokens
        self.page_size = page_size

        # Skip SWATokenToKVPoolAllocator.__init__ — call grand-parent base init
        # directly. The base's `self.size = size` call is absorbed by our
        # no-op size setter above.
        BaseTokenToKVPoolAllocator.__init__(
            self,
            size=full_max_total_num_tokens,
            page_size=page_size,
            dtype=shared_buffer.mha_spec("full").store_dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.shared_buffer = shared_buffer
        self._kvcache = kvcache
        self.lazy_compaction = lazy_compaction

        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            shared_buffer=shared_buffer,
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
            shared_buffer=shared_buffer,
            sub_pool_name="swa",
            device=device,
            is_id_owner=False,  # ← non-owner; consumes virtuals minted by full.
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        self.full_attn_allocator.bind_peer(self.swa_attn_allocator)
        self.swa_attn_allocator.bind_peer(self.full_attn_allocator)

        # Wire the pools to translate slot ids via their allocators.
        kvcache.full_kv_pool.attach_allocator(self.full_attn_allocator)
        kvcache.swa_kv_pool.attach_allocator(self.swa_attn_allocator)
        kvcache.attach_allocators(
            full_allocator=self.full_attn_allocator,
            swa_allocator=self.swa_attn_allocator,
        )

        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []
        # Empty (not None) for the leak checker — same as Mamba composite.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[shared-pool] SharedSWATokenToKVPoolAllocator ready: "
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
        """Tokens available for `alloc(N)` / `alloc_extend(N)`.

        Joint byte-budget at PAGE granularity. Each composite alloc(1)
        consumes one full-side page AND one swa-side page (same virtual
        page id, bound on both). For each composite alloc we can pick:
          * full from a hole (free) or from extension (cost: ``e_f`` bytes);
          * swa  from a hole (free) or from extension (cost: ``e_s`` bytes).
        Returns TOKENS (matches `available_size() == len(free_pages) *
        page_size`).

        Lazy compaction (fixed point (8)): the v1 implementation
        used `joint_drain = min(H_f, H_s)` (only counted holes when BOTH
        sides could drain). That under-promised: if H_f > H_s, the extra
        full-side holes could still be used asymmetrically (full drains,
        swa extends, costing only ``e_s`` bytes per such pair). The
        corrected formula consumes BOTH sides' holes maximally before
        extending either pool toward the gap:

        Let ``H_f, H_s`` = holes_full, holes_swa; ``e_f, e_s`` =
        bytes/page; ``R_f, R_s`` = extension room above each watermark;
        ``G`` = byte gap.

        Phase 1 (both drain, free):
            K_1 = min(H_f, H_s)
        Phase 2 (one side extends; the side with fewer holes runs out):
            if H_f <= H_s:  e_phase2 = e_f, K_phase2_max = H_s
            else:           e_phase2 = e_s, K_phase2_max = H_f
            K_2 = min(K_phase2_max - K_1, G // e_phase2)
            G -= K_2 * e_phase2
        Phase 3 (both extend, cost e_f + e_s per pair):
            K_3 = G // (e_f + e_s)
        Total K = K_1 + K_2 + K_3, capped by index-space rooms
        (H_f + R_f, H_s + R_s).

        For page_size == 1: collapses to plain slot math byte-identical.
        """
        fa, sa = self.full_attn_allocator, self.swa_attn_allocator
        e_f = fa.entry_bytes_per_page
        e_s = sa.entry_bytes_per_page
        # Direction-agnostic shared gap: the free byte band between the two
        # pools, regardless of which side grows up vs down. The grow-up side
        # sits at the low bytes (its high frontier is the lower edge of the
        # gap); the grow-down side sits at the high bytes (its low frontier is
        # the upper edge of the gap). Both perspectives yield the same band.
        if fa.grow_direction == "up":
            gap_bytes = max(0, sa._byte_low_frontier() - fa._byte_high_frontier())
        else:
            gap_bytes = max(0, fa._byte_low_frontier() - sa._byte_high_frontier())
        R_f = (
            fa.num_pages - fa.min_page_index - fa._allocated_pages()
        )
        R_s = (
            sa.num_pages - sa.min_page_index - sa._allocated_pages()
        )

        if not self.lazy_compaction:
            # Eager: no holes — collapse to the original joint formula.
            pages_by_bytes = gap_bytes // (e_f + e_s)
            return min(pages_by_bytes, R_f, R_s) * self.page_size

        H_f = len(fa._free_phys_pages)
        H_s = len(sa._free_phys_pages)

        # Phase 1: both sides drain.
        K1 = min(H_f, H_s)

        # Phase 2: the side with FEWER holes extends; the side with MORE
        # holes keeps draining.
        if H_f <= H_s:
            e_phase2 = e_f
            K_phase2_max = H_s
        else:
            e_phase2 = e_s
            K_phase2_max = H_f
        K2_room = K_phase2_max - K1
        K2 = min(K2_room, gap_bytes // e_phase2) if e_phase2 > 0 else K2_room
        gap_bytes -= K2 * e_phase2

        # Phase 3: both extend together.
        K3 = gap_bytes // (e_f + e_s)

        K_total = K1 + K2 + K3
        # Apply index-space caps (alloc consumes both an index on full and
        # an index on swa per page-pair).
        K_total = min(K_total, H_f + R_f, H_s + R_s)
        return K_total * self.page_size

    # Slot-conservation views — the ONLY views the leak invariant should see
    # (routed there via `pool_stats_observer._get_swa_token_info`). Under
    # shared SWA, the swa side can consume bytes that originally counted toward
    # the full side's static budget. Returning the byte-coordinated (dynamic,
    # peer-aware) value here would generate spurious leak detections.
    #
    # `allocated_count()` returns TOKENS (matching upstream convention), so
    # `cap_TOKENS - allocated_count()` is in TOKENS — the unit the leak
    # invariant expects. Returning pages here instead would trip the leak
    # checker.
    def _conserve_full_available_size(self) -> int:
        return (
            self._full_max_total_num_tokens
            - self.full_attn_allocator.allocated_count()
        )

    def _conserve_swa_available_size(self) -> int:
        return (
            self._swa_max_total_num_tokens
            - self.swa_attn_allocator.allocated_count()
        )

    # PHYSICAL per-side views — what every SCHEDULING / eviction consumer
    # (`evict_from_tree_cache`, `schedule_policy`, the radix caches, disagg)
    # reads. Upstream's `SWATokenToKVPoolAllocator.full_available_size()`
    # returns the sub-allocator's physical `available_size()`; we preserve that
    # CONTRACT (physical, not slot-conserve — the leak path uses the
    # `_conserve_*` methods above instead). The `min(...)` keeps the report
    # sound under dynamic borrowing: the static-conserve cap bounds the
    # lending side (which would otherwise report bytes the borrower physically
    # took), while the byte-coordinated `schedulable_*` view bounds the side
    # that has already grown into the shared gap. Whichever is tighter wins.
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
    # credited — see `MultiEndedAllocator.schedulable_available_size`). On the
    # non-shared `SWATokenToKVPoolAllocator` the sub-allocators have no peer,
    # so these collapse to the static physical view.
    def schedulable_full_available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def schedulable_swa_available_size(self) -> int:
        return self.swa_attn_allocator.schedulable_available_size()

    def _flush_both_for_alloc(self, need_tokens: int) -> bool:
        """SWA analogue of `MultiEndedAllocator._flush_peer_for_alloc`.

        Each composite alloc consumes a full-side AND a swa-side page, and
        either side's compaction opens shared gap for the other, so we flush
        BOTH. A single urgent flush per side retreats its watermark in one
        pass (full-pack compaction in `_flush`), so no loop is needed.
        """
        if not self.lazy_compaction:
            return need_tokens <= self.available_size()
        self.full_attn_allocator._flush(urgent=True)
        self.swa_attn_allocator._flush(urgent=True)
        return need_tokens <= self.available_size()

    # `size_full` / `size_swa` are inherited from `SWATokenToKVPoolAllocator`
    # — they read `_size_full` / `_size_swa`, which we set to the static
    # `full_max_total_num_tokens` / `swa_max_total_num_tokens` in __init__.
    # We deliberately do NOT report `max_slots - 1` here: under shared pool
    # `max_slots("full") ≈ full_max + swa_max`, which would over-promise to
    # any caller treating these as static budgets.

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
        Delegates to the full-side sub-allocator's ``translate_kv_loc``
        (page-math is in the base class).

        Supports ``out=`` for cuda-graph buffer stability.
        """
        result = self.full_attn_allocator.translate_kv_loc(loc, out=out)
        return result

    def translate_loc_from_full_to_swa(
        self,
        kv_indices: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SWA-layer read path: virtual TOKEN ids -> swa-physical TOKEN ids.

        For page_size == 1: direct v2p_swa lookup. For page_size > 1: page
        math, identical to ``translate_kv_loc`` but against the swa side's
        v2p table. Output is int32 to match the non-shared API contract.

        Supports ``out=`` for cuda-graph buffer stability. The
        ``out=`` buffer MUST be int32 and the same shape as ``kv_indices``.

        Note: the input semantics differ from the non-shared
        `SWATokenToKVPoolAllocator.translate_loc_from_full_to_swa`
        (which takes full-physical), but the output semantics (swa-physical
        int32) match — downstream consumers don't care.
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
        # Tombstone-safety clamp (mirrors the full-side clamp in
        # `MultiEndedAllocator.translate_kv_loc`): v2p_swa entries can be
        # tombstoned to -1 by `_compact_pending` / `free` / `free_swa`. The
        # captured SWA attention kernel reads `swa_k_buffer[result[i]]` at
        # replay; `swa_k_buffer[-1]` is illegal memory access. Negative
        # outputs are routed to physical slot 0 (the reserved padding sink
        # under the `min_slot_index` invariant — bytes `[0, entry_max)`
        # across all sub-pools hold no real data). For page_size > 1 a
        # tombstoned page
        # produces values in `[-ps, -1]` via `swa_phys * ps + offsets`; the
        # clamp covers that range too.
        if self.swa_attn_allocator.page_size == 1:
            if out is not None:
                # Two-step: gather into a transient int64 then cast into out.
                # The intermediate `tmp` is fresh per call but caching-
                # allocator-cached; the observable mutation is `out.copy_`.
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

    def translate_kv_loc_bounded(
        self,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        bs: int,
        *,
        src: Optional[torch.Tensor] = None,
    ) -> None:
        """GPU-bounded virtual->full-physical translate of the full-attention
        ``kv_indices[0:kv_indptr[bs]]``, for CAPTURE into the decode cuda-graph.
        See ``SharedMambaTokenToKVPoolAllocator.translate_kv_loc_bounded``.

        ``src`` (default None -> in place); when the out-of-place fix is on it is
        the dedicated VIRTUAL source and ``kv_indices`` is the PHYSICAL graph
        buffer the attention reads (idempotent under replay).
        """
        fa = self.full_attn_allocator
        translate_kv_indices_inplace(
            kv_indices, fa.virtual_to_physical, kv_indptr, bs, fa.page_size, src=src
        )

    def translate_loc_from_full_to_swa_bounded(
        self,
        window_kv_indices: torch.Tensor,
        window_kv_indptr: torch.Tensor,
        bs: int,
        *,
        src: Optional[torch.Tensor] = None,
    ) -> None:
        """In-place, GPU-bounded virtual->swa-physical translate of
        ``window_kv_indices[0:window_kv_indptr[bs]]``,
        for CAPTURE into the decode cuda-graph.

        Writes int64 directly into the int64 ``cuda_graph_window_kv_indices``
        buffer (the eager `translate_loc_from_full_to_swa` returned int32 only
        to assign into that int64 buffer; the swa attention kernel reads it as
        int64 either way). Reads the swa sub-pool's live ``virtual_to_physical``
        at replay (late-read invariant). Tombstone-clamped in the kernel.
        """
        sa = self.swa_attn_allocator
        translate_kv_indices_inplace(
            window_kv_indices,
            sa.virtual_to_physical,
            window_kv_indptr,
            bs,
            sa.page_size,
            src=src,
        )

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with record_function("SharedSWAAlloc.alloc"):
            # Joint pre-check. Both sides of the SWA
            # composite are mutual peers — each side's compaction releases
            # bytes into the shared gap that the OTHER side can extend
            # into, so under lazy we flush BOTH on shortfall.
            if need_size > self.available_size():
                # One joint urgent flush — each side retreats in one pass via its
                # full-pack compaction (see `_flush_both_for_alloc`).
                if not self._flush_both_for_alloc(need_size):
                    return None
            # Snapshot the virtual PAGES the full-side alloc is about to
            # consume, so we can bind them on the swa side too.
            num_pages = need_size // self.page_size
            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_pages].clone()

            v_tokens = fa.alloc(need_size)
            # With the tight joint pre-check (multi-phase formula for lazy;
            # original joint formula for eager), this can only fail under
            # internal-state inconsistency — assert rather than silently
            # rollback.
            assert v_tokens is not None, (
                "SharedSWA.alloc: full.alloc returned None after joint "
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
        """Paged extend allocation.

        Runs ``alloc_extend_kernel`` ONCE in virtual space (the kernel
        doesn't care whether its `free_page_ptr` is virtual or physical —
        it does `page_id * page_size + offset` math identically). Output
        is virtual TOKEN ids preserving the tail-page-reuse invariant in
        virtual space. The composite then snapshots the new virtual PAGES
        consumed by the kernel and binds them on the swa sub-allocator via
        `alloc_with_virtual`.

        Returns virtual TOKEN ids that respect:
        - the page-boundary tail-page-reuse contract
          `(last_loc + 1) % page_size == prefix_lens % page_size`
        - the cross-sub-pool identity (same virtual page id maps to
          full-physical-page on full side and swa-physical-page on swa side).
        """
        with record_function("SharedSWAAlloc.alloc_extend"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
            # Joint pre-check at page granularity.
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                # One joint urgent flush (see `_flush_both_for_alloc`).
                if not self._flush_both_for_alloc(need_tokens):
                    return None

            # Snapshot the virtual PAGES that the full-side kernel call is
            # about to consume — `free_virtual_ids[:num_new_pages]` is the
            # slice the kernel reads via `free_page_ptr`. Clone so the swa
            # side has its own view even after the slice is sliced off.
            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

            # Run the kernel ONCE in virtual space.
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
                "SharedSWA.alloc_extend: full.alloc_extend returned None "
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
        """Paged decode allocation. One new token per request;
        a page is consumed iff the decode wraps to a new page.

        Same one-kernel-in-virtual-space discipline as ``alloc_extend``.
        """
        with record_function("SharedSWAAlloc.alloc_decode"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )
            # Joint pre-check at page granularity. When num_new_pages == 0,
            # this is trivially true (kernel still runs to fill out_indices).
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                # One joint urgent flush (see `_flush_both_for_alloc`).
                if not self._flush_both_for_alloc(need_tokens):
                    return None

            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

            out_indices = fa.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
            assert out_indices is not None, (
                "SharedSWA.alloc_decode: full.alloc_decode returned None "
                "after joint pre-check passed — internal-state inconsistency"
            )

            if new_virtual_pages.numel() > 0:
                self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)

            return out_indices  # virtual TOKEN ids

    # `_rollback_full_alloc` was removed: with the tight multi-phase joint
    # pre-check in `available_size()`, `swa.alloc_with_virtual`
    # cannot fail after the pre-check passes (joint formula precisely bounds
    # what both sides can deliver). Assertions in the three alloc paths
    # surface any internal-state inconsistency rather than silently rolling
    # back to mask a bug.

    def is_slot_allocated(self, slot: int) -> bool:
        """Token-slot surface = the full side. SWARadixCache passes virtual
        ids (which the full sub-allocator owns) to validate before free."""
        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()

    # -- free --

    def free(self, free_index: torch.Tensor) -> None:
        with record_function("SharedSWAAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if not self.is_not_in_free_group:
                self.free_group.append(free_index)
                return
            # Free both peers. swa first (non-owner — only releases swa-physical;
            # doesn't touch the virtual id), then full (id-owner — recycles the
            # virtual id). Order is not load-bearing for correctness in v2 (no
            # cross-pool mapping coherence to maintain — there is no
            # `full_to_swa_index_mapping`; the per-sub-pool v2p IS the mapping).
            #
            # Filter the swa side to skip already-tombstoned virtuals (where
            # `swa.v2p_page[v_page] == -1` because `free_swa(...)` ran earlier).
            # Mirrors the v1 `swa_indices > 0` filter at
            # `old_design_and_impl/...:1387`. The full side does NOT need this
            # filter — under SWARadixCache the full side is the lifecycle owner,
            # so every value in `free_index` must still be bound on full.
            #
            # The filter operates at PAGE granularity (recovering v_pages via
            # `// page_size`) and emits TOKEN-granular `live_swa_tokens` so
            # `swa.free` can apply its own `unique(// page_size)` internally.
            v = free_index.detach().to(torch.int64)
            v_pages = v // self.page_size
            swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]
            # `> 0` (strict): -1 = tombstoned, 0 = padding-sink page — both
            # skipped. Mirrors the non-shared `swa_indices > 0`.
            live_token_mask = swa_v2p_pages > 0
            live_tokens = v[live_token_mask]
            if live_tokens.numel() > 0:
                self.swa_attn_allocator.free(live_tokens)
            self.full_attn_allocator.free(v)
            self.full_attn_allocator.clear_inverse_history()
            self.swa_attn_allocator.clear_inverse_history()

    def free_swa(self, free_index: torch.Tensor) -> None:
        """SWA tombstone path: swa-physical released, virtual id and
        full-physical stay live.

        Mirrors `SWATokenToKVPoolAllocator.free_swa`. Called by
        `SWARadixCache._evict_swa_only` when a tree node has aged past the
        sliding-window horizon — its swa state is no longer reachable but
        its full state still is, so the swa-side budget gets reclaimed
        without disturbing full bookkeeping. In v2 the SWA allocator's
        `virtual_to_physical_page[v_page] = -1` after this call IS the
        tombstone.

        Page-aware filter: recover `v_pages = v // page_size`, look up
        `swa.v2p_page[v_pages]`, keep token IDs whose page is still bound
        on the swa side. Token-granular output goes to ``swa.free`` which
        applies its own `unique(// page_size)` internally.
        """
        if free_index is None or free_index.numel() == 0:
            return
        # Filter to tokens whose virtual PAGE still has an swa-side binding —
        # under v2, `swa.v2p_page[v_page] == -1` means already-tombstoned;
        # calling `swa.free` on those would assert.
        v = free_index.detach().to(torch.int64)
        v_pages = v // self.page_size
        # `> 0` (strict): tombstoned entries have v2p_page[...] == -1; virtual
        # page 0 is the padding-sink page bound to physical page 0 — never
        # freeable. Mirrors the non-shared `free_swa`'s `swa_indices > 0`
        # filter (`swa_memory_pool.py:502`).
        swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]
        live = v[swa_v2p_pages > 0]
        if live.numel() == 0:
            return
        self.swa_attn_allocator.free(live)
        self.swa_attn_allocator.clear_inverse_history()

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        """No-op stub for HiCache load-back compatibility.

        On the non-shared `SWATokenToKVPoolAllocator`, this rewrites the
        `full_to_swa_index_mapping` after HiCache reallocates full + swa
        slots. In shared mode there is no mapping tensor — the swa
        sub-allocator's v2p table IS the mapping, and `alloc()` populates
        it automatically. HiCache for shared SWA is out of scope.
        """
        # HiCache for the shared SWA path is a follow-up; this stub
        # keeps the non-shared API surface compatible.
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

    def set_latest_forward_done_event(
        self, event: Optional["torch.cuda.Event"]
    ) -> None:
        """Forwards the per-batch `forward_done` event to BOTH sub-
        allocators. No-op when `lazy_compaction=False`."""
        with record_function("SharedSWAAlloc.set_latest_forward_done_event"):
            self.full_attn_allocator.set_latest_forward_done_event(event)
            self.swa_attn_allocator.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: "torch.cuda.Event",
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        """Hand the just-launched forward's metadata to BOTH sub-pools.

        Replaces the earlier `register_inflight_batch` design. Each sub-allocator stores the tensor reference;
        the actual write-set is materialized lazily inside `_flush`
        via the sub-allocator's OWN v2p (full ↦ full-physical,
        swa ↦ swa-physical) so the resulting write-sets are correct
        per side. The forward writes `set_kv_buffer` on BOTH sides
        for every new token, so both sub-pools have non-empty
        in-flight tensors per batch.
        """
        with record_function("SharedSWAAlloc.set_inflight_forward"):
            self.full_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )
            self.swa_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )

    def flush_opportunistic(self) -> int:
        """Non-urgent flush of BOTH sub-allocators. Sync-free. Called by
        the scheduler at quiescent points. No-op when
        `lazy_compaction=False`.

        Composite-level empty-set fast-path: see the Mamba
        composite above for the rationale. Closes the dominant per-tick
        overhead under `sched=overlap × cg=on` where
        cg_on's ~10× shorter GPU step amplified the CPU bookkeeping cost.
        """
        with record_function("SharedSWAAlloc.flush_opportunistic"):
            # Gate refinement: _inflight_batches is irrelevant
            # when there's no compaction or drain work to do.
            fa = self.full_attn_allocator
            sa = self.swa_attn_allocator
            if (
                fa._free_phys_pages.numel() == 0 and not fa._pending_reuse
                and sa._free_phys_pages.numel() == 0 and not sa._pending_reuse
            ):
                return 0
            return fa.flush_opportunistic() + sa.flush_opportunistic()
