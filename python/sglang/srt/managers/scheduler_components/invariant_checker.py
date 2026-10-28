from __future__ import annotations

import dataclasses
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
)

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    PoolStats,
    SchedulerPoolStatsObserver,
    kv_private_swa_tokens,
    kv_private_tokens,
)
from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_SANITY_CHECK_CACHE,
    SchedulerStageMetricsRecorder,
    scheduler_stage_method,
)
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.utils.common import (
    ceil_align,
    raise_error_or_warn,
)
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.request_lists import RequestList
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.bundles import KvCapacity, KvResources
    from sglang.srt.managers.scheduler_components.engine_state import (
        EngineFlags,
        LaunchCounters,
    )
    from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
    from sglang.srt.managers.scheduler_components.request_registry import (
        RequestRegistry,
    )


logger = logging.getLogger(__name__)

# Number of recent busy-check messages buffered for the level-1 dump-on-leak path.
BUSY_MEM_CHECK_LOG_RING_SIZE = 1000
# Read once: the check runs after every host commit.
CHECK_REQUEST_LISTS = envs.SGLANG_CHECK_REQUEST_LIST_INVARIANTS.get()


@dataclass(kw_only=True, slots=True)
class SchedulerInvariantChecker:
    kv: KvResources
    capacity: KvCapacity
    pool_stats_observer: SchedulerPoolStatsObserver
    # The step this iteration launched; None when it launched nothing.
    counters: LaunchCounters
    # The requests of the lists that hold KV own their KV rows.
    request_registry: RequestRegistry
    # Released requests' rows held for a pending step's commit.
    pending: PendingSteps
    scheduler_stage_metrics: SchedulerStageMetricsRecorder
    page_size: int = field(init=False, default_factory=lambda: get_schedule().page_size)
    count_req_pool_leak_warnings: int = 0
    count_memory_leak_warnings: int = 0
    recent_busy_msgs: Deque[str] = field(
        default_factory=lambda: deque(maxlen=BUSY_MEM_CHECK_LOG_RING_SIZE)
    )

    @staticmethod
    def _check_pool_invariant(
        pool_name: str,
        available: int,
        evictable: int,
        protected: int,
        session_held: int,
        total: int,
        uncached: int = 0,
    ) -> Tuple[bool, str]:
        """Check: available + evictable + protected + session_held + uncached == total."""
        total_accounted = available + evictable + protected + session_held + uncached
        leak = total_accounted != total
        msg = (
            f"[{pool_name}] {total=}, {available=}, {evictable=}, "
            f"{protected=}, {session_held=}, {uncached=}"
        )
        return leak, msg

    def _check_full_pool(self, ps: PoolStats, uncached: int = 0) -> Tuple[bool, str]:
        allocator = self.kv.token_to_kv_pool_allocator
        if self.capacity.is_hybrid_swa and not ps.full_capacity:
            return False, ""
        if self.capacity.is_hybrid_swa:
            protected = self.kv.tree_cache.full_protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            total = ps.full_capacity
        elif self.capacity.is_hybrid_ssm:
            # `total` is the allocator's `.size`: static for non-unified pools,
            # the byte-coordinated cap (matching `available_size`) for the unified pool.
            protected = self.kv.tree_cache.full_protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            total = self.kv.req_to_token_pool.schedulable_token_capacity(
                self.kv.token_to_kv_pool_allocator.size
            )
        else:
            protected = self.kv.tree_cache.protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            total = self.capacity.widened_max_total_num_tokens
        full_evictable_size = ps.full_evictable_size
        full_available = ps.full_available_size
        class_watermark_msg = ""
        kv_shard_size = page_interleave_shard_size(allocator)
        dcp_page_alloc = (
            kv_shard_size == 1
            and get_parallel().dcp_enabled
            and allocator.page_size > 1
        )
        if kv_shard_size > 1:
            # Conservation needs every free page, not the min-class capacity
            # floor used for admission. Sharding keeps the tree and allocator
            # page sizes equal, so cached counts must already be page-aligned.
            full_available = allocator.aggregate_free_size()
            class_watermark_msg = (
                f", class_free_pages={allocator.class_free_page_counts()}"
            )
        elif dcp_page_alloc:
            # Preserve the existing DCP-only accounting adjustment. DCP widens
            # allocator pages; its scheduler counters still use per-rank units.
            full_evictable_size = ceil_align(full_evictable_size, allocator.page_size)
        leak, msg = self._check_pool_invariant(
            "full",
            full_available,
            full_evictable_size,
            protected,
            session_held,
            total,
            uncached,
        )
        msg += class_watermark_msg
        if kv_shard_size > 1:
            # Partial active pages are already rounded in
            # _get_total_uncached_sizes. Rounding cached counts or accepting
            # arbitrary slack here would hide corruption and lost pages.
            unaligned = {
                name: value
                for name, value in (
                    ("available", full_available),
                    ("evictable", full_evictable_size),
                    ("protected", protected),
                    ("session_held", session_held),
                    ("uncached", uncached),
                    ("total", total),
                )
                if value % allocator.page_size != 0
            }
            if unaligned:
                return True, (
                    f"{msg}, unaligned_sharded_counts={unaligned}, "
                    f"page_size={allocator.page_size}"
                )
        if leak and dcp_page_alloc:
            # Legacy DCP counters use different page/capacity units. Keep that
            # compatibility path separate from exact KV-shard conservation.
            return False, f"{msg}, dcp_physical_page_slack_allowed=True"
        return leak, msg

    def _check_swa_pool(self, ps: PoolStats, uncached: int = 0) -> Tuple[bool, str]:
        allocator = self.kv.token_to_kv_pool_allocator
        if is_swa_req_ring(allocator):
            # Per-request SWA ring: there is no token pool to conserve; ring-slot
            # leaks are caught by the req_to_token check instead.
            return False, (
                "[swa] unified ring (leak-check skipped): "
                f"available={ps.swa_available_size}, "
                f"evictable={ps.swa_evictable_size}, "
                f"total={self.capacity.swa_tokens_per_layer}"
            )
        return self._check_pool_invariant(
            "swa",
            ps.swa_available_size,
            ps.swa_evictable_size,
            self.kv.tree_cache.swa_protected_size(),
            self.pool_stats_observer.session_held_swa_tokens(),
            ps.swa_capacity,
            uncached,
        )

    def _check_mamba_pool(self, ps: PoolStats) -> Tuple[bool, str]:
        ckpt_pool = getattr(self.kv.req_to_token_pool, "mamba_ckpt_pool", None)
        if ckpt_pool is not None:
            return self._check_mamba_pool_with_int8(ps, ckpt_pool)
        leak, msg = self._check_pool_invariant(
            "mamba",
            ps.mamba_available_size,
            ps.mamba_evictable_size,
            self.kv.tree_cache.mamba_protected_size(),
            self.pool_stats_observer.session_held_mamba_slots(),
            self.kv.req_to_token_pool.mamba_pool.size,
        )
        if leak:
            # Pools without a page free list return None; skip the census rather
            # than crash the watchdog thread that runs this dump.
            free_pages = self.kv.token_to_kv_pool_allocator.get_all_free_pages()
            if free_pages is None:
                return leak, msg
            free_full_pages = set(free_pages.tolist())
            cached_full_pages = set(self.kv.tree_cache.all_values_flatten().tolist())
            full_page_msg = ""
            if (
                self.kv.req_to_token_pool.schedulable_token_capacity(
                    self.kv.token_to_kv_pool_allocator.size
                )
                == self.kv.token_to_kv_pool_allocator.size
            ):
                expected_full_pages = set(
                    range(1, self.kv.token_to_kv_pool_allocator.size + 1)
                )
                leaked_full_pages = (
                    expected_full_pages - free_full_pages - cached_full_pages
                )
                full_page_msg = f", leaked_full_pages={leaked_full_pages or None}"
            mamba_allocator = self.kv.req_to_token_pool.mamba_allocator
            free_mamba_pages = set(mamba_allocator.free_slots.tolist())
            cached_mamba_pages = set(
                self.kv.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(1, mamba_allocator.size + 1))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            msg += full_page_msg
            msg += f", leaked_mamba_pages={leaked_mamba_pages or None}"
        return leak, msg

    def _check_mamba_pool_with_int8(self, ps: PoolStats, ckpt_pool) -> Tuple[bool, str]:
        """Two-pool invariant for int8 mamba checkpoints.

        The radix-cached states live in the int8 checkpoint pool, NOT the active
        bf16 pool. So the single-pool equation (active.available + radix_cached ==
        active.size) is wrong -- it double-counts the radix states against a pool
        that does not hold them. Instead check the two pools independently:

          * active bf16 pool: backs running requests only; the radix owns ZERO
            active slots. Checked at idle (in-flight == 0) -> available == total.
          * int8 checkpoint pool: backs the radix-cached states; its occupancy is
            exactly the radix evictable + protected counts.
        """
        active_leak, active_msg = self._check_pool_invariant(
            "mamba-active",
            ps.mamba_available_size,
            ps.mamba_evictable_size,  # 0 in int8 mode (radix owns no active slots)
            0,
            self.pool_stats_observer.session_held_mamba_slots(),
            self.kv.req_to_token_pool.mamba_pool.size,
        )
        int8_leak, int8_msg = self._check_pool_invariant(
            "mamba-int8",
            ckpt_pool.available_size(),
            self.kv.tree_cache.mamba_evictable_size(),
            self.kv.tree_cache.mamba_protected_size(),
            0,
            ckpt_pool.num_slots,
        )
        return active_leak or int8_leak, active_msg + "\n" + int8_msg

    def _requests_owning_rows(self) -> List[Req]:
        reqs = self.request_registry.reqs(lambda entry: entry.holds_kv)

        # A request running on a session's record does not own that row.
        session_kvs = {id(kv) for kv in self.kv.tree_cache.session_records().values()}
        owners, seen = [], set()
        for req in reqs:
            if id(req) in seen:
                continue
            seen.add(id(req))
            if req.kv.holds_kv and id(req.kv) not in session_kvs:
                owners.append(req)
        return owners

    def _get_total_uncached_sizes(self) -> Tuple[int, int]:
        """(full, swa) footprint of the KV rows requests own; 0 swa off hybrid-SWA."""
        full_uncached = 0
        swa_uncached = self.kv.tree_cache.swa_transient_size()
        for req in self._requests_owning_rows():
            assert req.kv.cache_protected_len % self.page_size == 0
            req.kv.check_lengths()
            full_uncached += kv_private_tokens(req.kv, self.page_size)
            if self.capacity.is_hybrid_swa:
                swa_uncached += kv_private_swa_tokens(req.kv, self.page_size)

            if req.beam_group is not None:
                full_uncached += req.beam_group.extra_uncached_tokens()

        for held in self.pending.held():
            kv = dataclasses.replace(held.kv, cache_protected_len=held.held_from)
            full_uncached += kv_private_tokens(kv, self.page_size)
            if self.capacity.is_hybrid_swa:
                swa_uncached += kv_private_swa_tokens(kv, self.page_size)

        return full_uncached, swa_uncached

    @scheduler_stage_method(SCHEDULER_STAGE_SANITY_CHECK_CACHE)
    def self_check_during_busy(self):
        if self.counters.cur_batch_for_debug is None:
            return

        ps = self.pool_stats_observer.get_pool_stats()
        full_uncached, swa_uncached = self._get_total_uncached_sizes()

        full_leak, full_msg = self._check_full_pool(ps, uncached=full_uncached)

        swa_leak, swa_msg = False, ""
        if self.capacity.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps, uncached=swa_uncached)

        level = envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get()
        full_line = f"[Mem Check (BUSY)] {full_msg}"
        swa_line = f"[Mem Check (BUSY)] {swa_msg}" if swa_msg else None

        if level > 1:
            # Verbose: log every iteration.
            logger.info(full_line)
            if swa_line:
                logger.info(swa_line)
        elif level == 1:
            # Quiet: buffer and stay silent; flush the recent ones only on a leak.
            self.recent_busy_msgs.append(full_line)
            if swa_line:
                self.recent_busy_msgs.append(swa_line)
            if full_leak or swa_leak:
                for msg in self.recent_busy_msgs:
                    logger.info(msg)

        assert not full_leak, f"Full Pool Mem Leak Detected! {full_msg}"
        assert not swa_leak, f"SWA Pool Mem Leak Detected! {swa_msg}"

        if envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.get():
            self._check_kv_page_invariants()

    @staticmethod
    def check_no_finished_reqs(lists: Dict[str, RequestList]) -> None:
        """After a host commit no request list holds a finished request: the
        commit moves each one out, except to a releasing list, which holds it
        until its KV is released."""
        for entry in lists.values():
            if entry.releasing:
                continue
            finished = [req.rid for req in entry.reqs if req.finished()]
            assert not finished, (
                f"Finished requests left in the {entry.name!r} list: {finished}"
            )

    def _check_kv_page_invariants(self):
        """The length order of every req/slot row, and no double free:
          A. no owner references a page that is in the free pool (use-after-free).
          B. the free pool has no duplicate pages (two owners freed the same page).
        All heavy work runs on GPU to avoid per-token device->host sync."""
        rtt = self.kv.req_to_token_pool.req_to_token
        row_width = rtt.shape[1]

        def _add_owner(label, kv):
            kv.check_lengths()
            assert kv.kv_allocated_len <= row_width
            owners.append((label, kv.req_pool_idx, kv.kv_allocated_len))

        owners: list[tuple[str, Optional[int], int]] = []
        for req in self._requests_owning_rows():
            _add_owner(f"req {req.rid}", req.kv)
        for sid, kv in self.kv.tree_cache.session_records().items():
            if kv.holds_kv:
                _add_owner(f"slot {sid[:8]}", kv)

        active = [
            (label, rpi, al) for label, rpi, al in owners if rpi is not None and al > 0
        ]
        if not active:
            return

        idx = torch.as_tensor([rpi for _, rpi, _ in active], device=rtt.device)
        allocs = torch.as_tensor([al for _, _, al in active], device=rtt.device)
        mask = torch.arange(row_width, device=rtt.device)[None, :] < allocs[:, None]
        owner_locs = rtt[idx][mask]

        # Sub-allocators to check: a flat allocator is its own single sub; a
        # hybrid-SWA wrapper exposes full_attn_allocator + swa_attn_allocator;
        # the classed sharding allocator keeps per-class lists instead of a
        # flat free_pages and exposes them through get_all_free_pages().
        alloc = self.kv.token_to_kv_pool_allocator
        sub_allocs = (
            [alloc]
            if getattr(alloc, "free_pages", None) is not None
            or page_interleave_shard_size(alloc) > 1
            else [
                sub
                for n in ("full_attn_allocator", "swa_attn_allocator")
                if (sub := getattr(alloc, n, None)) is not None
                and getattr(sub, "free_pages", None) is not None
            ]
        )
        if not sub_allocs:
            return

        # Page ids in the ALLOCATOR's page units (== the physical page for
        # the classed sharding allocator, whose free lists hold logical page
        # ids in the same unit).
        owner_pages = owner_locs // sub_allocs[0].page_size
        # Check B: every sub-pool's free set has no duplicate pages.
        for i, sub in enumerate(sub_allocs):
            free = sub.get_all_free_pages()
            uniq = torch.unique(free)
            if uniq.numel() != free.numel():
                raise_error_or_warn(
                    self,
                    envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                    "count_memory_leak_warnings",
                    f"KV double free: sub-pool {i} has {free.numel() - uniq.numel()} duplicate pages.",
                )

        # Check A: owner pages (full-pool indices) must not be in the full free
        # set (sub_allocs[0] is the full pool, even on hybrid-SWA).
        full_unique = torch.unique(sub_allocs[0].get_all_free_pages())
        stale = owner_pages[torch.isin(owner_pages, full_unique)]
        if stale.numel() > 0:
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_memory_leak_warnings",
                f"KV page use-after-free: {stale.numel()} owner page refs are in "
                f"the free pool, sample pages={torch.unique(stale)[:8].tolist()}.",
            )

    def _check_req_pool(self):
        # Account for pre-alloc headroom whenever the pool has it. The decode
        # pool always does; with runtime P<->D role switching a prefill instance
        # may also hold a decode-flavored (pre-alloc) pool, so key off the pool
        # itself rather than the current role.
        req_total_size = (
            self.kv.req_to_token_pool.size + self.kv.req_to_token_pool.pre_alloc_size
        )

        session_req_count = self.pool_stats_observer.session_held_req_count()
        if (
            len(self.kv.req_to_token_pool.free_slots) + session_req_count
            != req_total_size
        ):
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.kv.req_to_token_pool.free_slots)}, "
                f"session_held={session_req_count}, "
                f"total_size={self.kv.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def _report_leak(self, pool_name: str, token_msg: str):
        msg = f"{pool_name} memory leak detected! {token_msg}"
        raise_error_or_warn(
            self,
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
            "count_memory_leak_warnings",
            msg,
        )

    def _check_all_pools(
        self, ps: PoolStats, uncached: int = 0
    ) -> Tuple[bool, List[str]]:
        """Check memory invariant across all pools. Returns (has_leak, messages)."""
        has_leak = False
        messages = []

        full_leak, full_msg = self._check_full_pool(ps, uncached=uncached)
        has_leak |= full_leak
        messages.append(full_msg)

        if self.capacity.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps)
            has_leak |= swa_leak
            messages.append(swa_msg)

        if self.capacity.is_hybrid_ssm and self.kv.tree_cache.supports_mamba():
            mamba_leak, mamba_msg = self._check_mamba_pool(ps)
            has_leak |= mamba_leak
            messages.append(mamba_msg)

        return has_leak, messages

    def _check_tree_cache(self):
        if not envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.get():
            return
        if self.kv.tree_cache.supports_prefix_sharing() and (
            (self.capacity.is_hybrid_swa and self.kv.tree_cache.supports_swa())
            or (self.capacity.is_hybrid_ssm and self.kv.tree_cache.supports_mamba())
        ):
            self.kv.tree_cache.sanity_check()


class SchedulerWatchdog:
    """Fires when the launch counter stalls while the scheduler initializes or
    runs a step. Its stall dump reads the pools through the invariant checker,
    which the scheduler attaches before initialization ends."""

    def __init__(
        self,
        *,
        counters: LaunchCounters,
        flags: EngineFlags,
        watchdog_timeout: float,
        soft: bool = False,
    ):
        self.counters = counters
        self.flags = flags
        self.invariant_checker: Optional[SchedulerInvariantChecker] = None
        self.raw = WatchdogRaw(
            debug_name="Scheduler",
            get_counter=lambda: counters.forward_ct,
            is_active=lambda: (
                flags.is_initializing or counters.cur_batch_for_debug is not None
            ),
            watchdog_timeout=watchdog_timeout,
            soft=soft,
            dump_info=self._dump_info,
        )

    def _dump_info(self) -> str:
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            describe_registration_in_progress,
        )

        parts = []
        # Checked before is_initializing: CUDA graph capture registers too, and
        # only the soft watchdog runs that early.
        if (registration := describe_registration_in_progress()) is not None:
            parts.append(f"symmetric-memory registration in progress: {registration}")
        if not self.flags.is_initializing:
            checker = self.invariant_checker
            _, messages = checker._check_all_pools(
                checker.pool_stats_observer.get_pool_stats(),
            )
            batch = self.counters.cur_batch_for_debug
            parts += [
                f"scheduler.cur_batch_for_debug.batch_size()={batch.batch_size()!r}",
                f"scheduler.cur_batch_for_debug.reqs={batch.reqs!r}",
                *messages,
            ]
        return "\n".join(parts)
