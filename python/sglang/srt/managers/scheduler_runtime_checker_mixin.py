from __future__ import annotations

import dataclasses
import logging
import time
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache
from sglang.srt.observability.metrics_collector import QueueCount
from sglang.srt.utils.common import ceil_align, raise_error_or_warn
from sglang.srt.utils.request_logger import disable_request_logging
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.observability.metrics_collector import SchedulerStats

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PoolStats:
    # For full pools (required)
    full_num_used: int
    full_token_usage: float
    full_available_size: int
    full_evictable_size: int

    is_hybrid_swa: bool = False
    is_hybrid_ssm: bool = False
    is_hisparse: bool = False

    # For hybrid-swa pools
    swa_num_used: Optional[int] = None
    swa_token_usage: Optional[float] = None
    swa_available_size: Optional[int] = None
    swa_evictable_size: Optional[int] = None

    # For mamba pools
    mamba_num_used: Optional[int] = None
    mamba_usage: Optional[float] = None
    mamba_available_size: Optional[int] = None
    mamba_evictable_size: Optional[int] = None

    # HiSparse device/host breakdown for decode logs (plain KV pool only)
    hisparse_device_tokens: Optional[int] = None
    hisparse_device_token_usage: Optional[float] = None
    hisparse_host_tokens: Optional[int] = None
    hisparse_host_token_usage: Optional[float] = None

    def get_kv_token_stats(self) -> Tuple[int, float]:
        # NOTE: mamba pool is not included in the "token usage" calculation.
        if self.is_hybrid_swa:
            num_used = max(self.full_num_used, self.swa_num_used)
            token_usage = max(self.full_token_usage, self.swa_token_usage)
        else:
            num_used = self.full_num_used
            token_usage = self.full_token_usage

        return num_used, token_usage

    def get_max_pool_usage(self) -> float:
        usage = self.full_token_usage
        if self.is_hybrid_swa:
            usage = max(usage, self.swa_token_usage)
        if self.is_hybrid_ssm:
            usage = max(usage, self.mamba_usage)
        assert usage is not None and usage >= 0, f"{usage=} is not valid"
        return usage

    def get_prefill_usage_msg_parts(self) -> List[str]:
        parts = []
        if self.is_hybrid_swa:
            parts += [
                f"full token usage: {self.full_token_usage:.2f}",
                f"swa token usage: {self.swa_token_usage:.2f}",
            ]
        if self.is_hybrid_ssm:
            if not self.is_hybrid_swa:
                parts.append(f"full token usage: {self.full_token_usage:.2f}")
            parts.append(f"mamba usage: {self.mamba_usage:.2f}")
        if not parts:
            parts.append(f"token usage: {self.full_token_usage:.2f}")
        return parts

    def get_decode_usage_msg_parts(self) -> List[str]:
        parts = []
        if self.is_hybrid_swa:
            parts += [
                f"#full token: {self.full_num_used}",
                f"full token usage: {self.full_token_usage:.2f}",
                f"#swa token: {self.swa_num_used}",
                f"swa token usage: {self.swa_token_usage:.2f}",
            ]
        if self.is_hybrid_ssm:
            if not self.is_hybrid_swa:
                parts += [
                    f"#full token: {self.full_num_used}",
                    f"full token usage: {self.full_token_usage:.2f}",
                ]
            parts += [
                f"mamba num: {self.mamba_num_used}",
                f"mamba usage: {self.mamba_usage:.2f}",
            ]
        if self.is_hisparse:
            parts += [
                f"#gpu token: {self.hisparse_device_tokens}",
                f"gpu token usage: {self.hisparse_device_token_usage:.2f}",
                f"#cpu token: {self.hisparse_host_tokens}",
                f"cpu token usage: {self.hisparse_host_token_usage:.2f}",
            ]
        if not parts:
            parts.append(
                f"#token: {self.full_num_used}, token usage: {self.full_token_usage:.2f}"
            )
        return parts

    def update_scheduler_stats(self, stats: SchedulerStats) -> None:
        """Update pool-related fields on SchedulerStats."""
        num_used, _ = self.get_kv_token_stats()
        stats.num_used_tokens = num_used
        stats.token_usage = round(self.get_max_pool_usage(), 2)
        stats.full_token_usage = self.full_token_usage
        if self.is_hybrid_swa:
            stats.swa_token_usage = self.swa_token_usage
        if self.is_hybrid_ssm:
            stats.mamba_usage = self.mamba_usage
        stats.kv_available_tokens = self.full_available_size
        stats.kv_evictable_tokens = self.full_evictable_size
        stats.kv_used_tokens = self.full_num_used


class SchedulerRuntimeCheckerMixin:
    def _streaming_session_count(self: Scheduler) -> int:
        return sum(
            1
            for session in self.session_controller.sessions.values()
            if session.streaming
        )

    def _active_pool_idxs(self: Scheduler) -> set:
        """Pool idxs currently owned by reqs in last_batch / running_batch.

        Used to decide which session slots' KV is owned by batch reqs
        (and thus counted via uncached_size, not session_held).
        """
        idxs = set()
        for batch in [self.last_batch, self.running_batch]:
            if batch is None or batch.is_empty():
                continue
            for req in batch.reqs:
                if req.req_pool_idx is not None:
                    idxs.add(req.req_pool_idx)
        return idxs

    def _session_held_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_tokens(self._active_pool_idxs())
        return 0

    def _session_held_full_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_full_tokens(self._active_pool_idxs())
        return 0

    def _session_held_swa_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_swa_tokens(self._active_pool_idxs())
        return 0

    def _session_held_req_count(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_req_count()
        return 0

    def get_pool_stats(self: Scheduler) -> PoolStats:
        if self.is_hybrid_swa:
            pool_stats = self._get_swa_token_info()
        elif self.is_hybrid_ssm:
            return self._get_mamba_token_info()
        elif self.enable_hisparse:
            return self._get_hisparse_token_info()
        else:
            return self._get_token_info()

        # swa + ssm can coexist: overlay mamba fields onto swa stats
        if self.is_hybrid_ssm:
            mamba_stats = self._get_mamba_token_info()
            pool_stats.is_hybrid_ssm = True
            pool_stats.mamba_num_used = mamba_stats.mamba_num_used
            pool_stats.mamba_usage = mamba_stats.mamba_usage
            pool_stats.mamba_available_size = mamba_stats.mamba_available_size
            pool_stats.mamba_evictable_size = mamba_stats.mamba_evictable_size

        return pool_stats

    def _get_token_info(self: Scheduler) -> PoolStats:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return PoolStats(
            full_num_used=num_used,
            full_token_usage=token_usage,
            full_available_size=available_size,
            full_evictable_size=evictable_size,
        )

    def _get_hisparse_token_info(self: Scheduler) -> PoolStats:
        pool_stats = self._get_token_info()
        if self.enable_hisparse and self.hisparse_coordinator is not None:
            h = self.hisparse_coordinator.get_token_stats()
            return dataclasses.replace(
                pool_stats,
                is_hisparse=True,
                hisparse_device_tokens=h.device_tokens,
                hisparse_device_token_usage=h.device_token_usage,
                hisparse_host_tokens=h.host_tokens,
                hisparse_host_token_usage=h.host_token_usage,
            )
        return pool_stats

    def _get_mamba_token_info(self: Scheduler):
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_pool.available_size()
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size() if is_mamba_radix_cache else 0
        )
        full_num_used = self.token_to_kv_pool_allocator.size - (
            full_available_size + full_evictable_size
        )
        mamba_num_used = self.req_to_token_pool.mamba_pool.size - (
            mamba_available_size + mamba_evictable_size
        )
        full_token_usage = full_num_used / self.token_to_kv_pool_allocator.size
        mamba_usage = mamba_num_used / self.req_to_token_pool.mamba_pool.size

        return PoolStats(
            is_hybrid_ssm=True,
            full_num_used=full_num_used,
            full_token_usage=full_token_usage,
            full_available_size=full_available_size,
            full_evictable_size=full_evictable_size,
            mamba_num_used=mamba_num_used,
            mamba_usage=mamba_usage,
            mamba_available_size=mamba_available_size,
            mamba_evictable_size=mamba_evictable_size,
        )

    def _get_swa_token_info(self: Scheduler) -> PoolStats:
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer

        return PoolStats(
            is_hybrid_swa=True,
            full_num_used=full_num_used,
            full_token_usage=full_token_usage,
            full_available_size=full_available_size,
            full_evictable_size=full_evictable_size,
            swa_num_used=swa_num_used,
            swa_token_usage=swa_token_usage,
            swa_available_size=swa_available_size,
            swa_evictable_size=swa_evictable_size,
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

    def _check_full_pool(
        self: Scheduler, ps: PoolStats, uncached: int = 0
    ) -> Tuple[bool, str]:
        if self.is_hybrid_swa:
            protected = self.tree_cache.full_protected_size()
            session_held = self._session_held_full_tokens()
            total = self.full_tokens_per_layer
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            protected = self.tree_cache.full_protected_size()
            session_held = self._session_held_tokens()
            total = self.token_to_kv_pool_allocator.size
        else:
            protected = self.tree_cache.protected_size()
            session_held = self._session_held_tokens()
            total = self.max_total_num_tokens
        return self._check_pool_invariant(
            "full",
            ps.full_available_size,
            ps.full_evictable_size,
            protected,
            session_held,
            total,
            uncached,
        )

    def _check_swa_pool(
        self: Scheduler, ps: PoolStats, uncached: int = 0
    ) -> Tuple[bool, str]:
        return self._check_pool_invariant(
            "swa",
            ps.swa_available_size,
            ps.swa_evictable_size,
            self.tree_cache.swa_protected_size(),
            self._session_held_swa_tokens(),
            self.swa_tokens_per_layer,
            uncached,
        )

    def _check_mamba_pool(self: Scheduler, ps: PoolStats) -> Tuple[bool, str]:
        leak, msg = self._check_pool_invariant(
            "mamba",
            ps.mamba_available_size,
            ps.mamba_evictable_size,
            self.tree_cache.mamba_protected_size(),
            0,
            self.req_to_token_pool.mamba_pool.size,
        )
        if leak:
            # Page-level leak diagnosis for mamba
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            free_mamba_pages = set(
                self.req_to_token_pool.mamba_pool.free_slots.tolist()
            )
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(self.req_to_token_pool.mamba_pool.size))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            msg += (
                f", leaked_full_pages={leaked_full_pages or None}"
                f", leaked_mamba_pages={leaked_mamba_pages or None}"
            )
        return leak, msg

    def _get_total_uncached_sizes(self: Scheduler) -> Tuple[int, int]:
        """Sum uncached tokens for full and SWA pools across all active batches.

        Returns (full_uncached, swa_uncached). For non-SWA models, swa_uncached is 0.

        For full pool: uncached = allocated - cache_protected_len
        For SWA pool:  uncached = allocated - max(cache_protected_len, swa_evicted_seqlen)
        """
        # After decode: running_batch IS last_batch (same object), count once.
        # After prefill: they differ, both hold uncached tokens.
        batches = [self.last_batch]
        if (
            self.running_batch not in (None, self.last_batch)
            and not self.running_batch.is_empty()
        ):
            batches.append(self.running_batch)

        full_uncached = 0
        swa_uncached = 0
        for batch in batches:
            for req in batch.reqs:
                assert req.kv_committed_freed == req.kv_overallocated_freed
                if req.kv_committed_freed or req.req_pool_idx is None:
                    continue

                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0

                full_uncached += allocated_len - req.cache_protected_len
                if self.is_hybrid_swa:
                    swa_uncached += allocated_len - max(
                        req.cache_protected_len, req.swa_evicted_seqlen
                    )

        return full_uncached, swa_uncached

    def self_check_during_busy(self: Scheduler):
        if self.last_batch is None:
            return

        spec_topk = self.server_args.speculative_eagle_topk or 1
        if spec_topk > 1:
            warnings.warn(
                "Runtime memory check (busy) is not supported when speculation topk > 1."
            )
            return

        ps = self.get_pool_stats()
        full_uncached, swa_uncached = self._get_total_uncached_sizes()

        full_leak, full_msg = self._check_full_pool(ps, uncached=full_uncached)

        swa_leak, swa_msg = False, ""
        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps, uncached=swa_uncached)

        if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get() > 1:
            logger.info(f"[Mem Check (BUSY)] {full_msg}")
            if swa_msg:
                logger.info(f"[Mem Check (BUSY)] {swa_msg}")
        assert not full_leak, f"Full Pool Mem Leak Detected! {full_msg}"
        assert not swa_leak, f"SWA Pool Mem Leak Detected! {swa_msg}"

    def _check_req_pool(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        session_req_count = self._session_held_req_count()
        if len(self.req_to_token_pool.free_slots) + session_req_count != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"session_held={session_req_count}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def _report_leak(self: Scheduler, pool_name: str, token_msg: str):
        msg = f"{pool_name} memory leak detected! {token_msg}"
        raise_error_or_warn(
            self,
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
            "count_memory_leak_warnings",
            msg,
        )

    def _check_all_pools(
        self: Scheduler, ps: PoolStats, uncached: int = 0
    ) -> Tuple[bool, List[str]]:
        """Check memory invariant across all pools. Returns (has_leak, messages)."""
        has_leak = False
        messages = []

        full_leak, full_msg = self._check_full_pool(ps, uncached=uncached)
        has_leak |= full_leak
        messages.append(full_msg)

        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps)
            has_leak |= swa_leak
            messages.append(swa_msg)

        if self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            mamba_leak, mamba_msg = self._check_mamba_pool(ps)
            has_leak |= mamba_leak
            messages.append(mamba_msg)

        return has_leak, messages

    def _maybe_log_idle_metrics(self: Scheduler):
        """Collect and log metrics every 30 seconds during idle."""
        if (
            not self.current_scheduler_metrics_enabled
            or time.perf_counter() <= self.metrics_collector.last_log_time + 30
        ):
            return

        self.get_pool_stats().update_scheduler_stats(self.stats)
        self.stats.num_streaming_sessions = self._streaming_session_count()
        self.stats.streaming_session_held_tokens = self._session_held_tokens()

        priority_enabled = self.enable_priority_scheduling
        self.stats.num_running_reqs = QueueCount.from_reqs(
            self.running_batch.reqs, priority_enabled
        )
        self.stats.gen_throughput = 0
        self.stats.num_queue_reqs = QueueCount.from_reqs(
            self.waiting_queue, priority_enabled
        )
        self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.num_prefill_prealloc_queue_reqs = QueueCount.from_reqs(
                self.disagg_prefill_bootstrap_queue.queue, priority_enabled
            )
            self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                self.disagg_prefill_inflight_queue, priority_enabled
            )
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                self.disagg_decode_prealloc_queue.queue, priority_enabled
            )
            self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                self.disagg_decode_transfer_queue.queue, priority_enabled
            )
        self.metrics_collector.log_stats(self.stats)

    def _check_tree_cache(self: Scheduler):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()

    def on_idle(self: Scheduler):
        """Idle housekeeping: guard, check, metrics, reset, sleep."""
        if not self.is_fully_idle():
            return

        # memory leak check
        has_leak, messages = self._check_all_pools(self.get_pool_stats())
        if has_leak:
            self._report_leak("pool", "\n".join(messages))
        self._check_req_pool()

        # tree cache sanity check
        self._check_tree_cache()

        # metrics every 30s
        self._maybe_log_idle_metrics()

        # kv event publishing
        self._publish_kv_events()

        # reset token ratio
        self.new_token_ratio = self.init_new_token_ratio

        # sleep until next event
        self.maybe_sleep_on_idle()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing or disable_request_logging():
            return ""
        _, messages = scheduler._check_all_pools(scheduler.get_pool_stats())
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n" + "\n".join(messages)
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: getattr(scheduler, "forward_ct", 0),
        is_active=lambda: scheduler.is_initializing
        or getattr(scheduler, "cur_batch", None) is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
