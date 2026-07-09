from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Deque,
    List,
    Optional,
    Tuple,
)

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    PoolStats,
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import (
    ceil_align,
    raise_error_or_warn,
)
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


logger = logging.getLogger(__name__)

# Number of recent busy-check messages buffered for the level-1 dump-on-leak path.
BUSY_MEM_CHECK_LOG_RING_SIZE = 1000


@dataclass(kw_only=True, slots=True)
class SchedulerInvariantChecker:
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    disaggregation_mode: DisaggregationMode
    page_size: int
    full_tokens_per_layer: Optional[int]
    swa_tokens_per_layer: Optional[int]
    max_total_num_tokens: int
    server_args: ServerArgs
    tree_cache: BasePrefixCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    req_to_token_pool: ReqToTokenPool
    pool_stats_observer: SchedulerPoolStatsObserver
    get_last_batch: Callable
    get_running_batch: Callable
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
        if self.is_hybrid_swa and not self.full_tokens_per_layer:
            return False, ""
        if self.is_hybrid_swa:
            protected = self.tree_cache.full_protected_size()
            session_held = self.pool_stats_observer.session_held_full_tokens()
            total = self.full_tokens_per_layer
        elif self.is_hybrid_ssm:
            # Branch on cache type for the protected accessor (MambaRadixCache
            # splits full/mamba; ChunkCache only has the single protected_size).
            # Use the allocator's `.size` for `total`: static max_total_num_tokens for
            # non-unified pools, the dynamic byte-coordinated cap (matching
            # `available_size`) for the unified pool.
            if self.tree_cache.supports_mamba():
                protected = self.tree_cache.full_protected_size()
            else:
                protected = self.tree_cache.protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            total = self.token_to_kv_pool_allocator.size
        else:
            protected = self.tree_cache.protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            total = self.max_total_num_tokens
        full_evictable_size = ps.full_evictable_size
        allocator = self.token_to_kv_pool_allocator
        if getattr(self.server_args, "dcp_size", 1) > 1 and allocator.page_size > 1:
            # DCP stores logical tokens in widened physical pages.  Prefix cache
            # counters are logical-token based, while the allocator frees whole
            # physical pages, so round cached tokens up to physical page units.
            full_evictable_size = (
                (full_evictable_size + allocator.page_size - 1)
                // allocator.page_size
                * allocator.page_size
            )
        leak, msg = self._check_pool_invariant(
            "full",
            ps.full_available_size,
            full_evictable_size,
            protected,
            session_held,
            total,
            uncached,
        )
        if (
            leak
            and getattr(self.server_args, "dcp_size", 1) > 1
            and allocator.page_size > 1
        ):
            # Radix/Mamba cache accounting is logical-token based while DCP full
            # KV allocation is physical-page based. Partial physical pages can
            # leave a small page-level slack even when all pages are owned by
            # either the allocator or the prefix cache.
            return False, f"{msg}, dcp_physical_page_slack_allowed=True"
        return leak, msg

    def _check_swa_pool(self, ps: PoolStats, uncached: int = 0) -> Tuple[bool, str]:
        return self._check_pool_invariant(
            "swa",
            ps.swa_available_size,
            ps.swa_evictable_size,
            self.tree_cache.swa_protected_size(),
            self.pool_stats_observer.session_held_swa_tokens(),
            self.swa_tokens_per_layer,
            uncached,
        )

    def _check_mamba_pool(self, ps: PoolStats) -> Tuple[bool, str]:
        ckpt_pool = getattr(self.req_to_token_pool, "mamba_ckpt_pool", None)
        if ckpt_pool is not None:
            return self._check_mamba_pool_with_int8(ps, ckpt_pool)
        leak, msg = self._check_pool_invariant(
            "mamba",
            ps.mamba_available_size,
            ps.mamba_evictable_size,
            self.tree_cache.mamba_protected_size(),
            self.pool_stats_observer.session_held_mamba_slots(),
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
            mamba_allocator = self.req_to_token_pool.mamba_allocator
            free_mamba_pages = set(mamba_allocator.free_slots.tolist())
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(1, mamba_allocator.size + 1))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            msg += (
                f", leaked_full_pages={leaked_full_pages or None}"
                f", leaked_mamba_pages={leaked_mamba_pages or None}"
            )
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
            self.req_to_token_pool.mamba_pool.size,
        )
        int8_leak, int8_msg = self._check_pool_invariant(
            "mamba-int8",
            ckpt_pool.available_size(),
            self.tree_cache.mamba_evictable_size(),
            self.tree_cache.mamba_protected_size(),
            0,
            ckpt_pool.num_slots,
        )
        return active_leak or int8_leak, active_msg + "\n" + int8_msg

    def _get_total_uncached_sizes(
        self,
    ) -> Tuple[int, int]:
        """Sum uncached tokens for full and SWA pools across all active batches.

        Returns (full_uncached, swa_uncached). For non-SWA models, swa_uncached is 0.

        For full pool: uncached = allocated - cache_protected_len
        For SWA pool:  uncached = allocated - max(cache_protected_len, swa_evicted_seqlen)
        """
        # After decode: running_batch IS last_batch (same object), count once.
        # After prefill: they differ, both hold uncached tokens.
        # Use identity (is / is not), not membership or ==: ScheduleBatch's
        # dataclass __eq__ compares tensor fields and raises on ambiguous bools.
        last_batch = self.get_last_batch()
        running_batch = self.get_running_batch()
        batches = [last_batch]
        if (
            running_batch is not None
            and running_batch is not last_batch
            and not running_batch.is_empty()
        ):
            batches.append(running_batch)

        full_uncached = 0
        swa_uncached = 0
        for batch in batches:
            for req in batch.reqs:
                if req.kv is None:
                    continue

                allocated_len = req.kv.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0

                full_uncached += allocated_len - req.cache_protected_len
                if self.is_hybrid_swa:
                    swa_uncached += allocated_len - max(
                        req.cache_protected_len, req.kv.swa_evicted_seqlen
                    )

        return full_uncached, swa_uncached

    def self_check_during_busy(self):
        if self.get_last_batch() is None:
            return

        ps = self.pool_stats_observer.get_pool_stats()
        full_uncached, swa_uncached = self._get_total_uncached_sizes()

        full_leak, full_msg = self._check_full_pool(ps, uncached=full_uncached)

        swa_leak, swa_msg = False, ""
        if self.is_hybrid_swa:
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

    def _check_kv_page_invariants(self):
        """committed<=allocated for every req/slot, and no double free:
          A. no owner references a page that is in the free pool (use-after-free).
          B. the free pool has no duplicate pages (two owners freed the same page).
        All heavy work runs on GPU to avoid per-token device->host sync."""
        rtt = self.req_to_token_pool.req_to_token
        row_width = rtt.shape[1]

        def _add_owner(req_or_slot, label, rpi, committed, allocated):
            assert 0 <= committed <= allocated <= row_width
            owners.append((label, rpi, allocated))

        owners: list[tuple[str, Optional[int], int]] = []
        batch = self.get_last_batch()
        if batch is not None:
            for req in batch.reqs:
                if req.kv is None:
                    continue
                _add_owner(
                    req,
                    f"req {req.rid}",
                    req.req_pool_idx,
                    req.kv_committed_len,
                    req.kv.kv_allocated_len,
                )
        sess = getattr(self.tree_cache, "slots", None)
        if sess:
            for sid, slot in sess.items():
                if getattr(slot, "is_holding_kv", False):
                    _add_owner(
                        slot,
                        f"slot {sid[:8]}",
                        slot.req_pool_idx,
                        slot.kv_committed_len,
                        slot.kv.kv_allocated_len,
                    )

        active = [
            (label, rpi, al) for label, rpi, al in owners if rpi is not None and al > 0
        ]
        if not active:
            return

        idx = torch.as_tensor([rpi for _, rpi, _ in active], device=rtt.device)
        allocs = torch.as_tensor([al for _, _, al in active], device=rtt.device)
        mask = torch.arange(row_width, device=rtt.device)[None, :] < allocs[:, None]
        owner_pages = rtt[idx][mask] // self.page_size

        # Sub-allocators to check: a flat allocator is its own single sub; a
        # hybrid-SWA wrapper exposes full_attn_allocator + swa_attn_allocator.
        alloc = self.token_to_kv_pool_allocator
        sub_allocs = (
            [alloc]
            if getattr(alloc, "free_pages", None) is not None
            else [
                sub
                for n in ("full_attn_allocator", "swa_attn_allocator")
                if (sub := getattr(alloc, n, None)) is not None
                and getattr(sub, "free_pages", None) is not None
            ]
        )
        if not sub_allocs:
            return

        def _free_pages(a):
            free = a.free_pages
            release = getattr(a, "release_pages", None)
            return (
                torch.cat((free, release))
                if release is not None and len(release) > 0
                else free
            )

        # Check B: every sub-pool's free set has no duplicate pages.
        for i, sub in enumerate(sub_allocs):
            free = _free_pages(sub)
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
        full_unique = torch.unique(_free_pages(sub_allocs[0]))
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
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        session_req_count = self.pool_stats_observer.session_held_req_count()
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

        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps)
            has_leak |= swa_leak
            messages.append(swa_msg)

        if self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            mamba_leak, mamba_msg = self._check_mamba_pool(ps)
            has_leak |= mamba_leak
            messages.append(mamba_msg)

        return has_leak, messages

    def _check_tree_cache(self):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing:
            return ""
        _, messages = scheduler.invariant_checker._check_all_pools(
            scheduler.pool_stats_observer.get_pool_stats(),
        )
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n" + "\n".join(messages)
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: scheduler.forward_ct,
        is_active=lambda: scheduler.is_initializing or scheduler.cur_batch is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
