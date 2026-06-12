from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class SchedulerStats: ...  # type: ignore[no-redef]


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
            stats.swa_available_tokens = self.swa_available_size
            stats.swa_evictable_tokens = self.swa_evictable_size
            stats.swa_used_tokens = self.swa_num_used
        if self.is_hybrid_ssm:
            stats.mamba_usage = self.mamba_usage
            stats.mamba_available_tokens = self.mamba_available_size
            stats.mamba_evictable_tokens = self.mamba_evictable_size
            stats.mamba_used_tokens = self.mamba_num_used
        stats.kv_available_tokens = self.full_available_size
        stats.kv_evictable_tokens = self.full_evictable_size
        stats.kv_used_tokens = self.full_num_used


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerPoolStatsObserver:
    tree_cache: BasePrefixCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    req_to_token_pool: ReqToTokenPool
    session_controller: Any
    hisparse_coordinator: Any
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    enable_hisparse: bool
    full_tokens_per_layer: Any
    swa_tokens_per_layer: Any
    max_total_num_tokens: int
    get_last_batch: Callable
    get_running_batch: Callable

    def streaming_session_count(self) -> int:
        return sum(
            1
            for session in self.session_controller.sessions.values()
            if session.streaming
        )

    def active_pool_idxs(self) -> set:
        """Pool idxs currently owned by reqs in last_batch / running_batch.

        Used to decide which session slots' KV is owned by batch reqs
        (and thus counted via uncached_size, not session_held).
        """
        idxs = set()
        for batch in [self.get_last_batch(), self.get_running_batch()]:
            if batch is None or batch.is_empty():
                continue
            for req in batch.reqs:
                if req.req_pool_idx is not None:
                    idxs.add(req.req_pool_idx)
        return idxs

    def session_held_tokens(self) -> int:
        return self.tree_cache.session_held_tokens(self.active_pool_idxs())

    def session_held_full_tokens(self) -> int:
        return self.tree_cache.session_held_full_tokens(self.active_pool_idxs())

    def session_held_swa_tokens(self) -> int:
        return self.tree_cache.session_held_swa_tokens(self.active_pool_idxs())

    def session_held_req_count(self) -> int:
        return self.tree_cache.session_held_req_count()

    def session_held_mamba_slots(self) -> int:
        return self.tree_cache.session_held_mamba_slots(self.active_pool_idxs())

    def get_pool_stats(self) -> PoolStats:
        if self.is_hybrid_swa:
            pool_stats = self._get_swa_token_info()
        elif self.is_hybrid_ssm:
            pool_stats = self._get_mamba_token_info()
        else:
            pool_stats = self._get_token_info()

        if self.enable_hisparse:
            pool_stats = self._get_hisparse_token_info(pool_stats)

        # swa + ssm can coexist: overlay mamba fields onto swa stats
        if self.is_hybrid_ssm:
            mamba_stats = self._get_mamba_token_info()
            pool_stats.is_hybrid_ssm = True
            pool_stats.mamba_num_used = mamba_stats.mamba_num_used
            pool_stats.mamba_usage = mamba_stats.mamba_usage
            pool_stats.mamba_available_size = mamba_stats.mamba_available_size
            pool_stats.mamba_evictable_size = mamba_stats.mamba_evictable_size

        return pool_stats

    def _get_token_info(self) -> PoolStats:
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

    def _get_hisparse_token_info(self, pool_stats: PoolStats) -> PoolStats:
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

    def _get_mamba_token_info(self):
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_allocator.available_size()
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

    def _get_swa_token_info(self) -> PoolStats:
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
        # FIXME(hisparse): host-backup transiently over-releases the device pool
        # counter, producing negative full_num_used / swa_num_used. We clamp to 0
        # to keep token_usage / leak checks sane, but the underlying accounting
        # bug should be fixed so the clamp can go away.
        if self.enable_hisparse:
            full_num_used = max(0, full_num_used)
            swa_num_used = max(0, swa_num_used)
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
