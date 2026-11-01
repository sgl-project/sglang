from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Tuple,
)

from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ReqKvInfo
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.kv.storage import KVStorage, SubPoolUsage
    from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage


class SchedulerStats: ...  # type: ignore[no-redef]


# A KV record's footprint outside the tree, in whole pages. Each row is summed
# once, under its owner: uncached for a request's own, session-held for a session's.


def kv_private_tokens(kv: ReqKvInfo, page_size: int) -> int:
    return ceil_align(kv.kv_allocated_len, page_size) - kv.cache_protected_len


def kv_private_window_tokens(
    kv: ReqKvInfo, page_size: int, component: ComponentType
) -> int:
    # Window eviction frees only [window_dead_lo, evicted); [protected, window_dead_lo)
    # stays held until the request finishes, even after the cursor jumps past it.
    allocated = ceil_align(kv.kv_allocated_len, page_size)
    window_freed = max(
        0, kv.get_evicted_seqlen(component) - kv.window_dead_lo(page_size)
    )
    return allocated - kv.cache_protected_len - window_freed


def kv_mamba_slots(kv: ReqKvInfo) -> int:
    total = kv.mamba_pool_idx.numel() if kv.holds_mamba else 0
    if kv.mamba_ping_pong_track_buffer is not None:
        total += kv.mamba_ping_pong_empty.count(False)
    return total


@dataclasses.dataclass
class WindowPoolStats:
    """One window sub-pool's occupancy in tokens, under the name its numbers
    are published by (`SubPoolUsage.label`)."""

    name: str
    sub_pool: str
    # The tree slot whose cursors and locks describe this window; a tree that
    # keeps no window slots retires its one window as `ComponentType.SWA`.
    component: ComponentType
    capacity: int
    available_size: int
    evictable_size: int
    num_used: int
    token_usage: float


@dataclasses.dataclass
class PoolStats:
    # For full pools (required)
    full_num_used: int
    full_token_usage: float
    full_available_size: int
    full_evictable_size: int

    is_hybrid_ssm: bool = False
    is_hisparse: bool = False

    # Denominator of full_token_usage (always set)
    full_capacity: Optional[int] = None
    # Every window sub-pool, the tree's first window first; empty without one.
    windows: Tuple[WindowPoolStats, ...] = ()

    # For mamba pools
    mamba_capacity: Optional[int] = None
    mamba_num_used: Optional[int] = None
    mamba_usage: Optional[float] = None
    mamba_available_size: Optional[int] = None
    mamba_evictable_size: Optional[int] = None

    # HiSparse device/host breakdown for decode logs (plain KV pool only)
    hisparse_device_tokens: Optional[int] = None
    hisparse_device_token_usage: Optional[float] = None
    hisparse_host_tokens: Optional[int] = None
    hisparse_host_token_usage: Optional[float] = None

    def window(self, name: str) -> Optional[WindowPoolStats]:
        return next((w for w in self.windows if w.name == name), None)

    def get_kv_token_stats(self) -> Tuple[int, float]:
        # NOTE: mamba pool is not included in the "token usage" calculation.
        num_used = max([self.full_num_used, *(w.num_used for w in self.windows)])
        token_usage = max(
            [self.full_token_usage, *(w.token_usage for w in self.windows)]
        )
        return num_used, token_usage

    def get_max_pool_usage(self) -> float:
        usage = max([self.full_token_usage, *(w.token_usage for w in self.windows)])
        if self.is_hybrid_ssm:
            usage = max(usage, self.mamba_usage)
        assert usage is not None and usage >= 0, f"{usage=} is not valid"
        return usage

    def get_prefill_usage_msg_parts(self) -> List[str]:
        parts = []
        if self.windows:
            parts.append(f"full token usage: {self.full_token_usage:.2f}")
            parts += [
                f"{w.name} token usage: {w.token_usage:.2f}" for w in self.windows
            ]
        if self.is_hybrid_ssm:
            if not self.windows:
                parts.append(f"full token usage: {self.full_token_usage:.2f}")
            parts.append(f"mamba usage: {self.mamba_usage:.2f}")
        if not parts:
            parts.append(f"token usage: {self.full_token_usage:.2f}")
        return parts

    def get_decode_usage_msg_parts(self) -> List[str]:
        parts = []
        if self.windows:
            parts += [
                f"#full token: {self.full_num_used}",
                f"full token usage: {self.full_token_usage:.2f}",
            ]
            for w in self.windows:
                parts += [
                    f"#{w.name} token: {w.num_used}",
                    f"{w.name} token usage: {w.token_usage:.2f}",
                ]
        if self.is_hybrid_ssm:
            if not self.windows:
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
        stats.window_token_usage = {w.name: w.token_usage for w in self.windows}
        stats.window_available_tokens = {w.name: w.available_size for w in self.windows}
        stats.window_evictable_tokens = {w.name: w.evictable_size for w in self.windows}
        stats.window_used_tokens = {w.name: w.num_used for w in self.windows}
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
    # The sub-pools' usage, unified or static.
    storage: KVStorage | StaticKVStorage
    # Only for what pools built without unified memory still decide alone: a
    # per-request window ring and int8 state checkpoints.
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    session_controller: Any
    hisparse_coordinator: Any
    enable_hisparse: bool

    def streaming_session_count(self) -> int:
        return sum(
            1
            for session in self.session_controller.sessions.values()
            if session.streaming
        )

    def _session_kv_rows(self) -> List[ReqKvInfo]:
        records = self.tree_cache.session_records().values()
        return [kv for kv in records if kv.holds_kv]

    def session_held_tokens(self) -> int:
        page_size = self.tree_cache.page_size
        return sum(kv_private_tokens(kv, page_size) for kv in self._session_kv_rows())

    def session_held_window_tokens(self, component: ComponentType) -> int:
        page_size = self.tree_cache.page_size
        return sum(
            kv_private_window_tokens(kv, page_size, component)
            for kv in self._session_kv_rows()
        )

    def session_held_req_count(self) -> int:
        return len(self._session_kv_rows())

    def session_held_mamba_slots(self) -> int:
        return sum(kv_mamba_slots(kv) for kv in self._session_kv_rows())

    def get_pool_stats(self) -> PoolStats:
        usage = self.storage.usage()
        (full,) = (u for u in usage if u.metrics == "full")
        mamba = next((u for u in usage if u.metrics == "mamba"), None)
        windows = tuple(u for u in usage if u.metrics == "window")
        if windows:
            pool_stats = self._get_window_token_info(full, windows)
        elif mamba is not None:
            pool_stats = self._get_mamba_token_info(full, mamba)
        else:
            pool_stats = self._get_token_info(full)

        if self.enable_hisparse:
            pool_stats = self._get_hisparse_token_info(pool_stats)

        # windows + ssm can coexist: overlay mamba fields onto the window stats
        if mamba is not None:
            mamba_stats = self._get_mamba_token_info(full, mamba)
            pool_stats.is_hybrid_ssm = True
            pool_stats.mamba_capacity = mamba_stats.mamba_capacity
            pool_stats.mamba_num_used = mamba_stats.mamba_num_used
            pool_stats.mamba_usage = mamba_stats.mamba_usage
            pool_stats.mamba_available_size = mamba_stats.mamba_available_size
            pool_stats.mamba_evictable_size = mamba_stats.mamba_evictable_size

        return pool_stats

    def _get_token_info(self, full: SubPoolUsage) -> PoolStats:
        available_size = full.available
        evictable_size = self.tree_cache.evictable_size()
        num_used = full.capacity - (available_size + evictable_size)
        token_usage = num_used / full.capacity
        return PoolStats(
            full_num_used=num_used,
            full_token_usage=token_usage,
            full_available_size=available_size,
            full_evictable_size=evictable_size,
            full_capacity=full.capacity,
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

    def _get_mamba_token_info(
        self, full: SubPoolUsage, mamba: SubPoolUsage
    ) -> PoolStats:
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba()
            and self.tree_cache.supports_prefix_sharing()
        )
        full_available_size = full.available
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = mamba.available
        # `mamba_usage`/`mamba_num_used` track the ACTIVE bf16 pool occupancy (running
        # requests) -- this feeds throttle decisions (get_max_pool_usage) which asserts
        # usage >= 0. With int8 checkpoints the radix-cached states live in a SEPARATE
        # int8 pool, so they own ZERO active slots: report evictable=0 against the active
        # pool (otherwise active.size - (available + radix_cached) goes negative). The
        # int8 cache pool's own occupancy is validated separately in the invariant check.
        has_int8_ckpt = self.token_to_kv_pool_allocator.state.ckpt_pool is not None
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size()
            if (is_mamba_radix_cache and not has_int8_ckpt)
            else 0
        )
        full_capacity = full.capacity
        full_num_used = full_capacity - (full_available_size + full_evictable_size)
        mamba_num_used = mamba.capacity - (mamba_available_size + mamba_evictable_size)
        full_token_usage = full_num_used / full_capacity
        mamba_usage = mamba_num_used / mamba.capacity

        return PoolStats(
            is_hybrid_ssm=True,
            full_num_used=full_num_used,
            full_token_usage=full_token_usage,
            full_available_size=full_available_size,
            full_evictable_size=full_evictable_size,
            full_capacity=full_capacity,
            mamba_capacity=mamba.capacity,
            mamba_num_used=mamba_num_used,
            mamba_usage=mamba_usage,
            mamba_available_size=mamba_available_size,
            mamba_evictable_size=mamba_evictable_size,
        )

    def _window_slot(self, sub_pool: str) -> Optional[ComponentType]:
        """The tree slot keeping window sub-pool ``sub_pool``, if any."""
        for slot in self.tree_cache.window_slots():
            if self.storage.tree_sub_pool(slot) == sub_pool:
                return slot
        return None

    def _window_evictable_size(self, sub_pool: str) -> int:
        """Evictable tokens the prefix tree holds in window sub-pool
        ``sub_pool``: its slot's, or, for a tree keeping one pool and no window
        slots, everything it can evict."""
        if not self.tree_cache.window_slots():
            return self.tree_cache.evictable_size()
        slot = self._window_slot(sub_pool)
        return 0 if slot is None else self.tree_cache.window_evictable_size(slot)

    def _in_tree_order(
        self, windows: Tuple[SubPoolUsage, ...]
    ) -> Tuple[SubPoolUsage, ...]:
        """The window sub-pools in the prefix tree's window order. A sub-pool
        no slot keeps comes last."""
        order = {
            self.storage.tree_sub_pool(slot): i
            for i, slot in enumerate(self.tree_cache.window_slots())
        }
        return tuple(sorted(windows, key=lambda w: order.get(w.name, len(order))))

    def _get_window_token_info(
        self, full: SubPoolUsage, windows: Tuple[SubPoolUsage, ...]
    ) -> PoolStats:
        full_capacity, full_available_size = full.capacity, full.available
        full_evictable_size = self.tree_cache.full_evictable_size()
        full_num_used = full_capacity - (full_available_size + full_evictable_size)
        # FIXME(hisparse): host-backup transiently over-releases the device pool
        # counter, producing negative used counts. We clamp to 0 to keep
        # token_usage / leak checks sane, but the underlying accounting bug
        # should be fixed so the clamp can go away.
        if self.enable_hisparse:
            full_num_used = max(0, full_num_used)
        if not full_capacity:
            full_num_used = 0
            full_available_size = 0
            full_token_usage = 0.0
        else:
            full_token_usage = full_num_used / full_capacity
        # Per-request window ring: released with the req slot, yet cached radix
        # prefixes still report window evictable; counting it drives usage negative.
        req_ring = is_swa_req_ring(self.token_to_kv_pool_allocator)
        window_stats = []
        for window in self._in_tree_order(windows):
            capacity, available = window.capacity, window.available
            evictable = 0 if req_ring else self._window_evictable_size(window.name)
            num_used = capacity - (available + evictable)
            if self.enable_hisparse:
                num_used = max(0, num_used)
            if not capacity:
                num_used, available, token_usage = 0, 0, 0.0
            else:
                token_usage = num_used / capacity
            window_stats.append(
                WindowPoolStats(
                    name=window.label,
                    sub_pool=window.name,
                    component=self._window_slot(window.name) or ComponentType.SWA,
                    capacity=capacity,
                    available_size=available,
                    evictable_size=evictable,
                    num_used=num_used,
                    token_usage=token_usage,
                )
            )

        return PoolStats(
            full_capacity=full_capacity,
            full_num_used=full_num_used,
            full_token_usage=full_token_usage,
            full_available_size=full_available_size,
            full_evictable_size=full_evictable_size,
            windows=tuple(window_stats),
        )
