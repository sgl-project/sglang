from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class P2PAllocatorObserver:
    def __init__(
        self,
        *,
        pool_stats_observer,
        invariant_checker,
        req_to_token_pool,
        ps,
        interval_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        logger=logger,
    ):
        self.pool_stats_observer = pool_stats_observer
        self.invariant_checker = invariant_checker
        self.req_to_token_pool = req_to_token_pool
        self.ps = ps
        self.interval_s = interval_s
        self.clock = clock
        self.wall_clock = wall_clock
        self.logger = logger
        self._last_log_time: float | None = None

    @staticmethod
    def _call_int(obj, name: str, default: int = 0) -> int:
        value = getattr(obj, name, None)
        if value is None:
            return default
        if callable(value):
            value = value()
        return int(value)

    def _full_protected(self, pool_stats) -> int:
        tree_cache = self.pool_stats_observer.tree_cache
        if pool_stats.is_hybrid_swa or pool_stats.is_hybrid_ssm:
            return self._call_int(tree_cache, "full_protected_size")
        return self._call_int(tree_cache, "protected_size")

    def _full_session_held(self, pool_stats) -> int:
        if pool_stats.is_hybrid_swa or pool_stats.is_hybrid_ssm:
            return self._call_int(self.pool_stats_observer, "session_held_full_tokens")
        return self._call_int(self.pool_stats_observer, "session_held_tokens")

    def _build_snapshot(self) -> dict:
        pool_stats = self.pool_stats_observer.get_pool_stats()
        get_last_batch = getattr(self.invariant_checker, "get_last_batch", None)
        if callable(get_last_batch) and get_last_batch() is None:
            full_uncached = 0
            uncached_observation_ok = True
        else:
            try:
                full_uncached, _ = self.invariant_checker._get_total_uncached_sizes()
                uncached_observation_ok = True
            except Exception:
                self.logger.warning(
                    "p2p_allocator_snapshot_uncached_failed", exc_info=True
                )
                full_uncached = 0
                uncached_observation_ok = False
        token_allocator = self.pool_stats_observer.token_to_kv_pool_allocator
        full_total = self._call_int(
            token_allocator,
            "size",
            self._call_int(self.pool_stats_observer, "max_total_num_tokens"),
        )
        full_available = int(pool_stats.full_available_size)
        full_evictable = int(pool_stats.full_evictable_size)
        full_protected = self._full_protected(pool_stats)
        full_session_held = self._full_session_held(pool_stats)
        full_uncached = int(full_uncached)
        full_accounted = (
            full_available
            + full_evictable
            + full_protected
            + full_session_held
            + full_uncached
        )
        full_delta = full_total - full_accounted
        full_ok = (
            full_delta >= 0
            and min(
                full_total,
                full_available,
                full_evictable,
                full_protected,
                full_session_held,
                full_uncached,
            )
            >= 0
        )

        snapshot = {
            "timestamp": self.wall_clock(),
            "tp_rank": int(
                getattr(self.ps, "attn_tp_rank", getattr(self.ps, "tp_rank", 0))
            ),
            "pp_rank": int(getattr(self.ps, "pp_rank", 0)),
            "dp_rank": int(getattr(self.ps, "dp_rank", 0) or 0),
            "full_total": full_total,
            "full_available": full_available,
            "full_evictable": full_evictable,
            "full_protected": full_protected,
            "full_session_held": full_session_held,
            "full_uncached": full_uncached,
            "full_token_usage": float(pool_stats.full_token_usage),
            "full_accounting_delta": full_delta,
            "full_unobserved_allocated": max(full_delta, 0),
            "full_overaccounted": max(-full_delta, 0),
            "accounting_complete": full_delta == 0,
            "uncached_observation_ok": uncached_observation_ok,
        }

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        mamba_ok = True
        if mamba_allocator is not None:
            mamba_total = self._call_int(mamba_allocator, "size")
            mamba_available = self._call_int(mamba_allocator, "available_size")
            mamba_used = mamba_total - mamba_available
            mamba_evictable = int(pool_stats.mamba_evictable_size or 0)
            mamba_protected = self._call_int(
                self.pool_stats_observer.tree_cache, "mamba_protected_size"
            )
            mamba_session_held = self._call_int(
                self.pool_stats_observer, "session_held_mamba_slots"
            )
            mamba_ok = min(mamba_total, mamba_available, mamba_used) >= 0
            snapshot.update(
                mamba_total=mamba_total,
                mamba_available=mamba_available,
                mamba_used=mamba_used,
                mamba_evictable=mamba_evictable,
                mamba_protected=mamba_protected,
                mamba_session_held=mamba_session_held,
            )

        snapshot["accounting_ok"] = full_ok and mamba_ok
        return snapshot

    def maybe_log(self) -> dict | None:
        now = self.clock()
        if (
            self._last_log_time is not None
            and now - self._last_log_time < self.interval_s
        ):
            return None
        self._last_log_time = now
        try:
            snapshot = self._build_snapshot()
            self.logger.info(
                "p2p_allocator_snapshot %s", json.dumps(snapshot, sort_keys=True)
            )
            return snapshot
        except Exception:
            self.logger.exception("p2p_allocator_snapshot_failed")
