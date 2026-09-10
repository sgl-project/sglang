"""Deterministic no-IO storage simulator for HiCache benchmarking.

Stores KEYS ONLY (served KV is garbage — benchmark-only, for ignore_eos
workloads where nothing reads the generated text) and sleeps
``bytes / bandwidth + op latency`` on the calling backup/prefetch thread,
so pipeline dynamics are preserved while the medium is exactly
reproducible. Bandwidth is per scheduler rank (each rank ships its own
shard). extra_config knobs: ``sim_write_gbps`` (default 5.0; <=0 =
infinite), ``sim_read_gbps`` (default = write), ``sim_op_latency_us``
(default 100, applied to exists queries too).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)


class SimHiCacheStorage(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig):
        extra = storage_config.extra_config or {}
        self.write_gbps = float(extra.get("sim_write_gbps", 5.0))
        self.read_gbps = float(extra.get("sim_read_gbps", self.write_gbps))
        self.op_latency_s = float(extra.get("sim_op_latency_us", 100.0)) * 1e-6
        # Scoped key -> True. Keys only; there are no bytes to store.
        self._keys: set[str] = set()
        self._lock = threading.Lock()
        logger.info(
            "SimHiCacheStorage: write=%.2f GB/s read=%.2f GB/s latency=%.0fus "
            "(per rank; <=0 GB/s = infinite)",
            self.write_gbps,
            self.read_gbps,
            self.op_latency_s * 1e6,
        )

    # ---- timing model ----

    def _sleep_io(self, num_bytes: int, gbps: float) -> None:
        delay = self.op_latency_s
        if gbps > 0:
            delay += num_bytes / (gbps * 1e9)
        if delay > 0:
            time.sleep(delay)

    def _pool_bytes(self, name: PoolName, num_slots: int) -> int:
        return num_slots * self.registered_pools[name].size_per_token

    @staticmethod
    def _scoped(name: PoolName, key: str) -> str:
        return key if name == PoolName.KV else f"{key}.{name}"

    # ---- single-key surface (generic controller paths) ----

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        with self._lock:
            present = key in self._keys
        return target_location if present else None

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        with self._lock:
            self._keys.add(key)
        return True

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._keys

    # ---- batch v0/v1 (generic page funcs) ----

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        locations = target_locations or [None] * len(keys)
        with self._lock:
            present = [k in self._keys for k in keys]
        num_bytes = sum(
            loc.numel() * loc.element_size()
            for loc, hit in zip(locations, present)
            if hit and loc is not None
        )
        self._sleep_io(num_bytes, self.read_gbps)
        return [loc if hit else None for loc, hit in zip(locations, present)]

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        num_bytes = sum(v.numel() * v.element_size() for v in values or ())
        self._sleep_io(num_bytes, self.write_gbps)
        with self._lock:
            self._keys.update(keys)
        return True

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        with self._lock:
            present = [k in self._keys for k in keys]
        hit_slots = (len(host_indices) // max(1, len(keys))) * sum(present)
        self._sleep_io(hit_slots * self.mem_pool_host.size_per_token, self.read_gbps)
        return present

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        self._sleep_io(
            len(host_indices) * self.mem_pool_host.size_per_token, self.write_gbps
        )
        with self._lock:
            self._keys.update(keys)
        return [True] * len(keys)

    # ---- batch v2 (hybrid multi-pool paths) ----

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        self._sleep_io(0, self.read_gbps)
        with self._lock:
            for i, key in enumerate(keys):
                if key not in self._keys:
                    return i
        return len(keys)

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """Same fold semantics as HiCacheFile.batch_exists_v2, over the
        in-memory key set."""
        self._sleep_io(0, self.read_gbps)
        with self._lock:
            snapshot = self._keys.copy()

        kv_pages = next(
            (i for i in range(len(keys)) if keys[i] not in snapshot), len(keys)
        )
        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (
                        i
                        for i in range(kv_pages)
                        if self._scoped(name, keys[i]) not in snapshot
                    ),
                    kv_pages,
                )
            else:  # trailing_pages
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        self._scoped(name, keys[i]) in snapshot
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _batch_v2(
        self, transfers: List[PoolTransfer], gbps: float, record: bool
    ) -> dict[str, List[bool]]:
        results: dict[str, List[bool]] = {}
        num_bytes = 0
        for t in transfers:
            t_keys = t.keys or []
            if t.host_indices is not None:
                num_bytes += self._pool_bytes(t.name, len(t.host_indices))
            scoped = [self._scoped(t.name, k) for k in t_keys]
            with self._lock:
                if record:
                    self._keys.update(scoped)
                    results[t.name] = [True] * len(t_keys)
                else:
                    results[t.name] = [k in self._keys for k in scoped]
        self._sleep_io(num_bytes, gbps)
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_v2(transfers, self.read_gbps, record=False)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_v2(transfers, self.write_gbps, record=True)

    # ---- misc ----

    def clear(self) -> bool:
        with self._lock:
            self._keys.clear()
        return True

    def get_stats(self):
        return None
