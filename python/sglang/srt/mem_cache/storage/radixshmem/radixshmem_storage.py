"""RadixShmem as a HiCache L3 backend.

radixshmem keeps its own index and SlotStore; sglang keeps its host staging
pool. Pages are copied between the two, so neither side holds the other's
memory. radix-server starts with only a byte budget; this backend hands it
the slot geometry (page size, bytes of the largest host-pool page) on the
first storage call, once every host pool is registered. Keys are sglang's
chained page hashes folded to uint64 and salted per pool, PP/CP rank, and
TP rank unless the pool is rank-replicated (MLA). An ALL_PAGES pool is one
radix path per page (prefix query, run pull); a TRAILING_PAGES pool keeps
each page as its own one-block path, since only a window at the end of the
sequence exists. A pool without bytes (DeepSeek-V4's logical KV anchor) is
never stored.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np
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

def _salt(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "little")


def _page_bytes(host_pool: Any) -> int:
    meta = host_pool.get_page_buffer_meta(torch.arange(host_pool.page_size))
    return sum(meta[1]) if meta else 0


class _Pool:
    def __init__(self, host_pool: Any, salt: int, page_bytes: int):
        self.host_pool = host_pool
        self.salt = salt
        self.page_bytes = page_bytes


class RadixShmemStorage(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, mem_pool_host: Any = None):
        import shmradix

        self.shmradix = shmradix
        cfg = storage_config.extra_config or {}
        self.client = shmradix.RadixClient(
            cfg.get("radixshmem_name", "/shmradix"),
            endpoint=cfg.get("radixshmem_endpoint"),
            max_outstanding=int(cfg.get("radixshmem_max_outstanding", 256)),
        )
        self.ready_timeout_s = float(cfg.get("radixshmem_timeout_s", 300))
        self.pull_timeout_ms = int(cfg.get("radixshmem_pull_timeout_ms", 30000))
        self.full = shmradix.ComponentType.FULL
        self.full_mask = int(shmradix.COMPONENT_MASK_FULL)
        self.insert_ok = shmradix.InsertError.OK
        self._salt_parts = (
            storage_config.model_name,
            f"pp{storage_config.pp_rank}",
            f"cp{storage_config.attn_cp_rank}",
            "" if storage_config.is_mla_model else f"tp{storage_config.tp_rank}",
        )
        self.pools: Dict[Any, _Pool] = {}
        self.slot_bytes = 0
        self.index = None
        self._attach_lock = threading.Lock()

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        self.mem_pool_host = mem_pool_host
        self._register(PoolName.KV, mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool: Any, host_pool_name) -> None:
        self._register(host_pool_name, host_pool)

    def _register(self, name, host_pool: Any) -> None:
        page_bytes = _page_bytes(host_pool)
        if self.index is not None and page_bytes > self.slot_bytes:
            raise ValueError(
                f"pool {name}: {page_bytes} B per page exceeds the slot_bytes="
                f"{self.slot_bytes} the server was configured with"
            )
        salt = _salt(*self._salt_parts, str(name))
        self.pools[name] = _Pool(host_pool, salt, page_bytes)

    def _attach(self) -> None:
        """First storage call: hand the server the geometry of the registered
        pools and attach the index and SlotStore. Idempotent across ranks."""
        if self.index is not None:
            return
        with self._attach_lock:
            if self.index is not None:
                return
            self.block_size = int(self.pools[PoolName.KV].host_pool.page_size)
            slot_bytes = max(p.page_bytes for p in self.pools.values())
            if slot_bytes == 0:
                raise RuntimeError("radixshmem: no registered host pool carries data")
            geometry = self.shmradix.Geometry(
                block_size=self.block_size, full_slot_bytes=slot_bytes
            )
            self.client.configure(geometry)
            info = self.client.wait_ready(self.ready_timeout_s)
            self.store = self.client.store
            self._data_view = self.store.data_view()
            self._data_base = np.frombuffer(self._data_view, dtype=np.uint8).ctypes.data
            self.distributed = bool(self.client.is_distributed())
            self.slot_bytes = slot_bytes
            logger.info(
                "RadixShmemStorage ready: block_size=%d slot_bytes=%d slots=%d "
                "distributed=%s pools=%s",
                self.block_size,
                slot_bytes,
                info.geometry["pools"]["full"]["num_slots"],
                self.distributed,
                {str(name): pool.page_bytes for name, pool in self.pools.items()},
            )
            self.index = self.client.index

    # ---- HiCacheStorage: KV pool ----

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        return self._hit(self.pools[PoolName.KV], keys, extra_info)

    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        return self._store(self.pools[PoolName.KV], keys, host_indices, extra_info)

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        return self._load(self.pools[PoolName.KV], keys, host_indices, extra_info)

    # ---- HiCacheStorage: extra pools ----

    def batch_exists_v2(
        self, keys, pool_transfers=None, extra_info=None
    ) -> PoolTransferResult:
        kv_pages = self.batch_exists(keys, extra_info)
        hits = {PoolName.KV: kv_pages} if kv_pages else {}
        restorable = list(range(1, kv_pages + 1))
        for transfer in pool_transfers or []:
            if not restorable:
                break
            pool = self.pools[transfer.name]
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = self._hit(pool, keys[:kv_pages], extra_info)
                stops = set(range(1, boundary + 1))
            else:
                # A prefix is restorable when the window ending there is complete.
                present = self._present(pool, keys[:kv_pages])
                window = max(1, len(transfer.keys or ()))
                stops = {
                    n
                    for n in range(1, kv_pages + 1)
                    if all(present[max(0, n - window) : n])
                }
                boundary = max(stops, default=0)
            if boundary:
                hits[transfer.name] = boundary
            restorable = [n for n in restorable if n in stops]
        final = restorable[-1] if restorable else 0
        return PoolTransferResult(final, hits, restorable)

    def batch_set_v2(self, transfers, extra_info=None) -> dict:
        return {t.name: self._write(t, extra_info) for t in transfers}

    def batch_get_v2(self, transfers, extra_info=None) -> dict:
        return {t.name: self._read(t, extra_info) for t in transfers}

    def _write(self, transfer: PoolTransfer, extra_info) -> List[bool]:
        pool = self.pools[transfer.name]
        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            return self._store(pool, transfer.keys, transfer.host_indices, extra_info)
        return self._store_pages(pool, transfer.keys, transfer.host_indices)

    def _read(self, transfer: PoolTransfer, extra_info) -> List[bool]:
        pool = self.pools[transfer.name]
        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            return self._load(pool, transfer.keys, transfer.host_indices, extra_info)
        return self._load_pages(pool, transfer.keys, transfer.host_indices)

    # ---- keys ----

    def _chain(self, pool: _Pool, keys, extra_info) -> tuple[np.ndarray, int]:
        # sglang passes None for an empty prefix; without
        # hicache_storage_pass_prefix_keys it passes None always and every
        # write becomes its own root-anchored path (correct, fewer hits).
        prefix = (extra_info.prefix_keys if extra_info is not None else None) or []
        salt = pool.salt
        chain = np.fromiter(
            (int(k[:16], 16) ^ salt for k in (*prefix, *keys)),
            dtype=np.uint64,
            count=len(prefix) + len(keys),
        )
        return chain, len(prefix)

    def _single(self, pool: _Pool, key: str) -> np.ndarray:
        return np.array([int(key[:16], 16) ^ pool.salt], dtype=np.uint64)

    # ---- radixshmem primitives ----

    def _query(self, chain: np.ndarray) -> int:
        q = self.index.query(
            chain, self.full_mask, local_only=False, lock=False, update_meta=True
        )
        hit = int(q.common_hit) if int(q.status) == 0 else 0
        if q.finalize:
            q.finalize()
        return hit

    def _pull(self, chain: np.ndarray):
        """Pull the chain's remote run into local slots; None on timeout."""
        job = self.client.pull_async(
            chain, self.full_mask, lock=True, timeout_ms=self.pull_timeout_ms
        )
        try:
            return job.wait(self.pull_timeout_ms / 1000 + 5)
        except TimeoutError:
            job.cancel()
            return None

    def _allocate(self, n: int) -> Optional[np.ndarray]:
        slots = np.asarray(self.index.allocate_slots(n, self.full), dtype=np.int32)
        if len(slots) == n:
            return slots
        if len(slots):
            self.index.recycle_slots(slots, self.full)
        return None

    def _copy(self, host_pool: Any, host_indices: torch.Tensor, slots, to_slot: bool):
        ptrs, sizes = host_pool.get_page_buffer_meta(host_indices)
        per = len(ptrs) // len(slots)
        for i, slot in enumerate(slots):
            dst = self._data_base + int(self.store.slot_offset(int(slot), self.full))
            for j in range(i * per, (i + 1) * per):
                if to_slot:
                    ctypes.memmove(dst, ptrs[j], sizes[j])
                else:
                    ctypes.memmove(ptrs[j], dst, sizes[j])
                dst += sizes[j]

    # ---- ALL_PAGES pools: one radix path ----

    def _hit(self, pool: _Pool, keys, extra_info) -> int:
        self._attach()
        if not keys:
            return 0
        if pool.page_bytes == 0:
            return len(keys)
        chain, start = self._chain(pool, keys, extra_info)
        return max(0, min(self._query(chain) - start, len(keys)))

    def _store(self, pool: _Pool, keys, host_indices, extra_info) -> List[bool]:
        self._attach()
        n = len(keys)
        if n == 0:
            return []
        if pool.page_bytes == 0:
            return [True] * n
        chain, start = self._chain(pool, keys, extra_info)
        slots = self._allocate(n)
        if slots is None:
            return [False] * n
        self._copy(pool.host_pool, host_indices, slots, to_slot=True)
        result = self.index.insert(chain, slots, start, True, component=self.full)
        ok = result.error == self.insert_ok
        if ok and self.distributed:
            self.index.flush()
        return [ok] * n

    def _load(self, pool: _Pool, keys, host_indices, extra_info) -> List[bool]:
        self._attach()
        n = len(keys)
        if n == 0:
            return []
        if pool.page_bytes == 0:
            return [True] * n
        chain, start = self._chain(pool, keys, extra_info)
        result = self._pull(chain)
        if result is None:
            return [False] * n
        try:
            hit = max(0, min(int(result.common_hit) - start, n))
            if hit:
                page_size = pool.host_pool.page_size
                self._copy(
                    pool.host_pool,
                    host_indices[: hit * page_size],
                    result.full_slots[start : start + hit],
                    to_slot=False,
                )
        finally:
            result.finalize()
        return [True] * hit + [False] * (n - hit)

    # ---- TRAILING_PAGES pools: one-block path per page ----

    def _present(self, pool: _Pool, keys) -> List[bool]:
        self._attach()
        return [self._query(self._single(pool, key)) == 1 for key in keys]

    def _store_pages(self, pool: _Pool, keys, host_indices) -> List[bool]:
        self._attach()
        n = len(keys)
        if n == 0:
            return []
        slots = self._allocate(n)
        if slots is None:
            return [False] * n
        self._copy(pool.host_pool, host_indices, slots, to_slot=True)
        ok = [
            self.index.insert(
                self._single(pool, key), slots[i : i + 1], 0, True, component=self.full
            ).error
            == self.insert_ok
            for i, key in enumerate(keys)
        ]
        if any(ok) and self.distributed:
            self.index.flush()
        return ok

    def _load_pages(self, pool: _Pool, keys, host_indices) -> List[bool]:
        self._attach()
        page_size = pool.host_pool.page_size
        ok = []
        for i, key in enumerate(keys):
            result = self._pull(self._single(pool, key))
            hit = result is not None and int(result.common_hit) == 1
            if hit:
                self._copy(
                    pool.host_pool,
                    host_indices[i * page_size : (i + 1) * page_size],
                    result.full_slots[:1],
                    to_slot=False,
                )
            if result is not None:
                result.finalize()
            ok.append(hit)
        return ok

    # ---- misc ----

    def clear(self) -> bool:
        self._attach()
        self.index.reset()
        return True

    def close(self) -> None:
        self.client.close()

    def get(self, key, target_location=None, target_sizes=None):
        raise NotImplementedError("radixshmem uses the zero-copy v1/v2 interface")

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        raise NotImplementedError("radixshmem uses the zero-copy v1/v2 interface")

    def set(self, key, value=None, target_location=None, target_sizes=None):
        raise NotImplementedError("radixshmem uses the zero-copy v1/v2 interface")

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        raise NotImplementedError("radixshmem uses the zero-copy v1/v2 interface")

    def exists(self, key: str) -> bool:
        raise NotImplementedError("radixshmem needs the page chain; use batch_exists")
