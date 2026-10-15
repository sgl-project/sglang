"""RadixShmem as a HiCache L3 backend.

radixshmem keeps its own index and SlotStore; sglang keeps its host staging
pool and pages are copied between the two. One slot holds the whole page of
the attention TP/CP group: a head-sharded pool stores every attention-TP
rank's part side by side, a rank-replicated pool (MLA) stores one copy.
Keys are sglang's chained page hashes folded to uint64 and salted per model,
PP stage and pool, so every rank of the group reads the same slot.

Group rank 0 (attention TP 0, CP 0) is the only rank that touches the index:
it answers hit queries (the others report every page present and sglang
keeps the minimum), allocates and publishes slots, and pins slots while the
group copies them. Writes and reads of a shared slot are collective over the
group's gloo groups, handed over by the controller in set_shard_groups().
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


def _noop() -> None:
    pass


class _Solo:
    rank = 0
    size = 1

    def bcast(self, tensor: torch.Tensor) -> None:
        pass

    def min(self, tensor: torch.Tensor) -> None:
        pass


class _Comm:
    def __init__(self, group):
        import torch.distributed as dist

        self._dist = dist
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self._src = dist.get_global_rank(group, 0)

    def bcast(self, tensor: torch.Tensor) -> None:
        self._dist.broadcast(tensor, self._src, group=self.group)

    def min(self, tensor: torch.Tensor) -> None:
        self._dist.all_reduce(tensor, op=self._dist.ReduceOp.MIN, group=self.group)


def _as_comm(group):
    if group is None:
        return _Solo()
    return group if hasattr(group, "bcast") else _Comm(group)


class _Pool:
    def __init__(
        self,
        name,
        host_pool: Any,
        page_bytes: int,
        parts: int,
        part: int,
        salt: int,
    ):
        self.name = name
        self.host_pool = host_pool
        self.page_bytes = page_bytes
        self.sharded = parts > 1
        self.slot_bytes = page_bytes * parts
        self.offset = part * page_bytes
        self.salt = salt


class RadixShmemStorage(HiCacheStorage):
    shard_collective = True

    def __init__(self, storage_config: HiCacheStorageConfig, mem_pool_host: Any = None):
        import radixshmem

        self.radixshmem = radixshmem
        self.config = storage_config
        cfg = storage_config.extra_config or {}
        self.server_name = cfg.get("radixshmem_name", "/radixshmem")
        self.endpoint = cfg.get("radixshmem_endpoint")
        self.max_outstanding = int(cfg.get("radixshmem_max_outstanding", 256))
        self.ready_timeout_s = float(cfg.get("radixshmem_timeout_s", 300))
        self.pull_timeout_ms = int(cfg.get("radixshmem_pull_timeout_ms", 30000))
        self.full = radixshmem.ComponentType.FULL
        self.full_mask = int(radixshmem.COMPONENT_MASK_FULL)
        self.insert_ok = radixshmem.InsertError.OK
        self.host_pools: Dict[Any, Any] = {}
        self.pools: Dict[Any, _Pool] = {}
        self.client = None
        self.index = None
        self.slot_bytes = 0
        self._attach_lock = threading.Lock()
        self.set_shard_groups(None, None)

    def read_barrier(self, go: bool) -> bool:
        flag = torch.tensor([int(go)], dtype=torch.int64)
        self.read_comm.min(flag)
        return bool(flag[0])

    def set_shard_groups(self, write_group, read_group) -> None:
        if self.index is not None:
            raise RuntimeError("radixshmem: shard groups must be set before first use")
        self.write_comm = _as_comm(write_group)
        self.read_comm = _as_comm(read_group)
        cp_size = max(1, int(self.config.attn_cp_size)) if read_group else 1
        if self.read_comm.size % cp_size:
            raise ValueError(
                f"radixshmem: shard group of {self.read_comm.size} ranks is not "
                f"a multiple of attn_cp_size={cp_size}"
            )
        self.tp_parts = self.read_comm.size // cp_size
        self.tp_part = self.read_comm.rank % self.tp_parts
        self.cp_rank = self.read_comm.rank // self.tp_parts
        self.leader = self.read_comm.rank == 0

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        self.mem_pool_host = mem_pool_host
        self._register(PoolName.KV, mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool: Any, host_pool_name) -> None:
        self._register(host_pool_name, host_pool)

    def _register(self, name, host_pool: Any) -> None:
        self.host_pools[name] = host_pool
        if self.index is None:
            return
        pool = self._make_pool(name, host_pool)
        if pool.slot_bytes > self.slot_bytes:
            raise ValueError(
                f"pool {name}: {pool.slot_bytes} B per page exceeds the slot_bytes="
                f"{self.slot_bytes} the server was configured with"
            )
        self.pools[name] = pool

    def _sharded(self, name) -> bool:
        return not self.config.is_mla_model or name == PoolName.MAMBA

    def _make_pool(self, name, host_pool: Any) -> _Pool:
        sharded = self._sharded(name) and self.tp_parts > 1
        parts = self.tp_parts if sharded else 1
        salt = _salt(
            self.config.model_name,
            f"pp{self.config.pp_rank}/{self.config.pp_size}",
            f"tp{parts}",
            str(name),
        )
        part = self.tp_part if sharded else 0
        return _Pool(name, host_pool, _page_bytes(host_pool), parts, part, salt)

    def _attach(self) -> None:
        if self.index is not None:
            return
        with self._attach_lock:
            if self.index is not None:
                return
            pools = {
                name: self._make_pool(name, hp) for name, hp in self.host_pools.items()
            }
            slot_bytes = max(p.slot_bytes for p in pools.values())
            if slot_bytes == 0:
                raise RuntimeError("radixshmem: no registered host pool carries data")
            block_size = int(pools[PoolName.KV].host_pool.page_size)
            client = self.radixshmem.RadixClient(
                self.server_name,
                endpoint=self.endpoint,
                max_outstanding=self.max_outstanding,
                local_read_only=not self.leader,
            )
            client.configure(
                self.radixshmem.Geometry(
                    block_size=block_size, full_slot_bytes=slot_bytes
                )
            )
            info = client.wait_ready(self.ready_timeout_s)
            self.client = client
            self.store = client.store
            self._data_view = self.store.data_view()
            self._data_base = np.frombuffer(self._data_view, dtype=np.uint8).ctypes.data
            self.distributed = self.leader and bool(client.is_distributed())
            self.block_size = block_size
            self.slot_bytes = slot_bytes
            self.pools = pools
            logger.info(
                "RadixShmemStorage ready: rank %d/%d (tp part %d/%d, cp %d) "
                "block_size=%d slot_bytes=%d slots=%d distributed=%s pools=%s",
                self.read_comm.rank,
                self.read_comm.size,
                self.tp_part,
                self.tp_parts,
                self.cp_rank,
                block_size,
                slot_bytes,
                info.geometry["pools"]["full"]["num_slots"],
                self.distributed,
                {str(n): (p.page_bytes, p.slot_bytes) for n, p in pools.items()},
            )
            self.index = client.index

    # ---- HiCacheStorage: KV pool ----

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        self._attach()
        return self._hit(self.pools[PoolName.KV], keys, extra_info)

    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        self._attach()
        return self._store(self.pools[PoolName.KV], keys, host_indices, extra_info)

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        self._attach()
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
        self._attach()
        return {t.name: self._write_transfer(t, extra_info) for t in transfers}

    def batch_get_v2(self, transfers, extra_info=None) -> dict:
        self._attach()
        return {t.name: self._read_transfer(t, extra_info) for t in transfers}

    def _write_transfer(self, transfer: PoolTransfer, extra_info) -> List[bool]:
        pool = self.pools[transfer.name]
        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            return self._store(pool, transfer.keys, transfer.host_indices, extra_info)
        return self._store_pages(pool, transfer.keys, transfer.host_indices)

    def _read_transfer(self, transfer: PoolTransfer, extra_info) -> List[bool]:
        pool = self.pools[transfer.name]
        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            return self._load(pool, transfer.keys, transfer.host_indices, extra_info)
        return self._load_pages(pool, transfer.keys, transfer.host_indices)

    # ---- keys ----

    def _chain(self, pool: _Pool, keys, extra_info) -> Tuple[np.ndarray, int]:
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

    # ---- index (leader only) ----

    def _query(self, chain: np.ndarray) -> int:
        q = self.index.query(
            chain, self.full_mask, local_only=False, lock=False, update_meta=True
        )
        hit = int(q.common_hit) if int(q.status) == 0 else 0
        if q.finalize:
            q.finalize()
        return hit

    def _fetch(self, chain: np.ndarray) -> Tuple[np.ndarray, Callable]:
        job = self.client.pull_async(
            chain, self.full_mask, lock=True, timeout_ms=self.pull_timeout_ms
        )
        try:
            result = job.wait(self.pull_timeout_ms / 1000 + 5)
        except TimeoutError:
            job.cancel()
            return np.empty(0, dtype=np.int32), _noop
        return np.asarray(result.full_slots, dtype=np.int32), result.finalize

    def _allocate(self, n: int) -> Optional[np.ndarray]:
        slots = np.asarray(self.index.allocate_slots(n, self.full), dtype=np.int32)
        if len(slots) == n:
            return slots
        if len(slots):
            self.index.recycle_slots(slots, self.full)
        return None

    def _published(self, chain: np.ndarray, slots: np.ndarray, result) -> bool:
        if result.error != self.insert_ok:
            return False
        if len(getattr(result, "unused_slots", ())) < len(slots):
            return True
        q = self.index.query(
            chain, self.full_mask, local_only=True, lock=False, update_meta=False
        )
        return int(q.status) == 0 and int(q.common_hit) >= len(chain)

    # ---- collective slot traffic ----

    def _write(
        self,
        pool: _Pool,
        n: int,
        copy: Callable[[np.ndarray], None],
        publish: Callable[[np.ndarray], List[bool]],
    ) -> List[bool]:
        if not pool.sharded and not self.leader:
            return [True] * n
        comm = self.write_comm if pool.sharded else _Solo()
        buf = torch.full((n + 1,), -1, dtype=torch.int64)
        if self.leader:
            slots = self._allocate(n)
            if slots is not None:
                buf[0] = 1
                buf[1:] = torch.from_numpy(slots.astype(np.int64))
        comm.bcast(buf)
        if int(buf[0]) != 1:
            return [False] * n
        slots = buf[1:].numpy().astype(np.int32)
        ok = torch.ones(1, dtype=torch.int64)
        if self.cp_rank == 0:
            try:
                copy(slots)
            except Exception:
                logger.exception("radixshmem: copy into slots failed")
                ok[0] = 0
        comm.min(ok)
        result = torch.zeros(n, dtype=torch.int64)
        if self.leader:
            if int(ok[0]):
                try:
                    result[:] = torch.tensor(publish(slots), dtype=torch.int64)
                except Exception:
                    logger.exception("radixshmem: publish failed")
                if bool(result.any()) and self.distributed:
                    self.index.flush()
            else:
                self.index.recycle_slots(slots, self.full)
        comm.bcast(result)
        return [bool(x) for x in result.tolist()]

    def _read(
        self,
        n: int,
        fetch: Callable[[], Tuple[Sequence[int], List[Callable]]],
        copy: Callable[[List[Tuple[int, int]]], None],
    ) -> List[bool]:
        buf = torch.full((n,), -1, dtype=torch.int64)
        finalizers: List[Callable] = []
        if self.leader:
            try:
                slots, finalizers = fetch()
                if len(slots):
                    buf[: len(slots)] = torch.as_tensor(
                        np.asarray(slots, dtype=np.int64)
                    )
            except Exception:
                logger.exception("radixshmem: fetch failed")
        self.read_comm.bcast(buf)
        slots = buf.numpy()
        ok = torch.from_numpy((slots >= 0).astype(np.int64))
        try:
            copy([(i, int(s)) for i, s in enumerate(slots) if s >= 0])
        except Exception:
            logger.exception("radixshmem: copy out of slots failed")
            ok.zero_()
        self.read_comm.min(ok)
        for finalize in finalizers:
            finalize()
        return [bool(x) for x in ok.tolist()]

    def _copy(
        self,
        pool: _Pool,
        host_indices: torch.Tensor,
        pairs: Sequence[Tuple[int, int]],
        to_slot: bool,
    ) -> None:
        if not pairs:
            return
        ptrs, sizes = pool.host_pool.get_page_buffer_meta(host_indices)
        per = len(ptrs) // (len(host_indices) // pool.host_pool.page_size)
        for i, slot in pairs:
            at = (
                self._data_base
                + int(self.store.slot_offset(int(slot), self.full))
                + pool.offset
            )
            for j in range(i * per, (i + 1) * per):
                if to_slot:
                    ctypes.memmove(at, ptrs[j], sizes[j])
                else:
                    ctypes.memmove(ptrs[j], at, sizes[j])
                at += sizes[j]

    # ---- ALL_PAGES pools: one radix path ----

    def _hit(self, pool: _Pool, keys, extra_info) -> int:
        if not keys:
            return 0
        if pool.slot_bytes == 0 or not self.leader:
            return len(keys)
        chain, start = self._chain(pool, keys, extra_info)
        return max(0, min(self._query(chain) - start, len(keys)))

    def _store(self, pool: _Pool, keys, host_indices, extra_info) -> List[bool]:
        n = len(keys)
        if n == 0:
            return []
        if pool.slot_bytes == 0:
            return [True] * n
        chain, start = self._chain(pool, keys, extra_info)

        def publish(slots):
            result = self.index.insert(chain, slots, start, True, component=self.full)
            return [self._published(chain, slots, result)] * n

        return self._write(
            pool,
            n,
            lambda slots: self._copy(pool, host_indices, list(enumerate(slots)), True),
            publish,
        )

    def _load(self, pool: _Pool, keys, host_indices, extra_info) -> List[bool]:
        n = len(keys)
        if n == 0:
            return []
        if pool.slot_bytes == 0:
            return [True] * n
        chain, start = self._chain(pool, keys, extra_info)

        def fetch():
            slots, finalize = self._fetch(chain)
            return slots[start : start + n], [finalize]

        ok = self._read(
            n, fetch, lambda pairs: self._copy(pool, host_indices, pairs, False)
        )
        hit = ok.index(False) if False in ok else n
        return [True] * hit + [False] * (n - hit)

    # ---- TRAILING_PAGES pools: one-block path per page ----

    def _present(self, pool: _Pool, keys) -> List[bool]:
        if not self.leader:
            return [True] * len(keys)
        return [self._query(self._single(pool, key)) == 1 for key in keys]

    def _store_pages(self, pool: _Pool, keys, host_indices) -> List[bool]:
        n = len(keys)
        if n == 0:
            return []
        chains = [self._single(pool, key) for key in keys]

        def publish(slots):
            out = []
            for chain, slot in zip(chains, slots):
                one = np.array([slot], dtype=np.int32)
                result = self.index.insert(chain, one, 0, True, component=self.full)
                out.append(self._published(chain, one, result))
            return out

        return self._write(
            pool,
            n,
            lambda slots: self._copy(pool, host_indices, list(enumerate(slots)), True),
            publish,
        )

    def _load_pages(self, pool: _Pool, keys, host_indices) -> List[bool]:
        n = len(keys)
        if n == 0:
            return []

        def fetch():
            slots, finalizers = [], []
            for key in keys:
                got, finalize = self._fetch(self._single(pool, key))
                slots.append(int(got[0]) if len(got) else -1)
                finalizers.append(finalize)
            return slots, finalizers

        return self._read(
            n, fetch, lambda pairs: self._copy(pool, host_indices, pairs, False)
        )

    # ---- misc ----

    def clear(self) -> bool:
        self._attach()
        if self.leader:
            self.index.reset()
        return True

    def close(self) -> None:
        if self.client is not None:
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
