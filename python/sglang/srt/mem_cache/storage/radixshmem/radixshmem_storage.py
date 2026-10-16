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
    def __init__(self, group, src: int):
        import torch.distributed as dist

        self._dist = dist
        self.group = group
        self._src = src

    def bcast(self, tensor: torch.Tensor) -> None:
        self._dist.broadcast(tensor, self._src, group=self.group)

    def min(self, tensor: torch.Tensor) -> None:
        self._dist.all_reduce(tensor, op=self._dist.ReduceOp.MIN, group=self.group)


def _as_comm(group, src: int):
    if group is None:
        return _Solo()
    return group if hasattr(group, "bcast") else _Comm(group, src)


class _Part:
    def __init__(
        self,
        view: Callable[[int], torch.Tensor],
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        axis: Optional[int],
        spans: Sequence[Tuple[int, int, int]],
        write: bool,
    ):
        self.view = view
        self.shape = tuple(int(x) for x in shape)
        self.dtype = dtype
        self.axis = axis
        self.spans = list(spans)
        self.write = write
        self.nbytes = int(np.prod(self.shape)) * dtype.itemsize
        self.offset = 0


class _Layout:
    def __init__(self, parts: List[_Part], local_bytes: int):
        offset = 0
        for part in parts:
            part.offset = offset
            offset += part.nbytes
        self.parts = parts
        self.slot_bytes = offset
        self.local_bytes = local_bytes
        self.tag = ";".join(f"{p.dtype}{p.shape}@{p.axis}" for p in parts)


def _narrow(t: torch.Tensor, axis: Optional[int], start: int, length: int):
    return t if axis is None else t.narrow(axis, start, length)


def _shard_spans(
    local: int, parts: int, part: int, groups
) -> List[Tuple[int, int, int]]:
    if not groups:
        return [(0, part * local, local)]
    spans, loc, glob = [], 0, 0
    for full in groups:
        sub = full // parts
        spans.append((loc, glob + part * sub, sub))
        loc += sub
        glob += full
    return spans


class _Pool:
    def __init__(
        self,
        name,
        host_pool: Any,
        page_bytes: int,
        parts: int,
        part: int,
        salt: Callable[[int], int],
        kind,
        mask: int,
        layout: Optional[_Layout] = None,
        writes: bool = True,
    ):
        self.name = name
        self.host_pool = host_pool
        self.page_bytes = page_bytes
        self.sharded = parts > 1 or layout is not None
        self.writes = writes
        self.layout = layout
        self.slot_bytes = layout.slot_bytes if layout else page_bytes * parts
        self.offset = 0 if layout else part * page_bytes
        self.salt = salt
        self.kind = kind
        self.mask = mask


class RadixShmemStorage(HiCacheStorage):
    shard_collective = True
    sidecars_after_kv = True

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
        self.components = {
            PoolName.SWA: (
                radixshmem.ComponentType.SWA,
                int(radixshmem.COMPONENT_MASK_SWA),
            ),
            PoolName.MAMBA: (
                radixshmem.ComponentType.MAMBA,
                int(radixshmem.COMPONENT_MASK_MAMBA),
            ),
        }
        self.swa_window_pages = int(cfg.get("radixshmem_swa_window_pages", 0))
        self.kv_heads_override = cfg.get("radixshmem_kv_heads")
        self.device_pools: Dict[Any, Any] = {}
        self.insert_ok = radixshmem.InsertError.OK
        self.host_pools: Dict[Any, Any] = {}
        self.pools: Dict[Any, _Pool] = {}
        self.client = None
        self.index = None
        self.slot_bytes = 0
        self.kind_slot_bytes: Dict[Any, int] = {}
        self._attach_lock = threading.Lock()
        self.set_shard_groups(None, None)

    def read_barrier(self, go: bool) -> bool:
        flag = torch.tensor([int(go)], dtype=torch.int64)
        self.read_comm.min(flag)
        return bool(flag[0])

    def set_shard_groups(
        self,
        write_group,
        read_group,
        *,
        tp_part: int = 0,
        tp_parts: int = 1,
        cp_rank: int = 0,
        cp_parts: int = 1,
        leader_rank: int = 0,
    ) -> None:
        if self.index is not None:
            raise RuntimeError("radixshmem: shard groups must be set before first use")
        if (write_group is None) != (tp_parts * cp_parts == 1):
            raise ValueError("radixshmem: shard groups do not match the shard identity")
        self.write_comm = _as_comm(write_group, leader_rank)
        self.read_comm = _as_comm(read_group, leader_rank)
        self.tp_part, self.tp_parts = int(tp_part), int(tp_parts)
        self.cp_rank, self.cp_parts = int(cp_rank), int(cp_parts)
        self.leader = self.tp_part == 0 and self.cp_rank == 0

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        self.mem_pool_host = mem_pool_host
        self._register(PoolName.KV, mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool: Any, host_pool_name) -> None:
        self._register(host_pool_name, host_pool)

    def describe_pool(self, name, device_pool: Any) -> None:
        self.device_pools[name] = getattr(device_pool, "mamba_pool", device_pool)

    def _register(self, name, host_pool: Any) -> None:
        self.host_pools[name] = host_pool
        if self.index is None:
            return
        pool = self._make_pool(name, host_pool, self.pools[PoolName.KV])
        if pool.slot_bytes > self.kind_slot_bytes.get(pool.kind, 0):
            raise ValueError(
                f"pool {name}: {pool.slot_bytes} B per page exceeds the "
                f"{pool.kind} slot_bytes the server was configured with"
            )
        self.pools[name] = pool

    def _sharded(self, name) -> bool:
        return not self.config.is_mla_model or name == PoolName.MAMBA

    def _swa_window(self, block_size: int) -> int:
        if self.swa_window_pages:
            return self.swa_window_pages
        try:
            from sglang.srt.runtime_context import process_model_config

            window = process_model_config().sliding_window_size
        except Exception:
            window = None
        return -(-int(window) // block_size) if window else 0

    def _kv_heads(self, name, local: int) -> Optional[int]:
        if self.tp_parts == 1:
            return local
        if self.kv_heads_override:
            return int(self.kv_heads_override)
        if name not in (PoolName.KV, PoolName.SWA):
            return None
        try:
            from sglang.srt.runtime_context import process_model_config

            model = process_model_config()
            text = model.hf_text_config
            if name == PoolName.SWA and hasattr(text, "swa_num_key_value_heads"):
                return int(text.swa_num_key_value_heads)
            return int(model.get_total_num_kv_heads())
        except Exception:
            return None

    def _mha_layout(self, name, host_pool: Any) -> Optional[_Layout]:
        from sglang.srt.mem_cache.pool_host.mha import (
            AsymmetricMHATokenToKVPoolHost,
            MHATokenToKVPoolHost,
        )

        if not isinstance(host_pool, MHATokenToKVPoolHost) or isinstance(
            host_pool, AsymmetricMHATokenToKVPoolHost
        ):
            return None
        kv = getattr(host_pool, "kv_buffer", None)
        if not isinstance(kv, torch.Tensor):
            return None
        heads, ps = int(host_pool.head_num), int(host_pool.page_size)
        total = self._kv_heads(name, heads)
        if not total:
            return None
        replicas = max(1, self.tp_parts // total)
        if heads != max(1, total // self.tp_parts) or self.tp_parts % replicas:
            return None
        layout = host_pool.layout
        if layout == "page_first":
            view = lambda t: kv[:, t : t + ps].permute(0, 2, 1, 3, 4)
        elif layout == "layer_first":
            view = lambda t: kv[:, :, t : t + ps]
        elif layout == "page_first_direct":
            view = lambda t: kv[:, t // ps]
        elif layout == "page_head":
            view = lambda t: kv[:, t // ps].permute(0, 3, 2, 1, 4)
        else:
            return None
        shape = (2, int(host_pool.layer_num), ps, total, int(host_pool.head_dim))
        group = self.tp_part // replicas
        part = _Part(
            view,
            shape,
            kv.dtype,
            3,
            [(0, group * heads, heads)],
            self.tp_part % replicas == 0,
        )
        return _Layout([part], 2 * shape[1] * ps * heads * shape[4] * kv.itemsize)

    def _mamba_layout(self, name, host_pool: Any) -> Optional[_Layout]:
        from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost

        if not isinstance(host_pool, MambaPoolHost):
            return None
        side = list(getattr(host_pool, "slot_state_buffers", None) or ())
        if len(side) != len(getattr(host_pool, "slot_state_device_tensors", ())):
            return None
        device = self.device_pools.get(name)
        groups = getattr(device, "conv_shard_groups", None)
        conv_axis = getattr(device, "conv_slice_axis", 0)
        page_first = host_pool.layout in ("page_first", "page_first_direct")

        def at(buf):
            if page_first:
                return lambda t: buf[t]
            return lambda t: buf[:, t].unsqueeze(1)

        parts, local_bytes = [], 0
        tensors = (
            [(host_pool.temporal_buffer, 0, None, at)]
            + [(buf, conv_axis, groups, at) for buf in host_pool.conv_buffer]
            + [(buf, None, None, lambda b: lambda t: b[t]) for buf in side]
        )
        for buf, axis, subs, viewer in tensors:
            if buf is None or buf.numel() == 0:
                continue
            view = viewer(buf)
            local = tuple(view(0).shape)
            local_bytes += int(np.prod(local)) * buf.itemsize
            if axis is None:
                parts.append(_Part(view, local, buf.dtype, None, [], self.tp_part == 0))
                continue
            dim = 2 + axis
            if subs is not None and sum(subs) != local[dim] * self.tp_parts:
                return None
            shape = list(local)
            shape[dim] = local[dim] * self.tp_parts
            spans = _shard_spans(local[dim], self.tp_parts, self.tp_part, subs)
            parts.append(_Part(view, shape, buf.dtype, dim, spans, True))
        return _Layout(parts, local_bytes) if parts else None

    @staticmethod
    def _layer_split(host_pool: Any) -> bool:
        device = getattr(host_pool, "device_pool", None)
        return bool(getattr(device, "layer_shard_enabled", False))

    def _make_pool(self, name, host_pool: Any, anchor: Optional[_Pool]) -> _Pool:
        sharded = self._sharded(name) and self.tp_parts > 1
        layout = None
        if self._sharded(name):
            layout = self._mha_layout(name, host_pool) or self._mamba_layout(
                name, host_pool
            )
        parts = self.tp_parts if sharded else 1
        part = self.tp_part if sharded else 0
        writes = self.cp_rank == 0 if sharded or layout else self.leader
        tag = layout.tag if layout else f"tp{parts}"
        if layout is None and self._layer_split(host_pool) and self.cp_parts > 1:
            if sharded:
                raise ValueError(
                    f"radixshmem: pool {name} is split by layer and by TP heads"
                )
            parts, part = self.cp_parts, self.cp_rank
            writes = self.tp_part == 0
            tag = f"cp{parts}"
        if layout is None and getattr(host_pool, "slot_state_device_tensors", None):
            raise ValueError(
                f"radixshmem: pool {name} carries Mamba side state in a layout "
                "the backend cannot place (use page_first or layer_first)"
            )
        page_bytes = layout.local_bytes if layout else _page_bytes(host_pool)
        kind, mask = self.full, self.full_mask
        model, pp_size = self.config.model_name, self.config.pp_size
        salts: Dict[int, int] = {}

        def salt(pp_rank: int) -> int:
            if pp_rank not in salts:
                salts[pp_rank] = _salt(model, f"pp{pp_rank}/{pp_size}", tag, str(name))
            return salts[pp_rank]

        component = self.components.get(name)
        if (
            component is not None
            and anchor is not None
            and anchor.page_bytes > 0
            and page_bytes > 0
            and (name != PoolName.SWA or self.swa_window_pages > 0)
        ):
            kind, mask = component
            salt = anchor.salt
        return _Pool(
            name, host_pool, page_bytes, parts, part, salt, kind, mask, layout, writes
        )

    def _attach(self) -> None:
        if self.index is not None:
            return
        with self._attach_lock:
            if self.index is not None:
                return
            block_size = int(self.host_pools[PoolName.KV].page_size)
            if not self.swa_window_pages and PoolName.SWA in self.host_pools:
                self.swa_window_pages = self._swa_window(block_size)
            anchor = self._make_pool(PoolName.KV, self.host_pools[PoolName.KV], None)
            pools = {
                name: anchor
                if name == PoolName.KV
                else self._make_pool(name, hp, anchor)
                for name, hp in self.host_pools.items()
            }
            sizes: Dict[Any, int] = {}
            for p in pools.values():
                sizes[p.kind] = max(sizes.get(p.kind, 0), p.slot_bytes)
            slot_bytes = sizes.get(self.full, 0)
            if slot_bytes == 0:
                raise RuntimeError("radixshmem: no registered host pool carries data")
            swa_kind, _ = self.components[PoolName.SWA]
            mamba_kind, _ = self.components[PoolName.MAMBA]
            client = self.radixshmem.RadixClient(
                self.server_name,
                endpoint=self.endpoint,
                max_outstanding=self.max_outstanding,
                local_read_only=not self.leader,
            )
            client.configure(
                self.radixshmem.Geometry(
                    block_size=block_size,
                    full_slot_bytes=slot_bytes,
                    swa_slot_bytes=sizes.get(swa_kind, 0),
                    swa_window_blocks=self.swa_window_pages if swa_kind in sizes else 0,
                    mamba_slot_bytes=sizes.get(mamba_kind, 0),
                )
            )
            info = client.wait_ready(self.ready_timeout_s)
            self.client = client
            self.store = client.store
            self._data_view = self.store.data_view()
            self._data_base = np.frombuffer(self._data_view, dtype=np.uint8).ctypes.data
            self._data = torch.frombuffer(self._data_view, dtype=torch.uint8)
            self.distributed = self.leader and bool(client.is_distributed())
            self.block_size = block_size
            self.slot_bytes = slot_bytes
            self.kind_slot_bytes = sizes
            self.pools = pools
            logger.info(
                "RadixShmemStorage ready: tp part %d/%d, cp %d/%d, leader=%s "
                "block_size=%d slot_bytes=%d slots=%d distributed=%s pools=%s",
                self.tp_part,
                self.tp_parts,
                self.cp_rank,
                self.cp_parts,
                self.leader,
                block_size,
                slot_bytes,
                info.geometry["pools"]["full"]["num_slots"],
                self.distributed,
                {
                    str(n): (str(p.kind), p.page_bytes, p.slot_bytes)
                    for n, p in pools.items()
                },
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
            if transfer.hit_policy == PoolHitPolicy.BLOCK_GROUPS:
                pages = self._hit(pool, transfer.keys or keys, extra_info)
                groups = pages // max(1, transfer.group_pages)
                if groups:
                    hits[transfer.name] = groups * max(1, transfer.block_pages)
                continue
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = self._hit(pool, keys[:kv_pages], extra_info)
                stops = set(range(1, boundary + 1))
            elif pool.kind != self.full:
                boundary = self._component_hit(pool, keys[:kv_pages], extra_info)
                stops = (
                    set(range(1, kv_pages + 1)) if not self.leader else {boundary} - {0}
                )
            else:
                present = self._present(pool, keys[:kv_pages], extra_info)
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
        if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
            return self._store(pool, transfer.keys, transfer.host_indices, extra_info)
        if pool.kind != self.full and self._has_prefix(extra_info):
            return self._store_component(
                pool, transfer.keys, transfer.host_indices, extra_info
            )
        return self._store_pages(pool, transfer.keys, transfer.host_indices)

    def _read_transfer(self, transfer: PoolTransfer, extra_info) -> List[bool]:
        pool = self.pools[transfer.name]
        if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
            return self._load(pool, transfer.keys, transfer.host_indices, extra_info)
        if pool.kind != self.full and self._has_prefix(extra_info):
            return self._load_component(
                pool, transfer.keys, transfer.host_indices, extra_info
            )
        return self._load_pages(pool, transfer.keys, transfer.host_indices)

    @staticmethod
    def _has_prefix(extra_info) -> bool:
        return extra_info is not None and extra_info.prefix_keys is not None

    # ---- keys ----

    def _salt_of(self, pool: _Pool, extra_info) -> int:
        extra = (extra_info.extra_info if extra_info is not None else None) or {}
        return pool.salt(int(extra.get("pp_rank", self.config.pp_rank)))

    def _chain(self, pool: _Pool, keys, extra_info) -> Tuple[np.ndarray, int]:
        prefix = (extra_info.prefix_keys if extra_info is not None else None) or []
        salt = self._salt_of(pool, extra_info)
        chain = np.fromiter(
            (int(k[:16], 16) ^ salt for k in (*prefix, *keys)),
            dtype=np.uint64,
            count=len(prefix) + len(keys),
        )
        return chain, len(prefix)

    def _single(self, pool: _Pool, key: str, extra_info=None) -> np.ndarray:
        salt = self._salt_of(pool, extra_info)
        return np.array([int(key[:16], 16) ^ salt], dtype=np.uint64)

    # ---- index (leader only) ----

    def _query(self, chain: np.ndarray) -> int:
        q = self.index.query(
            chain, self.full_mask, local_only=False, lock=False, update_meta=True
        )
        hit = int(q.common_hit) if int(q.status) == 0 else 0
        if q.finalize:
            q.finalize()
        return hit

    def _pull(self, chain: np.ndarray, mask: int):
        job = self.client.pull_async(
            chain, mask, lock=True, timeout_ms=self.pull_timeout_ms
        )
        try:
            return job.wait(self.pull_timeout_ms / 1000 + 5)
        except TimeoutError:
            job.cancel()
            return None

    def _fetch(self, chain: np.ndarray) -> Tuple[np.ndarray, Callable]:
        result = self._pull(chain, self.full_mask)
        if result is None:
            return np.empty(0, dtype=np.int32), _noop
        return np.asarray(result.full_slots, dtype=np.int32), result.finalize

    def _allocate(self, n: int, kind) -> Optional[np.ndarray]:
        slots = np.asarray(self.index.allocate_slots(n, kind), dtype=np.int32)
        if len(slots) == n:
            return slots
        if len(slots):
            self.index.recycle_slots(slots, kind)
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
            slots = self._allocate(n, pool.kind)
            if slots is not None:
                buf[0] = 1
                buf[1:] = torch.from_numpy(slots.astype(np.int64))
        comm.bcast(buf)
        if int(buf[0]) != 1:
            return [False] * n
        slots = buf[1:].numpy().astype(np.int32)
        ok = torch.ones(1, dtype=torch.int64)
        if pool.writes:
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
                self.index.recycle_slots(slots, pool.kind)
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
        if pool.layout is not None:
            self._copy_layout(pool, host_indices, pairs, to_slot)
            return
        ptrs, sizes = pool.host_pool.get_page_buffer_meta(host_indices)
        per = len(ptrs) // (len(host_indices) // pool.host_pool.page_size)
        for i, slot in pairs:
            at = (
                self._data_base
                + int(self.store.slot_offset(int(slot), pool.kind))
                + pool.offset
            )
            for j in range(i * per, (i + 1) * per):
                if to_slot:
                    ctypes.memmove(at, ptrs[j], sizes[j])
                else:
                    ctypes.memmove(ptrs[j], at, sizes[j])
                at += sizes[j]

    def _copy_layout(self, pool: _Pool, host_indices, pairs, to_slot: bool) -> None:
        ps = int(pool.host_pool.page_size)
        for i, slot in pairs:
            token = int(host_indices[i * ps])
            base = int(self.store.slot_offset(int(slot), pool.kind))
            for part in pool.layout.parts:
                if to_slot and not part.write:
                    continue
                start = base + part.offset
                stored = (
                    self._data[start : start + part.nbytes]
                    .view(part.dtype)
                    .view(part.shape)
                )
                local = part.view(token)
                for loc, glob, length in part.spans or [(0, 0, 0)]:
                    src = _narrow(local, part.axis, loc, length)
                    dst = _narrow(stored, part.axis, glob, length)
                    if to_slot:
                        dst.copy_(src)
                    else:
                        src.copy_(dst)

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

    # ---- TRAILING_PAGES pools on the anchor path: SWA / MAMBA components ----

    def _component_hit(self, pool: _Pool, keys, extra_info) -> int:
        if not keys:
            return 0
        if not self.leader:
            return len(keys)
        chain, start = self._chain(pool, keys, extra_info)
        q = self.index.query(
            chain,
            self.full_mask | pool.mask,
            local_only=False,
            lock=False,
            update_meta=False,
        )
        hit = int(q.common_hit) if int(q.status) == 0 else 0
        if q.finalize:
            q.finalize()
        return max(0, hit - start)

    def _component_ends(self, pool: _Pool, chain: np.ndarray, n: int) -> List[int]:
        if pool.mask == self.components[PoolName.SWA][1]:
            return [len(chain)]
        return list(range(len(chain) - n + 1, len(chain) + 1))

    def _store_component(self, pool: _Pool, keys, host_indices, extra_info):
        n = len(keys)
        if n == 0:
            return []
        chain, _ = self._chain(pool, keys, extra_info)
        ends = self._component_ends(pool, chain, n)

        def publish(slots):
            if len(ends) == 1:
                result = self.index.insert(chain, slots, 0, True, component=pool.kind)
                return [result.error == self.insert_ok] * n
            return [
                self.index.insert(
                    chain[:end], slots[i : i + 1], 0, True, component=pool.kind
                ).error
                == self.insert_ok
                for i, end in enumerate(ends)
            ]

        return self._write(
            pool,
            n,
            lambda slots: self._copy(pool, host_indices, list(enumerate(slots)), True),
            publish,
        )

    def _load_component(self, pool: _Pool, keys, host_indices, extra_info):
        n = len(keys)
        if n == 0:
            return []
        chain, _ = self._chain(pool, keys, extra_info)
        ends = self._component_ends(pool, chain, n)
        mask = self.full_mask | pool.mask

        def fetch():
            slots, finalizers = [-1] * n, []
            for i, end in enumerate(ends):
                result = self._pull(chain[:end], mask)
                if result is None:
                    continue
                finalizers.append(result.finalize)
                if int(result.common_hit) != end:
                    continue
                if pool.mask != self.components[PoolName.SWA][1]:
                    slots[i] = int(result.mamba_slot)
                    continue
                window = np.asarray(result.swa_slots, dtype=np.int64)
                first = end - n
                for j in range(n):
                    at = first + j - int(result.swa_start)
                    if 0 <= at < len(window):
                        slots[j] = int(window[at])
            return slots, finalizers

        return self._read(
            n, fetch, lambda pairs: self._copy(pool, host_indices, pairs, False)
        )

    # ---- TRAILING_PAGES pools: one-block path per page ----

    def _present(self, pool: _Pool, keys, extra_info=None) -> List[bool]:
        if not self.leader:
            return [True] * len(keys)
        return [self._query(self._single(pool, key, extra_info)) == 1 for key in keys]

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
