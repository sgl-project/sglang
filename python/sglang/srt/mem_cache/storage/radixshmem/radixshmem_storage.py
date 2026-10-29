from __future__ import annotations

import ctypes
import hashlib
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)

_PREFIX_KEYS_REQUIRED = (
    "radixshmem stores each prefix as one radix path and needs "
    '"hicache_storage_pass_prefix_keys": true in '
    "--hicache-storage-backend-extra-config"
)


_PACK_POLICY = {
    PoolName.KV: PoolHitPolicy.ALL_PAGES,
    PoolName.SWA: PoolHitPolicy.TRAILING_PAGES,
    PoolName.MAMBA: PoolHitPolicy.TRAILING_PAGES,
}


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


def _shard_layout():
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import (
        get_attn_cp_group,
        get_attn_tp_group,
    )

    tp, cp = get_attn_tp_group(), get_attn_cp_group()
    rows = [list(tp.ranks)]
    if cp.world_size > 1:
        gathered = [None] * cp.world_size
        dist.all_gather_object(
            gathered, (cp.rank_in_group, list(tp.ranks)), group=cp.cpu_group
        )
        rows = [row for _, row in sorted(gathered)]
    members = [r for row in rows for r in row]
    if len(set(members)) != tp.world_size * cp.world_size or any(
        len(row) != tp.world_size for row in rows
    ):
        raise RuntimeError(
            f"attention TP groups {rows} do not tile the attention CP group"
        )
    identity = dict(
        tp_part=tp.rank_in_group,
        tp_parts=tp.world_size,
        cp_rank=cp.rank_in_group,
        cp_parts=cp.world_size,
        leader_rank=rows[0][0],
    )
    return members, identity


def _prefix_keys_enabled() -> Optional[bool]:
    try:
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )
        from sglang.srt.runtime_context import get_memory

        memory = get_memory()
        if memory.hicache_storage_backend != "radixshmem":
            return None
        raw = memory.hicache_storage_backend_extra_config
    except Exception:
        return None
    return bool(HybridCacheController.parse_storage_backend_extra_config(raw)[4])


def _decode_offload_enabled() -> bool:
    try:
        from sglang.srt.runtime_context import get_disagg

        disagg = get_disagg()
        return disagg.disaggregation_mode == "decode" and bool(
            disagg.disaggregation_decode_enable_offload_kvcache
        )
    except Exception:
        return False


class _Pack:
    def __init__(self, name, members: List[_Pool], kind, mask: int):
        self.name = name
        self.members = {m.name: m for m in members}
        self.base: Dict[Any, int] = {}
        offset = 0
        for m in members:
            offset = -(-offset // 64) * 64
            self.base[m.name] = offset
            offset += m.slot_bytes
        self.slot_bytes = offset
        self.sharded = any(m.sharded for m in members)
        self.writes = any(m.writes for m in members if m.slot_bytes)
        self.kind = kind
        self.mask = mask
        self.tag = ";".join(f"{m.name}:{m.tag}:{m.slot_bytes}" for m in members)


class _PoolMismatch(RuntimeError):
    pass


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
        tag: str,
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
        self.tag = tag


class RadixShmemStorage(HiCacheStorage):
    shard_collective = True

    def __init__(self, storage_config: HiCacheStorageConfig, mem_pool_host: Any = None):
        import radixshmem

        if storage_config.pp_size > 1:
            raise ValueError(
                "radixshmem does not support pipeline parallelism yet "
                f"(pp_size={storage_config.pp_size})"
            )
        if _prefix_keys_enabled() is False:
            raise ValueError(_PREFIX_KEYS_REQUIRED)
        if _decode_offload_enabled():
            raise ValueError(
                "radixshmem does not support --disaggregation-decode-enable-"
                "offload-kvcache: the decode offload path writes without prefix keys"
            )
        self.radixshmem = radixshmem
        self.config = storage_config
        cfg = storage_config.extra_config or {}
        self.server_name = cfg.get("radixshmem_name", "/radixshmem")
        self.endpoint = cfg.get("radixshmem_endpoint")
        self.max_outstanding = int(cfg.get("radixshmem_max_outstanding", 256))
        self.ready_timeout_s = float(cfg.get("radixshmem_timeout_s", 300))
        self.init_timeout_s = float(cfg.get("radixshmem_init_timeout_s", 30))
        self.pull_timeout_ms = int(cfg.get("radixshmem_pull_timeout_ms", 30000))
        self.retry_s = float(cfg.get("radixshmem_retry_s", 30))
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
        self.mamba_shape: Any = None
        self.insert_ok = radixshmem.InsertError.OK
        self.full_path_missing = radixshmem.InsertError.FULL_PATH_MISSING
        self.defer_s = float(cfg.get("radixshmem_defer_s", 60))
        self.host_pools: Dict[Any, Any] = {}
        self.pools: Dict[Any, _Pool] = {}
        self.packs: Dict[Any, _Pack] = {}
        self.pack_of: Dict[Any, _Pack] = {}
        self.sidecars: Dict[Any, Any] = {}
        self.client = None
        self.index = None
        self.slot_bytes = 0
        self.salt = 0
        self._local = threading.local()
        self.kind_slot_bytes: Dict[Any, int] = {}
        self._attach_lock = threading.Lock()
        self._retry_at = 0.0
        self._down_since: Optional[float] = None
        self._mismatch: Optional[Exception] = None
        self._deferred: List[Tuple[float, np.ndarray, np.ndarray, Any]] = []
        self.write_failures: Dict[str, int] = {}
        self._own_groups: List[Any] = []
        self.set_shard_groups(None, None)
        self._build_shard_groups()

    def _build_shard_groups(self) -> None:
        import torch.distributed as dist

        if not dist.is_initialized():
            return
        import socket

        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        members, identity = _shard_layout()
        groups = [
            create_custom_parallel_group(group_ranks=members, backend="gloo")
            for _ in range(2)
        ]
        if len(members) == 1:
            for group in groups:
                dist.destroy_process_group(group)
            return
        self._own_groups = groups
        hosts = [None] * len(members)
        dist.all_gather_object(hosts, socket.gethostname(), group=groups[0])
        if len(set(hosts)) > 1:
            raise RuntimeError(
                "radixshmem does not support an attention TP/CP group spanning "
                f"nodes yet: {sorted(set(hosts))}"
            )
        self.set_shard_groups(*groups, **identity)

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
        if self.pools:
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

    def describe_sidecar(self, spec) -> None:
        self.sidecars[spec.pool_name] = spec

    def _register(self, name, host_pool: Any) -> None:
        if self.pools and self.host_pools.get(name) is not host_pool:
            raise ValueError(
                f"radixshmem: pool {name} registered after the slot layout was fixed"
            )
        self.host_pools[name] = host_pool

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

    def _mamba_state_shape(self) -> Any:
        if self.mamba_shape is None:
            try:
                from sglang.srt.configs.hybrid_arch import mambaish_config
                from sglang.srt.runtime_context import process_model_config

                config = mambaish_config(process_model_config())
                self.mamba_shape = config.mamba2_cache_params.shape
            except Exception:
                self.mamba_shape = False
        return self.mamba_shape or None

    def _mamba_layout(self, name, host_pool: Any) -> Optional[_Layout]:
        from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost

        if not isinstance(host_pool, MambaPoolHost):
            return None
        side = list(getattr(host_pool, "slot_state_buffers", None) or ())
        if len(side) != len(getattr(host_pool, "slot_state_device_tensors", ())):
            return None
        shape = self._mamba_state_shape()
        groups = getattr(shape, "conv_shard_groups", None)
        conv_axis = getattr(shape, "conv_slice_axis", 0)
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

    def _make_pool(self, name, host_pool: Any) -> _Pool:
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
        return _Pool(name, host_pool, page_bytes, parts, part, tag, layout, writes)

    def _source(self, name):
        if name in _PACK_POLICY:
            return name
        spec = self.sidecars.get(name)
        if spec is None:
            raise ValueError(
                f"radixshmem: host pool {name} has no sidecar spec; the backend "
                "must be attached at launch so it sees every sidecar"
            )
        source = spec.indices_from_pool
        if _PACK_POLICY.get(source) != spec.hit_policy:
            raise ValueError(
                f"radixshmem: sidecar {name} ({source}, {spec.hit_policy}) does "
                "not ride the KV, SWA or Mamba slot"
            )
        return source

    def _plan(self) -> None:
        if self.pools:
            return
        with self._attach_lock:
            if self.pools:
                return
            self.block_size = int(self.host_pools[PoolName.KV].page_size)
            pools = {n: self._make_pool(n, hp) for n, hp in self.host_pools.items()}
            members: Dict[Any, List[_Pool]] = {}
            for name in sorted(pools, key=lambda n: (n not in _PACK_POLICY, str(n))):
                members.setdefault(self._source(name), []).append(pools[name])
            kinds = {PoolName.KV: (self.full, self.full_mask), **self.components}
            packs: Dict[Any, _Pack] = {}
            for source in _PACK_POLICY:
                group = members.get(source)
                if not group:
                    continue
                if source not in pools:
                    raise ValueError(
                        f"radixshmem: sidecars {[str(m.name) for m in group]} "
                        f"index into {source}, which is not registered"
                    )
                pack = _Pack(source, group, *kinds[source])
                if pack.slot_bytes == 0:
                    raise ValueError(f"radixshmem: the {source} pools carry no data")
                packs[source] = pack
            if PoolName.SWA in packs:
                self.swa_window_pages = self._swa_window(self.block_size)
                if self.swa_window_pages <= 0:
                    raise ValueError(
                        "radixshmem: the SWA window is unknown; set "
                        "radixshmem_swa_window_pages in the extra config"
                    )
            self.salt = _salt(
                self.config.model_name,
                *(f"{k}={p.tag}" for k, p in packs.items()),
            )
            self.packs = packs
            self.pack_of = {n: p for p in packs.values() for n in p.members}
            self.slot_bytes = packs[PoolName.KV].slot_bytes
            self.kind_slot_bytes = {p.kind: p.slot_bytes for p in packs.values()}
            self.pools = pools

    def _ready(self) -> bool:
        self._plan()
        if self.index is not None:
            return True
        if self._mismatch is not None:
            raise self._mismatch
        if time.monotonic() < self._retry_at:
            return False
        with self._attach_lock:
            if self.index is not None:
                return True
            if time.monotonic() < self._retry_at:
                return False
            try:
                self._connect()
            except _PoolMismatch as e:
                self._mismatch = e
                raise
            except Exception as e:
                if "_ratio is 0" in str(e):
                    self._mismatch = _PoolMismatch(self._mismatch_message(e))
                    raise self._mismatch from e
                self._retry_at = time.monotonic() + self.retry_s
                if self._down_since is None:
                    self._down_since = time.monotonic()
                    logger.warning(
                        "radixshmem: server %s unavailable (%s: %s); storage calls "
                        "miss, retrying every %.0f s",
                        self.server_name,
                        type(e).__name__,
                        str(e).splitlines()[0][:200] if str(e) else "",
                        self.retry_s,
                    )
                else:
                    logger.debug(
                        "radixshmem: server %s still unavailable: %s",
                        self.server_name,
                        e,
                    )
                return False
            if self._down_since is not None:
                logger.warning(
                    "radixshmem: server %s reachable after %.0f s",
                    self.server_name,
                    time.monotonic() - self._down_since,
                )
                self._down_since = None
            return True

    def connect(self) -> bool:
        self._plan()
        if self.index is not None:
            return True
        timeout, self.ready_timeout_s = self.ready_timeout_s, self.init_timeout_s
        try:
            return self._ready()
        finally:
            self.ready_timeout_s = timeout

    def _aux_kinds(self) -> List[str]:
        names = {
            self.components[PoolName.SWA][0]: "swa",
            self.components[PoolName.MAMBA][0]: "mamba",
        }
        return sorted(names[k] for k in self.kind_slot_bytes if k in names)

    def _mismatch_message(self, detail) -> str:
        return (
            f"radixshmem: this model uses component pools {self._aux_kinds()} but "
            f"server {self.server_name} {detail}; restart radix-server with "
            "--swa-ratio / --mamba-ratio > 0 exactly for the pools the model uses"
        )

    def _check_aux_pools(self, info) -> None:
        used = set(self._aux_kinds())
        for kind in ("swa", "mamba"):
            ratio = float(getattr(info, f"{kind}_ratio"))
            if (ratio > 0) != (kind in used):
                state = (
                    f"sets {kind}_ratio={ratio}" if ratio > 0 else f"has no {kind} pool"
                )
                raise _PoolMismatch(self._mismatch_message(state))

    def _connect(self) -> None:
        swa_kind, _ = self.components[PoolName.SWA]
        mamba_kind, _ = self.components[PoolName.MAMBA]
        sizes = self.kind_slot_bytes
        client = self.radixshmem.RadixClient(
            self.server_name,
            endpoint=self.endpoint,
            max_outstanding=self.max_outstanding,
            local_read_only=not self.leader,
        )
        try:
            client.configure(
                self.radixshmem.Geometry(
                    block_size=self.block_size,
                    full_slot_bytes=self.slot_bytes,
                    swa_slot_bytes=sizes.get(swa_kind, 0),
                    swa_window_blocks=self.swa_window_pages if swa_kind in sizes else 0,
                    mamba_slot_bytes=sizes.get(mamba_kind, 0),
                )
            )
            info = client.wait_ready(self.ready_timeout_s)
            self._check_aux_pools(info)
            store = client.store
            index = client.index
        except Exception:
            client.close()
            raise
        self.client = client
        self.store = store
        self._data_view = store.data_view()
        self._data_base = np.frombuffer(self._data_view, dtype=np.uint8).ctypes.data
        self._data = torch.frombuffer(self._data_view, dtype=torch.uint8)
        self.distributed = self.leader and bool(client.is_distributed())
        logger.info(
            "RadixShmemStorage ready: tp part %d/%d, cp %d/%d, leader=%s "
            "block_size=%d slot_bytes=%d slots=%d distributed=%s pools=%s",
            self.tp_part,
            self.tp_parts,
            self.cp_rank,
            self.cp_parts,
            self.leader,
            self.block_size,
            self.slot_bytes,
            info.geometry["pools"]["full"]["num_slots"],
            self.distributed,
            {
                str(k): ([str(n) for n in p.members], p.slot_bytes)
                for k, p in self.packs.items()
            },
        )
        self.index = index

    # ---- HiCacheStorage: KV pool ----

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        self._plan()
        self._local.recent = None
        return self._hit(keys, extra_info)

    def batch_set_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        self._require_prefix(extra_info)
        self._plan()
        self._local.recent = None
        return self._store_full(keys, host_indices, extra_info)

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> List[bool]:
        self._plan()
        return self._load_full(keys, host_indices, extra_info)

    # ---- HiCacheStorage: extra pools ----

    def batch_exists_v2(
        self, keys, pool_transfers=None, extra_info=None
    ) -> PoolTransferResult:
        kv_pages = self.batch_exists(keys, extra_info)
        hits = {PoolName.KV: kv_pages} if kv_pages else {}
        restorable = list(range(1, kv_pages + 1))
        boundaries: Dict[Any, int] = {}
        for transfer in pool_transfers or []:
            if not restorable:
                break
            pack = self._pack(transfer)
            if pack.kind == self.full:
                boundary = kv_pages
                stops = set(restorable)
            else:
                if pack.name not in boundaries:
                    boundaries[pack.name] = self._component_hit(
                        pack, keys[:kv_pages], extra_info
                    )
                boundary = boundaries[pack.name]
                stops = (
                    set(range(1, kv_pages + 1)) if not self.leader else {boundary} - {0}
                )
            if boundary:
                hits[transfer.name] = boundary
            restorable = [n for n in restorable if n in stops]
        final = restorable[-1] if restorable else 0
        return PoolTransferResult(final, hits, restorable)

    def batch_set_v2(self, transfers, extra_info=None) -> dict:
        self._require_prefix(extra_info)
        self._plan()
        self._local.recent = None
        out: Dict[Any, List[bool]] = {}
        for pack, group, keys, host_indices in self._grouped(transfers):
            if host_indices is None:
                result = [False] * len(keys)
            elif pack.kind == self.full:
                result = self._store_full(keys, host_indices, extra_info)
            else:
                result = self._store_component(pack, keys, host_indices, extra_info)
            out.update({t.name: result for t in group})
        return out

    def batch_get_v2(self, transfers, extra_info=None) -> dict:
        self._plan()
        out: Dict[Any, List[bool]] = {}
        for pack, group, keys, host_indices in self._grouped(transfers):
            if host_indices is None:
                result = [False] * len(keys)
            elif pack.kind == self.full:
                result = self._recent_full(keys, extra_info)
                if result is None:
                    result = self._load_full(keys, host_indices, extra_info)
            else:
                result = self._load_component(pack, keys, host_indices, extra_info)
            out.update({t.name: result for t in group})
        return out

    def _pack(self, transfer: PoolTransfer) -> _Pack:
        pack = self.pack_of.get(transfer.name)
        if pack is None:
            raise ValueError(f"radixshmem: pool {transfer.name} is not registered")
        return pack

    def _grouped(self, transfers):
        groups: Dict[Any, List[PoolTransfer]] = {}
        for t in transfers:
            groups.setdefault(self._pack(t).name, []).append(t)
        for name in sorted(groups, key=lambda n: n != PoolName.KV):
            group = groups[name]
            keys = list(group[0].keys or ())
            host_indices = next(
                (t.host_indices for t in group if t.host_indices is not None), None
            )
            if any(list(t.keys or ()) != keys for t in group):
                host_indices = None
            yield self.packs[name], group, keys, host_indices

    @staticmethod
    def _require_prefix(extra_info) -> None:
        if extra_info is None or extra_info.prefix_keys is None:
            raise ValueError(_PREFIX_KEYS_REQUIRED)

    # ---- keys ----

    def _chain(self, keys, extra_info) -> Tuple[np.ndarray, int]:
        prefix = (extra_info.prefix_keys if extra_info is not None else None) or []
        chain = np.fromiter(
            (int(k[:16], 16) ^ self.salt for k in (*prefix, *keys)),
            dtype=np.uint64,
            count=len(prefix) + len(keys),
        )
        return chain, len(prefix)

    # ---- index (leader only) ----

    def _query(self, chain: np.ndarray) -> int:
        q = self.index.query(
            chain, self.full_mask, local_only=False, lock=False, update_meta=True
        )
        hit = int(q.common_hit) if int(q.status) == 0 else 0
        if q.finalize:
            q.finalize()
        return hit

    def _local_query(self, chain: np.ndarray) -> int:
        q = self.index.query(
            chain, self.full_mask, local_only=True, lock=False, update_meta=False
        )
        return int(q.common_hit) if int(q.status) == 0 else 0

    def _count_write_failure(self, prefix_missing: bool, result) -> None:
        # prefix_missing: the request's earlier pages are only on a peer, so the
        # local insert cannot extend them.
        key = "prefix_missing" if prefix_missing else f"error_{int(result.error)}"
        self.write_failures[key] = self.write_failures.get(key, 0) + 1
        total = sum(self.write_failures.values())
        if total & (total - 1) == 0:
            logger.warning("radixshmem: write failures %s", self.write_failures)

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
        pack: _Pack,
        n: int,
        copy: Callable[[np.ndarray, int], None],
        publish: Callable[[np.ndarray, int], List[bool]],
        present: Optional[Callable[[], int]] = None,
    ) -> List[bool]:
        if not pack.sharded and not self.leader:
            return [True] * n
        ready = self._ready()
        comm = self.write_comm if pack.sharded else _Solo()
        buf = torch.full((n + 2,), -1, dtype=torch.int64)
        if self.leader and ready and self._deferred:
            self._publish_deferred()
        if self.leader and ready:
            skip = present() if present is not None else 0
            slots = self._allocate(n - skip, pack.kind) if skip < n else None
            if skip == n or slots is not None:
                buf[0], buf[1] = 1, skip
                if slots is not None:
                    buf[2 : n + 2 - skip] = torch.from_numpy(slots.astype(np.int64))
        comm.bcast(buf)
        if int(buf[0]) != 1:
            return [False] * n
        skip = int(buf[1])
        if skip == n:
            return [True] * n
        slots = buf[2 : n + 2 - skip].numpy().astype(np.int32)
        ok = torch.ones(1, dtype=torch.int64)
        if pack.writes and not ready:
            ok[0] = 0
        elif pack.writes:
            try:
                copy(slots, skip)
            except Exception:
                logger.exception("radixshmem: copy into slots failed")
                ok[0] = 0
        comm.min(ok)
        result = torch.zeros(n, dtype=torch.int64)
        result[:skip] = 1
        if self.leader:
            if int(ok[0]):
                try:
                    published = publish(slots, skip)
                    result[skip:] = torch.tensor(published, dtype=torch.int64)
                except Exception:
                    logger.exception("radixshmem: publish failed")
                if bool(result[skip:].any()) and self.distributed:
                    self.index.flush()
            else:
                self.index.recycle_slots(slots, pack.kind)
        comm.bcast(result)
        return [bool(x) for x in result.tolist()]

    def _read(
        self,
        n: int,
        fetch: Callable[[List[Callable]], Sequence[int]],
        copy: Callable[[List[Tuple[int, int]]], None],
    ) -> List[bool]:
        ready = self._ready()
        buf = torch.full((n,), -1, dtype=torch.int64)
        finalizers: List[Callable] = []
        if self.leader and ready:
            try:
                slots = fetch(finalizers)
                if len(slots):
                    buf[: len(slots)] = torch.as_tensor(
                        np.asarray(slots, dtype=np.int64)
                    )
            except Exception:
                logger.exception("radixshmem: fetch failed")
                buf.fill_(-1)
        self.read_comm.bcast(buf)
        slots = buf.numpy()
        ok = torch.from_numpy((slots >= 0).astype(np.int64))
        if not ready:
            ok.zero_()
        else:
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
        pack: _Pack,
        host_indices: torch.Tensor,
        n: int,
        pairs: Sequence[Tuple[int, int]],
        to_slot: bool,
    ) -> None:
        if not pairs:
            return
        stride = len(host_indices) // n
        offsets = {s: int(self.store.slot_offset(int(s), pack.kind)) for _, s in pairs}
        for m in pack.members.values():
            if m.slot_bytes == 0 or (to_slot and not m.writes):
                continue
            base = pack.base[m.name]
            if m.layout is not None:
                self._copy_layout(
                    m, base, offsets, host_indices, stride, pairs, to_slot
                )
                continue
            ptrs, sizes = m.host_pool.get_page_buffer_meta(host_indices)
            per = len(ptrs) // n
            for i, slot in pairs:
                at = self._data_base + offsets[slot] + base + m.offset
                for j in range(i * per, (i + 1) * per):
                    if to_slot:
                        ctypes.memmove(at, ptrs[j], sizes[j])
                    else:
                        ctypes.memmove(ptrs[j], at, sizes[j])
                    at += sizes[j]

    def _copy_layout(
        self, m: _Pool, base: int, offsets, host_indices, stride, pairs, to_slot
    ) -> None:
        for i, slot in pairs:
            token = int(host_indices[i * stride])
            for part in m.layout.parts:
                if to_slot and not part.write:
                    continue
                start = offsets[slot] + base + part.offset
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

    # ---- the FULL slot: KV and every KV-indexed sidecar on one radix path ----

    def _hit(self, keys, extra_info) -> int:
        if not keys:
            return 0
        if not self.leader:
            return len(keys)
        if not self._ready():
            return 0
        chain, start = self._chain(keys, extra_info)
        return max(0, min(self._query(chain) - start, len(keys)))

    def _store_full(self, keys, host_indices, extra_info) -> List[bool]:
        n = len(keys)
        if n == 0:
            return []
        pack = self.packs[PoolName.KV]
        chain, start = self._chain(keys, extra_info)
        out: List[bool] = []
        for lo in range(0, n, STORAGE_BATCH_SIZE):
            hi = min(n, lo + STORAGE_BATCH_SIZE)
            if out and not out[-1]:
                out += [False] * (hi - lo)
                continue
            out += self._store_chunk(
                pack, chain[: start + hi], start + lo, host_indices, n, lo, hi
            )
        return out

    def _store_chunk(self, pack, chain, first, host_indices, n, lo, hi) -> List[bool]:
        stride = len(host_indices) // n
        indices = host_indices[lo * stride : hi * stride]

        # The insert lands in the local tree, so the skip must come from the local
        # hit: a peer's copy of [first, hi) would move the start past the local
        # prefix and the insert would be rejected.
        local = [0]

        def present():
            local[0] = self._local_query(chain)
            return max(0, min(local[0] - first, hi - lo))

        def copy(slots, skip):
            pairs = [(skip + i, s) for i, s in enumerate(slots)]
            self._copy(pack, indices, hi - lo, pairs, True)

        def publish(slots, skip):
            result = self.index.insert(
                chain, slots, first + skip, True, component=self.full
            )
            ok = self._published(chain, slots, result)
            if not ok:
                self._count_write_failure(local[0] < first, result)
            if ok and self._deferred:
                self._publish_deferred()
            return [ok] * len(slots)

        return self._write(pack, hi - lo, copy, publish, present)

    def _load_full(self, keys, host_indices, extra_info) -> List[bool]:
        n = len(keys)
        if n == 0:
            return []
        pack = self.packs[PoolName.KV]
        chain, start = self._chain(keys, extra_info)

        def fetch(finalizers):
            slots, finalize = self._fetch(chain)
            finalizers.append(finalize)
            return slots[start : start + n]

        ok = self._read(
            n, fetch, lambda pairs: self._copy(pack, host_indices, n, pairs, False)
        )
        hit = ok.index(False) if False in ok else n
        result = [True] * hit + [False] * (n - hit)
        self._local.recent = (chain.tobytes(), result)
        return result

    def _recent_full(self, keys, extra_info) -> Optional[List[bool]]:
        recent = getattr(self._local, "recent", None)
        self._local.recent = None
        if recent is None or not keys:
            return None
        chain, _ = self._chain(keys, extra_info)
        return list(recent[1]) if recent[0] == chain.tobytes() else None

    # ---- the SWA / Mamba slots: components on the FULL path ----

    def _component_hit(self, pack: _Pack, keys, extra_info) -> int:
        if not keys:
            return 0
        if not self.leader:
            return len(keys)
        if not self._ready():
            return 0
        chain, start = self._chain(keys, extra_info)
        q = self.index.query(
            chain,
            self.full_mask | pack.mask,
            local_only=False,
            lock=False,
            update_meta=True,
        )
        hit = int(q.common_hit) if int(q.status) == 0 else 0
        if q.finalize:
            q.finalize()
        return max(0, hit - start)

    def _component_ends(self, pack: _Pack, chain: np.ndarray, n: int) -> List[int]:
        if pack.name == PoolName.SWA:
            return [len(chain)]
        return list(range(len(chain) - n + 1, len(chain) + 1))

    def _store_component(self, pack: _Pack, keys, host_indices, extra_info):
        n = len(keys)
        if n == 0:
            return []
        chain, _ = self._chain(keys, extra_info)
        ends = self._component_ends(pack, chain, n)

        def copy(slots, skip):
            self._copy(pack, host_indices, n, list(enumerate(slots)), True)

        def publish(slots, skip):
            if len(ends) == 1:
                return [self._publish_component(chain, slots, pack.kind)] * n
            return [
                self._publish_component(chain[:end], slots[i : i + 1], pack.kind)
                for i, end in enumerate(ends)
            ]

        return self._write(pack, n, copy, publish)

    def _publish_component(self, chain: np.ndarray, slots: np.ndarray, kind) -> bool:
        result = self.index.insert(chain, slots, 0, False, component=kind)
        if result.error == self.full_path_missing:
            self._deferred.append((time.monotonic(), chain, slots, kind))
            return True
        unused = np.asarray(getattr(result, "unused_slots", ()), dtype=np.int32)
        if result.error != self.insert_ok:
            unused = slots
        if len(unused):
            self.index.recycle_slots(unused, kind)
        return result.error == self.insert_ok

    def _publish_deferred(self, drop: bool = False) -> None:
        now, pending = time.monotonic(), self._deferred
        self._deferred = []
        for since, chain, slots, kind in pending:
            if drop:
                self.index.recycle_slots(slots, kind)
                continue
            result = self.index.insert(chain, slots, 0, False, component=kind)
            if result.error == self.full_path_missing and now - since < self.defer_s:
                self._deferred.append((since, chain, slots, kind))
                continue
            unused = np.asarray(getattr(result, "unused_slots", ()), dtype=np.int32)
            if result.error != self.insert_ok:
                unused = slots
            if len(unused):
                self.index.recycle_slots(unused, kind)

    def _load_component(self, pack: _Pack, keys, host_indices, extra_info):
        n = len(keys)
        if n == 0:
            return []
        chain, _ = self._chain(keys, extra_info)
        ends = self._component_ends(pack, chain, n)
        mask = self.full_mask | pack.mask

        def fetch(finalizers):
            slots = [-1] * n
            for i, end in enumerate(ends):
                result = self._pull(chain[:end], mask)
                if result is None:
                    continue
                finalizers.append(result.finalize)
                if int(result.common_hit) != end:
                    continue
                if pack.name != PoolName.SWA:
                    slots[i] = int(result.mamba_slot)
                    continue
                window = np.asarray(result.swa_slots, dtype=np.int64)
                first = end - n
                for j in range(n):
                    at = first + j - int(result.swa_start)
                    if 0 <= at < len(window):
                        slots[j] = int(window[at])
            return slots

        return self._read(
            n, fetch, lambda pairs: self._copy(pack, host_indices, n, pairs, False)
        )

    # ---- misc ----

    def clear(self) -> bool:
        if self.leader and self._ready():
            self.index.reset()
        return True

    def close(self) -> None:
        if self._deferred and self.index is not None:
            try:
                self._publish_deferred(drop=True)
            except Exception:
                logger.exception("radixshmem: recycling deferred slots failed")
        if self.client is not None:
            self.client.close()
        if self._own_groups:
            import torch.distributed as dist

            for group in self._own_groups:
                try:
                    dist.destroy_process_group(group)
                except Exception:
                    pass
            self._own_groups = []

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
