# SPDX-License-Identifier: Apache-2.0
"""HiCache storage over gRPC (``grpc_socket`` UDS) + mempool UDS for RegisterPool."""

from __future__ import annotations

import json
import logging
import mmap
import os
import socket
from typing import Any, Dict, List, Optional, Sequence, Tuple

import grpc
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
from sglang.srt.mem_cache.memory_pool_host import HostKVCache, HostTensorAllocator
from sglang.srt.mem_cache.storage.rpc import hicache_pb2_grpc
from sglang.srt.mem_cache.storage.rpc.hicache_pb2 import (
    BatchGetRequest,
    BatchPutRequest,
    ExistRequest,
    Iovec,
    Transfer,
)
from sglang.srt.mem_cache.storage.rpc.rpc_fd import (
    parse_unix_socket_uri,
    recv_pool_registration_reply_pool_id,
    send_pool_registration,
    unix_connect,
)

logger = logging.getLogger(__name__)


def _page_size() -> int:
    return getattr(mmap, "PAGESIZE", int(os.sysconf("SC_PAGE_SIZE")))


def _alloc_granularity() -> int:
    """mmap offset must be a multiple of this (often 4096 on Linux)."""
    return int(getattr(mmap, "ALLOCATIONGRANULARITY", _page_size()))


def _granularity_align_up(n: int) -> int:
    ag = _alloc_granularity()
    return (max(0, n) + ag - 1) // ag * ag


def _pool_registry_key(nm: Any) -> str:
    if isinstance(nm, PoolName):
        return nm.value
    return str(nm)


class MemfdTensorAllocator(HostTensorAllocator):
    """memfd + per-allocation mmap bump allocator (FD byte offset = ``memfd_offset``).

    Each :meth:`allocate` extends the memfd with :func:`os.ftruncate`, maps the new
    range with :class:`mmap.mmap`, and tags the returned tensor with
    ``memfd_offset`` (byte offset in the FD). Mappings are retained in
    :attr:`_mmap_maps` so backing memory stays valid (see TODO: alias / leak note).
    """

    def __init__(self):
        super().__init__()
        self._size = 0
        self._mmap_maps: List[mmap.mmap] = []
        self._fd = os.memfd_create(
            "sgl_hicache_rpc", getattr(os, "MFD_CLOEXEC", 0)
        )
        self.base_ptr: Optional[int] = None

    def offset_of(self, ptr: int) -> int:
        assert self.base_ptr is not None, "MemfdTensorAllocator has not allocated any tensor yet"
        assert self.base_ptr <= ptr <= self.base_ptr + self._size, (
            f"ptr {ptr} out of range [{self.base_ptr}, {self.base_ptr + self._size})"
        )
        return ptr - self.base_ptr

    @property
    def fd(self) -> int:
        return self._fd

    def allocate(
        self, dims: tuple, dtype: torch.dtype, device: str = "cpu"
    ) -> torch.Tensor:
        assert device == "cpu"
        self.dims = dims
        self.dtype = dtype
        need = 1
        for d in dims:
            need *= int(d)
        need *= torch.tensor([], dtype=dtype).element_size()
        need = int(need)
        mmap_off = _granularity_align_up(self._size)
        mmap_len = _granularity_align_up(need)
        new_eof = mmap_off + mmap_len
        os.ftruncate(self._fd, new_eof)
        # TODO(chaoshi): each mmap is retained in _mmap_maps (VMA leak vs. single mremap window).
        mm = mmap.mmap(
            self._fd,
            mmap_len,
            access=mmap.ACCESS_WRITE,
            offset=mmap_off,
        )
        self._mmap_maps.append(mm)
        mv = memoryview(mm)[:need]
        t = torch.frombuffer(mv, dtype=torch.uint8, count=need)
        if dtype != torch.uint8:
            es = torch.tensor([], dtype=dtype).element_size()
            assert need % es == 0
            t = t.view(dtype)
        out = t.view(dims)
        setattr(out, "memfd_offset", mmap_off)
        self._size = mmap_off + need
        if self.base_ptr is None:
            self.base_ptr = int(out.data_ptr())
        return out


class HiCacheRpcStorage(HiCacheStorage):
    """L3 via gRPC StorageServer; host pages are FD-backed offsets (zero-copy)."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool: Optional[HostKVCache] = None,
    ):
        extra = (
            storage_config.extra_config
            if storage_config and storage_config.extra_config
            else {}
        )
        if isinstance(extra, str):
            extra = json.loads(extra)
        self._mempool_socket_uri = extra.get("mempool_socket")
        self._grpc_socket_uri = extra.get("grpc_socket")
        if not self._mempool_socket_uri or not self._grpc_socket_uri:
            raise ValueError(
                'HiCacheRpcStorage requires extra_config["mempool_socket"] and '
                '["grpc_socket"] (e.g. unix:///tmp/hicache_mempool.sock and '
                "unix:///tmp/hicache_grpc.sock)"
            )
        self._grpc_timeout = float(extra.get("grpc_timeout_s", 120.0))
        self.storage_config = storage_config
        self.mem_pool_host = mem_pool
        self.enable_storage_metrics = bool(
            getattr(storage_config, "enable_storage_metrics", False)
        )
        self.is_mla_backend = storage_config.is_mla_model
        self.local_rank = storage_config.tp_rank
        self.pp_rank = storage_config.pp_rank
        self.pp_size = storage_config.pp_size
        self.attn_cp_rank = storage_config.attn_cp_rank
        self.attn_cp_size = storage_config.attn_cp_size
        self.enable_pp = self.pp_size > 1
        self.enable_cp = self.attn_cp_size > 1
        if self.enable_pp or self.enable_cp:
            self.mha_suffix = f"{self.local_rank}_{self.pp_rank}_{self.attn_cp_rank}"
            self.mla_suffix = f"{self.pp_rank}_{self.attn_cp_rank}"
        else:
            self.mha_suffix = f"{self.local_rank}"
            self.mla_suffix = ""
        self.split_factor = 0
        if storage_config.should_split_heads:
            self.split_factor = (
                (storage_config.tp_lcm_size or 1) // storage_config.tp_size
            )
        self.extra_backend_tag: Optional[str] = extra.get("extra_backend_tag")

        self.registered_pools: Dict[Any, HostKVCache] = {}
        self._pool_server_ids: Dict[str, int] = {}
        self._mempool_sessions: List[socket.socket] = []
        self._stub: Optional[hicache_pb2_grpc.StorageServerStub] = None
        self._channel: Optional[grpc.Channel] = None
        self._closed = False

        self._init_channel()

    def _page_km_kv(self) -> int:
        """How many flat storage keys comprise one logical KV page for aggregation."""
        if self.is_mla_backend:
            return 1
        if self.storage_config.should_split_heads:
            return 2 * max(1, self.split_factor)
        pool = getattr(self, "mem_pool_host", None)
        # layer_first layout stores one K/V Put per transformer layer per page (see get_page_buffer_meta).
        if pool is not None and getattr(pool, "layout", None) == "layer_first":
            return 2 * int(getattr(pool, "layer_num"))
        return 2

    def _init_channel(self) -> None:
        path = parse_unix_socket_uri(self._grpc_socket_uri)
        target = f"unix://{path}"
        self._channel = grpc.insecure_channel(
            target,
            options=(
                ("grpc.max_send_message_length", 512 * 1024 * 1024),
                ("grpc.max_receive_message_length", 512 * 1024 * 1024),
            ),
        )
        self._stub = hicache_pb2_grpc.StorageServerStub(self._channel)

    def close(self) -> None:
        self._closed = True
        for s in self._mempool_sessions:
            try:
                s.close()
            except OSError:
                pass
        self._mempool_sessions.clear()
        if self._channel is not None:
            self._channel.close()

    def _tag_keys(self, keys: List[str]) -> List[str]:
        if self.extra_backend_tag is None:
            return keys
        return [f"{self.extra_backend_tag}_{k}" for k in keys]

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        self.registered_pools[_pool_registry_key(PoolName.KV)] = mem_pool_host
        self._register_pool_fd(str(PoolName.KV.value), mem_pool_host)

    def register_mem_host_pool_v2(
        self, host_pool: HostKVCache, host_pool_name: Any
    ) -> None:
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        name_str = (
            host_pool_name.value
            if isinstance(host_pool_name, PoolName)
            else str(host_pool_name)
        )
        self.registered_pools[name_str] = host_pool
        if name_str == str(PoolName.KV.value) or host_pool_name == PoolName.KV:
            return
        self._register_pool_fd(name_str, host_pool)

    def _register_pool_fd(self, pool_name: str, host_pool: HostKVCache):
        allocator = host_pool.allocator
        assert isinstance(allocator, MemfdTensorAllocator)
        try:
            us = unix_connect(self._mempool_socket_uri)
            send_pool_registration(us, pool_name, allocator.fd)
            pool_id = recv_pool_registration_reply_pool_id(us)
            self._pool_server_ids[pool_name] = pool_id
            self._mempool_sessions.append(us)
        except OSError as e:
            logger.warning("FD registration failed for pool %s: %s", pool_name, e)

    def _server_pool_id(self, pool: str) -> int:
        pid = self._pool_server_ids.get(pool)
        if pid is None:
            raise RuntimeError(
                f"pool {pool!r} has no server pool_id; mempool RegisterPool did not complete"
            )
        return int(pid)

    def _offset(self, pool: str, ptr: int) -> int:
        host_pool = self.registered_pools.get(pool)
        if host_pool is None:
            raise RuntimeError(f"unknown pool for FD offset: {pool!r}")
        allocator = host_pool.allocator
        assert isinstance(allocator, MemfdTensorAllocator), (
            f"expected MemfdTensorAllocator, got {type(allocator).__name__}"
        )
        return allocator.offset_of(int(ptr))

    def _stub_call(self):
        stub = self._stub
        if stub is None or self._closed:
            raise RuntimeError("HiCacheRpcStorageStub not initialized or closed")
        return stub

    def _grpc_exist_ok(self, id_bytes: Sequence[bytes]) -> bool:
        if not id_bytes:
            return True
        stub = self._stub_call()
        try:
            resp = stub.Exist(
                ExistRequest(id=list(id_bytes)),
                timeout=self._grpc_timeout,
            )
        except grpc.RpcError as e:
            logger.warning("Exist rpc error: %s", e)
            return False
        got = list(resp.results)
        return len(got) == len(id_bytes) and all(got)

    def _grpc_put(
        self, pool: str, id_bytes: bytes, ptr: int, nbytes: int
    ) -> bool:
        stub = self._stub_call()
        iov = Iovec(
            offset=self._offset(pool, int(ptr)),
            length=max(0, int(nbytes)),
        )
        t = Transfer(
            id=id_bytes,
            iovec=[iov],
            pool_id=self._server_pool_id(pool),
        )
        try:
            resp = stub.BatchPut(
                BatchPutRequest(puts=[t]),
                timeout=self._grpc_timeout,
            )
        except grpc.RpcError:
            return False
        res = list(resp.results)
        return len(res) == 1 and res[0] == 0

    def _grpc_get(self, pool: str, id_bytes: bytes, ptr: int, nbytes: int) -> bool:
        stub = self._stub_call()
        iov = Iovec(
            offset=self._offset(pool, int(ptr)),
            length=max(0, int(nbytes)),
        )
        t = Transfer(
            id=id_bytes,
            iovec=[iov],
            pool_id=self._server_pool_id(pool),
        )
        try:
            resp = stub.BatchGet(
                BatchGetRequest(gets=[t]),
                timeout=self._grpc_timeout,
            )
        except grpc.RpcError:
            return False
        res = list(resp.results)
        return len(res) == 1 and res[0] == 0

    # ---- v2 ---------------------------------------------------------------
    def _get_hybrid_page_component_keys(
        self,
        page_keys: List[str],
        transfer: PoolTransfer,
    ) -> Tuple[List[str], int]:
        """Same layout as MooncakeStore for interoperability."""
        suffixes = []
        name = transfer.name
        if name == PoolName.INDEXER:
            suffixes = [f"_{self.mla_suffix}_{PoolName.INDEXER}"]
        elif name == PoolName.MAMBA:
            pools = getattr(self, "registered_pools", {})
            mamba_pool = pools.get(PoolName.MAMBA)
            conv_num = len(getattr(mamba_pool, "conv_buffer", None) or [])
            base_suffix = f"_{self.mha_suffix}"
            suffixes = [f"{base_suffix}_temporal"] + [
                f"{base_suffix}_conv_{i}" for i in range(conv_num)
            ]
        key_multiplier = max(1, len(suffixes))
        component_keys = [
            f"{page_key}{suffix}" for page_key in page_keys for suffix in suffixes
        ]
        return component_keys, key_multiplier

    def _batch_preprocess(self, keys, host_indices):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        if self.storage_config.should_split_heads:
            return self._get_mha_split_heads_buffer_meta(keys, host_indices)
        return self._get_mha_buffer_meta(keys, host_indices)

    def _get_mha_split_heads_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = (
            self.mem_pool_host.get_split_heads_page_buffer_meta(
                indices, self.split_factor
            )
        )
        key_list = []
        for key_ in keys:
            for suffix in self._mha_suffix_iter():
                key_list.append(f"{key_}_{suffix}_k")
                key_list.append(f"{key_}_{suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _mha_suffix_iter(self) -> List[str]:
        if isinstance(self.mha_suffix, list):
            return self.mha_suffix
        return [self.mha_suffix]

    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        suffix = self.mha_suffix
        pool = self.mem_pool_host
        if pool.layout == "layer_first":
            # One (k, v) pair per layer per page — must match get_page_buffer_meta ordering.
            ln = int(pool.layer_num)
            if len(ptr_list) != len(keys) * 2 * ln:
                raise RuntimeError(
                    f"HiCacheRpcStorage MHA key/meta mismatch: ptrs={len(ptr_list)} "
                    f"pages={len(keys)} layers={ln}"
                )
            for key_ in keys:
                for layer_id in range(ln):
                    key_list.append(f"{key_}_{suffix}_L{layer_id}_k")
                    key_list.append(f"{key_}_{suffix}_L{layer_id}_v")
        else:
            # page_first / page_first_direct / page_head: one contiguous K and V blob per page.
            for key_ in keys:
                key_list.append(f"{key_}_{suffix}_k")
                key_list.append(f"{key_}_{suffix}_v")
            if len(key_list) != len(ptr_list):
                raise RuntimeError(
                    f"HiCacheRpcStorage MHA key/meta mismatch: keys={len(key_list)} ptrs={len(ptr_list)}"
                )
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        pool = self.mem_pool_host
        if pool.layout == "layer_first":
            # One key per layer per page — must match get_page_buffer_meta ordering.
            ln = int(pool.layer_num)
            if len(ptr_list) != len(keys) * ln:
                raise RuntimeError(
                    f"HiCacheRpcStorage MLA key/meta mismatch: ptrs={len(ptr_list)} "
                    f"pages={len(keys)} layers={ln}"
                )
            for key_ in keys:
                for layer_id in range(ln):
                    key_list.append(f"{key_}_{self.mla_suffix}_L{layer_id}_k")
        else:
            # page_first / page_first_direct: one contiguous blob per page.
            for key_ in keys:
                key_list.append(f"{key_}_{self.mla_suffix}_k")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        qkeys = self._tag_keys(keys)
        kv_pages = self.batch_exists(qkeys, extra_info)

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            component_keys, key_multiplier = self._get_hybrid_page_component_keys(
                qkeys, transfer
            )
            ex = self._batch_exist_rpc(component_keys)
            if key_multiplier > 0:
                page_exists = [
                    all(
                        r
                        for r in ex[
                            i * key_multiplier : (i + 1) * key_multiplier
                        ]
                    )
                    for i in range(kv_pages)
                ]
            else:
                page_exists = [False] * kv_pages
            boundary = 0
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                try:
                    boundary = page_exists.index(False)
                except ValueError:
                    boundary = kv_pages
            else:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        page_exists[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[transfer.name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _batch_exist_rpc(self, component_keys: List[str]) -> List[bool]:
        if not component_keys:
            return []
        stub = self._stub_call()
        try:
            resp = stub.Exist(
                ExistRequest(id=[ck.encode("utf-8") for ck in component_keys]),
                timeout=self._grpc_timeout,
            )
        except grpc.RpcError:
            return [False] * len(component_keys)
        got = list(resp.results)
        if len(got) != len(component_keys):
            return [False] * len(component_keys)
        return [bool(x) for x in got]

    def _batch_io_v2(self, transfers: List[PoolTransfer], is_set: bool):
        results: Dict[str, List[bool]] = {}
        for transfer in transfers:
            pk = _pool_registry_key(transfer.name)
            host_pool = self.registered_pools.get(pk)
            if host_pool is None:
                logger.error("missing host pool %s", transfer.name)
                results[pk] = []
                continue
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            host_indices = transfer.host_indices
            if (
                host_indices is None
                or host_indices.numel() != len(keys) * page_size
            ):
                results[pk] = [False] * len(keys)
                continue

            if transfer.name == PoolName.KV:
                key_strs, ptr_list, element_size_list = self._batch_preprocess(
                    self._tag_keys(keys), host_indices
                )
                km = self._page_km_kv()
            else:
                ptr_list, element_size_list = host_pool.get_page_buffer_meta(
                    host_indices
                )
                key_strs, km = self._get_hybrid_page_component_keys(keys, transfer)
                key_strs = self._tag_keys(key_strs)

            pool_name_str = pk
            pid = self._server_pool_id(pool_name_str)
            xfer_pb: List[Transfer] = []
            for ks, ptr, nbytes in zip(key_strs, ptr_list, element_size_list):
                xfer_pb.append(
                    Transfer(
                        id=ks.encode("utf-8"),
                        iovec=[
                            Iovec(
                                offset=self._offset(pool_name_str, int(ptr)),
                                length=max(0, int(nbytes)),
                            )
                        ],
                        pool_id=pid,
                    )
                )
            stub = self._stub_call()
            row_flags: List[bool]
            try:
                if is_set:
                    resp = stub.BatchPut(
                        BatchPutRequest(puts=xfer_pb),
                        timeout=self._grpc_timeout,
                    )
                    codes = list(resp.results)
                else:
                    resp = stub.BatchGet(
                        BatchGetRequest(gets=xfer_pb),
                        timeout=self._grpc_timeout,
                    )
                    codes = list(resp.results)
            except grpc.RpcError:
                row_flags = [False] * len(key_strs)
            else:
                if len(codes) != len(key_strs):
                    row_flags = [False] * len(key_strs)
                else:
                    row_flags = [c == 0 for c in codes]

            results[pk] = self._aggregate_flat_flags(row_flags, km)
        return results

    def _aggregate_flat_flags(
        self, flags: List[bool], key_multiplier: Optional[int]
    ) -> List[bool]:
        km = (
            key_multiplier
            if key_multiplier is not None
            else (1 if self.is_mla_backend else 2)
        )
        return [
            all(flags[i : i + km]) for i in range(0, len(flags), km)
        ]

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        return self._batch_io_v2(transfers, is_set=False)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        return self._batch_io_v2(transfers, is_set=True)

    # ---- v1 / legacy ------------------------------------------------------
    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        keys = self._tag_keys(keys)
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(
            keys, host_indices
        )
        results = []
        for ks, ptr, nbytes in zip(key_strs, buffer_ptrs, buffer_sizes):
            ok = self._grpc_get(
                str(PoolName.KV.value), ks.encode(), int(ptr), int(nbytes)
            )
            results.append(ok)
        return self._aggregate_flat_flags(results, self._page_km_kv())

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        keys = self._tag_keys(keys)
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(
            keys, host_indices
        )
        results = []
        exist_f = self._batch_exist_rpc(key_strs)
        for i, (ks, ptr, nbytes) in enumerate(
            zip(key_strs, buffer_ptrs, buffer_sizes)
        ):
            if exist_f[i]:
                results.append(True)
                continue
            ok = self._grpc_put(
                str(PoolName.KV.value), ks.encode(), int(ptr), int(nbytes)
            )
            results.append(ok)
        return self._aggregate_flat_flags(results, self._page_km_kv())

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        assert target_location is not None
        loc = target_location
        nbytes = (
            loc.numel() * loc.element_size()
            if isinstance(loc, torch.Tensor)
            else int(target_sizes or 0)
        )
        if nbytes <= 0:
            return None
        ptr = int(loc.data_ptr()) if isinstance(loc, torch.Tensor) else int(loc)
        ok = self._grpc_get(
            str(PoolName.KV.value),
            key.encode("utf-8"),
            ptr,
            nbytes,
        )
        return loc if ok else None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        assert target_locations is not None and target_sizes is not None
        if len(keys) == 0:
            return 0
        mult = 1 if self.is_mla_backend else 2
        for i, k in enumerate(keys):
            loc = target_locations[i]
            nbytes = (
                loc.numel() * loc.element_size()
                if isinstance(loc, torch.Tensor)
                else int(target_sizes[i])
            )
            ptr = (
                int(loc.data_ptr())
                if isinstance(loc, torch.Tensor)
                else int(loc)
            )
            if not self._grpc_get(
                str(PoolName.KV.value),
                k.encode("utf-8"),
                ptr,
                nbytes,
            ):
                return i // mult
        return len(keys) // mult

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        assert target_location is not None
        loc = target_location
        nbytes = (
            loc.numel() * loc.element_size()
            if isinstance(loc, torch.Tensor)
            else int(target_sizes or 0)
        )
        if nbytes <= 0:
            return False
        ptr = int(loc.data_ptr()) if isinstance(loc, torch.Tensor) else int(loc)
        if self.exists(key):
            return True
        return self._grpc_put(
            str(PoolName.KV.value),
            key.encode("utf-8"),
            ptr,
            nbytes,
        )

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if len(keys) == 0:
            return False
        if values is not None:
            raise RuntimeError(
                "HiCacheRpcStorage does not support HiCacheFile-style batch_set(keys, page_tensors); "
                "use batch_set_v1(keys, host_indices). The rpc backend stores each page as k/v "
                "components with FD offsets (see _batch_preprocess)."
            )
        if target_locations is None or target_sizes is None:
            raise RuntimeError(
                "HiCacheRpcStorage batch_set requires target_locations and target_sizes; "
                "for host pool pages use batch_set_v1(keys, host_indices)."
            )
        for i, k in enumerate(keys):
            if not self.set(
                k,
                target_location=target_locations[i],
                target_sizes=target_sizes[i],
            ):
                return False
        return True

    def exists(self, key: str) -> bool:
        return self._grpc_exist_ok([key.encode("utf-8")])

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        qkeys = self._tag_keys(keys)
        if self.is_mla_backend:
            pool = getattr(self, "mem_pool_host", None)
            if pool is not None and getattr(pool, "layout", None) == "layer_first":
                ln = int(getattr(pool, "layer_num"))
                query_keys = []
                for key in qkeys:
                    for lid in range(ln):
                        query_keys.append(f"{key}_{self.mla_suffix}_L{lid}_k")
                km = ln
            else:
                query_keys = [f"{key}_{self.mla_suffix}_k" for key in qkeys]
                km = 1
        else:
            query_keys = []
            if self.storage_config.should_split_heads:
                for key in qkeys:
                    for suffix in self._mha_suffix_iter():
                        query_keys.append(f"{key}_{suffix}_k")
                        query_keys.append(f"{key}_{suffix}_v")
                km = 2 * max(1, self.split_factor)
            else:
                pool = getattr(self, "mem_pool_host", None)
                if pool is not None and getattr(pool, "layout", None) == "layer_first":
                    ln = int(getattr(pool, "layer_num"))
                    for key in qkeys:
                        for lid in range(ln):
                            query_keys.append(f"{key}_{self.mha_suffix}_L{lid}_k")
                            query_keys.append(f"{key}_{self.mha_suffix}_L{lid}_v")
                    km = 2 * ln
                else:
                    for key in qkeys:
                        query_keys.append(f"{key}_{self.mha_suffix}_k")
                        query_keys.append(f"{key}_{self.mha_suffix}_v")
                    km = 2

        max_pages = len(qkeys)
        if max_pages == 0 or not query_keys:
            return 0

        best = 0
        for prefix_len in range(1, max_pages + 1):
            chunk = query_keys[: prefix_len * km]
            if not self._grpc_exist_ok([c.encode("utf-8") for c in chunk]):
                break
            best = prefix_len
        return best

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


__all__ = [
    "HiCacheRpcStorage",
    "MemfdTensorAllocator",
]
