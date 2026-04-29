# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""HiCache L3 storage backend for Ascend MemCache.

This backend is implemented in parallel to Mooncake (not inheriting MooncakeStore).
It follows the same HiCacheStorage contract and key layout strategy, while using
`memcache_hybrid.DistributedObjectStore` as the underlying object store.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, List, Optional, Tuple

import requests
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.observability.metrics_collector import StorageMetrics

SETUP_TIMEOUT = 600  # seconds

logger = logging.getLogger(__name__)

# Keys consumed by SGLang wrapper logic, not forwarded to memcache_hybrid.LocalConfig.
# memcache 不用的接口
_MEMCACHE_CTRL_KEYS = frozenset(
    {
        "device_id",
        "init_bm",
        "check_server",
        "metrics_url",
        "memcache_metrics_url",
        "extra_backend_tag",
    }
)


#------------------------------------------------------------
# 对应MooncakeConfig
#Mooncake通过SGLANG_HICACHE_MOONCAKE_CONFIG_PATH和extra_config来获取配置文件
#Memcache仿照Mooncake，通过SGLANG_HICACHE_MEMCACHE_CONFIG_PATH和extra_config来获取配置文件
#memcache LocalConfig 接口https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_python_api.md#localconfig-%E7%B1%BB

def _merge_memcache_config_dict(storage_config: Optional[HiCacheStorageConfig]) -> dict:
    merged: dict = {}
    if envs.SGLANG_HICACHE_MEMCACHE_CONFIG_PATH.is_set():
        path = envs.SGLANG_HICACHE_MEMCACHE_CONFIG_PATH.get()
        with open(path, encoding="utf-8") as fin:
            merged.update(json.load(fin))
        logger.info("Memcache configuration loaded from %s", path)
    extra = getattr(storage_config, "extra_config", None) or {}
    merged.update(extra)
    return merged



#把merged字段分为两部分，一部分是memcache真正需要用的字段，另一部分是sglang自己控制的
def _split_memcache_ctrl(merged: dict) -> tuple[dict, dict]:
    ctrl = {k: merged[k] for k in _MEMCACHE_CTRL_KEYS if k in merged}
    local_fields = {k: v for k, v in merged.items() if k not in _MEMCACHE_CTRL_KEYS}
    return local_fields, ctrl


#将字典写入对象，把未知字段忽略，若字段需要严格匹配则需要重改
def _apply_dict_to_local_config(local_cfg: Any, data: dict) -> List[str]:
    unknown: List[str] = []
    for key, value in data.items():
        if not hasattr(local_cfg, key):
            unknown.append(key)
            continue
        setattr(local_cfg, key, value)
    return unknown


#------------------------------------------------------------

class AscendMemcacheStore(HiCacheStorage):
    """HiCache storage backend backed by Ascend MemCache (`memcache_hybrid`)."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig = None,
        _mem_pool: HostKVCache = None,
    ):
        self.store = None
        self.storage_config = storage_config
        

        # 导入memcache接口，Mooncake通过MooncakeBaseStore导入
        try:
            from memcache_hybrid import DistributedObjectStore, LocalConfig
        except ImportError as e:
            raise ImportError(
                "Ascend Memcache HiCache backend requires `memcache_hybrid`. "
                "Install it with `pip install memcache_hybrid` and deploy "
                "MetaService/LocalService according to https://gitcode.com/Ascend/memcache"
            ) from e

        try:
            #-----
            # 合并配置文件和extra_config
            merged = _merge_memcache_config_dict(storage_config)
            local_fields, ctrl = _split_memcache_ctrl(merged)

            local_cfg = LocalConfig()
            unknown_fields = _apply_dict_to_local_config(local_cfg, local_fields)
            if unknown_fields:
                # Keep Mooncake-like tolerance, but surface invalid config keys
                # clearly to avoid silent setup/init failures.
                logger.warning(
                    "Ignoring unknown Memcache LocalConfig keys: %s", unknown_fields
                )
            #-----

            #创建对象
            self.store = DistributedObjectStore()
            if self.store.setup(local_cfg) != 0:
                raise RuntimeError(
                    "memcache_hybrid.DistributedObjectStore.setup failed"
                )

            device_id = int(ctrl.get("device_id", 0))
            init_bm = bool(ctrl.get("init_bm", True))
            if self.store.init(device_id, init_bm) != 0:
                raise RuntimeError("memcache_hybrid.DistributedObjectStore.init failed")
            #都与mooncake类似
            self._memcache_metrics_url = ctrl.get("metrics_url") or ctrl.get(
                "memcache_metrics_url"
            )
            self._check_server_enabled = bool(ctrl.get("check_server", False))
            self.extra_backend_tag = ctrl.get("extra_backend_tag")

            if self._check_server_enabled:
                self.check_server()

            # memcache_hybrid documents init_bm=False as pure client mode without
            # data read/write capability; skip warmup put/get in that mode.
            if init_bm:
                self.warmup()
            else:
                logger.info(
                    "Memcache init_bm is False; skip warmup because read/write is unavailable in pure client mode."
                )
            self._init_runtime_fields(storage_config)

        except ValueError as e:
            logger.error("Ascend Memcache configuration failed: %s", e)
            raise
        except Exception as exc:
            logger.error("Ascend Memcache store initialization failed: %s", exc)
            raise

    def _init_runtime_fields(self, storage_config: Optional[HiCacheStorageConfig]) -> None:
        self.enable_storage_metrics = False
        if storage_config is not None:
            self.is_mla_backend = storage_config.is_mla_model
            self.local_rank = storage_config.tp_rank
            self.pp_rank = storage_config.pp_rank
            self.pp_size = storage_config.pp_size
            self.attn_cp_rank = storage_config.attn_cp_rank
            self.attn_cp_size = storage_config.attn_cp_size
            self.enable_storage_metrics = storage_config.enable_storage_metrics
        else:
            self.is_mla_backend = False
            self.local_rank = 0
            self.pp_rank = 0
            self.pp_size = 1
            self.attn_cp_rank = 0
            self.attn_cp_size = 1

        self.enable_pp = self.pp_size > 1
        self.enable_cp = self.attn_cp_size > 1
        if self.enable_pp or self.enable_cp:
            self.mha_suffix = f"{self.local_rank}_{self.pp_rank}_{self.attn_cp_rank}"
            self.mla_suffix = f"{self.pp_rank}_{self.attn_cp_rank}"
        else:
            self.mha_suffix = f"{self.local_rank}"
            self.mla_suffix = ""

        self.split_factor = 0
        if self.storage_config is not None and self.storage_config.should_split_heads:
            self.split_factor = (
                self.storage_config.tp_lcm_size // self.storage_config.tp_size
            )
            base_rank = self.local_rank * self.split_factor
            target_ranks = [base_rank + i for i in range(self.split_factor)]
            if self.enable_pp or self.enable_cp:
                self.mha_suffix = [
                    f"{rank}_{self.pp_rank}_{self.attn_cp_rank}" for rank in target_ranks
                ]
            else:
                self.mha_suffix = [f"{rank}" for rank in target_ranks]

        self.registered_pools = {}
        self.gb_per_page = None
        self.prefetch_pgs = []
        self.backup_pgs = []
        self.prefetch_bandwidth = []
        self.backup_bandwidth = []

    #与Mooncake一致
    def register_buffer(self, tensor: torch.Tensor):
        if self.store is None:
            raise RuntimeError("Ascend Memcache store is not initialized.")
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        ret_code = self.store.register_buffer(ptr, size)
        if ret_code != 0:
            logger.error("Failed to register buffer, error code: %s", ret_code)
            raise RuntimeError(
                f"Failed to register buffer to Ascend Memcache, error code: {ret_code}"
            )
    #与Mooncake一致
    def check_server(self) -> None:
        url = self._memcache_metrics_url
        if not url:
            logger.warning(
                "Memcache check_server is true but no metrics_url/memcache_metrics_url was provided; skipping readiness wait."
            )
            return

        start = time.perf_counter()
        while time.perf_counter() - start < SETUP_TIMEOUT:
            try:
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    logger.info("Memcache metrics endpoint is reachable.")
                    return
            except Exception:
                pass
            logger.info(
                "Waiting for Memcache metrics endpoint at %s (%.1fs elapsed).",
                url,
                time.perf_counter() - start,
            )
            time.sleep(3)

        raise TimeoutError(
            f"Timed out after {SETUP_TIMEOUT}s waiting for Memcache metrics URL {url}"
        )

    def warmup(self):
        # warmup_key = "sglang_ascend_memcache_store_warmup_key" + uuid.uuid4().hex
        # warmup_value = bytes(4 * 1024)
        # assert self.store.put(warmup_key, warmup_value) == 0
        # assert self.store.is_exist(warmup_key) == 1
        # assert self.store.get(warmup_key) == warmup_value

        warmup_key = "sglang_ascend_memcache_store_warmup_key" + uuid.uuid4().hex

        buf = torch.zeros(4096, dtype=torch.uint8)
        ptr = buf.data_ptr()
        size = buf.numel() * buf.element_size()

        put_ret = self.store.batch_put_from([warmup_key], [ptr], [size])
        assert put_ret[0] == 0, f"warmup batch_put_from failed: {put_ret}"

        exist_ret = self.store.batch_is_exist([warmup_key])
        assert exist_ret[0] == 1, f"warmup batch_is_exist failed: {exist_ret}"

        out = torch.empty_like(buf)
        get_ret = self.store.batch_get_into([warmup_key], [out.data_ptr()], [size])
        assert get_ret[0] == 0, f"warmup batch_get_into failed: {get_ret}"
        

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        assert self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
            "page_head",
            "page_first_kv_split",
            "page_first_kv_spilt",
        ], (
            "ascend_memcache storage backend only supports page_first, "
            "page_first_direct, page_head and page_first_kv_split layout"
        )
        buffer = self.mem_pool_host.kv_buffer
        self.register_buffer(buffer)

        bytes_per_page = mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size
        self.gb_per_page = bytes_per_page / (1 << 30)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if host_pool_name == PoolName.KV:
            return
        self.registered_pools[host_pool_name] = host_pool
        buf_list = host_pool.get_hybrid_pool_buffer()
        for buf in buf_list:
            self.register_buffer(buf)

    def _tag_keys(self, keys: List[str]) -> List[str]:
        if self.extra_backend_tag is None:
            return keys
        return [f"{self.extra_backend_tag}_{key}" for key in keys]

    def _get_hybrid_page_component_keys(
        self, page_keys: List[str], transfer: PoolTransfer
    ) -> Tuple[List[str], int]:
        name = transfer.name
        suffixes = []
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
        key_multiplier = len(suffixes)
        component_keys = [
            f"{page_key}{suffix}" for page_key in page_keys for suffix in suffixes
        ]
        return component_keys, key_multiplier

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
            ex = self._batch_exist(component_keys)
            if key_multiplier > 0:
                page_exists = [
                    all(
                        r == 1
                        for r in ex[i * key_multiplier : (i + 1) * key_multiplier]
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
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
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

    def _batch_io_v2(self, transfers: List[PoolTransfer], is_set: bool):
        results: dict = {}
        for transfer in transfers:
            host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
            if host_pool is None:
                raise RuntimeError(
                    f"Host pool '{transfer.name}' is not registered. "
                    "Call register_mem_host_pool_v2() before batch_get_v2/batch_set_v2."
                )
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            host_indices = transfer.host_indices
            if len(keys) == 0:
                raise ValueError(
                    f"PoolTransfer '{transfer.name}' has empty keys in batch v2 I/O."
                )
            if host_indices is None:
                raise ValueError(
                    f"PoolTransfer '{transfer.name}' has null host_indices in batch v2 I/O."
                )
            if len(keys) != len(host_indices) // page_size:
                raise ValueError(
                    f"PoolTransfer '{transfer.name}' keys/host_indices mismatch: "
                    f"len(keys)={len(keys)}, len(host_indices)={len(host_indices)}, page_size={page_size}."
                )

            ptr_list, element_size_list = host_pool.get_page_buffer_meta(host_indices)
            key_strs, key_multiplier = self._get_hybrid_page_component_keys(
                keys, transfer
            )
            key_strs = self._tag_keys(key_strs)

            if is_set:
                exist_result = self._batch_exist(key_strs)
                io_results = [0 if state == 1 else -1 for state in exist_result]
                missing_idx = [i for i, state in enumerate(exist_result) if state != 1]
                if missing_idx:
                    put_results = self._put_batch_zero_copy_impl(
                        [key_strs[i] for i in missing_idx],
                        [ptr_list[i] for i in missing_idx],
                        [element_size_list[i] for i in missing_idx],
                    )
                    for i, res in zip(missing_idx, put_results):
                        io_results[i] = res
            else:
                io_results = self._get_batch_zero_copy_impl(
                    key_strs, ptr_list, element_size_list
                )
            results[transfer.name] = self._batch_postprocess(
                io_results, is_set_operate=is_set, key_multiplier=key_multiplier
            )
        return results

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

    def _get_mha_split_heads_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_split_heads_page_buffer_meta(
            indices, self.split_factor
        )
        key_list = []
        for key_ in keys:
            for suffix in self.mha_suffix:
                key_list.append(f"{key_}_{suffix}_k")
                key_list.append(f"{key_}_{suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mha_suffix}_k")
            key_list.append(f"{key_}_{self.mha_suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mla_suffix}_k")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        if self.storage_config and self.storage_config.should_split_heads:
            return self._get_mha_split_heads_buffer_meta(keys, host_indices)
        return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(
        self, results: List[int], is_set_operate: bool = False, key_multiplier=None
    ):
        if key_multiplier is None:
            if self.is_mla_backend:
                key_multiplier = 1
            else:
                key_multiplier = 2
                if self.storage_config and self.storage_config.should_split_heads:
                    key_multiplier *= self.split_factor

        result_groups = [
            results[i : i + key_multiplier] for i in range(0, len(results), key_multiplier)
        ]
        return [
            all(res == 0 for res in group)
            if is_set_operate
            else all(res > 0 for res in group)
            for group in result_groups
        ]

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        keys = self._tag_keys(keys)
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)

        start_time = time.perf_counter()
        get_results = self._get_batch_zero_copy_impl(
            key_strs, buffer_ptrs, buffer_sizes
        )
        end_time = time.perf_counter()

        if self.enable_storage_metrics and end_time > start_time:
            self.prefetch_pgs.append(len(keys))
            self.prefetch_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        return self._batch_postprocess(get_results, is_set_operate=False)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        keys = self._tag_keys(keys)
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        exist_result = self._batch_exist(key_strs)

        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [-1] * len(key_strs)
        for i in range(len(key_strs)):
            if exist_result[i] != 1:
                set_keys.append(key_strs[i])
                set_buffer_ptrs.append(buffer_ptrs[i])
                set_buffer_sizes.append(buffer_sizes[i])
                set_indices.append(i)
            else:
                set_results[i] = 0

        if set_keys:
            start_time = time.perf_counter()
            put_results = self._put_batch_zero_copy_impl(
                set_keys, set_buffer_ptrs, set_buffer_sizes
            )
            end_time = time.perf_counter()

            if self.enable_storage_metrics and end_time > start_time:
                self.backup_pgs.append(len(set_keys))
                self.backup_bandwidth.append(
                    len(set_keys) / (end_time - start_time) * self.gb_per_page
                )

            for i in range(len(set_indices)):
                set_results[set_indices[i]] = put_results[i]

        return self._batch_postprocess(set_results, is_set_operate=True)

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        _ = value
        assert target_location is not None and target_sizes is not None
        exist_result = self._batch_exist([key])
        if exist_result[0] == 1:
            return True
        put_result = self._put_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return put_result[0] == 0

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        _ = values
        assert target_locations is not None and target_sizes is not None
        assert len(keys) == len(target_locations) == len(target_sizes)

        if len(keys) == 0:
            return False

        for i in range(len(keys)):
            if (
                keys[i] is None
                or target_locations[i] is None
                or target_sizes[i] is None
            ):
                return False

        exist_result = self._batch_exist(keys)
        set_keys = []
        set_target_locations = []
        set_target_sizes = []
        set_indices = []
        for i in range(len(keys)):
            if exist_result[i] != 1:
                set_keys.append(keys[i])
                set_target_locations.append(target_locations[i])
                set_target_sizes.append(target_sizes[i])
                set_indices.append(i)

        start_time = time.perf_counter()
        put_result = self._put_batch_zero_copy_impl(
            set_keys, set_target_locations, set_target_sizes
        )
        end_time = time.perf_counter()

        if self.enable_storage_metrics and set_keys and end_time > start_time:
            self.backup_pgs.append(len(set_keys))
            self.backup_bandwidth.append(
                len(set_keys) / (end_time - start_time) * self.gb_per_page
            )

        for i in range(len(set_indices)):
            if put_result[i] == 0:
                exist_result[set_indices[i]] = 1

        success_count = 0
        for i in range(len(keys)):
            if exist_result[i] == 0:
                break
            success_count += 1
        return success_count == len(keys)

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        assert target_location is not None and target_sizes is not None
        get_result = self._get_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return get_result[0] >= 0

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        assert len(keys) == len(target_locations) == len(target_sizes)
        if len(keys) == 0:
            return 0

        start_time = time.perf_counter()
        get_result = self._get_batch_zero_copy_impl(
            keys, target_locations, target_sizes
        )
        end_time = time.perf_counter()

        key_multiplier = 1 if self.is_mla_backend else 2

        if self.enable_storage_metrics and end_time > start_time:
            self.prefetch_pgs.append(len(keys))
            self.prefetch_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        for i in range(len(keys)):
            if get_result[i] < 0:
                return i // key_multiplier
        return len(keys) // key_multiplier

    def exists(self, key: str) -> bool:
        exist_result = self._batch_exist([key])
        return exist_result[0] == 1

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        keys = self._tag_keys(keys)

        if self.is_mla_backend:
            query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]
            key_multiplier = 1
        else:
            query_keys = []
            if self.storage_config and self.storage_config.should_split_heads:
                for key in keys:
                    for suffix in self.mha_suffix:
                        query_keys.append(f"{key}_{suffix}_k")
                        query_keys.append(f"{key}_{suffix}_v")
                key_multiplier = 2 * self.split_factor
            else:
                for key in keys:
                    query_keys.append(f"{key}_{self.mha_suffix}_k")
                    query_keys.append(f"{key}_{self.mha_suffix}_v")
                key_multiplier = 2

        exist_result = self._batch_exist(query_keys)
        for i in range(len(query_keys)):
            if exist_result[i] != 1:
                return i // key_multiplier
        return len(query_keys) // key_multiplier

    def clear(self) -> None:
        self.store.remove_all()

    def close(self) -> None:
        if self.store is None:
            return
        try:
            self.store.close()
        except Exception as e:
            logger.warning("Ascend Memcache store.close failed: %s", e)
        self.store = None

    def _put_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[int]:
        return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)

    def _get_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[int]:
        raw = self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)
        # memcache_hybrid reports 0 on success, but HiCache read postprocess expects
        # positive values for success and negative values for failures.
        out: List[int] = []
        for code, sz in zip(raw, buffer_sizes):
            out.append(int(sz) if code == 0 else int(code))
        return out

    def _batch_exist(self, key_strs: List[str]) -> List[int]:
        return self.store.batch_is_exist(key_strs)

    def get_stats(self):
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return storage_metrics
