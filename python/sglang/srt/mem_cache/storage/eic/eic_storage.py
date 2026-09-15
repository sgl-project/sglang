import json
import logging
import os
import time
from typing import Any, List, Optional, Tuple

import eic
import torch
import yaml

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host import HostKVCache

logger = logging.getLogger(__name__)


TensorPoolSize = 2048

REMOTE_EIC_YAML_ENV_VAR = "REMOTE_EIC_YAML"

# gpu direct rdma for kv set
G_EnableKVSetGPUDirect = False

# gpu direct rdma for kv get
G_EnableKVGetGPUDirect = False

# gpu nic affinity
G_EnableGPUNicAffinity = False

# default H20 gpu nic affinity
GPUNicAffinity = {
    "cuda:0": "eth1",
    "cuda:1": "eth1",
    "cuda:2": "eth2",
    "cuda:3": "eth2",
    "cuda:4": "eth3",
    "cuda:5": "eth3",
    "cuda:6": "eth4",
    "cuda:7": "eth4",
}

# default H20 cpu nic affinity
CPUNicAffinity = {
    "cuda:0": "cpu",
    "cuda:1": "cpu",
    "cuda:2": "cpu",
    "cuda:3": "cpu",
    "cuda:4": "cpu",
    "cuda:5": "cpu",
    "cuda:6": "cpu",
    "cuda:7": "cpu",
}


def get_eic_config_file_path():
    if os.environ.get(REMOTE_EIC_YAML_ENV_VAR) is not None:
        logger.info(f"eic init with env var {REMOTE_EIC_YAML_ENV_VAR}")
        config_file = os.environ.get(REMOTE_EIC_YAML_ENV_VAR)
    else:
        config_file = "/sgl-workspace/config/remote-eic.yaml"
        logger.info(f"eic init with default config, config_file {config_file}")
    return config_file


class FlexibleKVCacheMemoryPool:
    def __init__(self, conn, kvcache_shape, kvcache_dtype, device):
        self.connection = conn

        if device.startswith("cpu") and G_EnableGPUNicAffinity:
            gpu_id = torch.cuda.current_device()
            self.device = CPUNicAffinity["cuda:" + str(gpu_id)]
            # current memory pool size is 5 times of CPU TensorPoolSize
            mempool_size = TensorPoolSize * 5
        else:
            self.device = device
            mempool_size = TensorPoolSize

        self.kvcache_shape = kvcache_shape
        self.kvcache_dtype = kvcache_dtype

        self.kv_cache_numel = 1
        for i in self.kvcache_shape:
            self.kv_cache_numel *= i

        self.free_data_addr = set()
        self.data_ptr_to_index = dict()

        if self.device.startswith("cpu"):
            self.kvcache_mempool = torch.zeros(
                (mempool_size,) + kvcache_shape,
                dtype=kvcache_dtype,
                device=self.device,
                pin_memory=True,
            )
        else:
            self.kvcache_mempool = torch.zeros(
                (mempool_size,) + kvcache_shape, dtype=kvcache_dtype, device=self.device
            )

        for i in range(mempool_size):
            self.free_data_addr.add(i)
            self.data_ptr_to_index[self.kvcache_mempool[i].data_ptr()] = i

        meminfo = eic.MemoryInfo()
        meminfo.type = eic.MemoryType.MEMORY_CUDA
        meminfo.cuda_id = 0
        vals = eic.IOBuffers()
        vals.append(
            self.kvcache_mempool.data_ptr(),
            self.kvcache_mempool.numel() * self.kvcache_mempool.element_size(),
            True,
        )
        self.connection.register_memory(vals, meminfo)
        logger.info(
            f"allocate memory pool, size {self.kvcache_mempool.numel() * self.kvcache_mempool.element_size()}, device {self.device}"
        )

    def try_allocate_kv_cache(self, shape, dtype, count=1):
        if len(self.free_data_addr) < count:
            return None

        numel = 1
        for i in shape:
            numel *= i
        if numel != self.kv_cache_numel or dtype != self.kvcache_dtype:
            logger.error(
                f"allocate from mempool failed, self.kvcache_shape {self.kvcache_shape}, dtype {self.kvcache_dtype}, require shape {shape}, dtype {dtype}"
            )
            return None

        ret = []
        for _ in range(count):
            free_index = self.free_data_addr.pop()
            ret.append(self.kvcache_mempool[free_index])
        return ret

    def free_to_mempool(self, data_ptr):
        if data_ptr not in self.data_ptr_to_index:
            logger.error(
                f"free_to_mempool failed, data_ptr {data_ptr} not in allocated_data_addr"
            )
            return
        self.free_data_addr.add(self.data_ptr_to_index[data_ptr])

    def check_data_ptr_allocated(self, data_ptr):
        return data_ptr in self.data_ptr_to_index

    def left_count(self):
        return len(self.free_data_addr)


class EICStorage(HiCacheStorage):
    def __init__(
        self, hicache_config: HiCacheStorageConfig, memory_pool_host: HostKVCache
    ):
        global G_EnableKVSetGPUDirect, G_EnableKVGetGPUDirect
        global GPUNicAffinity, CPUNicAffinity, G_EnableGPUNicAffinity

        config_file = get_eic_config_file_path()
        if os.path.exists(config_file) is False:
            logger.error(f"config file {config_file} not exists")
            raise RuntimeError(f"eic config file {config_file} not exists")

        with open(config_file, "r") as fin:
            config = yaml.safe_load(fin)

        remote_url = config.get("remote_url", None)
        if remote_url is None:
            AssertionError("remote_url is None")

        endpoint = remote_url[len("eic://") :]

        logger.info(f"eic remote_url:" + remote_url + " endpoint: " + endpoint)

        eic_instance_id = config.get("eic_instance_id", None)
        logger.info(f"eic instance_id: {eic_instance_id}")

        eic_thread_num = config.get("eic_thread_num", 1)
        logger.info(f"eic thread_num: {eic_thread_num}")

        eic_log_dir = config.get("eic_log_dir", None)
        logger.info(f"eic log_dir: {eic_log_dir}")

        eic_log_level = config.get("eic_log_level", 2)
        logger.info(f"eic log_level: {eic_log_level}")

        eic_trans_type = config.get("eic_trans_type", 3)
        logger.info(f"eic trans_type: {eic_trans_type}")

        eic_flag_file = config.get("eic_flag_file", None)
        logger.info(f"eic flag_file: {eic_flag_file}")

        # GDR now is not used
        G_EnableKVSetGPUDirect = (
            config.get("enable_kvset_gpu_direct", False) and torch.cuda.is_available()
        )
        logger.debug(f"eic enable_kvset_gpu_direct: {G_EnableKVSetGPUDirect}")

        G_EnableKVGetGPUDirect = (
            config.get("enable_kvget_gpu_direct", False) and torch.cuda.is_available()
        )
        logger.debug(f"eic enable_kvget_gpu_direct: {G_EnableKVGetGPUDirect}")

        self.model_name = hicache_config.model_name

        # rdma
        enable_kv_set_direct = config.get("enable_kvset_direct", True)
        logger.info(f"eic enable_kv_set_direct: {enable_kv_set_direct}")
        self.enable_kv_set_direct = enable_kv_set_direct

        enable_kv_get_direct = config.get("enable_kvget_direct", True)
        logger.info(f"eic enable_kv_get_direct: {enable_kv_get_direct}")
        self.enable_kv_get_direct = enable_kv_get_direct

        # gpu nic affinity
        G_EnableGPUNicAffinity = config.get("enable_gpu_nic_affinity", False)
        logger.info(f"eic enable_gpu_nic_affinity: {G_EnableGPUNicAffinity}")
        self.enable_gpu_nic_affinity = G_EnableGPUNicAffinity

        if G_EnableGPUNicAffinity:
            if "gpu_nic_affinity_config" in config:
                GPUNicAffinity = json.loads(config["gpu_nic_affinity_config"])
            if "cpu_nic_affinity_config" in config:
                CPUNicAffinity = json.loads(config["cpu_nic_affinity_config"])
            logger.info(f"eic gpu nic affinity {GPUNicAffinity}")
            logger.info(f"eic cpu nic affinity {CPUNicAffinity}")

        eic_namespace = config.get("eic_namespace", "")
        logger.info(f"eic namespace: {eic_namespace}")
        self.eic_namespace = eic_namespace

        if not os.path.exists(eic_log_dir) and not os.path.isdir(eic_log_dir):
            os.makedirs(eic_log_dir, exist_ok=True)

        self.connection = eic.Client()
        init_option = eic.InitOption()
        init_option.log_dir = eic_log_dir
        init_option.log_level = eic.LogLevel(eic_log_level)
        init_option.transport_type = eic.TransportType(eic_trans_type)
        init_option.flag_file = eic_flag_file

        if G_EnableGPUNicAffinity:
            gpu_id = torch.cuda.current_device()
            init_option.multi_net_local_interface_names = GPUNicAffinity[
                "cuda:" + str(gpu_id)
            ]
            logger.info(
                f"gpu {gpu_id} set gpu nic affinity to {init_option.multi_net_local_interface_names}"
            )

        ret = self.connection.init(eic_instance_id, endpoint, init_option)
        if ret != 0:
            logger.error(f"fail to init eic client, ret: {ret}")
            raise RuntimeError("EIC Client Init Failed.")
        self.warmup()

        self.memory_pool_host = memory_pool_host
        self.host_kvcache_layout = self.memory_pool_host.layout
        self.trans_type = eic.TransportType(eic_trans_type)
        self.kv_cache_dtype = self.memory_pool_host.dtype
        self.is_mla_model = hicache_config.is_mla_model
        self.rank = hicache_config.tp_rank
        self.world_size = hicache_config.tp_size
        self.pp_rank = hicache_config.pp_rank
        self.pp_size = hicache_config.pp_size
        self.page_size = self.memory_pool_host.page_size
        self.registered_pools = {}
        self.logical_anchor = self.memory_pool_host.kv_buffer is None
        self.use_zero_copy = self.memory_pool_host.layout == "page_first"
        if self.logical_anchor:
            self.use_zero_copy = True
            self.anchor_marker = torch.ones(
                (1,), dtype=torch.uint8, device="cpu", pin_memory=True
            )
        self.mha_zero_copy = self.use_zero_copy and not self.is_mla_model
        if not self.use_zero_copy:
            self.kv_cache_shape = self.memory_pool_host.get_data_page(
                0, flat=True
            ).shape
            if self.enable_kv_set_direct:
                self.kv_cache_write_mem_pool = FlexibleKVCacheMemoryPool(
                    self.connection, self.kv_cache_shape, self.kv_cache_dtype, "cpu"
                )
            if self.enable_kv_get_direct:
                self.kv_cache_get_mem_pool = FlexibleKVCacheMemoryPool(
                    self.connection, self.kv_cache_shape, self.kv_cache_dtype, "cpu"
                )
        self._init_eic_prefix()

    def warmup(self):
        logger.info("begin warm up eic client")
        start_time = time.perf_counter()
        num_warmup = 1024
        preheat_keys = ["warmup_key_" + str(i) for i in range(num_warmup)]
        batch_size = 32
        for i in range(0, num_warmup, batch_size):
            keys_vec = eic.StringVector()
            for key in preheat_keys[i : i + batch_size]:
                keys_vec.append(key)
            exist_option = eic.ExistOption()
            _, _ = self.connection.mexist(keys_vec, exist_option)
        logger.info(
            f"finish eic client warm up, warm up cost {time.perf_counter() - start_time:.2f} seconds"
        )

    def _register_tensors(self, tensors: List[torch.Tensor]) -> None:
        tensors = [
            buffer for buffer in tensors if buffer is not None and buffer.numel()
        ]
        if not tensors:
            return
        # no need judge meminfo type, cuda_id, etc.
        meminfo = eic.MemoryInfo()
        meminfo.type = eic.MemoryType.MEMORY_CUDA
        meminfo.cuda_id = 0
        vals = eic.IOBuffers()
        for buffer in tensors:
            vals.append(
                buffer.data_ptr(),
                buffer.numel() * buffer.element_size(),
                True,
            )
        self.connection.register_memory(vals, meminfo)

    def _pool_buffers(self, memory_pool_host: HostKVCache) -> List[torch.Tensor]:
        # NSA indexer host has no kv_buffer; use get_hybrid_pool_buffer() if present.
        get_hybrid = getattr(memory_pool_host, "get_hybrid_pool_buffer", None)
        if callable(get_hybrid):
            buffer = get_hybrid()
        else:
            buffer = memory_pool_host.kv_buffer
        if buffer is None:
            return []
        if isinstance(buffer, list):
            return buffer
        return [buffer]

    def register_mem_pool_host(self, memory_pool_host: HostKVCache) -> None:
        self._register_tensors(self._pool_buffers(memory_pool_host))
        if self.logical_anchor:
            self._register_tensors([self.anchor_marker])

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if host_pool_name == PoolName.KV:
            return
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        self._register_tensors(self._pool_buffers(host_pool))

    def _init_eic_prefix(self):
        # Same content hash means different bytes per stage; without this the
        # stages overwrite each other's KV. Suffix only when pp_size > 1 so
        # existing single-stage keys stay valid.
        pp = f"_pp{self.pp_size}_{self.pp_rank}" if self.pp_size > 1 else ""
        if self.is_mla_model:
            self.eic_prefix = (
                f"{self.model_name}_mla_att_{self.host_kvcache_layout}{pp}@sglang"
            )
        else:
            self.eic_prefix = f"{self.model_name}_mha_attn_{self.host_kvcache_layout}_{self.rank}_{self.world_size}{pp}_@sglang"

    def _get_eic_key(self, keys: List[str]) -> str:
        return [f"{self.eic_prefix}_{key}" for key in keys]

    def _get_component_keys(
        self, page_keys: List[str], transfer: PoolTransfer
    ) -> Tuple[List[str], int]:
        host_pool = self.registered_pools.get(transfer.name)
        key_multiplier = len(self._pool_buffers(host_pool)) if host_pool else 0
        if key_multiplier <= 0:
            return [], 0
        if key_multiplier == 1:
            return [f"{key}_{transfer.name}" for key in page_keys], key_multiplier
        return [
            f"{key}_{transfer.name}_{i}"
            for key in page_keys
            for i in range(key_multiplier)
        ], key_multiplier

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_size: Optional[Any] = None,
    ) -> bool:
        # now is not used
        if self.use_zero_copy:
            return self.zero_copy_batch_set([key], [target_location])
        else:
            return self.generic_batch_set([key], [value])

    # target_locations and target_sizes are not used for now
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if len(keys) == 0:
            return True
        if self.use_zero_copy:
            return self.zero_copy_batch_set(keys, values)
        else:
            return self.generic_batch_set(keys, values)

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_size: Optional[Any] = None,
    ) -> torch.Tensor | None:
        # now is not used
        if self.use_zero_copy:
            return self.zero_copy_batch_get([key], [target_location])
        else:
            return self.generic_batch_get([key], [target_location])

    # use for v1 interface, and shound not be called directly
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        assert len(keys) == len(target_locations)
        if len(keys) == 0:
            return None
        if self.use_zero_copy:
            return self.zero_copy_batch_get(keys, target_locations)
        else:
            return self.generic_batch_get(keys, target_locations)

    def _batch_exists_impl(self, keys) -> List[bool]:
        if len(keys) == 0:
            return 0
        eic_keys = self._get_eic_key(keys)
        logger.debug(f"eic exists {len(keys)}")
        result = []
        exist_bs = 1024
        for i in range(0, len(eic_keys), exist_bs):
            batch_keys = eic_keys[i : i + exist_bs]
            keys_vec = eic.StringVector()
            for key in batch_keys:
                keys_vec.append(key)
            exist_option = eic.ExistOption()
            exist_option.ns = self.eic_namespace
            status_code, exist_outcome = self.connection.mexist(keys_vec, exist_option)
            if status_code != eic.StatusCode.SUCCESS:
                logger.error(
                    f"eic exists {len(keys)} failed, status_code {status_code}"
                )
                result.extend([False] * len(batch_keys))
                continue
            for err_code in exist_outcome.status_codes:
                result.append(err_code == eic.StatusCode.SUCCESS)
        return result

    def exists(self, key) -> bool:
        exist_num = self.batch_exists([key])
        return exist_num == 1

    def batch_exists(
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        if len(keys) == 0:
            return 0
        if self.logical_anchor:
            exist_mask = self._batch_exists_impl(keys)
            prefix_success = 0
            for exist in exist_mask:
                if exist:
                    prefix_success += 1
                else:
                    break
            return prefix_success
        if self.mha_zero_copy:
            keys = self._get_mha_zero_copy_keys(keys)
        exist_mask = self._batch_exists_impl(keys)
        prefix_success = 0
        for exist in exist_mask:
            if exist:
                prefix_success += 1
            else:
                break
        if self.mha_zero_copy:
            prefix_success = prefix_success // 2
        return prefix_success

    def delete(self, key) -> None:
        eic_keys = self._get_eic_key([key])
        keys_vec = eic.StringVector()
        for eic_key in eic_keys:
            keys_vec.append(eic_key)
        del_option = eic.DelOption()
        self.connection.mdel(keys_vec, del_option)

    def clear(self) -> None:
        return

    # Not used for now
    def _filter_kv_cache(self, total_len) -> Tuple[int, int]:
        mean_len = total_len // self.world_size
        remainder = total_len % self.world_size
        tp_keys_len = mean_len + (1 if self.rank < remainder else 0)
        start = self.rank * mean_len + min(self.rank, remainder)
        end = start + tp_keys_len
        logger.debug(f"start: {start}, end: {end}, tp_keys_len: {tp_keys_len}")
        return start, end

    def zero_copy_batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        logger.debug(f"eic zero copy set {len(keys)} keys")
        if len(keys) == 0:
            return True
        eic_keys = self._get_eic_key(keys)
        keys_vec = eic.StringVector()
        vals_vec = eic.IOBuffers()
        # set data key & value
        for i, key in enumerate(eic_keys):
            # set data key & value
            keys_vec.append(key)
            vals_vec.append(
                values[i].data_ptr(),
                values[i].element_size() * values[i].numel(),
                True,
            )
        # set options
        set_option = eic.SetOption()
        set_option.ns = self.eic_namespace
        set_option.ttl_second = -1
        status_code, set_outcome = self.connection.mset(keys_vec, vals_vec, set_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mset {len(keys)} failed, status_code {status_code}")
            return [False] * len(keys)
        else:
            logger.debug(f"eic zero copy mset {len(keys)} success")
        return [True] * len(keys)

    def zero_copy_batch_get(
        self, keys: List[str], values: List[torch.Tensor]
    ) -> List[bool]:
        logger.debug(f"eic zero copy get {len(keys)} keys")
        # Get Data: generate data keys and vals
        get_data_start_time = time.perf_counter()
        eic_keys = self._get_eic_key(keys)
        data_keys = eic.StringVector()
        data_vals = eic.IOBuffers()
        success_mask = [True] * len(keys)
        count = len(keys)
        for i, key in enumerate(eic_keys):
            data_keys.append(key)
            data_vals.append(
                values[i].data_ptr(),
                values[i].element_size() * values[i].numel(),
                True,
            )

        # Get data: recv data buffer tensor
        get_option = eic.GetOption()
        get_option.ns = self.eic_namespace
        status_code, data_vals, get_outcome = self.connection.mget(
            data_keys, get_option, data_vals
        )

        if status_code != eic.StatusCode.SUCCESS:
            if status_code == eic.StatusCode.PARTIAL_FAILED:
                for i, err_code in enumerate(get_outcome.status_codes):
                    success = err_code == eic.StatusCode.SUCCESS
                    if success:
                        logger.debug(f"eic get data {eic_keys[i]} success")
                    else:
                        logger.error(
                            f"eic get data {eic_keys[i]} failed, err_code {err_code}"
                        )
                        success_mask[i] = False
            else:
                logger.error(
                    f"eic mget {len(eic_keys)} keys failed, status_code {status_code}"
                )
                success_mask = [False] * len(keys)
                return success_mask

        get_data_end_time = time.perf_counter()
        get_data_execution_time = (get_data_end_time - get_data_start_time) * 1e6
        logger.debug(f"eic get {count} keys data cost %.2f us", get_data_execution_time)
        return success_mask

    def _zero_copy_batch_set_ptrs(
        self, keys: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        if len(keys) == 0:
            return []
        eic_keys = self._get_eic_key(keys)
        keys_vec = eic.StringVector()
        vals_vec = eic.IOBuffers()
        for key, ptr, size in zip(eic_keys, ptrs, sizes):
            keys_vec.append(key)
            vals_vec.append(ptr, size, True)
        set_option = eic.SetOption()
        set_option.ns = self.eic_namespace
        set_option.ttl_second = -1
        status_code, set_outcome = self.connection.mset(keys_vec, vals_vec, set_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mset {len(keys)} failed, status_code {status_code}")
        return [
            err_code == eic.StatusCode.SUCCESS for err_code in set_outcome.status_codes
        ]

    def _zero_copy_batch_get_ptrs(
        self, keys: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        if len(keys) == 0:
            return []
        eic_keys = self._get_eic_key(keys)
        keys_vec = eic.StringVector()
        vals_vec = eic.IOBuffers()
        for key, ptr, size in zip(eic_keys, ptrs, sizes):
            keys_vec.append(key)
            vals_vec.append(ptr, size, True)
        get_option = eic.GetOption()
        get_option.ns = self.eic_namespace
        status_code, _data_vals, get_outcome = self.connection.mget(
            keys_vec, get_option, vals_vec
        )
        if status_code not in (eic.StatusCode.SUCCESS, eic.StatusCode.PARTIAL_FAILED):
            logger.error(f"eic mget {len(keys)} failed, status_code {status_code}")
            return [False] * len(keys)
        return [
            err_code == eic.StatusCode.SUCCESS for err_code in get_outcome.status_codes
        ]

    def _object_results_to_page_results(
        self, results: List[bool], key_multiplier: int
    ) -> List[bool]:
        if key_multiplier <= 0:
            return []
        return [
            all(results[i : i + key_multiplier])
            for i in range(0, len(results), key_multiplier)
        ]

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_pages = self.batch_exists(keys, extra_info)
        hit_count = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            component_keys, key_multiplier = self._get_component_keys(
                keys[:kv_pages], transfer
            )
            object_exists = self._batch_exists_impl(component_keys)
            page_exists = self._object_results_to_page_results(
                object_exists, key_multiplier
            )
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

    def _batch_io_v2(
        self, transfers: List[PoolTransfer], is_set: bool
    ) -> dict[str, List[bool]]:
        results = {}
        for transfer in transfers:
            host_pool = self.registered_pools.get(transfer.name)
            keys = transfer.keys or []
            host_indices = transfer.host_indices
            if host_pool is None or host_indices is None or not keys:
                results[transfer.name] = []
                continue

            page_size = getattr(host_pool, "page_size", 1) or 1
            assert len(keys) == len(host_indices) // page_size
            ptrs, sizes = host_pool.get_page_buffer_meta(host_indices)
            component_keys, key_multiplier = self._get_component_keys(keys, transfer)
            assert len(component_keys) == len(ptrs)

            if is_set:
                exists = self._batch_exists_impl(component_keys)
                object_results = [True if x else False for x in exists]
                missing = [i for i, x in enumerate(exists) if not x]
                if missing:
                    set_results = self._zero_copy_batch_set_ptrs(
                        [component_keys[i] for i in missing],
                        [ptrs[i] for i in missing],
                        [sizes[i] for i in missing],
                    )
                    for i, ok in zip(missing, set_results):
                        object_results[i] = ok
            else:
                object_results = self._zero_copy_batch_get_ptrs(
                    component_keys, ptrs, sizes
                )
            results[transfer.name] = self._object_results_to_page_results(
                object_results, key_multiplier
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

    def generic_batch_set(
        self,
        keys: List[str],
        values: List[torch.Tensor],
    ) -> List[bool]:
        assert len(keys) == len(values)
        logger.debug(f"eic generic set {len(keys)} keys")
        if len(keys) == 0:
            return True
        eic_keys = self._get_eic_key(keys)
        keys_vec = eic.StringVector()
        vals_vec = eic.IOBuffers()
        count = len(keys)
        registered = False
        items = []
        if self.enable_kv_set_direct:
            values_data_ptrs = []
            items = self.kv_cache_write_mem_pool.try_allocate_kv_cache(
                self.kv_cache_shape, self.kv_cache_dtype, count
            )
            if items is None:
                logger.warning("can not allocate tensor from pool")
                for i, value in enumerate(values):
                    values_data_ptrs.append(
                        (value.data_ptr(), value.element_size() * value.numel(), False)
                    )
            else:
                objs = items
                registered = True
                for i, key in enumerate(eic_keys):
                    temp = objs[i].reshape(values[i].shape).contiguous()
                    temp.copy_(values[i])
                    if temp.data_ptr() != objs[i].data_ptr():
                        registered = False
                        temp = temp.cpu()
                    values_data_ptrs.append(
                        (
                            temp.data_ptr(),
                            temp.element_size() * temp.numel(),
                            registered,
                        )
                    )

            for i, key in enumerate(eic_keys):
                keys_vec.append(key)
                data_ptr, data_size, registered = values_data_ptrs[i]
                vals_vec.append(data_ptr, data_size, registered)
        else:
            # use tensor direct
            for i, key in enumerate(eic_keys):
                keys_vec.append(key)
                vals_vec.append(
                    values[i].data_ptr(),
                    values[i].element_size() * values[i].numel(),
                    False,
                )

        # set options
        set_option = eic.SetOption()
        set_option.ns = self.eic_namespace
        set_option.ttl_second = -1
        status_code, set_outcome = self.connection.mset(keys_vec, vals_vec, set_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mset {len(eic_keys)} failed, status_code {status_code}")
        else:
            logger.debug(f"eic mset {len(eic_keys)} success")

        if self.enable_kv_set_direct and items is not None:
            for item in items:
                self.kv_cache_write_mem_pool.free_to_mempool(item.data_ptr())

        err_code = set_outcome.status_codes[0]
        if err_code != eic.StatusCode.SUCCESS:
            logger.error(f"set data key {len(eic_keys)} failed, err_code {err_code}")
            return [False] * len(keys)

        logger.debug(f"set data key {len(eic_keys)} success")
        return [True] * len(keys)

    def generic_batch_get(
        self, keys: List[str], buffers: List[torch.Tensor]
    ) -> List[bool]:
        # all success or all fail
        logger.debug(f"eic generic get {len(keys)} keys")
        eic_keys = self._get_eic_key(keys)
        get_data_start_time = time.perf_counter()
        data_keys = eic.StringVector()
        data_vals = eic.IOBuffers()
        count = len(eic_keys)
        registered = False
        items = []
        success_mask = [True] * len(keys)
        if self.enable_kv_get_direct:
            items = self.kv_cache_get_mem_pool.try_allocate_kv_cache(
                self.kv_cache_shape, self.kv_cache_dtype, count
            )
            if items is None:
                logger.warning("can not allocate tensor from pool")
                for i, key in enumerate(eic_keys):
                    data_keys.append(key)
                    data_vals.append(
                        buffers[i].data_ptr(),
                        buffers[i].element_size() * buffers[i].numel(),
                        False,
                    )
            else:
                registered = True
                for i, key in enumerate(eic_keys):
                    data_keys.append(key)
                    data_vals.append(
                        items[i].data_ptr(),
                        items[i].element_size() * items[i].numel(),
                        registered,
                    )

        else:
            for i, key in enumerate(eic_keys):
                data_keys.append(key)
                data_vals.append(
                    buffers[i].data_ptr(),
                    buffers[i].element_size() * buffers[i].numel(),
                    False,
                )

        # Get data: recv data buffer tensor
        get_option = eic.GetOption()
        get_option.ns = self.eic_namespace
        status_code, data_vals, get_outcome = self.connection.mget(
            data_keys, get_option, data_vals
        )

        if status_code != eic.StatusCode.SUCCESS:
            if status_code == eic.StatusCode.PARTIAL_FAILED:
                for i, err_code in enumerate(get_outcome.status_codes):
                    success = err_code == eic.StatusCode.SUCCESS
                    if success:
                        logger.debug(f"eic get data {eic_keys[i]} success")
                    else:
                        logger.error(
                            f"eic get data {eic_keys[i]} failed, err_code {err_code}"
                        )
                        success_mask[i] = False
            else:
                logger.error(
                    f"eic mget {len(eic_keys)} keys failed, status_code {status_code}"
                )
                success_mask = [False] * len(keys)

        if registered:
            for i, item in enumerate(items):
                if success_mask[i]:
                    buffers[i].copy_(item)
                self.kv_cache_get_mem_pool.free_to_mempool(item.data_ptr())

        get_data_end_time = time.perf_counter()
        get_data_execution_time = (get_data_end_time - get_data_start_time) * 1e6
        logger.debug(f"eic get {count} keys data cost %.2f us", get_data_execution_time)
        return success_mask

    def _get_mha_zero_copy_keys(self, keys: List[str]) -> List[str]:
        new_keys = []
        for k in keys:
            new_keys.append(f"{k}_k")
            new_keys.append(f"{k}_v")
        return new_keys

    def _get_mha_zero_copy_values(
        self, values: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        new_values = []
        for value in values:
            new_values.append(value[0])
            new_values.append(value[1])
        return new_values

    def _batch_get_preprocess(self, keys, host_indices):
        page_num = len(host_indices) // self.page_size
        # use memory pool directly or dummy page
        values = (
            [
                self.memory_pool_host.get_data_page(
                    host_indices[i * self.page_size], flat=False
                )
                for i in range(page_num)
            ]
            if self.use_zero_copy
            else [
                self.memory_pool_host.get_dummy_flat_data_page()
                for _ in range(page_num)
            ]
        )

        if self.mha_zero_copy:
            keys = self._get_mha_zero_copy_keys(keys)
            values = self._get_mha_zero_copy_values(values)

        return keys, values

    def _batch_get_postprocess(self, host_indices, values, results):
        page_num = len(host_indices) // self.page_size

        if self.use_zero_copy:
            if not self.is_mla_model:
                results = [
                    (results[2 * i] and results[2 * i + 1]) for i in range(page_num)
                ]
                results = results[:page_num]
            return results

        # dummy page copy to host memory pool
        for i in range(page_num):
            if not results[i]:
                break
            self.memory_pool_host.set_from_flat_data_page(
                host_indices[i * self.memory_pool_host.page_size], values[i]
            )

        return results

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if self.logical_anchor:
            return self._batch_exists_impl(keys)
        keys, values = self._batch_get_preprocess(keys, host_indices)
        results = self.batch_get(keys, values)
        return self._batch_get_postprocess(host_indices, values, results)

    def _batch_set_preprocess(self, keys, host_indices):
        page_num = len(host_indices) // self.page_size
        flat = not self.use_zero_copy
        values = [
            self.memory_pool_host.get_data_page(
                host_indices[i * self.page_size], flat=flat
            )
            for i in range(page_num)
        ]

        if self.mha_zero_copy:
            keys = self._get_mha_zero_copy_keys(keys)
            values = self._get_mha_zero_copy_values(values)

        return keys, values

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if self.logical_anchor:
            ptr = self.anchor_marker.data_ptr()
            size = self.anchor_marker.numel() * self.anchor_marker.element_size()
            return self._zero_copy_batch_set_ptrs(
                keys, [ptr] * len(keys), [size] * len(keys)
            )
        keys, values = self._batch_set_preprocess(keys, host_indices)
        results = self.batch_set(keys, values)
        return results
