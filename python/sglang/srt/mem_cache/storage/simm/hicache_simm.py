import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

# Third Party
try:
    from simm.kv import BlockView, Store, register_mr, set_flag
except ImportError as e:
    raise ImportError(
        "Please install simm by following the instructions at "
        "to run vLLM with SimmConnector."
    ) from e

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB
SETUP_TIMEOUT = 600  # 10min

logger = logging.getLogger(__name__)


@dataclass
class SiMMConfig:
    manager_address: str
    clnt_threadpool_size: int
    enable_profile: bool

    @staticmethod
    def from_file() -> "SiMMConfig":
        """Load the config from a JSON file."""
        if not envs.SGLANG_HICACHE_SIMM_CONFIG_PATH.is_set():
            raise RuntimeError(
                f"Config file path not set. Please set {envs.SGLANG_HICACHE_SIMM_CONFIG_PATH.name}"
            )
        file_path = envs.SGLANG_HICACHE_SIMM_CONFIG_PATH.get()
        try:
            with open(file_path) as fin:
                config = json.load(fin)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {str(e)}")

        if "manager_address" not in config:
            raise ValueError("Manager_address is required in config file")

        return SiMMConfig(
            manager_address=config.get(
                "manager_address", envs.SIMM_CLUSTER_MANAGER.default
            ),
            clnt_threadpool_size=config.get(
                "clnt_threadpool_size", envs.SIMM_CLNT_THREADPOOL_SIZE.default
            ),
            enable_profile=config.get(
                "enable_profile", envs.SIMM_ENABLE_PROFILE.default
            ),
        )

    @staticmethod
    def load_from_env() -> "SiMMConfig":
        """Load config from a file specified in the environment variable."""
        # other required environment variables...
        if not envs.SIMM_CLUSTER_MANAGER.is_set():
            raise ValueError(
                "The environment variable 'SIMM_CLUSTER_MANAGER' is not set."
            )

        return SiMMConfig(
            manager_address=envs.SIMM_CLUSTER_MANAGER.get(),
            clnt_threadpool_size=envs.SIMM_CLNT_THREADPOOL_SIZE.get(),
            enable_profile=envs.SIMM_ENABLE_PROFILE.get(),
        )

    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "SiMMConfig":
        """Load config from extra_config dictionary."""
        if "manager_address" not in extra_config:
            raise ValueError("manager_address is required in extra_config")

        return SiMMConfig(
            manager_address=extra_config.get(
                "manager_address", envs.SIMM_CLUSTER_MANAGER.default
            ),
            clnt_threadpool_size=extra_config.get(
                "clnt_threadpool_size", envs.SIMM_CLNT_THREADPOOL_SIZE.default
            ),
            enable_profile=extra_config.get(
                "enable_profile", envs.SIMM_ENABLE_PROFILE.default
            ),
        )


def get_current_process_numa() -> int:
    """
    Return value: numa_node of current process, failed return -1
    """
    try:
        # get current cpu
        with open("/proc/self/stat", "r") as f:
            stat_data = f.read()

        # the 39th field is processor
        fields = stat_data.split()
        if len(fields) < 39:
            return -1
        current_cpu = int(fields[38])
        numa_path = f"/sys/devices/system/cpu/cpu{current_cpu}/node0"
        if os.path.exists(numa_path) and os.path.islink(numa_path):
            link_target = os.readlink(numa_path)
            # parse numa node from path
            match = re.search(r"node(\d+)$", link_target)
            if match:
                return int(match.group(1))

        return -1
    except Exception:
        return -1


def get_numa_nic_mapping() -> Dict[int, List[str]]:
    """
    Return value: Dict[numa_node, List(rdma_device_name)]
    """
    ib_root = "/sys/class/infiniband"
    device_map = defaultdict(list)

    if not os.path.exists(ib_root):
        logger.error(f"SiMM ERROR: {ib_root} not found. Are RDMA drivers loaded?")
        return []

    for device_name in os.listdir(ib_root):
        numa_path = os.path.join(ib_root, device_name, "device", "numa_node")
        numa_node = -1  # default value, if system is UMA.

        try:
            if os.path.exists(numa_path):
                with open(numa_path, "r") as f:
                    content = f.read().strip()
                    numa_node = int(content)
        except (IOError, ValueError):
            pass
        device_map[numa_node].append(device_name)

    return device_map


class HiCacheSiMM(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig = None, mem_pool: HostKVCache = None
    ):
        try:
            extra_config = (
                getattr(storage_config, "extra_config", None)
                if storage_config
                else None
            )
            # Load configuration with manager_address prioritized from extra_config if available
            if (
                extra_config is not None
                and extra_config.get("manager_address") is not None
            ):
                # Load from extra_config
                self.config = SiMMConfig.load_from_extra_config(extra_config)
                logger.info("SiMM Configuration loaded from extra_config successfully.")
            elif envs.SGLANG_HICACHE_SIMM_CONFIG_PATH.is_set():
                # Load from config file
                self.config = SiMMConfig.from_file()
                logger.info("SiMM Configuration loaded from file successfully.")
            else:
                # Load from environment variables
                self.config = SiMMConfig.load_from_env()
                logger.info("SiMM Configuration loaded from env successfully.")

            # Check if extra_backend_tag should be passed to SiMM data server
            self.extra_backend_tag = None
            if extra_config and "extra_backend_tag" in extra_config:
                self.extra_backend_tag = extra_config["extra_backend_tag"]
                logger.info(f"Using extra_backend_tag: {self.extra_backend_tag}")

            # Set nic device according to current process numa node
            nic_mapping = get_numa_nic_mapping()
            logger.info(f"SiMM NUMA-awared allocation: {nic_mapping}")
            current_numa = get_current_process_numa()
            if current_numa >= 0:
                rdma_devices = nic_mapping.get(current_numa)
                if rdma_devices is not None and len(rdma_devices) > 0:
                    rdma_device_str = ",".join(rdma_devices)
                    os.environ["SICL_NET_DEVICES"] = rdma_device_str
                    logger.info(f"SiMM using rdma {rdma_device_str}")

            # Set simm log path: /var/log/simm/{filename_ts}-{pid}/simm_clnt.log
            filename_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file_path: str = (
                f"/var/log/simm/{filename_ts}-{os.getpid()}/simm_clnt.log"
            )

            cm_ip = self.config.manager_address.split(":")[0]
            cm_port = self.config.manager_address.split(":")[1]
            set_flag("cm_primary_node_ip", cm_ip)
            set_flag("cm_primary_node_port", cm_port)
            set_flag("clnt_log_file", log_file_path)
            set_flag("clnt_thread_pool_size", str(self.config.clnt_threadpool_size))

            self.store = Store()
            logger.info("SiMM store setup successfully.")
            self.mr_ext = None

            self.warmup()
            logger.info("SiMM store warmup successfully.")

            if storage_config is not None:
                self.model_name = storage_config.model_name
                self.is_mla_backend = storage_config.is_mla_model
                self.local_rank = storage_config.tp_rank
                self.pp_rank = storage_config.pp_rank
                self.pp_size = storage_config.pp_size
            else:
                self.model_name = ""
                self.is_mla_backend = False
                self.local_rank = 0
                self.pp_rank = 0
                self.pp_size = 1

            self.enable_pp = self.pp_size > 1
            if self.enable_pp:
                self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
                self.mla_suffix = f"{self.pp_rank}"
            else:
                self.mha_suffix = f"{self.local_rank}"
                self.mla_suffix = ""

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    def warmup(self):
        """Dryrun a key to warmup SiMM client"""
        logger.info("begin warm up SiMM client")
        start_time = time.perf_counter_ns()
        warmup_key = "sglang_simm_warmup_key" + uuid.uuid4().hex
        warmup_tensor = torch.frombuffer(
            bytearray(warmup_key.encode()), dtype=torch.uint8
        )
        warmup_size = 4 * 1024  # 4 KB
        block = self.store.allocate(warmup_size)
        block_ = block.as_ref()
        block_[: len(warmup_key)] = warmup_tensor
        if self.store.put(warmup_key, block.view()) != 0:
            logger.warning(f"SiMM client warmup put key {warmup_key} failed")
        if not self.store.exists(warmup_key):
            logger.warning(f"SiMM client warmup key {warmup_key} not exists")
        got_block = self.store.allocate(warmup_size)
        if self.store.get(warmup_key, got_block.view()) < 0:
            logger.warning(f"SiMM client warmup get key {warmup_key} failed")
        if not all(got_block.as_ref()[: len(warmup_key)] == warmup_tensor):
            logger.warning(f"SiMM client warmup key {warmup_key} data wrong")
        logger.info(
            f"finish SiMM client warm up, cost {(time.perf_counter_ns() - start_time)/1000:.2f} us"
        )

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        assert self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
        ], "simm storage backend only support page first or page first direct layout"
        buffer = self.mem_pool_host.kv_buffer
        try:
            self.mr_ext = register_mr(buffer)
            if self.mr_ext is None:
                logger.error(f"Failed to register buffer")
                raise RuntimeError(f"Failed to register buffer to SiMM")
        except TypeError as err:
            logger.error("Failed to register buffer to SiMM: %s", err)
            raise TypeError("SiMM Register Buffer Error.") from err

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
        else:
            return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(self, results: List[int], is_set_operate=False):
        """
        for batch_get_into, results is Vector of integers,
            where each element is the number of bytes read on success, or a negative value on error
        for batch_put_from, results is Vector of integers,
            where each element is 0 on success, or a negative value on error
        """
        if self.is_mla_backend:
            return [k_res == 0 if is_set_operate else k_res > 0 for k_res in results]
        else:
            kv_pairs = zip(results[::2], results[1::2])
            return [
                (
                    (k_res == 0 and v_res == 0)
                    if is_set_operate
                    else (k_res > 0 and v_res > 0)
                )
                for k_res, v_res in kv_pairs
            ]

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Apply extra_backend_tag prefix if available
        if self.extra_backend_tag is not None:
            prefix = self.extra_backend_tag
            keys = [f"{prefix}_{key}" for key in keys]

        t1 = time.perf_counter_ns()
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        get_results = self._get_batch_zero_copy_impl(
            key_strs, buffer_ptrs, buffer_sizes
        )
        t2 = time.perf_counter_ns()
        total_size = sum([k_res if k_res > 0 else 0 for k_res in get_results])
        if self.config.enable_profile:
            logger.info(
                f"SiMM batch_get_v1 {len(keys)} keys, total size: {total_size / 1024**2} MiB, \
                    using {(t2 - t1)/1000} us, Throughput: {total_size / 1024**3 / ((t2 - t1) / 1000**3):.2f} GiB/s"
            )
        return self._batch_postprocess(get_results, is_set_operate=False)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Apply extra_backend_tag prefix if available
        if self.extra_backend_tag is not None:
            prefix = self.extra_backend_tag
            keys = [f"{prefix}_{key}" for key in keys]

        t1 = time.perf_counter_ns()
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        exist_result = self._batch_exist_impl(key_strs)
        t2 = time.perf_counter_ns()
        if self.config.enable_profile:
            logger.info(
                f"SiMM batch exists {len(keys)} keys, using {(t2 - t1)/1000} us"
            )

        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [-1] * len(key_strs)
        total_size = 0
        for i in range(len(key_strs)):
            if not exist_result[i]:
                set_keys.append(key_strs[i])
                set_buffer_ptrs.append(buffer_ptrs[i])
                set_buffer_sizes.append(buffer_sizes[i])
                set_indices.append(i)
                total_size += buffer_sizes[i]
            else:
                set_results[i] = 0

        # Only set non-existing keys to storage
        if len(set_keys) > 0:
            put_results = self._put_batch_zero_copy_impl(
                set_keys, set_buffer_ptrs, set_buffer_sizes
            )
            for i in range(len(set_indices)):
                set_results[set_indices[i]] = put_results[i]
        t3 = time.perf_counter_ns()
        if self.config.enable_profile:
            logger.info(
                f"SiMM batch_put_v1 {len(keys)} keys, total size: {total_size / 1024**2} MiB, \
                    using {(t3 - t2)/1000} us, Throughput: {total_size / 1024**3 / ((t3 - t2) / 1000**3):.2f} GiB/s"
            )

        return self._batch_postprocess(set_results, is_set_operate=True)

    def set(
        self,
        key,
        value: Optional[Any] = None,
        target_location: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        # Only support zero copy set for now
        assert target_location is not None and target_sizes is not None
        exist_result = self._batch_exist_impl([key])
        if exist_result[0]:
            return True
        put_result = self._put_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return put_result[0] == 0

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        # Only support zero copy set for now
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

        exist_result = self._batch_exist_impl(keys)
        set_keys = []
        set_target_locations = []
        set_target_sizes = []
        set_indices = []
        for i in range(len(keys)):
            if not exist_result[i]:
                set_keys.append(keys[i])
                set_target_locations.append(target_locations[i])
                set_target_sizes.append(target_sizes[i])
                set_indices.append(i)
        # Only set non-existing keys to storage
        put_result = self._put_batch_zero_copy_impl(
            set_keys, set_target_locations, set_target_sizes
        )
        for i in range(len(set_indices)):
            if put_result[i] == 0:
                exist_result[set_indices[i]] = 1

        # return the number of consecutive successful operations from the start.
        success_count = 0
        for i in range(len(keys)):
            if exist_result[i] == 0:
                break
            success_count += 1
        return success_count == len(keys)

    def get(
        self,
        key,
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
        get_result = self._get_batch_zero_copy_impl(
            keys, target_locations, target_sizes
        )
        if self.is_mla_backend:
            key_multiplier = 1
        else:
            key_multiplier = 2
        for i in range(len(keys)):
            if get_result[i] < 0:
                return i // key_multiplier
        return len(keys) // key_multiplier

    def exists(self, key) -> bool:
        exist_result = self._batch_exist_impl([key])
        return exist_result[0]

    def batch_exists(
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        if self.is_mla_backend:
            query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]
            key_multiplier = 1
        else:
            query_keys = []
            for key in keys:
                query_keys.append(f"{key}_{self.mha_suffix}_k")
                query_keys.append(f"{key}_{self.mha_suffix}_v")
            key_multiplier = 2

        t1 = time.perf_counter_ns()
        exist_result = self._batch_exist_impl(query_keys)
        t2 = time.perf_counter_ns()
        if self.config.enable_profile:
            logger.info(
                f"SiMM batch exists {len(keys)} keys, using {(t2 - t1)/1000} us"
            )
        for i in range(len(query_keys)):
            if not exist_result[i]:
                return i // key_multiplier
        return len(query_keys) // key_multiplier

    def _put_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[int]:
        block_views = []
        for i in range(len(buffer_ptrs)):
            block_view = BlockView.from_buffer(
                buffer_ptrs[i], buffer_sizes[i], self.mr_ext
            )
            block_views.append(block_view)
        return self.store.mput(key_strs, block_views)

    def _get_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[int]:
        block_views = []
        for i in range(len(buffer_ptrs)):
            block_view = BlockView.from_buffer(
                buffer_ptrs[i], buffer_sizes[i], self.mr_ext
            )
            block_views.append(block_view)
        return self.store.mget(key_strs, block_views)

    def _batch_exist_impl(self, key_strs: List[str]) -> List[bool]:
        return self.store.mexists(key_strs)
