import logging
import os
import time
import uuid
from enum import Enum
from itertools import chain
from typing import Any, List, Optional

import torch

from sglang.srt.distributed.parallel_state import get_world_group, get_world_size
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class MmcDirect(Enum):
    D_2_L3 = 0
    L3_2_D = 1
    L3_2_H = 2
    H_2_L3 = 3


class AscendMemCacheStore(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig):
        try:
            from memcache_hybrid import DistributedObjectStore
        except ImportError as e:
            logger.error("Import Ascend MemCache failed: %s", e)
            raise ImportError(
                "Please install ascend memcache_hybrid by following the instructions at "
                "https://gitcode.com/Ascend/memcache/blob/master/doc/build.md "
                "to run SGLang with MemCache."
            ) from e

        try:
            self.local_rank = storage_config.tp_rank
            self.tp_size = storage_config.tp_size
            self.device_id = storage_config.extra_config["device_id"]
            self.is_mla_model = storage_config.is_mla_model

            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            # Initialize hccl in advance through all_gather to avoid conflicts with rdma initialization.
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )

            self.store = DistributedObjectStore()
            ret = self.store.init(self.device_id)
            if ret:
                msg = f"Failed to init Ascend MemCache, error code: device_id={self.device_id}, {ret=}"
                logger.error(msg)
                raise Exception(msg)

            logger.info("Init Ascend MemCache successfully.")
            self.warmup()
            logger.info("Ascend MemCache warmup successfully.")

            self.is_device_rdma = self._check_device_rdma()
        except ValueError as e:
            logger.error("Init Ascend MemCache failed: %s", e)
            raise e
        except Exception as e:
            logger.error(
                "An error occurred while Ascend MemCache initialization: %s", e
            )
            raise e

    def _check_device_rdma(self):
        config_path = os.environ.get("MMC_LOCAL_CONFIG_PATH")
        if not config_path:
            logger.error("MMC_LOCAL_CONFIG_PATH is not set")
            return False

        try:
            with open(config_path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            logger.error(f"FileNotFoundError: {config_path}")
            return False

        protocol_value = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("ock.mmc.local_service.protocol"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    protocol_value = parts[1].strip()
                break

        if protocol_value == "device_rdma":
            return True
        return False

    def warmup(self):
        warmup_key = "sglang_memcache_warmup_key" + uuid.uuid4().hex
        logger.info(f"{warmup_key=}")

        warmup_value = bytes(4 * 1024)  # 4 KB
        assert self.store.put(warmup_key, warmup_value) == 0
        assert self.store.is_exist(warmup_key) == 1
        assert self.store.get(warmup_key) == warmup_value

    def register_mem_pool_device(self, mem_pool_device: KVCache):
        if not self.is_device_rdma:
            return

        super().register_mem_pool_device(mem_pool_device)
        try:
            if self.is_mla_model:
                k_buffer = self.mem_pool_device.k_buffer
                ret_code = self.store.register_buffer(
                    k_buffer.data_ptr(),
                    k_buffer.numel() * k_buffer.element_size(),
                )
                v_buffer = self.mem_pool_device.v_buffer
                ret_code |= self.store.register_buffer(
                    v_buffer.data_ptr(),
                    v_buffer.numel() * v_buffer.element_size(),
                )

                if self.mem_pool_device.index_head_dim is not None:
                    index_k_buffer = self.mem_pool_device.index_k_buffer
                    ret_code |= self.store.register_buffer(
                        index_k_buffer.data_ptr(),
                        index_k_buffer.numel() * index_k_buffer.element_size(),
                    )
            else:
                buffer = self.mem_pool_device.kv_buffer
                ret_code = self.store.register_buffer(
                    buffer.data_ptr(),
                    buffer.numel() * buffer.element_size(),
                )
            if ret_code:
                logger.error(f"failed to register kv buffer for device rdma, error code: {ret_code}")
            else:
                logger.info(f"register kv buffer for device rdma success: {ret_code=}")
        except TypeError as err:
            logger.error("Failed to register buffer to Ascend MemCache Store: %s", err)
            raise TypeError("Ascend MemCache Store Register Buffer Error.") from err

    def set(
        self,
        key,
        value: Optional[Any] = None,
        target_location: Optional[int] = None,
        target_sizes: Optional[int] = None,
    ) -> bool:
        return False

    def _batch_preprocess(self, keys):
        local_rank = 0 if self.is_mla_model else self.local_rank
        return [f"{key}_{local_rank}" for key in keys]

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[List[List[int]]] = None,
        target_sizes: Optional[List[List[int]]] = None,
    ) -> int:

        assert len(keys) > 0
        assert len(keys) == len(target_locations) == len(target_sizes)
        key_strs = self._batch_preprocess(keys)

        start = time.time()
        exist_result = self.store.batch_is_exist(key_strs)
        end = time.time()
        assert len(exist_result) == len(key_strs)
        non_exist_indices = [i for i, exist in enumerate(exist_result) if exist != 1]
        if len(non_exist_indices) == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"batch exist before put, ret=True, total keys is {len(keys)}, "
                    f"all keys is already in memcache, "
                    f"duration {float((end - start) * 1000):.3f}ms"
                )
            return len(keys)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"batch exist before put, total keys is {len(keys)}, "
                f"{len(non_exist_indices)} keys is not in memcache, "
                f"duration {float((end - start) * 1000):.3f}ms"
            )

        new_keys = [key_strs[i] for i in non_exist_indices]
        new_src_locations = [target_locations[i] for i in non_exist_indices]
        new_src_sizes = [target_sizes[i] for i in non_exist_indices]

        # Only set non-existing keys to storage
        direct = int(MmcDirect.D_2_L3.value)
        start = time.time()
        put_result = self.store.batch_put_from_layers(
            new_keys, new_src_locations, new_src_sizes, direct
        )
        end = time.time()
        for i, idx in enumerate(non_exist_indices):
            exist_result[idx] = 1 if put_result[i] == 0 else 0

        if logger.isEnabledFor(logging.DEBUG):
            layers_per_key = len(new_src_locations[0])
            total_size = sum(chain.from_iterable(new_src_sizes))
            logger.debug(
                f"batch put finished, origin keys is {len(keys)}, "
                f"non exist keys is {len(new_keys)}, "
                f"success is {put_result.count(0)}, "
                f"layers per key is {layers_per_key}, "
                f"copy_total_size={total_size}, "
                f"duration {(end - start) * 1000:.3f}ms, "
                f"speed {float(total_size) / 1024 / 1024 / 1024 / (end - start):.3f}GB/s"
            )

        for i in range(len(keys)):
            if exist_result[i] != 1:
                return i
        return len(keys)

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        return False

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[List[int]]] = None,
        target_sizes: Optional[List[List[int]]] = None,
    ) -> int:
        assert len(keys) > 0
        assert len(keys) == len(target_locations) == len(target_sizes)
        key_strs = self._batch_preprocess(keys)

        direct = int(MmcDirect.L3_2_D.value)
        start = time.time()
        get_result = self.store.batch_get_into_layers(
            key_strs, target_locations, target_sizes, direct
        )
        end = time.time()

        if logger.isEnabledFor(logging.DEBUG):
            layers_per_key = len(target_locations[0])
            total_size = sum(chain.from_iterable(target_sizes))
            logger.debug(
                f"batch get finished, origin keys is {len(keys)}, "
                f"success is {get_result.count(0)}, "
                f"layers per key is {layers_per_key}, "
                f"copy_total_size={total_size}, "
                f"duration {(end - start) * 1000:.3f}ms, "
                f"speed {float(total_size) / 1024 / 1024 / 1024 / (end - start):.3f}GB/s"
            )
        for i in range(len(keys)):
            if get_result[i] != 0:
                return i
        return len(keys)

    def exists(self, key) -> bool:
        return self.store.is_exist(key)

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        assert len(keys) > 0
        query_keys = self._batch_preprocess(keys)
        start = time.time()
        exist_result = self.store.batch_is_exist(query_keys)
        end = time.time()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"batch exist finished, len is {len(query_keys)}, "
                f"exist is {exist_result.count(1)}, "
                f"duration {(end - start) * 1000:.3f}ms"
            )
        for i in range(len(query_keys)):
            if exist_result[i] != 1:
                return i
        return len(query_keys)

    def delete(self, key) -> None:
        d_key = f"{key}_{self.local_rank}"
        ret = self.store.remove(d_key)
        if ret != 0:
            logger.error(f"MemCache client close failed, errcode: {ret}")

    def close(self):
        ret = self.store.close()
        if ret == 0:
            logger.info("MemCache client close successfully")
        else:
            logger.error(f"MemCache client close failed, errcode: {ret}")

    def clear(self) -> None:
        ret = self.store.remove_all()
        if ret == 0:
            logger.info("MemCache clear all keys successfully")
        else:
            logger.error(f"MemCache clear all keys failed, errcode: {ret}")
