import logging
from typing import Any, List, Optional

import torch
from aibrix_kvcache import (
    BaseKVCacheManager,
    BlockHashes,
    KVCacheBlockLayout,
    KVCacheBlockSpec,
    KVCacheConfig,
    KVCacheTensorSpec,
    ModelSpec,
)
from aibrix_kvcache.common.absl_logging import log_every_n_seconds

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


class AibrixKVCacheStorage(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, mem_pool: HostKVCache):
        if storage_config is not None:
            self.is_mla_backend = storage_config.is_mla_model
            self.local_rank = storage_config.tp_rank
        else:
            self.is_mla_backend = False
            self.local_rank = 0
        kv_cache = mem_pool.device_pool
        self.page_size = mem_pool.page_size
        self.kv_cache_dtype = kv_cache.dtype
        self.layer_num = kv_cache.layer_num
        self.kv_head_ids = [
            self.local_rank * kv_cache.head_num + i for i in range(kv_cache.head_num)
        ]
        if not self.is_mla_backend:
            self.layer_ids = range(
                kv_cache.start_layer, kv_cache.end_layer
            )  # for pipeline parallel

            self.block_spec = KVCacheBlockSpec(
                block_ntokens=self.page_size,
                block_dtype=self.kv_cache_dtype,
                block_layout=KVCacheBlockLayout(KVCacheBlockLayout.NCLD),
                tensor_spec=KVCacheTensorSpec(
                    heads=self.kv_head_ids,
                    layers=self.layer_ids,
                    head_size=kv_cache.head_dim,
                ),
            )
            logger.info(self.block_spec)
            config = KVCacheConfig(
                block_spec=self.block_spec, model_spec=ModelSpec(102400)
            )
            self.kv_cache_manager = BaseKVCacheManager(config)
        else:
            raise NotImplementedError(
                "MLA is not supported by AibrixKVCacheStorage yet."
            )

    def _aibrix_kvcache_metrics_report(self):
        self.kv_cache_manager.metrics.summary()
        self.kv_cache_manager.metrics.reset()

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        block_hash = BlockHashes(keys, self.page_size)
        status = self.kv_cache_manager.acquire(None, block_hash)
        log_every_n_seconds(
            logger, logging.INFO, self._aibrix_kvcache_metrics_report(), 1
        )
        if status.is_ok():
            num_fetched_tokens, handle = status.value
            kv_blocks = handle.to_tensors()
            assert len(kv_blocks) == len(target_locations)
            for i in range(len(kv_blocks)):
                assert (
                    target_locations[i].nbytes == kv_blocks[i].nbytes
                ), f"{target_locations[i].nbytes}, {kv_blocks[i].nbytes}"
                target_locations[i].copy_(kv_blocks[i].flatten())
            handle.release()
            return target_locations

        return [None] * len(keys)

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_size: Optional[Any] = None,
    ) -> torch.Tensor | None:
        return self.batch_get([key], [target_location], [target_size])[0]

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        block_hash = BlockHashes(keys, self.page_size)
        status = self.kv_cache_manager.allocate_for(None, block_hash)
        if not status.is_ok():
            logger.warning(
                f"aibrix_kvcache set allocate failed, error_code {status.error_code}"
            )
            return False
        handle = status.value
        tensors = handle.to_tensors()
        if len(tensors) != len(values):
            logger.warning("aibrix_kvcache set allocate not enough")
            return False
        for i in range(len(tensors)):
            assert (
                tensors[i].nbytes == values[i].nbytes
            ), f"{tensors[i].nbytes}, {values[i].nbytes}"
            tensors[i].reshape(values[i].shape).copy_(values[i]).reshape(
                tensors[i].shape
            )
        status = self.kv_cache_manager.put(None, block_hash, handle)
        if not status.is_ok():
            logger.info(
                f"AIBrix KVCache Storage set failed, error_code {status.error_code}"
            )
            return False
        completed = status.value
        return completed == len(keys) * self.page_size

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_size: Optional[Any] = None,
    ) -> bool:
        return self.batch_set([key], [value], [target_location], [target_size])

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        block_hash = BlockHashes(keys, self.page_size)
        status = self.kv_cache_manager.exists(None, block_hash)
        if status.is_ok():
            return status.value // self.page_size
        return 0

    def exists(self, key: str) -> bool | dict:
        return self.batch_exists([key]) > 0
