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

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
)

logger = logging.getLogger(__name__)


class AibrixKVCacheStorage(HiCacheStorage):
    def __init__(self, config: HiCacheStorageConfig, kv_cache: KVCache):
        tp_rank = config.tp_rank
        tp_size = config.tp_size
        self.page_size = kv_cache.page_size
        self.kv_cache_dtype = kv_cache.dtype
        self.kv_cache = kv_cache
        self.layer_num = self.kv_cache.layer_num
        self.kv_head_ids = [
            tp_rank * self.kv_cache.head_num + i for i in range(self.kv_cache.head_num)
        ]
        if isinstance(kv_cache, MLATokenToKVPool):
            self.kv_cache_shape = (
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache.kv_lora_rank + self.kv_cache.qk_rope_head_dim,
            )
            raise NotImplementedError(
                "MLA is not supported by AibrixKVCacheStorage yet."
            )
        elif isinstance(kv_cache, MHATokenToKVPool):
            self.kv_cache_shape = (
                2,
                self.layer_num,
                self.page_size,
                self.kv_cache.head_num,
                self.kv_cache.head_dim,
            )

            self.layer_ids = range(
                self.kv_cache.start_layer, self.kv_cache.end_layer
            )  # for pipeline parallel

            self.block_spec = KVCacheBlockSpec(
                block_ntokens=self.page_size,
                block_dtype=self.kv_cache_dtype,
                block_layout=KVCacheBlockLayout(KVCacheBlockLayout.NCLD),
                tensor_spec=KVCacheTensorSpec(
                    heads=self.kv_head_ids,
                    layers=self.layer_ids,
                    head_size=self.kv_cache.head_dim,
                ),
            )
            logger.info(self.block_spec)
            config = KVCacheConfig(
                block_spec=self.block_spec, model_spec=ModelSpec(102400)
            )
            self.kv_cache_manager = BaseKVCacheManager(config)

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        block_hash = BlockHashes(keys, self.page_size)
        status = self.kv_cache_manager.acquire(None, block_hash)
        log_every_n_seconds(
            logger, logging.INFO, self.kv_cache_manager.metrics.summary(), 1
        )
        if status.is_ok():
            num_fetched_tokens, handle = status.value
            kv_blocks = handle.to_tensors()
            assert len(kv_blocks) == len(target_locations)
            for i in range(len(kv_blocks)):
                target_locations[i].reshape(kv_blocks[i].shape).copy_(
                    kv_blocks[i]
                ).reshape(target_locations[i].shape)
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
            logger.warning("aibrix_kvcache set allocate failed")
            return False
        handle = status.value
        tensors = handle.to_tensors()
        if len(tensors) != len(values):
            logger.warning("aibrix_kvcache set allocate not enough")
            return False
        for i in range(len(tensors)):
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

    def batch_exists(self, keys: List[str]) -> int:
        block_hash = BlockHashes(keys, self.page_size)
        status = self.kv_cache_manager.exists(None, block_hash)
        if status.is_ok():
            return status.value // self.page_size
        return 0

    def exists(self, key: str) -> bool | dict:
        return self.batch_exists([key]) > 0
