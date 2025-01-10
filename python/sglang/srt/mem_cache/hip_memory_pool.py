from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union, Optional, Dict
import logging

import math
import torch
import triton
from hip.models.hip_attention.gen3.attention_metadata import (
    HiPAttentionOutputMetadata,
    HiPAttentionCacheAccessStatistics
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.hip_attention.hip_config import HiPAttentionConfig

logger = logging.getLogger(__name__)


class HiPMetadataCachePool:
    cache: List[Dict[str, torch.Tensor]]

    def __init__(
        self,
        query_head_num: int,
        layer_num: int,
        device: str,
        hip_config: HiPAttentionConfig,
    ):
        self.hip_config = hip_config
        self.layer_num = layer_num
        self.cache = [{} for _ in range(layer_num)]
        self.head_num = query_head_num
        self.max_batch_size = hip_config.metadata_cache_max_batch_size
        self.device = device
        self.allocated_gpu_bytes = 0

        for layer_idx in range(layer_num):
            require_dense = layer_idx in hip_config.dense_layers
            if len(hip_config.layers) == 2:
                layer_config = hip_config.layers[0 if require_dense else 1]
            else:
                layer_config = hip_config.layers[layer_idx]

            n_chunks = triton.cdiv(layer_config.second_stage_k, layer_config.stages[-1].stage_chunk_size)
            
            num_q_blocks = 1
            self.init_buffer(layer_idx, 'indices', (num_q_blocks, n_chunks,), torch.int64)
            self.init_buffer(layer_idx, 'ks', (num_q_blocks,), torch.int64)
            self.init_buffer(layer_idx, 'ks_count', (num_q_blocks, 1,), torch.int64)
            self.init_buffer(layer_idx, 'ks_start_end', (num_q_blocks, 2,), torch.int64)
            
            self.init_buffer(layer_idx, 'mask_access_count', (num_q_blocks,), torch.int64)
            self.init_buffer(layer_idx, 'mask_unique_access_count', (num_q_blocks,), torch.int64)
            self.init_buffer(layer_idx, 'mask_cache_miss_count', (num_q_blocks,), torch.int64)

            self.init_buffer(layer_idx, 'sa_access_count', (num_q_blocks,), torch.int64)
            self.init_buffer(layer_idx, 'sa_unique_access_count', (num_q_blocks,), torch.int64)
            self.init_buffer(layer_idx, 'sa_cache_miss_count', (num_q_blocks,), torch.int64)

        self.allocated_gpu_bytes = self.compute_allocated_bytes()
        logger.info(f"Allocated HiP metadata cache pool size: {self.allocated_gpu_bytes / 1024 / 1024:.2f} MB")

    def compute_allocated_bytes(self):
        t = 0
        for layer_buffer in self.cache:
            for v in layer_buffer.values():
                t += v.numel() * v.element_size()
        return t

    def init_buffer(self, layer_idx: int, name: str, shape: List[int], dtype: torch.dtype):
        layer_buffer = self.cache[layer_idx]
        layer_buffer[name] = torch.zeros(
            (self.max_batch_size * self.head_num, *shape), 
            device=self.device, 
            dtype=dtype
        )
    
    def get_buffer(self, layer_idx: int, name: str, batch_size: int):
        return self.cache[layer_idx][name][:batch_size * self.head_num]

    def set_buffer(self, layer_idx: int, name: str, value: torch.Tensor):
        target = self.cache[layer_idx][name][:value.shape[0]]
        target.copy_(value)

    def get_hip_metadata_cache(
        self, 
        layer_id: int, 
        size: int, 
        batch_size: int
    ) -> HiPAttentionOutputMetadata:
        assert size == batch_size
        
        return HiPAttentionOutputMetadata(
            indices=self.get_buffer(layer_id, 'indices', batch_size),
            ks=self.get_buffer(layer_id, 'ks', batch_size),
            ks_count=self.get_buffer(layer_id, 'ks_count', batch_size),
            ks_start_end=self.get_buffer(layer_id, 'ks_start_end', batch_size),
            mask_cache_statistics=None,
            sa_cache_statistics=None,
        )

    def set_hip_metadata_cache(
        self,
        layer_id: int,
        size: int,
        batch_size: int,
        metadata: HiPAttentionOutputMetadata
    ):
        assert size == batch_size

        self.set_buffer(layer_id, 'indices', metadata.indices)
        self.set_buffer(layer_id, 'ks', metadata.ks)
        self.set_buffer(layer_id, 'ks_count', metadata.ks_count)
        self.set_buffer(layer_id, 'ks_start_end', metadata.ks_start_end)
        
        def update_cache_stats(stats: HiPAttentionCacheAccessStatistics, prefix: str):
            if stats is None:
                access_count = torch.zeros((1,), dtype=torch.int64, device=self.device)
                unique_access_count = torch.zeros((1,), dtype=torch.int64, device=self.device)
                cache_miss_count = torch.zeros((1,), dtype=torch.int64, device=self.device)
            else:
                computed_statistics = stats.compute_statistics()
                access_count = computed_statistics['access_count']
                unique_access_count = computed_statistics['unique_access_count']
                cache_miss_count = computed_statistics['cache_miss_count']
            
            self.set_buffer(layer_id, f'{prefix}_access_count', access_count.view(1, 1).expand(self.max_batch_size, 1))
            self.set_buffer(layer_id, f'{prefix}_unique_access_count', unique_access_count.view(1, 1).expand(self.max_batch_size, 1))
            self.set_buffer(layer_id, f'{prefix}_cache_miss_count', cache_miss_count.view(1, 1).expand(self.max_batch_size, 1))

        update_cache_stats(metadata.sa_cache_statistics, 'sa')
        update_cache_stats(metadata.mask_cache_statistics, 'mask')
    
    def compute_cache_statistics(self, batch_size: int):
        def compute(prefix):
            total_access = 0
            total_miss = 0
            for idx_layer in range(self.layer_num):
                access_count = self.get_buffer(idx_layer, f'{prefix}_access_count', batch_size)
                miss_count = self.get_buffer(idx_layer, f'{prefix}_cache_miss_count', batch_size)
                total_access += access_count.sum()
                total_miss += miss_count.sum()
            return {
                f'{prefix}_access': total_access,
                f'{prefix}_miss': total_miss,
                f'{prefix}_hit_ratio': 1 - (total_miss / total_access),
            }

        result = {}
        result.update(compute('sa'))
        result.update(compute('mask'))
        return result