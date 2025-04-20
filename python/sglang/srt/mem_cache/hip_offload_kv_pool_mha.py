from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
from torch import Tensor

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from hip_attn.v1_2 import HiPAttentionConfig, HiPOffloadCache

logger = logging.getLogger(__name__)


class MHATokenToHiPOffloadKVPool(KVCache):

    def __init__(
        self,
        max_token_size: int,
        max_mask_cache_factor: float,
        max_mask_cache_size: Optional[int],
        max_sa_cache_factor: float,
        max_sa_cache_size: Optional[int],
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: torch.device,
        hip_config: HiPAttentionConfig,
    ):
        super().__init__()
        self.size = max_token_size
        self.dtype = dtype
        self.device = device

        assert isinstance(device, torch.device)
        assert device.index is not None

        from hip_attn.v1_2 import HiPModelOffloadCache

        self.offload_cache = HiPModelOffloadCache(
            max_token_size=max_token_size,
            max_mask_cache_factor=max_mask_cache_factor,
            max_mask_cache_token_size=max_mask_cache_size,
            max_sa_cache_factor=max_sa_cache_factor,
            max_sa_cache_token_size=max_sa_cache_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            hip_config=hip_config,
        )

    def get_key_buffer(self, layer_id: int):
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int):
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[HiPOffloadCache, Any]:
        return self.offload_cache.get_kv_buffer(layer_id)

    def get_fetched_prefix_kv_buffer(
        self,
        layer_id: int,
        extend_seq_lens: Tensor,
        extend_seq_lens_cpu: List[int],
        # you need to pass KV for extend
        cache_k: Tensor,
        cache_v: Tensor,
    ) -> Tuple[Tensor, Tensor, Any]:
        return self.offload_cache.get_fetched_prefix_kv_buffer(
            layer_id,
            cache_k=cache_k,
            cache_v=cache_v,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        table: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        async_copy: bool = False,
        push_to_gpu_cache: bool = False,
    ):
        self.offload_cache.set_kv_buffer(
            layer.layer_id, table, cache_k, cache_v, async_copy, push_to_gpu_cache
        )

    def get_flat_data(self, indices):
        raise NotImplementedError()

    def transfer(self, indices, flat_data):
        raise NotImplementedError()

    def transfer_per_layer(self, indices, flat_data, layer_id):
        raise NotImplementedError()

    def on_model_start(self, forward_batch: ForwardBatch):
        assert forward_batch.token_to_kv_pool == self

        self.offload_cache.on_model_start(
            forward_batch.forward_mode.is_extend(),
            forward_batch.batch_size,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
        )

    def on_model_end(self, forward_batch: ForwardBatch):
        assert forward_batch.token_to_kv_pool == self

        self.offload_cache.on_model_end(
            forward_batch.forward_mode.is_extend(),
        )

    def on_layer_start(self, forward_batch: ForwardBatch, layer_id: int):
        assert forward_batch.token_to_kv_pool == self

        self.offload_cache.on_layer_start(
            layer_id,
            forward_batch.forward_mode.is_extend(),
            forward_batch.batch_size,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
        )

    def on_layer_end(self, forward_batch: ForwardBatch, layer_id: int):
        assert forward_batch.token_to_kv_pool == self

        self.offload_cache.on_layer_end(
            layer_id,
            forward_batch.forward_mode.is_extend(),
        )

    def is_online_cache_update_enabled(self):
        return self.offload_cache.is_online_cache_update_enabled()
