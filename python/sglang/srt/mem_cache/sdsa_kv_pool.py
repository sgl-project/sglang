# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KV pool carrying the index keys that SDSA's indexer scores.

Every full-attention layer keeps one BF16 index key per KV slot, so a key's
cache row is its KV slot and its lifecycle rides the KV allocator and radix
tree. With a KV page of 64 tokens the cache reads as FlashInfer SDSA's
``[pages, 64, index_head_dim]`` key layout. PD transfer and HiCache offload do
not carry these keys (create_sdsa_backend rejects both).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import GB, HybridLinearKVPool, MambaPool

SDSA_INDEX_DTYPE = torch.bfloat16
# FlashInfer's SDSA indexer reads keys in pages of 64.
SDSA_INDEX_PAGE = 64


class SdsaTokenToKVPool(HybridLinearKVPool):
    """Hybrid full/linear KV pool plus a per-slot SDSA index-key cache."""

    @staticmethod
    def sdsa_bytes_per_token(*, index_head_dim: int, num_layers: int) -> int:
        return index_head_dim * SDSA_INDEX_DTYPE.itemsize * num_layers

    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        sdsa_index_head_dim: int,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
    ):
        if page_size % SDSA_INDEX_PAGE != 0:
            raise ValueError(
                f"SDSA needs a KV page size that is a multiple of {SDSA_INDEX_PAGE} "
                f"(FlashInfer's SDSA key pages), got {page_size}."
            )
        # super().__init__ sizes mem_usage through the overridden
        # get_kv_size_bytes, so the index buffers need a placeholder first.
        self.sdsa_index_k_buffer_pool: list[torch.Tensor] = []
        super().__init__(
            size=size,
            dtype=dtype,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            mamba_pool=mamba_pool,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            use_mla=False,
            start_layer=start_layer,
            full_kv_pool_class=full_kv_pool_class,
            quant_method=quant_method,
            post_capture_active=post_capture_active,
        )
        self.sdsa_index_head_dim = sdsa_index_head_dim
        # Same slot space as the KV buffers (slot 0 is the padding slot).
        num_slots = size + page_size
        allocation_pool = self.full_kv_pool
        with (
            allocation_pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(allocation_pool.custom_mem_pool)
                if allocation_pool.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            self.sdsa_index_k_flat = torch.zeros(
                (len(full_attention_layer_ids), num_slots, sdsa_index_head_dim),
                dtype=SDSA_INDEX_DTYPE,
                device=device,
            )
        self.sdsa_index_k_buffer_pool = list(self.sdsa_index_k_flat.unbind(0))
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        """``[num_slots, index_head_dim]``, indexed by KV slot."""
        self._wait_for_layer(layer_id)
        return self.sdsa_index_k_buffer_pool[self._transfer_full_attention_id(layer_id)]

    def set_index_k_buffer(
        self, layer_id: int, loc: torch.Tensor, index_k: torch.Tensor
    ) -> None:
        buffer = self.sdsa_index_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]
        buffer[loc.long()] = index_k.to(buffer.dtype)

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        index_size = sum(
            t.numel() * t.element_size() for t in self.sdsa_index_k_buffer_pool
        )
        return k_size + index_size, v_size
