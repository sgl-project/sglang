"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.mem_cache.elasticmem_orchestrator import (
    ElasticMempool,
    ElasticMempoolOrchestrator,
    use_elasticmem,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, SWAKVPool

if use_elasticmem:
    from kvcached.etensor import ETensor


logger = logging.getLogger(__name__)


class ElasticMHATokenToKVPool(MHATokenToKVPool, ElasticMempool):
    def __init__(
        self,
        *args,
        pool_name: Optional[str] = None,
        **kwargs,
    ):
        assert use_elasticmem
        assert pool_name is not None
        self.pool_name = pool_name
        super().__init__(*args, **kwargs)

    def _create_buffers(self):
        self.create_elastic_buffers()

    def create_elastic_buffers(self):
        current_device_id = f"cuda:{torch.cuda.current_device()}"

        assert self.size % self.page_size == 0
        self.esize = (
            ((self.size + self.page_size) + self.page_size - 1)
            // self.page_size
            * self.page_size
        )
        page_size = self.page_size
        state_shape = (self.head_num, self.head_dim)
        dtype = self.store_dtype
        self.ek_buffer = [
            ETensor(
                f"{self.pool_name}-k-{i}",
                self.esize,
                page_size,
                state_shape,
                dtype,
                current_device_id,
            )
            for i in range(self.layer_num)
        ]
        self.ev_buffer = [
            ETensor(
                f"{self.pool_name}-v-{i}",
                self.esize,
                page_size,
                state_shape,
                dtype,
                current_device_id,
            )
            for i in range(self.layer_num)
        ]
        self.k_buffer = [etensor.etensor for etensor in self.ek_buffer]
        self.v_buffer = [etensor.etensor for etensor in self.ev_buffer]
        self.state_memsize = self.ek_buffer[0].state_memsize
        logger.info(f"{self.esize=}, {self.k_buffer[0].shape=}")

    def disable(self, indices: List[int]) -> Tuple[int, List[int], List[int]]:
        total_unmap_num = 0
        for ek, ev in zip(self.ek_buffer, self.ev_buffer):
            unmap_num, pass_indices, proc_indices = ek.disable(indices)
            total_unmap_num += unmap_num
            unmap_num, pass_indices, proc_indices = ev.disable(indices)
            total_unmap_num += unmap_num
        self.size -= len(proc_indices)
        return total_unmap_num, pass_indices, proc_indices

    def enable(self, indices: List[int]) -> Tuple[int, List[int], List[int]]:
        total_map_num = 0
        for ek, ev in zip(self.ek_buffer, self.ev_buffer):
            map_num, pass_indices, proc_indices = ek.enable(indices)
            total_map_num += map_num
            map_num, pass_indices, proc_indices = ev.enable(indices)
            total_map_num += map_num
        self.size += len(proc_indices)
        return total_map_num, pass_indices, proc_indices


class ElasticSWAKVPool(SWAKVPool):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.emem_orch = ElasticMempoolOrchestrator()
        super().__init__(*args, **kwargs)

    def _create_buffers(self, token_to_kv_pool_class, **kwargs):
        token_to_kv_pool_class = ElasticMHATokenToKVPool
        self.swa_kv_pool = token_to_kv_pool_class(
            size=self.size_swa,
            dtype=self.dtype,
            layer_num=self.swa_layer_nums,
            pool_name="swa",
            **kwargs,
        )
        self.full_kv_pool = token_to_kv_pool_class(
            size=self.size,
            dtype=self.dtype,
            layer_num=self.full_layer_nums,
            pool_name="full",
            **kwargs,
        )
