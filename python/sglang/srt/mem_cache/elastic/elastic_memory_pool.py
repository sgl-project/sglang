"""
Copyright 2025 SGLang Team
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
from typing import Optional, Tuple, override

import torch

from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
    ElasticMempool,
    cu_page_size,
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
        logger.info(f"{self.pool_name=} initialized")

    def _create_buffers(self):
        self.create_elastic_buffers()

    def _size_to_esize(self, size: int):
        return size + self.page_size

    def _esize_to_size(self, esize: int):
        return esize - self.page_size

    @override
    def create_elastic_buffers(self):
        current_device_id = f"cuda:{torch.cuda.current_device()}"

        free_memory, total_memory = torch.cuda.mem_get_info()

        self.esize = self._size_to_esize(self.size)
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
                free_memory,
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
                free_memory,
            )
            for i in range(self.layer_num)
        ]
        self.k_buffer = [etensor.etensor for etensor in self.ek_buffer]
        self.v_buffer = [etensor.etensor for etensor in self.ev_buffer]
        self.state_memsize = self.ek_buffer[0].state_memsize
        logger.debug(f"{self.esize=}, {self.k_buffer[0].shape=}, {self.state_memsize=}")

    # TODO: exception handler, consistency in mempool size for each kv layer
    @override
    def reduce(self, new_size: int) -> Tuple[int, int]:
        new_esize = self._size_to_esize(new_size)
        total_unmap_num = 0
        for ek, ev in zip(self.ek_buffer, self.ev_buffer):
            unmap_num, self.esize = ek.reduce(new_esize)
            total_unmap_num += unmap_num
            unmap_num, self.esize = ev.reduce(new_esize)
            total_unmap_num += unmap_num
        self.size = self._esize_to_size(self.esize)
        return total_unmap_num, self.size

    @override
    def expand(self, new_size: int) -> Tuple[int, int]:
        new_esize = self._size_to_esize(new_size)
        total_map_num = 0
        for ek, ev in zip(self.ek_buffer, self.ev_buffer):
            map_num, self.esize = ek.expand(new_esize)
            total_map_num += map_num
            map_num, self.esize = ev.expand(new_esize)
            total_map_num += map_num
        self.size = self._esize_to_size(self.esize)
        return total_map_num, self.size

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        cu_page_num_per_kv_layer = cu_page_num // (2 * self.layer_num)
        cu_mem_per_kv_layer = cu_page_num_per_kv_layer * cu_page_size
        token_num = cu_mem_per_kv_layer // self.state_memsize
        token_num = token_num // self.page_size * self.page_size
        return token_num


class ElasticSWAKVPool(SWAKVPool):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        logger.info(f"ElasticSWAKVPool initialized")

    def _create_buffers(self, token_to_kv_pool_class, **kwargs):
        self.create_elastic_buffers(token_to_kv_pool_class, **kwargs)

    @override
    def create_elastic_buffers(self, token_to_kv_pool_class, **kwargs):
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

    @override
    def reduce(self, new_size: int) -> Tuple[int, int]:
        raise NotImplementedError()

    @override
    def expand(self, new_size: int) -> Tuple[int, int]:
        raise NotImplementedError()

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        raise NotImplementedError()
