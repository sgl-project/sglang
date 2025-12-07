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
from sglang.srt.mem_cache.memory_pool import (
    GB,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MambaPool,
    MHATokenToKVPool,
    SWAKVPool,
    get_tensor_size_bytes,
)

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

    @override
    def _finalize_allocation_log(self, num_tokens: int):
        k_size = self.state_memsize * self.size * self.layer_num
        v_size = k_size
        k_size_GB = k_size / GB
        v_size_GB = v_size / GB
        logger.info(
            f"KV Cache is allocated. #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
        )
        self.mem_usage = k_size_GB + v_size_GB

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


##################################################


class ElasticHybridReqToTokenPool(HybridReqToTokenPool):
    @override
    def _create_mamba_pool(
        self, size, cache_params, device, speculative_num_draft_tokens
    ):
        self.mamba_pool = ElasticMambaPool(
            size=size,
            cache_params=cache_params,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            pool_name="mamba",
        )


class ElasticMambaPool(MambaPool, ElasticMempool):
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
        return size + 1

    def _esize_to_size(self, esize: int):
        return esize - 1

    def sort_free(self):
        self.free_slots, _ = torch.sort(self.free_slots)

    # TODO: Optimize tensor memory layout (layer-first) for compatibility with PD disaggregation and speculative decoding.
    @override  # ElasticMempool
    def create_elastic_buffers(self):
        current_device_id = f"cuda:{torch.cuda.current_device()}"

        free_memory, total_memory = torch.cuda.mem_get_info()
        free_memory = free_memory // self.num_mamba_layers

        self.esize = self._size_to_esize(self.size)

        self.state_memsize = 0
        self.econv_state = []
        for i, conv_shape in enumerate(self.conv_state_shape):
            self.econv_state.append(
                [
                    ETensor(
                        f"{self.pool_name}-conv-{i}-{j}",
                        self.esize,
                        1,
                        conv_shape,
                        self.conv_dtype,
                        current_device_id,
                        free_memory,
                    )
                    for j in range(self.num_mamba_layers)
                ]
            )
            self.state_memsize = max(
                self.state_memsize, self.econv_state[i][0].state_memsize
            )
        conv_state = [
            [etensor.etensor for etensor in econv_state]
            for econv_state in self.econv_state
        ]
        self.etemporal_state = [
            ETensor(
                f"{self.pool_name}-ssm-{i}",
                self.esize,
                1,
                self.temporal_state_shape,
                self.ssm_dtype,
                current_device_id,
                free_memory,
            )
            for i in range(self.num_mamba_layers)
        ]
        self.state_memsize = max(
            self.state_memsize, self.etemporal_state[0].state_memsize
        )
        temporal_state = [etensor.etensor for etensor in self.etemporal_state]

        # TODO: speculative
        assert self.speculative_num_draft_tokens is None

        self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
        logger.info(
            f"Mamba Cache is allocated. "
            f"max_mamba_cache_size: {self.size}, "
            f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
            f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
        )

    @override  # MambaPool
    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))
        for j in range(self.num_mamba_layers):
            for i in range(len(self.mamba_cache.conv)):
                self.mamba_cache.conv[i][j][free_index] = 0
            self.mamba_cache.temporal[j][free_index] = 0

    @override  # MambaPool
    def clear(self):
        for j in range(self.num_mamba_layers):
            for i in range(len(self.mamba_cache.conv)):
                self.mamba_cache.conv[i][j].zero_()
            self.mamba_cache.temporal[j].zero_()

        self.free_slots = torch.arange(self.size, dtype=torch.int64, device=self.device)

    @override  # MambaPool
    def copy_from(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        for j in range(self.num_mamba_layers):
            for i in range(len(self.mamba_cache.conv)):
                self.mamba_cache.conv[i][j][dst_index] = self.mamba_cache.conv[i][j][
                    src_index
                ]
            self.mamba_cache.temporal[j][dst_index] = self.mamba_cache.temporal[j][
                src_index
            ]
        return

    @override  # ElasticMempool
    def reduce(self, new_size: int) -> Tuple[int, int]:
        new_esize = self._size_to_esize(new_size)
        total_unmap_num = 0
        for _econv_state in self.econv_state:
            for _econv_state_layer in _econv_state:
                unmap_num, self.esize = _econv_state_layer.reduce(new_esize)
                total_unmap_num += unmap_num
        for _etemporal_state_layer in self.etemporal_state:
            unmap_num, self.esize = _etemporal_state_layer.reduce(new_esize)
            total_unmap_num += unmap_num

        self.size = self._esize_to_size(self.esize)
        return total_unmap_num, self.size

    @override  # ElasticMempool
    def expand(self, new_size: int) -> Tuple[int, int]:
        new_esize = self._size_to_esize(new_size)
        total_map_num = 0
        for _econv_state in self.econv_state:
            for _econv_state_layer in _econv_state:
                map_num, self.esize = _econv_state_layer.expand(new_esize)
                total_map_num += map_num
        for _etemporal_state_layer in self.etemporal_state:
            map_num, self.esize = _etemporal_state_layer.expand(new_esize)
            total_map_num += map_num

        self.size = self._esize_to_size(self.esize)
        return total_map_num, self.size

    @override  # ElasticMempool
    def cu_page_to_token(self, cu_page_num: int) -> int:
        cu_page_num_per_state_layer = cu_page_num // (
            (len(self.econv_state) + 1)  # self.econv_state + self.etemporal_state
            * self.num_mamba_layers
        )
        cu_mem_per_state_layer = cu_page_num_per_state_layer * cu_page_size
        token_num = cu_mem_per_state_layer // self.state_memsize
        token_num = token_num
        return token_num


class ElasticHybridLinearKVPool(HybridLinearKVPool, ElasticMempool):
    def _create_buffers(self):
        self.create_elastic_buffers()

    @override  # ElasticMempool
    def create_elastic_buffers(self):
        assert not self.use_mla
        self.pool_name = "full"
        # TODO: mla
        self.full_kv_pool = ElasticMHATokenToKVPool(
            size=self.size,
            page_size=self.page_size,
            dtype=self.dtype,
            head_num=self.head_num,
            head_dim=self.head_dim,
            layer_num=self.full_layer_nums,
            pool_name=self.pool_name,
            device=self.device,
            enable_memory_saver=self.enable_memory_saver,
        )
        self.state_memsize = self.full_kv_pool.state_memsize

    @override  # ElasticMempool
    def reduce(self, new_size: int) -> Tuple[int, int]:
        total_unmap_num, size = self.full_kv_pool.reduce(new_size)
        self.size = size
        return total_unmap_num, size

    @override  # ElasticMempool
    def expand(self, new_size: int) -> Tuple[int, int]:
        total_map_num, size = self.full_kv_pool.expand(new_size)
        self.size = size
        return total_map_num, size

    @override  # ElasticMempool
    def cu_page_to_token(self, cu_page_num: int) -> int:
        return self.full_kv_pool.cu_page_to_token(cu_page_num)
