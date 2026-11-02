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
from typing import Tuple

import torch

from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
    ElasticMempool,
    cu_page_size,
    init_emem_once,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)


class ElasticMHATokenToKVPool(MHATokenToKVPool, ElasticMempool):
    """MHA pool whose K/V buffers are kvcached ETensors: a large virtual range
    with only the first `size + page_size` rows physically backed.

    Used as the full/SWA sub-pool class of `SWAKVPool`; `allocation_label`
    ("Full" / "SWA") names the ETensors.
    """

    def _create_buffers(self):
        # The configurator refuses a hybrid-SWA split with an empty side.
        assert self.layer_num > 0, "elasticmem needs at least one layer per pool"
        # Both branches would bypass _create_buffers_normal and leave plain tensors.
        assert not self.is_quantized_kv_cache, (
            "elasticmem does not support quantized KV"
        )
        assert not self.post_capture_active, (
            "elasticmem does not support post-capture KV sizing"
        )
        super()._create_buffers()

    def _create_buffers_normal(self):
        assert self.kv_cache_layout == "nhd", (
            f"elasticmem supports the NHD KV layout only, got {self.kv_cache_layout}"
        )
        from kvcached.etensor import ETensor

        init_emem_once()
        current_device_id = f"cuda:{torch.cuda.current_device()}"
        free_memory, _ = torch.cuda.mem_get_info()
        pool_name = (self.allocation_label or "kv").lower()

        self.esize = self._size_to_esize(self.size)

        def make(kind: str, layer: int, head_dim: int) -> ETensor:
            return ETensor(
                f"{pool_name}-{kind}-{layer}",
                self.esize,
                self.page_size,
                (self.head_num, head_dim),
                self.store_dtype,
                current_device_id,
                free_memory,
            )

        self.ek_buffer = [make("k", i, self.head_dim) for i in range(self.layer_num)]
        self.ev_buffer = [make("v", i, self.v_head_dim) for i in range(self.layer_num)]
        self.k_buffer = [etensor.etensor for etensor in self.ek_buffer]
        self.v_buffer = [etensor.etensor for etensor in self.ev_buffer]
        self.state_memsize = (
            self.ek_buffer[0].state_memsize + self.ev_buffer[0].state_memsize
        ) // 2
        logger.debug(
            f"elasticmem {pool_name}: {self.esize=}, "
            f"{self.k_buffer[0].shape=}, {self.state_memsize=}"
        )

    def _size_to_esize(self, size: int) -> int:
        # The padded page (slot 0's page) absorbs dummy padded-token writes.
        return size + self.page_size

    def _esize_to_size(self, esize: int) -> int:
        return esize - self.page_size

    def get_kv_size_bytes(self):
        # Physically backed bytes; the ETensor views span the whole virtual range.
        k_size = self.ek_buffer[0].state_memsize * self.size * self.layer_num
        v_size = self.ev_buffer[0].state_memsize * self.size * self.layer_num
        return k_size, v_size

    # TODO(pansicheng): exception handler, consistency in mempool size for each kv layer
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

    def cu_page_to_token(self, cu_page_num: int) -> int:
        cu_page_num_per_kv_layer = cu_page_num // (2 * self.layer_num)
        cu_mem_per_kv_layer = cu_page_num_per_kv_layer * cu_page_size()
        token_num = cu_mem_per_kv_layer // self.state_memsize
        return token_num // self.page_size * self.page_size
