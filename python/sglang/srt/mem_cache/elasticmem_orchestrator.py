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

import atexit
import signal
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from sglang.srt.utils import get_bool_env_var, get_int_env_var

use_elasticmem = get_bool_env_var("SGLANG_ELASTIC_MEM_POOL", "false")
cu_page_size = get_int_env_var("SGLANG_CU_PAGE_SIZE", 2 << 20)

if use_elasticmem:
    import kvcached.vmm_ops as vmm_ops


class ElasticMempool(ABC):
    @abstractmethod
    def create_elastic_buffers(self):
        pass

    @abstractmethod
    def disable(self, indices: List[int]) -> Tuple[int, List[int], List[int]]:
        pass

    @abstractmethod
    def enable(self, indices: List[int]) -> Tuple[int, List[int], List[int]]:
        pass


class ElasticAllocator(ABC):
    @abstractmethod
    def disable(self, need_size: int) -> int:
        pass

    @abstractmethod
    def enable(self, need_size: int) -> int:
        pass


class ElasticMempoolOrchestrator:
    def __init__(self):
        assert use_elasticmem
        atexit.register(vmm_ops.shutdown_emem)
        signal.signal(signal.SIGINT, lambda sig, frame: vmm_ops.shutdown_emem())
        signal.signal(signal.SIGTERM, lambda sig, frame: vmm_ops.shutdown_emem())
        signal.signal(signal.SIGQUIT, lambda sig, frame: vmm_ops.shutdown_emem())

        current_device_id = f"cuda:{torch.cuda.current_device()}"
        vmm_ops.init_emem(current_device_id, cu_page_size)

        self.elastic_mempools = []

    def register_pool(self, mem_pool: ElasticMempool, allocator: ElasticAllocator):
        self.elastic_mempools.append((mem_pool, allocator))
