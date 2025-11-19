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
import logging
import signal
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from sglang.srt.utils import get_bool_env_var, get_int_env_var

logger = logging.getLogger(__name__)

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

    @abstractmethod
    def cu_page_to_token(self, cu_page_num: int) -> int:
        pass


class ElasticAllocator(ABC):
    def register_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler

    def register_emem_orch(self, emem_orch):
        self.emem_orch = emem_orch

    @abstractmethod
    def can_unmap(self) -> bool:
        pass

    @abstractmethod
    def can_map(self) -> bool:
        pass

    @abstractmethod
    def disable(self) -> int:
        pass

    @abstractmethod
    def enable(self, need_size: int) -> int:
        pass

    @abstractmethod
    def cu_page_to_token(self, cu_page_num: int) -> int:
        pass

    @abstractmethod
    def evictable_size(self) -> int:
        pass

    @abstractmethod
    def evict(self, evictable_size: int) -> None:
        pass

    @abstractmethod
    def update_size(self) -> None:
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

        self.allocators = []

    def register_allocator(self, allocator: ElasticAllocator):
        allocator.register_emem_orch(self)
        self.allocators.append(allocator)

    def try_resize(self) -> None:
        map_candidate = None
        unmap_candidate = None
        for allocator in self.allocators:
            if allocator.can_map() and (
                (map_candidate is None)
                or (allocator.token_usage() > map_candidate.token_usage())
            ):
                map_candidate = allocator
            if allocator.can_unmap() and (
                (unmap_candidate is None)
                or (allocator.token_usage() < unmap_candidate.token_usage())
            ):
                unmap_candidate = allocator

        if (
            map_candidate is not None
            and unmap_candidate is not None
            and map_candidate != unmap_candidate
        ):
            logger.info(
                "ElasticMempoolOrchestrator try_resize "
                f"{map_candidate.token_usage()=}, "
                f"{unmap_candidate.token_usage()=}"
            )
            self.do_resize(map_candidate, unmap_candidate)

    def do_resize(
        self, map_allocator: ElasticAllocator, unmap_allocator: ElasticAllocator
    ) -> None:
        logger.info("ElasticMempoolOrchestrator do_resize")
        unmap_page = 0
        unmap_allocator.evict(unmap_allocator.evictable_size())
        unmap_page = unmap_allocator.disable()
        logger.info(f"{unmap_allocator._kvcache.pool_name} unmap {unmap_page} cu_page")

        map_token = map_allocator.cu_page_to_token(unmap_page)
        map_page = 0
        if map_token > 0:
            map_page = map_allocator.enable(map_token)
        logger.info(f"{map_allocator._kvcache.pool_name} map {map_page} cu_page")

        for _allocator in self.allocators:
            _allocator.update_size()
