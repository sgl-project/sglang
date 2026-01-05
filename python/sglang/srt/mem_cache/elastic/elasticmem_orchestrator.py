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
import time
from abc import ABC, abstractmethod
from typing import Tuple

import torch

from sglang.srt.utils import get_bool_env_var, get_float_env_var, get_int_env_var

logger = logging.getLogger(__name__)

use_elasticmem = get_bool_env_var("SGLANG_ELASTIC_MEM_POOL", "false")
# page size of device memory, default 2MB
cu_page_size = get_int_env_var("SGLANG_CU_PAGE_SIZE", 2 << 20)

can_map_threshold = get_float_env_var("SGLANG_CAN_MAP", 0.9)
can_unmap_threshold = get_float_env_var("SGLANG_CAN_UNMAP", 0.8)
resize_trigger_diff_ratio = get_float_env_var("SGLANG_RESIZE_TRIGGER_DIFF_RATIO", 0.2)

if use_elasticmem:
    import kvcached.vmm_ops as vmm_ops


class ElasticMempool(ABC):
    @abstractmethod
    def create_elastic_buffers(self):
        pass

    @abstractmethod
    def reduce(self, new_size: int) -> Tuple[int, int]:
        """Reduce the memory pool to a new size.

        Args:
            new_size: Target size for the memory pool.

        Returns:
            A tuple containing (released_cu_pages, mempool_size_after_reduce).
        """
        pass

    @abstractmethod
    def expand(self, new_size: int) -> Tuple[int, int]:
        """Expand the memory pool to a new size.

        Args:
            new_size: Target size for the memory pool.

        Returns:
            A tuple containing (allocated_cu_pages, mempool_size_after_expand).
        """
        pass

    @abstractmethod
    def cu_page_to_token(self, cu_page_num: int) -> int:
        # The approximate number of tokens that can be expanded by mapping cu_page_num cu_pages.
        pass


class ElasticAllocator(ABC):
    def register_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler

    def register_emem_orch(self, emem_orch):
        self.emem_orch = emem_orch

    def can_be_candidate(self) -> bool:
        return False

    def mark_unmap_candidate(self, is_candidate: bool) -> "ElasticAllocator":
        raise NotImplementedError()

    @abstractmethod
    def can_unmap(self) -> bool:
        pass

    def can_do_unmap(self) -> bool:
        return False

    @abstractmethod
    def can_map(self) -> bool:
        pass

    @abstractmethod
    def reduce(self) -> int:
        """Reduce the allocated memory.

        Returns:
            Number of cu_page unmapped.
        """
        pass

    @abstractmethod
    def expand(self, expand_size: int) -> int:
        """Expand the allocated memory.

        Args:
            need_size: Required additional size.

        Returns:
            Number of cu_page mapped.
        """
        pass

    @abstractmethod
    def cu_page_to_token(self, cu_page_num: int) -> int:
        # The approximate number of tokens that can be expanded by mapping cu_page_num cu_pages.
        pass

    @abstractmethod
    def register_evict_func(self, func_evictable_size, func_evict) -> None:
        pass

    @abstractmethod
    def token_usage(self) -> float:
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
    """Orchestrator for managing elastic memory pools.
    Coordinates between different allocators to dynamically resize memory pools
    based on demand.
    """

    def __init__(self):
        assert use_elasticmem
        atexit.register(vmm_ops.shutdown_emem)
        signal.signal(signal.SIGINT, lambda sig, frame: vmm_ops.shutdown_emem())
        signal.signal(signal.SIGTERM, lambda sig, frame: vmm_ops.shutdown_emem())
        signal.signal(signal.SIGQUIT, lambda sig, frame: vmm_ops.shutdown_emem())

        current_device_id = f"cuda:{torch.cuda.current_device()}"
        vmm_ops.init_emem(current_device_id, cu_page_size)

        self.allocators = []

        self.remaining_page = 0

        self.unmap_candidate = None

    def register_allocator(self, allocator: ElasticAllocator):
        allocator.register_emem_orch(self)
        self.allocators.append(allocator)

    def register_scheduler(self, scheduler) -> None:
        for allocator in self.allocators:
            allocator.register_scheduler(scheduler)

    def _get_candidate_allocator(self):
        map_candidate = None
        unmap_candidate = None

        for allocator in self.allocators:
            if not allocator.can_be_candidate():
                continue

            if (map_candidate is None) or (
                allocator.token_usage() > map_candidate.token_usage()
            ):
                map_candidate = allocator

            if (unmap_candidate is None) or (
                allocator.token_usage() < unmap_candidate.token_usage()
            ):
                unmap_candidate = allocator

        return map_candidate, unmap_candidate

    def _reset_unmap_cadidate(self):
        if self.unmap_candidate is None:
            return
        self.unmap_candidate = self.unmap_candidate.mark_unmap_candidate(
            is_candidate=False
        )

    def _set_unmap_cadidate(self, unmap_candidate):
        if self.unmap_candidate == unmap_candidate:
            return

        self._reset_unmap_cadidate()
        self.unmap_candidate = unmap_candidate.mark_unmap_candidate(is_candidate=True)

    def try_resize(self) -> None:
        map_candidate, unmap_candidate = self._get_candidate_allocator()

        diff = map_candidate.token_usage() - unmap_candidate.token_usage()
        if diff < resize_trigger_diff_ratio:
            self._reset_unmap_cadidate()
            return

        if map_candidate is None or unmap_candidate is None:
            self._reset_unmap_cadidate()
            return

        if not map_candidate.can_map() or not unmap_candidate.can_unmap():
            self._reset_unmap_cadidate()
            return

        if self.unmap_candidate is None or not self.unmap_candidate.can_unmap():
            self._set_unmap_cadidate(unmap_candidate)

        if self.unmap_candidate is None or not self.unmap_candidate.can_do_unmap():
            return

        self._do_resize(map_candidate, self.unmap_candidate)

        self._reset_unmap_cadidate()

    def _do_resize(
        self, map_allocator: ElasticAllocator, unmap_allocator: ElasticAllocator
    ) -> None:
        start_time = time.perf_counter()

        logger.info("ElasticMempoolOrchestrator do_resize")
        unmap_page = 0
        unmap_page = unmap_allocator.reduce()
        logger.info(
            f"{unmap_allocator._kvcache.pool_name} unmap {unmap_page}, remain {self.remaining_page} cu_page"
        )

        # The approximate number of tokens that can be expanded by mapping cu_page_num cu_pages.
        to_map_page = unmap_page + self.remaining_page
        map_token = map_allocator.cu_page_to_token(to_map_page)
        map_page = 0
        if map_token > 0:
            map_page = map_allocator.expand(map_token)
        self.remaining_page = to_map_page - map_page
        logger.info(
            f"{map_allocator._kvcache.pool_name} map {map_page}, remain {self.remaining_page} cu_page"
        )

        for _allocator in self.allocators:
            _allocator.update_size()

        logger.debug(f"do_resize took {(time.perf_counter() - start_time) * 1000} ms")
