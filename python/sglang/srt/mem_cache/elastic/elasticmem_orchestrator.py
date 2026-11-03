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

Elastic memory (kvcached VMM) for hybrid-SWA models.

The full-attention and SWA KV pools each reserve a large virtual range and back
only their current size with physical pages. When one pool runs hot and the
other cold, the orchestrator shrinks the cold pool's tail and maps the released
pages into the hot one.
"""

from __future__ import annotations

import atexit
import functools
import logging
import signal
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def is_elasticmem_enabled() -> bool:
    return envs.SGLANG_ENABLE_ELASTIC_MEM_POOL.get()


def cu_page_size() -> int:
    """Physical page granularity of device memory map/unmap, in bytes."""
    return envs.SGLANG_ELASTIC_MEM_CU_PAGE_SIZE.get()


_emem_initialized = False


def init_emem_once() -> None:
    """Initialize kvcached on the current device; idempotent.

    Must run before the first ETensor is created.
    """
    global _emem_initialized
    if _emem_initialized:
        return
    assert is_elasticmem_enabled()

    import kvcached.vmm_ops as vmm_ops

    vmm_ops.init_emem(f"cuda:{torch.cuda.current_device()}", cu_page_size())
    install_shutdown_handlers(vmm_ops.shutdown_emem)
    _emem_initialized = True


def install_shutdown_handlers(shutdown: Callable[[], None]) -> None:
    """Run shutdown at exit, and before a terminating signal kills the process."""
    atexit.register(shutdown)
    # signal.signal only works on the main thread; atexit still covers it elsewhere.
    if threading.current_thread() is not threading.main_thread():
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
        # A Python handler or SIG_IGN keeps the process alive past the signal,
        # so atexit runs; only the default disposition kills it first.
        if signal.getsignal(sig) == signal.SIG_DFL:
            signal.signal(sig, functools.partial(_shutdown_then_die, shutdown))


def _shutdown_then_die(shutdown: Callable[[], None], signum: int, frame) -> None:
    shutdown()
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)


class ElasticMempool(ABC):
    @abstractmethod
    def reduce(self, new_size: int) -> Tuple[int, int]:
        """Shrink to new_size tokens; returns (released_cu_pages, size_after)."""

    @abstractmethod
    def expand(self, new_size: int) -> Tuple[int, int]:
        """Grow to new_size tokens; returns (mapped_cu_pages, size_after)."""

    @abstractmethod
    def cu_page_to_token(self, cu_page_num: int) -> int:
        """Approximate number of tokens gained by mapping cu_page_num cu_pages."""


class ElasticAllocator(ABC):
    def can_be_candidate(self) -> bool:
        return False

    def mark_unmap_candidate(self, is_candidate: bool) -> Optional[ElasticAllocator]:
        raise NotImplementedError()

    @abstractmethod
    def can_unmap(self) -> bool: ...

    def can_do_unmap(self) -> bool:
        return False

    @abstractmethod
    def can_map(self) -> bool: ...

    @abstractmethod
    def reduce(self) -> int:
        """Shrink to the marked candidate size; returns cu_pages unmapped."""

    @abstractmethod
    def expand(self, expand_size: int) -> int:
        """Grow by expand_size tokens; returns cu_pages mapped."""

    @abstractmethod
    def cu_page_to_token(self, cu_page_num: int) -> int: ...

    @abstractmethod
    def token_usage(self) -> float: ...

    @abstractmethod
    def update_size(self) -> None: ...

    @property
    def pool_name(self) -> str:
        return type(self).__name__


class ElasticMempoolOrchestrator:
    """Orchestrator for managing elastic memory pools.
    Coordinates between different allocators to dynamically resize memory pools
    based on demand.
    """

    def __init__(self):
        assert is_elasticmem_enabled()
        self.allocators: List[ElasticAllocator] = []
        self.remaining_page = 0
        self.unmap_candidate: Optional[ElasticAllocator] = None
        self.resize_trigger_diff_ratio = (
            envs.SGLANG_ELASTIC_MEM_RESIZE_TRIGGER_DIFF_RATIO.get()
        )

    def register_allocator(self, allocator: ElasticAllocator) -> None:
        self.allocators.append(allocator)

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

    def _reset_unmap_candidate(self):
        if self.unmap_candidate is None:
            return
        self.unmap_candidate = self.unmap_candidate.mark_unmap_candidate(
            is_candidate=False
        )

    def _set_unmap_candidate(self, unmap_candidate: ElasticAllocator):
        if self.unmap_candidate is unmap_candidate:
            return

        self._reset_unmap_candidate()
        self.unmap_candidate = unmap_candidate.mark_unmap_candidate(is_candidate=True)

    def try_resize(self) -> None:
        map_candidate, unmap_candidate = self._get_candidate_allocator()
        if (
            map_candidate is None
            or unmap_candidate is None
            or map_candidate is unmap_candidate
        ):
            self._reset_unmap_candidate()
            return

        diff = map_candidate.token_usage() - unmap_candidate.token_usage()
        if diff < self.resize_trigger_diff_ratio:
            self._reset_unmap_candidate()
            return

        if not map_candidate.can_map() or not unmap_candidate.can_unmap():
            self._reset_unmap_candidate()
            return

        # Keep a pending candidate while it stays cold: its tail may be draining.
        if self.unmap_candidate is None or not self.unmap_candidate.can_unmap():
            self._set_unmap_candidate(unmap_candidate)

        if self.unmap_candidate is None or not self.unmap_candidate.can_do_unmap():
            return

        self._do_resize(map_candidate, self.unmap_candidate)

        self._reset_unmap_candidate()

    def _do_resize(
        self, map_allocator: ElasticAllocator, unmap_allocator: ElasticAllocator
    ) -> None:
        start_time = time.perf_counter()

        unmap_page = unmap_allocator.reduce()
        logger.info(
            f"elasticmem: {unmap_allocator.pool_name} unmap {unmap_page}, "
            f"remain {self.remaining_page} cu_page"
        )

        to_map_page = unmap_page + self.remaining_page
        map_token = map_allocator.cu_page_to_token(to_map_page)
        map_page = 0
        if map_token > 0:
            map_page = map_allocator.expand(map_token)
        self.remaining_page = to_map_page - map_page
        logger.info(
            f"elasticmem: {map_allocator.pool_name} map {map_page}, "
            f"remain {self.remaining_page} cu_page"
        )
        if unmap_page > 0 and self.remaining_page > 0:
            self._return_remaining_pages(unmap_allocator)

        for allocator in self.allocators:
            allocator.update_size()

        logger.debug(
            f"elasticmem: resize took {(time.perf_counter() - start_time) * 1000:.2f} ms"
        )

    def _return_remaining_pages(self, allocator: ElasticAllocator) -> None:
        # Pages too few for the hot pool go back to the cold pool where they fit,
        # instead of staying unmapped where another process could take them.
        return_token = allocator.cu_page_to_token(self.remaining_page)
        if return_token <= 0:
            return
        self.remaining_page -= allocator.expand(return_token)
        logger.info(
            f"elasticmem: {allocator.pool_name} took back pages, "
            f"remain {self.remaining_page} cu_page"
        )
