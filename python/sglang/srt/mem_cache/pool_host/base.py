from __future__ import annotations

import abc
import logging
import os
import threading
from functools import wraps
from typing import Optional

import psutil
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.pool_host.common import (
    _cuda_host_unregister,
    get_allocator_from_storage,
)
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()

# Host RAM to leave free when sizing HiCache pools (OS, other processes).
HICACHE_HOST_MEMORY_RESERVE_BYTES: int = 10 * (1024**3)


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


def get_local_hicache_process_count() -> int:
    """Best-effort count of ranks sharing the same host memory budget."""
    local_process_count = envs.SGLANG_HICACHE_LOCAL_PROCESS_COUNT.get()
    if local_process_count > 0:
        return local_process_count

    try:
        from sglang.srt.distributed.parallel_state import get_world_group

        local_size = getattr(get_world_group(), "local_size", 0)
        if local_size > 0:
            return local_size
    except Exception:
        pass

    for env_name in ("LOCAL_SIZE", "LOCAL_WORLD_SIZE"):
        try:
            local_size = int(os.environ.get(env_name, "0"))
            if local_size > 0:
                return local_size
        except ValueError:
            pass

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return max(1, torch.distributed.get_world_size())
    return 1


def validate_hicache_host_memory(
    requested_bytes: int,
    *,
    description: str,
) -> tuple[int, int, int]:
    host_mem = psutil.virtual_memory()
    available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
    local_process_count = get_local_hicache_process_count()
    aggregate_requested_bytes = requested_bytes * local_process_count
    if aggregate_requested_bytes > available_bytes:
        raise ValueError(
            f"Not enough host memory available for {description}. "
            f"Requesting {requested_bytes / 1e9:.2f} GB per local process "
            f"across {local_process_count} local processes "
            f"({aggregate_requested_bytes / 1e9:.2f} GB total), but only have "
            f"{available_bytes / 1e9:.2f} GB free. Please reduce the size of "
            f"the hierarchical cache."
        )
    return available_bytes, aggregate_requested_bytes, local_process_count


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.can_use_write_back_jit = False

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        requested_bytes = self.size * self.size_per_token
        (
            _,
            aggregate_requested_bytes,
            local_process_count,
        ) = validate_hicache_host_memory(
            requested_bytes,
            description="hierarchical KV cache",
        )
        logger.info(
            "Allocating %.2f GB host memory for hierarchical KV cache "
            "(%.2f GB aggregate across %d local processes).",
            requested_bytes / 1e9,
            aggregate_requested_bytes / 1e9,
            local_process_count,
        )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    def destroy(self):
        """Unregister pinned host buffers in userspace before process exit.

        Large cudaHostRegister'd buffers are otherwise unpinned by the kernel
        during SIGKILL reclaim, which can stall teardown in uninterruptible
        sleep for tens of seconds. Idempotent. (Only the host_register path
        needs this; npu/musa pin_memory buffers are freed by torch.)
        """
        if getattr(self, "_destroyed", False):
            return
        self._destroyed = True
        buffers = getattr(self, "kv_buffer", None)
        if buffers is not None and self.pin_memory and (_is_cuda or _is_hip):
            if not isinstance(buffers, (list, tuple)):
                buffers = [buffers]
            for buf in buffers:
                if buf is not None:
                    _cuda_host_unregister(buf)
        self.kv_buffer = None

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        """
        Load KV data from the host memory pool to the device memory pool for a specific layer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        raise NotImplementedError()

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """Return True if per-page strides are multiples of *page_size_bytes*.

        Subclasses should override this with a layout-specific stride formula.
        This base implementation logs a warning and returns False (safe default).
        """
        logger.warning(
            "%s does not implement is_stride_page_aligned(); assuming not aligned. "
            "O_DIRECT with a file-based NIXL backend will fall back to copy mode for this pool.",
            type(self).__name__,
        )
        return False

    @synchronized
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices.cpu()])
        return len(indices)
