from __future__ import annotations

import abc
import logging
import threading
from functools import wraps
from typing import Optional

import psutil
import torch

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


def sync_fixed_hicache_size(size: int, host_size: int) -> int:
    """Sync fixed-size HiCache token capacity across PP ranks.

    A fixed --hicache-size is specified in GB, but each PP stage may have a
    different bytes/token because it owns different layers. Use the global
    minimum token capacity within the PP group so all stages expose the same
    host-cache capacity.
    Ratio-based sizing already derives from the synced device pool size.
    """
    if host_size <= 0 or not torch.distributed.is_available():
        return size

    if not torch.distributed.is_initialized():
        return size

    try:
        from sglang.srt.distributed.parallel_state import get_pp_group

        pp_group = get_pp_group()
    except AssertionError:
        return size

    if pp_group.world_size <= 1:
        return size

    tensor = torch.tensor(size, dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.MIN,
        group=pp_group.cpu_group,
    )
    synced_size = int(tensor.item())

    if synced_size != size:
        logger.info(
            "Sync fixed-size HiCache host token capacity from %d to %d.",
            size,
            synced_size,
        )
    return synced_size


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


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
            self.size = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.size_per_token), host_size
            )
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        if self.size <= device_pool.size:
            logger.warning(
                "HiCache host KV pool (%d tokens) is smaller than the device pool (%d tokens);"
                "L2 cache effectiveness is reduced."
                "Consider increasing --hicache-ratio (or --hicache-size) for higher L2 cache hit rate.",
                self.size,
                device_pool.size,
            )

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
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
        Get a dummy flat data page shaped like one host page, for use as a scratch
        read buffer (e.g. non-zero-copy prefetch). The contents are uninitialized:
        callers must fully overwrite the buffer before reading from it.
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
        # Per-slot flag used to detect double-free.
        # slot_used[k] is true if slot k is allocated.
        self.slot_used = torch.zeros(self.size, dtype=torch.bool)

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

        assert not self.slot_used[select_index].any(), (
            f"Double-alloc detected: slots already allocated: "
            f"{select_index[self.slot_used[select_index]].tolist()}."
        )
        self.slot_used[select_index] = True

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.cpu()
        assert self.slot_used[indices_cpu].all(), (
            f"Double-free detected: slots not currently allocated: "
            f"{indices_cpu[~self.slot_used[indices_cpu]].tolist()}."
        )
        self.slot_used[indices_cpu] = False
        self.free_slots = torch.cat([self.free_slots, indices_cpu])
        return len(indices)
