from __future__ import annotations

import logging
import math
from collections import defaultdict
from multiprocessing import shared_memory
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.mmap_allocator import alloc_mmap
from sglang.srt.utils.stale_shm_cleanup import make_shm_name

logger = logging.getLogger(__name__)


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class CpSharedHostTensorAllocator(HostTensorAllocator):
    def __init__(self, cpu_group, owner_rank: int = 0, kind: str = "dsa_l2"):
        super().__init__()
        self.cpu_group = cpu_group
        self.owner_rank = owner_rank
        self.kind = kind
        self.rank = torch.distributed.get_rank(group=cpu_group)
        self.group_ranks = torch.distributed.get_process_group_ranks(cpu_group)
        self.is_owner = self.rank == owner_rank
        self._segments = []

    @property
    def shared_group_key(self) -> str:
        return f"{self.kind}:{self.owner_rank}:{','.join(map(str, self.group_ranks))}"

    def log_host_allocation(
        self,
        requested_bytes: int,
        logger: logging.Logger,
        *,
        pool_name: str,
        token_capacity: int,
        page_num: int,
        page_size: int,
    ) -> None:
        action = "create" if self.is_owner else "attach"
        logger.info(
            "DSA CP shared L2 host memory: %s %.2f GB host memory for %s "
            "(cp_rank=%s owner_rank=%s is_owner=%s token_capacity=%s "
            "page_num=%s page_size=%s kind=%s shared_group_key=%s)",
            action,
            requested_bytes / 1e9,
            pool_name,
            self.rank,
            self.owner_rank,
            self.is_owner,
            token_capacity,
            page_num,
            page_size,
            self.kind,
            self.shared_group_key,
        )

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"CpSharedHostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        numel = math.prod(dims)
        nbytes = numel * torch.empty([], dtype=dtype).element_size()
        payload = [None]
        if self.is_owner:
            segment = shared_memory.SharedMemory(
                create=True, size=nbytes, name=make_shm_name(self.kind)
            )
            payload[0] = {"name": segment.name, "nbytes": nbytes}

        torch.distributed.broadcast_object_list(
            payload, src=self.group_ranks[self.owner_rank], group=self.cpu_group
        )

        if not self.is_owner:
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                segment = shared_memory.SharedMemory(name=payload[0]["name"])

        self._segments.append(segment)
        tensor = torch.frombuffer(segment.buf, dtype=dtype, count=numel).reshape(dims)
        action = "create" if self.is_owner else "attach"
        logger.info(
            "DSA CP shared L2 host memory segment: %s shm_name=%s nbytes=%s "
            "data_ptr=%#x (cp_rank=%s owner_rank=%s is_owner=%s kind=%s "
            "shared_group_key=%s)",
            action,
            segment.name,
            payload[0]["nbytes"],
            tensor.data_ptr(),
            self.rank,
            self.owner_rank,
            self.is_owner,
            self.kind,
            self.shared_group_key,
        )
        return tensor

    def destroy(self) -> None:
        for segment in self._segments:
            segment.close()
            if self.is_owner:
                segment.unlink()
        self._segments.clear()


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    else:
        return HostTensorAllocator()


def _cuda_host_register(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    n_bytes = buffer.numel() * buffer.element_size()
    rc = cudart.cudaHostRegister(buffer.data_ptr(), n_bytes, 0)
    if int(rc) != 0:
        raise RuntimeError(
            f"cudaHostRegister failed (rc={int(rc)}, "
            f"{cudart.cudaGetErrorString(rc)}) for ptr={buffer.data_ptr():#x} "
            f"size={n_bytes}; host buffer is not pinned and device transfers "
            f"may silently return stale data."
        )


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostUnregister(buffer.data_ptr())
    if int(rc) != 0:
        # Best-effort on shutdown: warn, don't raise -- a leak is reclaimed at exit.
        logger.warning(
            "cudaHostUnregister failed (rc=%d, %s) for ptr=%#x",
            int(rc),
            cudart.cudaGetErrorString(rc),
            buffer.data_ptr(),
        )


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
