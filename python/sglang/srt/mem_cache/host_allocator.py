from __future__ import annotations

import abc
import logging
from collections import defaultdict
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


class HostTensorAllocator(abc.ABC):
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Allocate a tensor of given dims and dtype on the memory."""
        self.dtype = dtype
        self.dims = dims
        tensor = torch.empty(dims, dtype=dtype, device=device)
        return tensor


@lru_cache(maxsize=1)
def get_mooncake_host_tensor_allocator_cls():
    try:
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeHostTensorAllocator,
        )
    except ImportError:
        return None

    return MooncakeHostTensorAllocator


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        mooncake_allocator_cls = get_mooncake_host_tensor_allocator_cls()
        if mooncake_allocator_cls is not None:
            return mooncake_allocator_cls()

        logger.warning(
            "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
            "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
            "Fallback to use default allocator."
        )
        return HostTensorAllocator()
    else:
        return HostTensorAllocator()


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
        torch.cuda.cudart().cudaHostRegister(
            buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
        )
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


def alloc_with_host_register_npu(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor for NPU devices.

    For MooncakeHostTensorAllocator, memory is allocated directly from Mooncake's
    managed pool and should not be pinned again (would break zero-copy semantics).

    For standard allocators, use PyTorch's pin_memory flag for efficient NPU-CPU transfers.
    """
    mooncake_allocator_cls = get_mooncake_host_tensor_allocator_cls()
    if mooncake_allocator_cls is not None and isinstance(
        allocator, mooncake_allocator_cls
    ):
        # Mooncake standalone storage requires buffers to remain in Mooncake's
        # shared-memory pool so register_buffer() can preserve zero-copy access.
        return allocator.allocate(dims, dtype=dtype, device=device)

    # Standard NPU allocation with pin_memory
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_host_register_npu,
        "musa": alloc_with_pin_memory,
    },
)
