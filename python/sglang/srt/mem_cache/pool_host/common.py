from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import alloc_mmap
from sglang.srt.runtime_context import get_memory
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

_CUDA_HOST_REGISTERED_RANGES_ATTR = "_sglang_cuda_host_registered_ranges"


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == "cpu", (
            f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        )
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class ShmHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == "cpu", (
            f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        )
        self.dtype = dtype
        self.dims = dims
        from sglang.srt.mem_cache.storage.mmap import alloc_shm

        tensor, fd, mm = alloc_shm(dims, dtype)
        self.fds.append(fd)
        self.mms.append(mm)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


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
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type() -> str:
    """The host-allocator kind the published HiCache configuration asks for."""

    backend = get_memory().hicache_storage_backend
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = get_memory().hicache_storage_backend_extra_config
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _clear_sticky_cuda_error() -> None:
    """Consume the sticky CUDA error left by failed cudart calls.

    A failed cudaHostRegister leaves cudaErrorInvalidValue in the context's
    last-error slot. torch wheels statically link cudart, so the error cannot
    be cleared via ctypes; instead run an unrelated CUDA call so torch's error
    check reads (and clears) it. Without this, the next torch CUDA op raises
    the leftover error and looks like the failing party.
    """
    try:
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
    except Exception:
        pass


def _register_chunk_with_retry(
    cudart, ptr: int, size: int, *, offset: int, total: int
) -> int:
    """Register [ptr, ptr + size), halving the chunk on transient failures.

    Single-shot cudaHostRegister calls of tens of GB were observed to fail
    intermittently with cudaErrorInvalidValue in processes under heavy GPU
    memory usage, while smaller registrations succeed. Returns the chunk size
    that was actually registered. Raises RuntimeError when even a 4 KiB chunk
    cannot be registered.
    """
    orig_size = size
    rc_obj = cudart.cudaHostRegister(ptr, size, 0)
    while int(rc_obj) != 0 and size > 4096:
        size //= 2
        rc_obj = cudart.cudaHostRegister(ptr, size, 0)
    if int(rc_obj) == 0:
        if size < orig_size:
            _clear_sticky_cuda_error()
            logger.warning(
                "cudaHostRegister degraded: %d -> %d bytes at offset=%d "
                "(total=%d, ptr=%#x)",
                orig_size,
                size,
                offset,
                total,
                ptr,
            )
        return size
    _clear_sticky_cuda_error()
    raise RuntimeError(
        f"cudaHostRegister failed for every chunk size down to 4 KiB "
        f"(last rc={int(rc_obj)}, {cudart.cudaGetErrorString(rc_obj)}) "
        f"at offset={offset} size={orig_size} ptr={ptr:#x} (total={total}); "
        f"host buffer is not pinned and device transfers may silently return "
        f"stale data."
    )


def _cuda_host_register(
    buffer: torch.Tensor, registration_granularity_bytes: int | None = None
) -> None:
    # Avoid oversized cudaHostRegister calls on large host pools.
    cudart = torch.cuda.cudart()
    base = buffer.data_ptr()
    total = buffer.numel() * buffer.element_size()
    chunk_limit_bytes = (
        max(envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB.get(), 1) * 1024**3
    )
    # Preserve the legacy single-call behavior unless the caller provides a
    # copy granularity. Splitting an unknown page-first layout at an arbitrary
    # byte offset can make one cudaMemcpyBatchAsync span two registrations.
    chunk_bytes = total
    if registration_granularity_bytes is not None:
        if registration_granularity_bytes <= 0:
            raise ValueError(
                "registration_granularity_bytes must be positive, got "
                f"{registration_granularity_bytes}"
            )
        if registration_granularity_bytes > chunk_limit_bytes:
            raise ValueError(
                "Host registration granularity exceeds the configured chunk limit: "
                f"granularity={registration_granularity_bytes}, "
                f"chunk_limit={chunk_limit_bytes}"
            )
        chunk_bytes = (
            chunk_limit_bytes // registration_granularity_bytes
        ) * registration_granularity_bytes
    # Honor the chunk limit for every buffer, not only for callers that pass
    # a copy granularity; oversized single-shot registrations are unreliable.
    chunk_bytes = min(chunk_bytes, chunk_limit_bytes)
    registered_ranges: list[tuple[int, int]] = []
    try:
        offset = 0
        while offset < total:
            size = min(chunk_bytes, total - offset)
            ptr = base + offset
            size = _register_chunk_with_retry(
                cudart, ptr, size, offset=offset, total=total
            )
            # Once a chunk had to be shrunk, keep the smaller size for the
            # remaining chunks instead of failing over and over again.
            chunk_bytes = min(chunk_bytes, size)
            registered_ranges.append((ptr, size))
            offset += size

        # Keep the exact registration bases alive with the tensor. CUDA requires
        # cudaHostUnregister to receive each base pointer, not just the tensor's
        # original base once after several independent registrations.
        setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, registered_ranges)
    except Exception:
        remaining_ranges = _cuda_host_unregister_ranges(
            cudart, registered_ranges, operation="registration rollback"
        )
        if remaining_ranges:
            setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)
        raise


def _cuda_host_unregister_ranges(
    cudart, registered_ranges: list[tuple[int, int]], *, operation: str
) -> list[tuple[int, int]]:
    failed_ranges = []
    for ptr, size in reversed(registered_ranges):
        rc = int(cudart.cudaHostUnregister(ptr))
        if rc != 0:
            failed_ranges.append((ptr, size))
            logger.warning(
                "cudaHostUnregister failed during %s (rc=%d, %s) for ptr=%#x size=%d",
                operation,
                rc,
                cudart.cudaGetErrorString(rc),
                ptr,
                size,
            )
    failed_ranges.reverse()
    return failed_ranges


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    registered_ranges = getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, None)
    if registered_ranges is None:
        # Compatibility for buffers registered before range metadata was added.
        registered_ranges = [
            (buffer.data_ptr(), buffer.numel() * buffer.element_size())
        ]
    if not registered_ranges:
        return

    remaining_ranges = _cuda_host_unregister_ranges(
        cudart, registered_ranges, operation="host-pool destroy"
    )
    setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer, registration_granularity_bytes)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


@lru_cache(maxsize=1)
def _resolve_device_accessible_ptr_fn():
    try:
        from sgl_kernel.kvcacheio import get_device_accessible_ptr
    except ImportError:
        get_device_accessible_ptr = None
    else:
        if not hasattr(torch.ops.sgl_kernel, "get_device_accessible_ptr"):
            get_device_accessible_ptr = None

    if get_device_accessible_ptr is None:
        # CUDA's UVA makes host and device addresses equal; on HIP they differ.
        if _is_hip:
            raise ImportError(
                "sgl_kernel.kvcacheio.get_device_accessible_ptr is missing from the "
                "installed sglang-kernel. It is required on ROCm, where registered "
                "host memory carries a distinct device address. Rebuild sglang-kernel "
                "from python/sglang/kernels/aot (setup_rocm.py)."
            )
        logger.warning(
            "sgl_kernel.kvcacheio.get_device_accessible_ptr is missing from the "
            "installed sglang-kernel; using raw host addresses for kernel pointer "
            "tables. Build sglang-kernel from python/sglang/kernels/aot to enable it."
        )
    return get_device_accessible_ptr


def make_kernel_ptr_table(
    tensors: list[torch.Tensor],
    target_device: torch.device | str,
    *,
    host_memory_registered: bool,
) -> torch.Tensor:
    device = torch.device(target_device)
    get_device_accessible_ptr = (
        _resolve_device_accessible_ptr_fn()
        if host_memory_registered and device.type == "cuda"
        else None
    )
    if get_device_accessible_ptr is not None:
        if device.index is None:
            device_index = torch.cuda.current_device()
        else:
            device_index = device.index
        pointers = [
            get_device_accessible_ptr(tensor, device_index) for tensor in tensors
        ]
    else:
        pointers = [tensor.data_ptr() for tensor in tensors]
    return torch.tensor(
        pointers,
        dtype=torch.uint64,
        device=device,
    )


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
