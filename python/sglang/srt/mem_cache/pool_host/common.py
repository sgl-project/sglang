from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import torch

from sglang.srt.mem_cache.storage.mmap import alloc_mmap

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
        assert (
            device == "cpu"
        ), f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
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


def get_allocator_type(server_args) -> str:
    backend = getattr(server_args, "hicache_storage_backend", None)
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = getattr(
            server_args, "hicache_storage_backend_extra_config", None
        )
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_host_register(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    n_bytes = buffer.numel() * buffer.element_size()
    rc = cudart.cudaHostRegister(buffer.data_ptr(), n_bytes, 2)  # _L1_DEVPTR Mapped
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


_HIP_LIB = None
def cuda_host_get_device_pointer(buffer):
    """_L1_DEVPTR: device-mapped addr for a Mapped-registered host tensor, via
    ctypes into HIP (torch cudart lacks this on ROCm)."""
    import ctypes, torch
    global _HIP_LIB
    if _HIP_LIB is None:
        for nm in ("libamdhip64.so","libamdhip64.so.7","libamdhip64.so.6"):
            try:
                _HIP_LIB=ctypes.CDLL(nm); break
            except OSError: pass
    if _HIP_LIB is None:
        raise RuntimeError(
            "cuda_host_get_device_pointer: libamdhip64 not loadable; cannot obtain a "
            "device alias for host-registered memory. Falling back to the host VA would "
            "hand an unmapped address to a GPU kernel and fault later, far from here."
        )
    dptr = ctypes.c_void_p()
    rc = _HIP_LIB.hipHostGetDevicePointer(
        ctypes.byref(dptr), ctypes.c_void_p(int(buffer.data_ptr())), ctypes.c_uint(0)
    )
    if int(rc) != 0 or not dptr.value:
        raise RuntimeError(
            f"hipHostGetDevicePointer failed (rc={int(rc)}) for host buffer "
            f"{hex(buffer.data_ptr())}. The host VA is not a valid device address on "
            "this kernel, so the transfer kernels cannot use it."
        )
    return int(dptr.value)


class _CudaAliasView:
    """Minimal __cuda_array_interface__ provider so torch can wrap a raw device address."""

    def __init__(self, dev_ptr: int, shape, typestr: str):
        self.__cuda_array_interface__ = {
            "data": (int(dev_ptr), False),
            "shape": tuple(shape),
            "typestr": typestr,
            "version": 3,
            "strides": None,
        }


_ALIAS_TYPESTR = {
    torch.uint8: "|u1",
    torch.int8: "|i1",
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.int32: "<i4",
    torch.int64: "<i8",
}


def device_alias_view(dev_ptr: int, like: torch.Tensor, gpu_device) -> torch.Tensor:
    """Wrap a device-mapped address as a CUDA tensor shaped/typed like ``like``.

    Host-registered pool memory is reachable from the GPU only through its device
    alias. Wrapping that alias once lets the fast per-layer transfer kernels keep
    taking a plain tensor argument, instead of switching to the pointer-array
    (all-layer) kernels which are far slower when driven one layer at a time.
    """
    typestr = _ALIAS_TYPESTR.get(like.dtype)
    if typestr is not None:
        return torch.as_tensor(
            _CudaAliasView(dev_ptr, tuple(like.shape), typestr), device=gpu_device
        )
    n_bytes = like.numel() * like.element_size()
    raw = torch.as_tensor(
        _CudaAliasView(dev_ptr, (n_bytes,), "|u1"), device=gpu_device
    )
    return raw.view(like.dtype).reshape(like.shape)


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
