from __future__ import annotations

import ctypes
import logging
import math
import os
import weakref
from functools import lru_cache

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _hip_runtime() -> ctypes.CDLL:
    from sglang.srt.distributed.device_communicators.cuda_wrapper import (
        find_loaded_library,
    )

    # Bind the copy torch loaded; dlopen by soname could load a second HIP runtime.
    path = find_loaded_library("libamdhip64")
    if path is None:
        raise RuntimeError("libamdhip64 is not loaded in this process")
    lib = ctypes.CDLL(path)
    lib.hipHostMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    lib.hipHostMalloc.restype = ctypes.c_int
    lib.hipHostFree.argtypes = [ctypes.c_void_p]
    lib.hipHostFree.restype = ctypes.c_int
    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    lib.hipGetErrorString.restype = ctypes.c_char_p
    return lib


def _hip_host_malloc(nbytes: int) -> int:
    lib = _hip_runtime()
    ptr = ctypes.c_void_p()
    # hipHostMallocDefault, the flags torch's pinned allocator uses.
    rc = lib.hipHostMalloc(ctypes.byref(ptr), nbytes, 0)
    if rc != 0:
        raise RuntimeError(
            f"hipHostMalloc of {nbytes / 1e9:.2f} GB for a HiCache host pool failed "
            f"(rc={rc}, {lib.hipGetErrorString(rc).decode()}). With "
            "HSA_USERPTR_FOR_PAGED_MEM=0 host pools are GTT memory, which all ranks "
            "on a node share up to /sys/class/drm/card*/device/mem_info_gtt_total. "
            "Reduce --hicache-size or unset HSA_USERPTR_FOR_PAGED_MEM."
        )
    return ptr.value


def _hip_host_free(ptr: int) -> None:
    rc = _hip_runtime().hipHostFree(ptr)
    if rc != 0:
        logger.warning("hipHostFree failed (rc=%d) for ptr=%#x", rc, ptr)


@lru_cache(maxsize=1)
def _log_hip_owned_host_pool() -> None:
    logger.info(
        "HSA_USERPTR_FOR_PAGED_MEM=0: allocating HiCache host pools with "
        "hipHostMalloc (GTT) instead of hipHostRegister (USERPTR)."
    )
    if envs.SGLANG_HUGEPAGE_SIZE.get():
        logger.warning(
            "SGLANG_HUGEPAGE_SIZE is ignored for hipHostMalloc HiCache host pools."
        )


def _uses_hip_owned_host_memory(
    *, pin_memory: bool, is_default_allocator: bool
) -> bool:
    # hipHostRegister memory is USERPTR even with HSA_USERPTR_FOR_PAGED_MEM=0; any page
    # migration in it stalls every GPU queue while KFD revalidates. GTT never migrates.
    return (
        pin_memory
        # Storage allocators own their backing (shm fds, registered transfer buffers).
        and is_default_allocator
        and os.environ.get("HSA_USERPTR_FOR_PAGED_MEM") == "0"
    )


def _alloc_hip_host_tensor(dims: tuple, dtype: torch.dtype) -> torch.Tensor:
    # Not torch pin_memory: its host allocator rounds each block up to a power of
    # two, so a 180 GB pool would pin 256 GiB of GTT.
    numel = math.prod(dims)
    nbytes = numel * dtype.itemsize
    ptr = _hip_host_malloc(nbytes)
    try:
        array = (ctypes.c_uint8 * nbytes).from_address(ptr)
        buffer = torch.frombuffer(array, dtype=dtype, count=numel).reshape(dims)
    except BaseException:
        _hip_host_free(ptr)
        raise
    # Freed with the tensor's last view; at interpreter exit the driver reclaims it.
    weakref.finalize(array, _hip_host_free, ptr).atexit = False
    return buffer


def maybe_alloc_hip_owned_host_tensor(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    *,
    pin_memory: bool,
    is_default_allocator: bool,
) -> torch.Tensor | None:
    if not _uses_hip_owned_host_memory(
        pin_memory=pin_memory, is_default_allocator=is_default_allocator
    ):
        return None

    assert device == "cpu", f"HiCache host pools are CPU memory; got {device!r}"
    _log_hip_owned_host_pool()
    return _alloc_hip_host_tensor(dims=dims, dtype=dtype)
