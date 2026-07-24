import ctypes
import ctypes.util
import logging
import math
import mmap
import os
import weakref

import psutil
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Load libc once at module level so munmap is callable safely at GC/shutdown time.
# Resolve the SONAME via find_library so the allocator also works on systems
# whose libc is not named "libc.so.6" (e.g. musl / Alpine).
try:
    _libc_name = ctypes.util.find_library("c") or "libc.so.6"
    _libc = ctypes.CDLL(_libc_name, use_errno=True)
    _libc.mmap.restype = ctypes.c_void_p
    _libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    _libc.munmap.restype = ctypes.c_int
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
except OSError:
    _libc = None

# MAP_POPULATE is in Python's mmap module only since 3.11.
_MAP_POPULATE = getattr(mmap, "MAP_POPULATE", 0x08000)
# MAP_HUGETLB and MAP_HUGE_* are Linux-specific and not in Python's mmap module.
_MAP_HUGETLB = 0x40000
_MAP_HUGE_2MB = 21 << 26  # 0x1400000
_MAP_HUGE_1GB = 30 << 26  # 0x78000000
_MAP_FAILED = ctypes.c_void_p(-1).value

MEM_BACKEND_UNKNOWN = 0
MEM_BACKEND_MMAP = 1
MEM_BACKEND_HUGEPAGE = 2

HUGEPAGE_BYTES_2MB = 2 * 1024 * 1024
HUGEPAGE_BYTES_1GB = 1024 * 1024 * 1024

_HUGEPAGE_SIZE_BYTES = {
    "2MB": HUGEPAGE_BYTES_2MB,
    "1GB": HUGEPAGE_BYTES_1GB,
}
_HUGEPAGE_MMAP_FLAGS = {
    HUGEPAGE_BYTES_2MB: _MAP_HUGETLB | _MAP_HUGE_2MB,
    HUGEPAGE_BYTES_1GB: _MAP_HUGETLB | _MAP_HUGE_1GB,
}
_HUGEPAGE_SYSFS_PATHS = {
    HUGEPAGE_BYTES_2MB: "/sys/kernel/mm/hugepages/hugepages-2048kB",
    HUGEPAGE_BYTES_1GB: "/sys/kernel/mm/hugepages/hugepages-1048576kB",
}

HUGEPAGE_MODE_OFF = "off"
HUGEPAGE_MODE_PREFER = "prefer"
HUGEPAGE_MODE_REQUIRED = "required"
_HUGEPAGE_MODES = {
    HUGEPAGE_MODE_OFF,
    HUGEPAGE_MODE_PREFER,
    HUGEPAGE_MODE_REQUIRED,
}

# Host RAM to leave free when sizing HiCache pools (OS, other processes).
# This reserve applies only to normal RAM. Pre-reserved hugetlb pages are
# dedicated to hugepage allocations and must not be reduced by it.
HICACHE_HOST_MEMORY_RESERVE_BYTES = 10 * (1024**3)

_tensor_mem_backend: dict[int, int] = {}


def hugepage_size_requested() -> int:
    configured_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()
    if not configured_size:
        return 0
    hugepage_size = _HUGEPAGE_SIZE_BYTES.get(configured_size)
    if hugepage_size is None:
        logger.warning(
            "Unrecognized SGLANG_HUGEPAGE_SIZE=%r; expected '2MB' or '1GB'.",
            envs.SGLANG_HUGEPAGE_SIZE.get(),
        )
        return 0
    return hugepage_size


def hugepage_mode(hugepage_size: int) -> str:
    default_mode = HUGEPAGE_MODE_PREFER if hugepage_size > 0 else HUGEPAGE_MODE_OFF
    configured_mode = (envs.SGLANG_HUGEPAGE_MODE.get() or "").strip().lower()
    if not configured_mode:
        return default_mode
    if configured_mode not in _HUGEPAGE_MODES:
        logger.warning(
            "Unrecognized SGLANG_HUGEPAGE_MODE=%r; expected off, prefer, or "
            "required. Using default mode %s.",
            configured_mode,
            default_mode,
        )
        return default_mode
    return configured_mode


def hugepage_available_bytes(hugepage_size: int) -> int:
    sysfs_path = _HUGEPAGE_SYSFS_PATHS.get(hugepage_size)
    if sysfs_path is None:
        return 0
    try:
        with open(os.path.join(sysfs_path, "free_hugepages")) as f:
            return int(f.read().strip()) * hugepage_size
    except (OSError, ValueError) as e:
        logger.warning("Failed to read free_hugepages from %s: %s", sysfs_path, e)
        return 0


def memory_available_bytes() -> int:
    """Bytes available for HiCache host pool preflight.

    ``off`` uses normal RAM after keeping a system reserve. ``prefer`` returns
    the larger of usable normal RAM and free hugetlb because either allocation
    path may satisfy the request. ``required`` returns only free hugetlb and
    never counts normal RAM.
    """
    hugepage_size = hugepage_size_requested()
    mode = hugepage_mode(hugepage_size)
    if mode == HUGEPAGE_MODE_REQUIRED:
        return hugepage_available_bytes(hugepage_size)

    normal_available_bytes = max(
        psutil.virtual_memory().available - HICACHE_HOST_MEMORY_RESERVE_BYTES,
        0,
    )
    if mode == HUGEPAGE_MODE_OFF:
        return normal_available_bytes
    if mode == HUGEPAGE_MODE_PREFER:
        return max(
            normal_available_bytes,
            hugepage_available_bytes(hugepage_size),
        )
    raise AssertionError(f"Unexpected hugepage mode: {mode}")


def _tensor_storage_key(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _track_tensor_backend(tensor: torch.Tensor, backend: int) -> torch.Tensor:
    key = _tensor_storage_key(tensor)
    _tensor_mem_backend[key] = backend
    weakref.finalize(tensor, _tensor_mem_backend.pop, key, None)
    return tensor


def tensor_mem_backend(tensor: torch.Tensor) -> int:
    return _tensor_mem_backend.get(_tensor_storage_key(tensor), MEM_BACKEND_UNKNOWN)


def _mmap_page_size_and_flags(mode: str, hugepage_size: int) -> tuple[int, int]:
    if mode == HUGEPAGE_MODE_OFF or hugepage_size == 0:
        return mmap.PAGESIZE, 0
    return hugepage_size, _HUGEPAGE_MMAP_FLAGS[hugepage_size]


def _alloc_hugepage(n_bytes: int, alloc_bytes: int, extra_flags: int) -> ctypes.Array:
    """Call mmap via libc with hugepage flags and return an owning ctypes array.

    munmap fires automatically via weakref.finalize when the array is
    garbage-collected (i.e. when the tensor that wraps it is freed).
    """
    ptr = _libc.mmap(
        None,
        alloc_bytes,
        mmap.PROT_READ | mmap.PROT_WRITE,
        mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | _MAP_POPULATE | extra_flags,
        -1,
        0,
    )
    if ptr is None or ptr == _MAP_FAILED:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    array = (ctypes.c_uint8 * n_bytes).from_address(ptr)
    weakref.finalize(array, _libc.munmap, ctypes.c_void_p(ptr), alloc_bytes)
    return array


def alloc_mmap(dims: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a host tensor via anonymous mmap.

    MAP_SHARED + MAP_POPULATE are both required so cudaHostRegister pins real,
    pre-faulted physical pages (otherwise pinning can race with COW or page
    faults and the device ends up reading stale data).

    ``SGLANG_HUGEPAGE_MODE=prefer`` falls back to normal pages when hugetlb
    allocation fails. ``required`` raises instead of falling back.

    The tensor owns the mapping; munmap fires when the tensor is freed.
    """
    # Re-read per call (not cached) so that envs.SGLANG_HUGEPAGE_SIZE.override()
    # works correctly in tests.
    hugepage_size = hugepage_size_requested()
    mode = hugepage_mode(hugepage_size)
    page_size, extra_flags = _mmap_page_size_and_flags(mode, hugepage_size)
    if mode == HUGEPAGE_MODE_REQUIRED and not extra_flags:
        raise ValueError(
            "SGLANG_HUGEPAGE_MODE=required requires " "SGLANG_HUGEPAGE_SIZE=2MB or 1GB."
        )
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    if extra_flags:
        if _libc is None:
            error_message = (
                "Hugepage mmap requested but libc.so.6 could not be loaded; "
                f"SGLANG_HUGEPAGE_SIZE={envs.SGLANG_HUGEPAGE_SIZE.get()}."
            )
            if mode == HUGEPAGE_MODE_REQUIRED:
                raise RuntimeError(error_message)
            logger.error("%s Falling back to plain mmap.", error_message)
        else:
            try:
                array = _alloc_hugepage(n_bytes, alloc_bytes, extra_flags)
                return _track_tensor_backend(
                    torch.frombuffer(array, dtype=dtype, count=math.prod(dims)).reshape(
                        dims
                    ),
                    MEM_BACKEND_HUGEPAGE,
                )
            except OSError as e:
                error_message = (
                    f"Hugepage mmap via libc failed ({e}); "
                    f"SGLANG_HUGEPAGE_SIZE={envs.SGLANG_HUGEPAGE_SIZE.get()}."
                )
                if mode == HUGEPAGE_MODE_REQUIRED:
                    raise RuntimeError(error_message) from e
                logger.error("%s Falling back to plain mmap.", error_message)
        alloc_bytes = math.ceil(n_bytes / mmap.PAGESIZE) * mmap.PAGESIZE

    # Plain mmap path -- used directly when no hugepages requested, or as fallback.
    # torch.frombuffer keeps a reference to mm inside the tensor storage, so mm
    # stays alive until the tensor is freed and mmap.mmap.__del__ calls munmap.
    mm = mmap.mmap(
        -1,
        alloc_bytes,
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | _MAP_POPULATE,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    return _track_tensor_backend(
        torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims),
        MEM_BACKEND_MMAP,
    )
