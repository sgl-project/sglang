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

_HUGEPAGE_SYSFS_PATH = (
    "/sys/kernel/mm/hugepages/hugepages-2048kB",
    "/sys/kernel/mm/hugepages/hugepages-1048576kB",
)

_tensor_mem_backend: dict[int, int] = {}


def hugepage_size_requested() -> int:
    hugepage_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()
    if hugepage_size == "2MB":
        return HUGEPAGE_BYTES_2MB
    elif hugepage_size == "1GB":
        return HUGEPAGE_BYTES_1GB
    else:
        return 0


def hugepage_available_bytes(hugepage_size: int) -> int:
    if hugepage_size == HUGEPAGE_BYTES_2MB:
        sysfs_path = _HUGEPAGE_SYSFS_PATH[0]
    elif hugepage_size == HUGEPAGE_BYTES_1GB:
        sysfs_path = _HUGEPAGE_SYSFS_PATH[1]
    else:
        return 0
    try:
        with open(os.path.join(sysfs_path, "free_hugepages")) as f:
            return int(f.read().strip()) * hugepage_size
    except (OSError, ValueError) as e:
        logger.warning("Failed to read free_hugepages from %s: %s", sysfs_path, e)
        return 0


def memory_available_bytes() -> int:
    """Bytes available for HiCache host pool preflight.

    Without ``SGLANG_HUGEPAGE_SIZE``, uses free host RAM. With hugepages requested,
    uses ``max(RAM, free hugetlb)`` so preflight can succeed when normal RAM is
    low but the reserved hugetlb pool is large (``alloc_mmap`` still may fall
    back to normal pages if the pool is exhausted at allocation time).
    """
    available_bytes = psutil.virtual_memory().available
    hugepage_size = hugepage_size_requested()
    if hugepage_size > 0:
        available_bytes = max(available_bytes, hugepage_available_bytes(hugepage_size))
    return available_bytes


def _tensor_storage_key(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _track_tensor_backend(tensor: torch.Tensor, backend: int) -> torch.Tensor:
    key = _tensor_storage_key(tensor)
    _tensor_mem_backend[key] = backend
    weakref.finalize(tensor, _tensor_mem_backend.pop, key, None)
    return tensor


def tensor_mem_backend(tensor: torch.Tensor) -> int:
    return _tensor_mem_backend.get(_tensor_storage_key(tensor), MEM_BACKEND_UNKNOWN)


def _mmap_page_size_and_flags() -> tuple[int, int]:
    hugepage_bytes = hugepage_size_requested()
    if hugepage_bytes == HUGEPAGE_BYTES_2MB:
        return hugepage_bytes, _MAP_HUGETLB | _MAP_HUGE_2MB
    elif hugepage_bytes == HUGEPAGE_BYTES_1GB:
        return hugepage_bytes, _MAP_HUGETLB | _MAP_HUGE_1GB
    elif (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip():
        logger.warning(
            "Unrecognized SGLANG_HUGEPAGE_SIZE=%r; expected '2MB' or '1GB'. "
            "Falling back to plain page-size mmap.",
            envs.SGLANG_HUGEPAGE_SIZE.get(),
        )
    return mmap.PAGESIZE, 0


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
    """Allocate a host tensor via anonymous mmap. Set SGLANG_HUGEPAGE_SIZE=2MB or 1GB for hugepages.

    MAP_SHARED + MAP_POPULATE are both required so cudaHostRegister pins real,
    pre-faulted physical pages (otherwise pinning can race with COW or page
    faults and the device ends up reading stale data).

    The tensor owns the mapping; munmap fires when the tensor is freed.
    """
    # Re-read per call (not cached) so that envs.SGLANG_HUGEPAGE_SIZE.override()
    # works correctly in tests.
    page_size, extra_flags = _mmap_page_size_and_flags()
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    if extra_flags:
        if _libc is None:
            logger.error(
                "Hugepage mmap requested but libc.so.6 could not be loaded; "
                "falling back to plain mmap. SGLANG_HUGEPAGE_SIZE=%s will be ignored.",
                envs.SGLANG_HUGEPAGE_SIZE.get(),
            )
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
                logger.error(
                    "Hugepage mmap via libc failed (%s); falling back to plain mmap. "
                    "SGLANG_HUGEPAGE_SIZE=%s will be ignored.",
                    e,
                    envs.SGLANG_HUGEPAGE_SIZE.get(),
                )
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
