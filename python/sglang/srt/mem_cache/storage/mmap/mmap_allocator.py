import ctypes
import ctypes.util
import functools
import logging
import math
import mmap
import os
import uuid
import weakref

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
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
_PROT_RW = mmap.PROT_READ | mmap.PROT_WRITE

# Hugetlb pools as the kernel exposes them: one sysfs directory per page size.
_HUGEPAGE_SYSFS_DIR = "/sys/kernel/mm/hugepages"
_HUGEPAGE_SIZES = {"2MB": 2 * 1024 * 1024, "1GB": 1024 * 1024 * 1024}
_HUGEPAGE_MMAP_FLAGS = {
    2 * 1024 * 1024: _MAP_HUGETLB | _MAP_HUGE_2MB,
    1024 * 1024 * 1024: _MAP_HUGETLB | _MAP_HUGE_1GB,
}

HUGEPAGE_MODE_OFF = "off"
HUGEPAGE_MODE_PREFER = "prefer"
HUGEPAGE_MODE_REQUIRED = "required"
_HUGEPAGE_MODES = {
    HUGEPAGE_MODE_OFF,
    HUGEPAGE_MODE_PREFER,
    HUGEPAGE_MODE_REQUIRED,
}


def hugepage_size_requested() -> int:
    """Hugepage size in bytes that SGLANG_HUGEPAGE_SIZE asks alloc_mmap() for.

    Return 0 when the variable is unset or unrecognized (the latter with a
    warning). Re-read per call, not cached, so that
    envs.SGLANG_HUGEPAGE_SIZE.override() works in tests.
    """
    raw = envs.SGLANG_HUGEPAGE_SIZE.get() or ""
    key = raw.strip().upper()
    if key == "":
        return 0
    size = _HUGEPAGE_SIZES.get(key)
    if size is None:
        logger.warning(
            "Unrecognized SGLANG_HUGEPAGE_SIZE=%r; expected '2MB' or '1GB'. "
            "Treating it as unset.",
            raw,
        )
        return 0
    return size


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


def _hugetlb_count(pool_dir: str, counter: str) -> int:
    with open(os.path.join(pool_dir, counter)) as f:
        return int(f.read().strip())


def hugetlb_pool_free_bytes() -> int:
    """Bytes a new alloc_mmap() mapping could take from the hugetlb pool, else 0.

    That is the pool of the size SGLANG_HUGEPAGE_SIZE names, provided the
    selected mode enables hugepages and libc is loadable. Read from sysfs,
    which reports every pool size (the whole hugetlb pool is excluded from
    MemAvailable). Pages a mapping has reserved but not yet faulted in still
    count as free, so only ``free - resv`` can back a new mapping.
    """
    size = hugepage_size_requested()
    if hugepage_mode(size) == HUGEPAGE_MODE_OFF or size == 0 or _libc is None:
        return 0
    pool_dir = os.path.join(_HUGEPAGE_SYSFS_DIR, f"hugepages-{size // 1024}kB")
    try:
        free = _hugetlb_count(pool_dir, "free_hugepages")
        resv = _hugetlb_count(pool_dir, "resv_hugepages")
    except (OSError, ValueError) as e:
        logger.warning(
            "Cannot read the hugetlb pool at %s (%s); not crediting it.", pool_dir, e
        )
        return 0
    return max(free - resv, 0) * size


def _mmap_page_size_and_flags(mode: str, hugepage_size: int) -> tuple[int, int]:
    if mode == HUGEPAGE_MODE_OFF or hugepage_size == 0:
        return mmap.PAGESIZE, 0
    return hugepage_size, _HUGEPAGE_MMAP_FLAGS[hugepage_size]


@functools.cache
def _has_madv_populate_write() -> bool:
    """Whether this kernel implements MADV_POPULATE_WRITE (Linux 5.14+).

    Probed once on a single page. Probing on the real allocation is not an
    option: the answer decides how that allocation gets pre-faulted.
    """
    try:
        probe = mmap.mmap(
            -1,
            mmap.PAGESIZE,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            prot=_PROT_RW,
        )
    except OSError:
        return False
    try:
        probe.madvise(_MADV_POPULATE_WRITE)
        return True
    except (OSError, ValueError):
        return False
    finally:
        probe.close()


def _mmap_prefaulted(fileno: int, alloc_bytes: int, flags: int) -> mmap.mmap:
    """mmap `alloc_bytes` with every page already faulted in and writable.

    cudaHostRegister has to pin real, pre-faulted pages, so these mappings can
    never be handed back lazily. MAP_POPULATE and MADV_POPULATE_WRITE each give
    that guarantee on their own, but asking for both makes the kernel walk the
    whole mapping twice. Prefer the madvise, which additionally reports a
    failure (e.g. ENOMEM) instead of leaving pages quietly unpopulated, and fall
    back to MAP_POPULATE only where the kernel lacks it.
    """
    if _has_madv_populate_write():
        mm = mmap.mmap(fileno, alloc_bytes, flags=flags, prot=_PROT_RW)
        mm.madvise(_MADV_POPULATE_WRITE)
        return mm
    return mmap.mmap(fileno, alloc_bytes, flags=flags | _MAP_POPULATE, prot=_PROT_RW)


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
    hugepage_size = hugepage_size_requested()
    mode = hugepage_mode(hugepage_size)
    page_size, extra_flags = _mmap_page_size_and_flags(mode, hugepage_size)
    if mode == HUGEPAGE_MODE_REQUIRED and not extra_flags:
        raise ValueError(
            "SGLANG_HUGEPAGE_MODE=required requires SGLANG_HUGEPAGE_SIZE=2MB or 1GB."
        )
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    if extra_flags:
        if _libc is None:
            error_message = (
                "Hugepage mmap requested but the C library could not be loaded; "
                f"SGLANG_HUGEPAGE_SIZE={envs.SGLANG_HUGEPAGE_SIZE.get()}."
            )
            if mode == HUGEPAGE_MODE_REQUIRED:
                raise RuntimeError(error_message)
            logger.error("%s Falling back to plain mmap.", error_message)
        else:
            try:
                array = _alloc_hugepage(n_bytes, alloc_bytes, extra_flags)
                return torch.frombuffer(
                    array, dtype=dtype, count=math.prod(dims)
                ).reshape(dims)
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
    mm = _mmap_prefaulted(-1, alloc_bytes, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
    return torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)


def alloc_shm(dims: tuple, dtype: torch.dtype) -> tuple[torch.Tensor, int, mmap.mmap]:
    """Allocate a host tensor via shared memory (/dev/shm).

    Returns a tuple of (tensor, fd, mm).
    The caller is responsible for keeping the fd open if they need to share it,
    and closing it when they are done.
    """
    hugepage_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    # Note: hugepages are not directly supported with /dev/shm mmap files
    # without mounting hugetlbfs there, so we fall back to plain page size.
    if hugepage_size:
        logger.warning(
            "Hugepages are not supported with SHM allocator. "
            "Falling back to plain page-size mmap."
        )

    page_size = mmap.PAGESIZE
    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    # Create an anonymous shared memory file descriptor via memfd_create
    fd = None
    try:
        # MFD_CLOEXEC is standard on Linux 3.17+
        fd = os.memfd_create(
            f"sglang_host_pool_{uuid.uuid4().hex}",
            flags=getattr(os, "MFD_CLOEXEC", 1),
        )
    except (AttributeError, OSError):
        # Fallback to creating a file in /dev/shm if memfd_create is not supported
        shm_path = f"/dev/shm/sglang_host_pool_{uuid.uuid4().hex}.mmap"
        try:
            fd = os.open(shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            try:
                os.unlink(shm_path)
            except OSError:
                pass
        except Exception as e:
            raise OSError(f"Failed to create shm file: {e}")

    try:
        os.ftruncate(fd, alloc_bytes)
        mm = _mmap_prefaulted(fd, alloc_bytes, mmap.MAP_SHARED)
    except Exception as e:
        if fd is not None:
            os.close(fd)
        raise e

    tensor = torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)
    return tensor, fd, mm
