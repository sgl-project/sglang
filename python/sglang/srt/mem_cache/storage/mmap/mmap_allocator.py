import ctypes
import ctypes.util
import functools
import logging
import math
import mmap
import os
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor

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
    _libc.madvise.restype = ctypes.c_int
    _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _libc.memset.restype = ctypes.c_void_p
    _libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
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
_MADV_DONTFORK = getattr(mmap, "MADV_DONTFORK", 10)
_MADV_HUGEPAGE = getattr(mmap, "MADV_HUGEPAGE", 14)
_PROT_RW = mmap.PROT_READ | mmap.PROT_WRITE
# Prefault chunks are aligned to the THP size so no 2 MB page straddles two
# threads (a straddled page is still correct, just faulted as small pages).
_THP_SIZE = 2 * 1024 * 1024


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


def _mmap_private_parallel_prefaulted(alloc_bytes: int, num_threads: int) -> mmap.mmap:
    """Private anonymous mapping, THP-backed, prefaulted by `num_threads` threads.

    The default MAP_SHARED|MAP_ANONYMOUS mapping is shmem: every fault takes
    the backing shmem file's locks, so prefaulting runs at ~2 GB/s however many
    threads touch it, and shmem THP is off by default (shmem_enabled=never).
    For a multi-hundred-GB HiCache pool that is minutes of startup per rank.

    A private anonymous mapping faults without that contention and picks up
    2 MB transparent hugepages under the common enabled=madvise policy, so the
    prefault scales with threads (measured on B300: 80 GB/s with 32 threads
    vs 2.5 GB/s for the shared mapping).

    Why MAP_SHARED was used: a MAP_PRIVATE page goes copy-on-write in a forked
    child, and a parent write after fork would move the parent off the page
    cudaHostRegister pinned. MADV_DONTFORK removes the mapping from any child
    altogether, so there is no COW to race with -- the same thing the CUDA
    driver does for cudaHostAlloc memory. Consequence: a fork()ed child (not a
    spawned one) cannot see this buffer.

    Pages are prefaulted before return, as cudaHostRegister requires.
    """
    if _libc is None:
        raise OSError("libc unavailable")
    mm = mmap.mmap(
        -1, alloc_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=_PROT_RW
    )
    base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    if _libc.madvise(base, alloc_bytes, _MADV_DONTFORK) != 0:
        errno = ctypes.get_errno()
        mm.close()
        raise OSError(errno, "madvise(MADV_DONTFORK): " + os.strerror(errno))
    # Best effort: THP may be disabled on this host (enabled=never); the
    # mapping is still correct with 4 KB pages, just slower to fault.
    _libc.madvise(base, alloc_bytes, _MADV_HUGEPAGE)

    chunk = max(_THP_SIZE, -(-alloc_bytes // max(num_threads, 1)))
    chunk = -(-chunk // _THP_SIZE) * _THP_SIZE
    ranges = [
        (off, min(chunk, alloc_bytes - off)) for off in range(0, alloc_bytes, chunk)
    ]

    def _fault(rng):
        off, size = rng
        # Touch, don't MADV_POPULATE_WRITE: per-chunk populate measured 2.5-10
        # GB/s on B300 vs 12+ GB/s for memset with the same 32 threads. ctypes
        # releases the GIL around foreign calls, so the threads really fault in
        # parallel. Writing zeros over zero-filled pages is harmless.
        _libc.memset(base + off, 0, size)

    try:
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            list(pool.map(_fault, ranges))
    except BaseException:
        mm.close()
        raise
    return mm


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

    SGLANG_HICACHE_HOST_PARALLEL_ALLOC=1 swaps the plain (non-hugetlb) path for
    a private, MADV_DONTFORK, THP-backed mapping prefaulted by
    SGLANG_HICACHE_HOST_ALLOC_THREADS threads -- see
    _mmap_private_parallel_prefaulted for why that is equally safe to pin and
    an order of magnitude faster to fault. Any failure falls back to the
    shared path.

    The tensor owns the mapping; munmap fires when the tensor is freed.
    """
    # Re-read per call (not cached) so that envs.SGLANG_HUGEPAGE_SIZE.override()
    # works correctly in tests.
    hugepage_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    if hugepage_size == "":
        page_size, extra_flags = mmap.PAGESIZE, 0
    elif hugepage_size == "2MB":
        page_size, extra_flags = 2 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_2MB
    elif hugepage_size == "1GB":
        page_size, extra_flags = 1024 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_1GB
    else:
        logger.warning(
            "Unrecognized SGLANG_HUGEPAGE_SIZE=%r; expected '2MB' or '1GB'. "
            "Falling back to plain page-size mmap.",
            envs.SGLANG_HUGEPAGE_SIZE.get(),
        )
        page_size, extra_flags = mmap.PAGESIZE, 0

    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    if extra_flags:
        if _libc is None:
            logger.error(
                "Hugepage mmap requested but libc.so.6 could not be loaded; "
                "falling back to plain mmap. SGLANG_HUGEPAGE_SIZE=%s will be ignored.",
                hugepage_size,
            )
        else:
            try:
                array = _alloc_hugepage(n_bytes, alloc_bytes, extra_flags)
                return torch.frombuffer(
                    array, dtype=dtype, count=math.prod(dims)
                ).reshape(dims)
            except OSError as e:
                logger.error(
                    "Hugepage mmap via libc failed (%s); falling back to plain mmap. "
                    "SGLANG_HUGEPAGE_SIZE=%s will be ignored.",
                    e,
                    hugepage_size,
                )
        alloc_bytes = math.ceil(n_bytes / mmap.PAGESIZE) * mmap.PAGESIZE

    # Plain mmap path -- used directly when no hugepages requested, or as fallback.
    # torch.frombuffer keeps a reference to mm inside the tensor storage, so mm
    # stays alive until the tensor is freed and mmap.mmap.__del__ calls munmap.
    if envs.SGLANG_HICACHE_HOST_PARALLEL_ALLOC.get():
        try:
            mm = _mmap_private_parallel_prefaulted(
                alloc_bytes, envs.SGLANG_HICACHE_HOST_ALLOC_THREADS.get()
            )
            return torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(
                dims
            )
        except OSError as e:
            logger.error(
                "Parallel private host allocation failed (%s); falling back to the "
                "shared prefaulted mmap.",
                e,
            )
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
    if hugepage_size != "":
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
