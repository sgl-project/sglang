"""Prefer 1 GiB hugepages for Qwen4-Exp host slabs, and fall back otherwise.

PLE pinned tables and, once a Qwen4-Exp model has been constructed in this
process, HiCache host pools of at least 1 GiB try the boot-reserved hugetlb
pool. The GPU's NUMA node wins when it has enough free pages; otherwise the
node with the most free pages is used. A missing pool, a short pool, or a
failed mmap/register returns ``None`` so the caller keeps ordinary pinned
memory. This is the opposite of Engram, which fails the launch when its pool
is short.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import weakref

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.dsv41_host_placement import GIB, read_node_hugepages
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import alloc_1g_hugepage

logger = logging.getLogger(__name__)

# cudaHostRegisterMapped. Device kernels (the PLE gather) load through this
# pointer; a default register only pins the pages for cudaMemcpy.
_CUDA_HOST_REGISTER_MAPPED = 0x02

# x86_64. A c_ulong nodemask covers nodes 0..63.
_SYS_GET_MEMPOLICY = 239
_SYS_SET_MEMPOLICY = 238
_MPOL_DEFAULT = 0
_MPOL_BIND = 2
_MAX_NODE = 64

_prefer_qwen = False
# Mappings whose cudaHostUnregister rollback did not finish. Dropping them
# would munmap memory CUDA still has registered.
_retained_after_failed_unregister: list[torch.Tensor] = []


def set_qwen_host_hugepage_preference(enabled: bool) -> None:
    """Record that this process is serving Qwen4-Exp and may use the pool."""
    global _prefer_qwen
    _prefer_qwen = bool(enabled)


def qwen_host_hugepage_preferred() -> bool:
    """HiCache may try hugepages only after Qwen4-Exp construction, and only if enabled."""
    return _prefer_qwen and bool(envs.SGLANG_QWEN_HOST_HUGETLB.get())


def select_1g_hugepage_node(
    nbytes: int,
    *,
    gpu_node: int | None,
    free_by_node: dict[int, int],
) -> int | None:
    """Node whose free 1 GiB pages can hold ``nbytes``, or ``None``.

    Slabs under 1 GiB stay on ordinary pages so a small buffer is not rounded
    up to a whole hugepage. The GPU node is used when it fits. Otherwise the
    fullest node that fits is used.
    """
    if nbytes < GIB:
        return None
    need = (nbytes + GIB - 1) // GIB
    usable = {
        node: free
        for node, free in free_by_node.items()
        if 0 <= node < _MAX_NODE and free >= need
    }
    if gpu_node is not None and gpu_node in usable:
        return gpu_node
    if not usable:
        return None
    return max(usable, key=lambda node: (usable[node], -node))


def free_1g_pages_by_node() -> dict[int, int]:
    """Free 1 GiB hugepages per NUMA node that has the pool configured."""
    base = "/sys/devices/system/node"
    free_by_node: dict[int, int] = {}
    try:
        names = os.listdir(base)
    except OSError:
        return free_by_node
    for name in names:
        if not (name.startswith("node") and name[4:].isdigit()):
            continue
        node = int(name[4:])
        nr, free = read_node_hugepages(node)
        if nr > 0:
            free_by_node[node] = free
    return free_by_node


def try_alloc_pinned_1g_hugepage(
    dims: tuple,
    dtype: torch.dtype,
    *,
    purpose: str,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor | None:
    """A DMA-registered 1 GiB hugepage tensor, or ``None`` to use ordinary pages."""
    if not envs.SGLANG_QWEN_HOST_HUGETLB.get():
        return None
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()
    if n_bytes < GIB:
        return None
    need = (n_bytes + GIB - 1) // GIB
    free_by_node = free_1g_pages_by_node()
    node = select_1g_hugepage_node(
        n_bytes,
        gpu_node=_preferred_gpu_node(),
        free_by_node=free_by_node,
    )
    if node is None:
        if free_by_node:
            logger.info(
                "Qwen %s: %.2f GiB needs %d free 1 GiB hugepages, have %s; "
                "using ordinary pinned memory",
                purpose,
                n_bytes / GIB,
                need,
                free_by_node,
            )
        else:
            logger.info(
                "Qwen %s (%.2f GiB): 1 GiB hugepage pool is not configured; "
                "using ordinary pinned memory",
                purpose,
                n_bytes / GIB,
            )
        return None
    try:
        with _bind_mempolicy(node):
            tensor = alloc_1g_hugepage(dims, dtype)
    except OSError as exc:
        logger.info(
            "Qwen %s: 1 GiB hugepage mmap on NUMA node %d failed (%s); "
            "using ordinary pinned memory",
            purpose,
            node,
            exc,
        )
        return None
    try:
        from sglang.srt.mem_cache.pool_host.common import _cuda_host_register

        # The mapping is rounded up to whole 1 GiB pages. Register that span,
        # not the shorter tensor, or the last page is only partly mapped.
        mapped_bytes = ((n_bytes + GIB - 1) // GIB) * GIB
        _cuda_host_register(
            tensor,
            registration_granularity_bytes,
            flags=_CUDA_HOST_REGISTER_MAPPED,
            nbytes=mapped_bytes,
        )
    except Exception as exc:
        from sglang.srt.mem_cache.pool_host.common import (
            _CUDA_HOST_REGISTERED_RANGES_ATTR,
        )

        ranges = getattr(tensor, _CUDA_HOST_REGISTERED_RANGES_ATTR, None) or []
        if ranges:
            _retained_after_failed_unregister.append(tensor)
            logger.warning(
                "Qwen %s: cudaHostRegister rollback left memory registered; "
                "keeping the hugepage mapping alive",
                purpose,
            )
        else:
            logger.info(
                "Qwen %s: cudaHostRegister of the 1 GiB hugepage mapping failed "
                "(%s); using ordinary pinned memory",
                purpose,
                exc,
            )
        return None
    ranges = getattr(tensor, "_sglang_cuda_host_registered_ranges", None)
    if isinstance(ranges, list):
        weakref.finalize(tensor, _unregister_registered_ranges, ranges)
    tensor._sglang_hugetlb_node = node  # type: ignore[attr-defined]
    logger.info(
        "Qwen %s: %.2f GiB on NUMA node %d via 1 GiB hugepages (%d pages)",
        purpose,
        n_bytes / GIB,
        node,
        need,
    )
    return tensor


def _unregister_registered_ranges(ranges: list) -> None:
    if not ranges:
        return
    snapshot = list(ranges)
    ranges.clear()
    try:
        cudart = torch.cuda.cudart()
    except Exception:
        logger.warning("cudaHostUnregister at hugepage free skipped", exc_info=True)
        return
    from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister_ranges

    _cuda_host_unregister_ranges(cudart, snapshot, operation="hugepage free")


def _preferred_gpu_node() -> int | None:
    try:
        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
    except Exception:
        return None
    try:
        from sglang.srt.utils.numa_utils import _query_numa_node_for_gpu

        nodes = _query_numa_node_for_gpu(device)
    except Exception:
        logger.debug("GPU NUMA node lookup failed", exc_info=True)
        return None
    if not nodes:
        return None
    node = int(nodes[0])
    if node < 0 or node >= _MAX_NODE:
        return None
    return node


class _bind_mempolicy:
    """Bind this thread's mempolicy to ``node`` for one mmap, then restore it.

    1 GiB pages are taken from the policy in force at mmap and cannot be moved
    afterwards. Restoring the previous policy keeps a ``numactl --interleave``
    launch from becoming node-local for every later ordinary allocation.
    """

    def __init__(self, node: int):
        self.node = node
        self._mode: int | None = None
        self._mask: int | None = None
        self._maxnode = max(_MAX_NODE, node + 1)

    def __enter__(self):
        libc = _libc()
        mode = ctypes.c_int()
        mask = ctypes.c_ulong()
        rc = libc.syscall(
            ctypes.c_long(_SYS_GET_MEMPOLICY),
            ctypes.byref(mode),
            ctypes.byref(mask),
            ctypes.c_ulong(self._maxnode),
            ctypes.c_void_p(0),
            ctypes.c_ulong(0),
        )
        if rc == 0:
            self._mode = int(mode.value)
            self._mask = int(mask.value)
        nodemask = ctypes.c_ulong(1 << self.node)
        rc = libc.syscall(
            ctypes.c_long(_SYS_SET_MEMPOLICY),
            ctypes.c_int(_MPOL_BIND),
            ctypes.byref(nodemask),
            ctypes.c_ulong(self._maxnode),
        )
        if rc != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        return self

    def __exit__(self, *exc):
        libc = _libc()
        if self._mode is None or self._mode == _MPOL_DEFAULT:
            libc.syscall(
                ctypes.c_long(_SYS_SET_MEMPOLICY),
                ctypes.c_int(_MPOL_DEFAULT),
                ctypes.c_void_p(0),
                ctypes.c_ulong(0),
            )
        else:
            mask = ctypes.c_ulong(self._mask or 0)
            libc.syscall(
                ctypes.c_long(_SYS_SET_MEMPOLICY),
                ctypes.c_int(self._mode),
                ctypes.byref(mask),
                ctypes.c_ulong(self._maxnode),
            )
        return False


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc
