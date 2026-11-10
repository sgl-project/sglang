"""NUMA placement for DSV4.1 Engram host tables and expert-spill pins.

Bind mappings to the GPU-local node (``gpu_numa_node``, or
``SGLANG_DSV41_ENGRAM_NUMA_NODE`` when set).
Engram host tables may consume the boot-reserved 1 GiB hugetlb pool.
Routed-expert spill must not take those pages: anonymous + THP on the
preferred node. Fail loud rather than let Linux silently place pages
off-node.

``PRE_H1_NODE_TOTAL_GIB`` / ``DOCUMENTED_NODE_TOTAL_GIB`` are fixtures for
the V100 budget unit tests and dry-run, not runtime defaults.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import mmap
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
    _MADV_POPULATE_WRITE,
    _has_madv_populate_write,
)

logger = logging.getLogger(__name__)

GIB = 1024**3

# x86_64
_SYS_MBIND = 237
_SYS_SET_MEMPOLICY = 238
_MPOL_DEFAULT = 0
_MPOL_BIND = 2
_MPOL_MF_STRICT = 1
_MPOL_MF_MOVE = 2

# Python's mmap module does not export MAP_HUGETLB on this distro.
MAP_ANONYMOUS = getattr(mmap, "MAP_ANONYMOUS", 0x20)
MAP_POPULATE = 0x8000
MAP_HUGETLB = 0x40000
MAP_HUGE_SHIFT = 26
MAP_HUGE_2MB = 21 << MAP_HUGE_SHIFT
MAP_HUGE_1GB = 30 << MAP_HUGE_SHIFT
MADV_HUGEPAGE = 14
# This venv's Python 3.12 was built without os.memfd_create / os.MFD_*.
MFD_CLOEXEC = 0x0001
MFD_HUGETLB = 0x0004
MFD_HUGE_2MB = 21 << MAP_HUGE_SHIFT
MFD_HUGE_1GB = 30 << MAP_HUGE_SHIFT
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
_libc.memfd_create.restype = ctypes.c_int


def memfd_create(name: str, flags: int) -> int:
    fd = int(_libc.memfd_create(name.encode("utf-8"), flags))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), "memfd_create")
    return fd

# GPU-local node of the measured 8x V100 box; default for the offline dry-run
# planner only. Runtime placement uses gpu_numa_node().
DOCUMENTED_GPU_NUMA_NODE = 1


@functools.cache
def gpu_numa_node() -> int:
    """NUMA node of CUDA device 0 per sysfs, 0 when the host reports none.

    Every TP rank sees the same visible-device list, so all ranks agree.
    """
    props = torch.cuda.get_device_properties(0)
    bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    try:
        with open(f"/sys/bus/pci/devices/{bdf}/numa_node") as f:
            node = int(f.read().strip())
    except (OSError, ValueError):
        logger.warning("DSV4.1 host placement: no NUMA node for GPU %s; using node 0", bdf)
        return 0
    # Single-node hosts report -1.
    return max(node, 0)


def engram_numa_node() -> int:
    """SGLANG_DSV41_ENGRAM_NUMA_NODE when set, else the GPU-local node. <0 disables binding."""
    node = envs.SGLANG_DSV41_ENGRAM_NUMA_NODE.get()
    return gpu_numa_node() if node is None else int(node)

# Fixtures for test/registered/unit/mem_cache/test_dsv41_v100_budget.py.
PRE_H1_NODE_TOTAL_GIB = {0: 177.0, 1: 173.0}
DOCUMENTED_NODE_TOTAL_GIB = {0: 62.3, 1: 279.5}


class EngramNumaError(RuntimeError):
    """Engram host tables would silently cross UPI or do not fit the preferred node."""


@dataclass(frozen=True)
class NumaNodeMem:
    node: int
    total_bytes: int
    free_bytes: int

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def free_gib(self) -> float:
        return self.free_bytes / GIB


@dataclass
class HostMapping:
    name: str
    nbytes: int
    node: int
    layout: str
    rank: int = 0
    layer_id: int = -1
    crosses_upi: bool = False


@dataclass
class HostPlacementPlan:
    mappings: list[HostMapping] = field(default_factory=list)
    huge_pages_total: int = 0
    huge_page_kB: int = 2048
    preferred_node: int = DOCUMENTED_GPU_NUMA_NODE
    allow_split: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def bytes_per_node(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for m in self.mappings:
            out[m.node] = out.get(m.node, 0) + m.nbytes
        return out

    @property
    def crosses_upi(self) -> bool:
        return any(m.crosses_upi for m in self.mappings)


def read_huge_pages() -> tuple[int, int]:
    """Return (HugePages_Total, Hugepagesize_kB) from /proc/meminfo."""
    total, size_kb = 0, 2048
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("HugePages_Total:"):
                    total = int(line.split()[1])
                elif line.startswith("Hugepagesize:"):
                    size_kb = int(line.split()[1])
    except OSError:
        pass
    return total, size_kb


def read_numa_nodes(*, documented_fallback: bool = False) -> dict[int, NumaNodeMem]:
    """MemTotal/MemFree per NUMA node from sysfs.

    Planning uses MemTotal (installed capacity). MemFree on a loaded box is
    not a placement signal. ``documented_fallback`` is test-only; production
    must not invent another machine's NUMA map.
    """
    nodes: dict[int, NumaNodeMem] = {}
    sysfs = "/sys/devices/system/node"
    try:
        names = sorted(
            n for n in os.listdir(sysfs) if re.fullmatch(r"node\d+", n)
        )
    except OSError:
        names = []
    for name in names:
        node = int(name[4:])
        total = free = 0
        try:
            with open(os.path.join(sysfs, name, "meminfo")) as f:
                for line in f:
                    if "MemTotal:" in line:
                        total = int(line.split()[3]) * 1024
                    elif "MemFree:" in line:
                        free = int(line.split()[3]) * 1024
        except OSError:
            continue
        if total:
            nodes[node] = NumaNodeMem(node, total, free)
    if nodes:
        return nodes
    if not documented_fallback:
        raise EngramNumaError(
            "cannot read NUMA node memory from /sys/devices/system/node; "
            "refusing to place Engram (would silently cross UPI)"
        )
    return {
        n: NumaNodeMem(n, int(round(gib * GIB)), int(round(gib * GIB)))
        for n, gib in DOCUMENTED_NODE_TOTAL_GIB.items()
    }


def _fits(node: NumaNodeMem, used: int, nbytes: int) -> bool:
    return used + nbytes <= node.total_bytes


def plan_engram_host_tables(
    *,
    table_nbytes: Iterable[tuple[int, int]],
    layout: str,
    tp_size: int,
    preferred_node: int = DOCUMENTED_GPU_NUMA_NODE,
    allow_split: bool = False,
    extra_per_rank_bytes: int = 0,
    nodes: Optional[dict[int, NumaNodeMem]] = None,
    huge_pages_total: Optional[int] = None,
) -> HostPlacementPlan:
    """Place Engram host tables (and optional per-rank expert-spill pins).

    ``table_nbytes`` is ``(layer_id, full_table_bytes)`` for each Engram layer
    (the unsharded size). ``layout`` is ``shared`` (one copy of each table) or
    ``private`` (one shard per TP rank).
    """
    if layout not in ("shared", "private"):
        raise ValueError(f"engram host layout must be shared or private, got {layout!r}")
    nodes = nodes if nodes is not None else read_numa_nodes()
    if preferred_node not in nodes:
        raise EngramNumaError(
            f"preferred Engram NUMA node {preferred_node} is not visible "
            f"(have {sorted(nodes)}). Set SGLANG_DSV41_ENGRAM_NUMA_NODE to a "
            f"node that exists on this host."
        )
    huge_pages_total, huge_page_kB = (
        (huge_pages_total, 2048)
        if huge_pages_total is not None
        else read_huge_pages()
    )
    plan = HostPlacementPlan(
        huge_pages_total=huge_pages_total,
        huge_page_kB=huge_page_kB,
        preferred_node=preferred_node,
        allow_split=allow_split,
    )
    if huge_pages_total == 0:
        plan.warnings.append(
            "HugePages_Total is 0. Host Engram gathers will run on 4 KiB pages "
            "(high TLB miss tax). Reserve 1G pages at boot if the tables are large."
        )

    used = {n: 0 for n in nodes}
    tables = list(table_nbytes)

    def assign(name, nbytes, node, layout, rank, layer_id) -> None:
        if nbytes <= 0:
            return
        cap = nodes[node]
        if not _fits(cap, used[node], nbytes):
            raise EngramNumaError(
                f"{name} is {nbytes / GIB:.1f} GiB but NUMA node {node} only "
                f"has {cap.total_gib:.1f} GiB total ({used[node] / GIB:.1f} GiB "
                f"already planned). Increase DRAM on node {preferred_node} "
                f"or pass allow_split to place tables across nodes — never "
                f"silently cross sockets."
            )
        used[node] += nbytes
        plan.mappings.append(
            HostMapping(
                name=name,
                nbytes=nbytes,
                node=node,
                layout=layout,
                rank=rank,
                layer_id=layer_id,
                crosses_upi=node != preferred_node,
            )
        )

    if layout == "shared":
        # One copy of each layer's table. Split = one layer per NUMA node,
        # not a silent mix of pages inside one mapping.
        other_nodes = [n for n in sorted(nodes) if n != preferred_node]
        for i, (layer_id, nbytes) in enumerate(tables):
            node = preferred_node
            if not _fits(nodes[preferred_node], used[preferred_node], nbytes):
                if not allow_split:
                    raise EngramNumaError(
                        f"shared Engram layer {layer_id} is {nbytes / GIB:.1f} GiB; "
                        f"NUMA node {preferred_node} is {nodes[preferred_node].total_gib:.1f} GiB "
                        f"with {used[preferred_node] / GIB:.1f} GiB already planned. "
                        f"so a node-local shared table does not fit. "
                        f"Set SGLANG_DSV41_ENGRAM_NUMA_SPLIT=1 to put different "
                        f"layers on different nodes (UPI tax on the off-node table), "
                        f"or keep host-sharded (private) layout."
                    )
                if not other_nodes:
                    raise EngramNumaError("allow_split is set but there is no other NUMA node")
                node = other_nodes[i % len(other_nodes)]
                plan.warnings.append(
                    f"Engram layer {layer_id} ({nbytes / GIB:.1f} GiB) placed on "
                    f"NUMA node {node}, not GPU node {preferred_node}: gathers cross UPI."
                )
            assign(f"engram_layer_{layer_id}", nbytes, node, layout, 0, layer_id)
    else:
        # Host-sharded: each rank maps only its row range. Prefer packing every
        # shard onto the GPU node; overflow ranks go to the other node if split.
        for layer_id, full_bytes in tables:
            shard = full_bytes // tp_size
            rem = full_bytes % tp_size
            for rank in range(tp_size):
                nbytes = shard + (rem if rank == tp_size - 1 else 0)
                node = preferred_node
                if not _fits(nodes[preferred_node], used[preferred_node], nbytes):
                    if not allow_split:
                        raise EngramNumaError(
                            f"private Engram shard rank {rank} layer {layer_id} is "
                            f"{nbytes / GIB:.1f} GiB and does not fit the remaining "
                            f"{(nodes[preferred_node].total_bytes - used[preferred_node]) / GIB:.1f} GiB "
                            f"on NUMA node {preferred_node} (node total "
                            f"{nodes[preferred_node].total_gib:.1f} GiB). "
                            f"Set SGLANG_DSV41_ENGRAM_NUMA_SPLIT=1 to place "
                            f"overflow shards on another node (cross-socket tax)."
                        )
                    overflow = [n for n in sorted(nodes) if n != preferred_node]
                    if not overflow:
                        raise EngramNumaError("allow_split is set but there is no other NUMA node")
                    node = overflow[0]
                    plan.warnings.append(
                        f"Engram private shard rank={rank} layer={layer_id} "
                        f"({nbytes / GIB:.1f} GiB) on NUMA node {node}, not GPU "
                        f"node {preferred_node}: that rank's gathers cross UPI."
                    )
                assign(
                    f"engram_layer_{layer_id}_rank_{rank}",
                    nbytes,
                    node,
                    layout,
                    rank,
                    layer_id,
                )

    if extra_per_rank_bytes > 0:
        for rank in range(tp_size):
            node = preferred_node
            if not _fits(nodes[preferred_node], used[preferred_node], extra_per_rank_bytes):
                if not allow_split:
                    raise EngramNumaError(
                        f"expert-spill pin of {extra_per_rank_bytes / GIB:.1f} GiB/rank "
                        f"does not fit remaining capacity on NUMA node {preferred_node}. "
                        f"Lower SGLANG_DSV41_EXPERT_SPILL_GB, add DRAM on that node, "
                        f"or set SGLANG_DSV41_ENGRAM_NUMA_SPLIT=1."
                    )
                overflow = [n for n in sorted(nodes) if n != preferred_node]
                node = overflow[0]
                plan.warnings.append(
                    f"expert-spill rank {rank} pinned on NUMA node {node} "
                    f"(crosses UPI to the GPU complex on node {preferred_node})"
                )
            assign(
                f"expert_spill_rank_{rank}",
                extra_per_rank_bytes,
                node,
                "private",
                rank,
                -1,
            )

    if plan.crosses_upi:
        plan.warnings.append(
            "one or more host mappings sit off the GPU NUMA node; "
            "gathers will be slower than node-local huge pages"
        )
    return plan


def node_for_mapping(plan: HostPlacementPlan, *, layer_id: int, rank: int) -> int:
    for m in plan.mappings:
        if m.layer_id == layer_id and (m.layout == "shared" or m.rank == rank):
            if m.name.startswith("engram_"):
                return m.node
    raise EngramNumaError(
        f"no Engram mapping in the placement plan for layer {layer_id} rank {rank}"
    )


def bind_buffer_to_numa_node(
    ptr: int, nbytes: int, node: int, *, move_existing: bool = True
) -> None:
    """mbind(MPOL_BIND) the mapping to ``node``. Fails loud.

    ``move_existing`` adds STRICT|MOVE for ranges that already hold pages; the
    call then fails with EIO when any page cannot be moved. Bind a range before
    its first touch and pass False.
    """
    if nbytes <= 0:
        return
    page = mmap.PAGESIZE
    start = ptr & ~(page - 1)
    end = (ptr + nbytes + page - 1) & ~(page - 1)
    length = end - start
    nodemask = ctypes.c_ulong(1 << node)
    # maxnode is the highest node index plus one; 64 covers a ulong bitmask.
    maxnode = ctypes.c_ulong(max(64, node + 1))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    rc = libc.syscall(
        ctypes.c_long(_SYS_MBIND),
        ctypes.c_void_p(start),
        ctypes.c_ulong(length),
        ctypes.c_int(_MPOL_BIND),
        ctypes.byref(nodemask),
        maxnode,
        ctypes.c_uint((_MPOL_MF_STRICT | _MPOL_MF_MOVE) if move_existing else 0),
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise EngramNumaError(
            f"mbind(MPOL_BIND, node={node}, {nbytes} bytes) failed: "
            f"errno={err} ({os.strerror(err)})"
        )


def node_available_bytes(node: int) -> int:
    """MemFree plus file-backed LRU pages of ``node``: what an allocation bound
    there can get without swapping. 0 when the node is unknown."""
    keys = ("MemFree:", "Active(file):", "Inactive(file):")
    total = 0
    try:
        with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4 and parts[2] in keys:
                    total += int(parts[3]) * 1024
    except OSError:
        return 0
    return total


def read_node_hugepages(node: int, pagesize_kb: int = 1048576) -> tuple[int, int]:
    """Return (nr, free) hugepages of ``pagesize_kb`` on a NUMA node."""
    path = (
        f"/sys/devices/system/node/node{node}/hugepages/"
        f"hugepages-{pagesize_kb}kB"
    )
    try:
        with open(os.path.join(path, "nr_hugepages")) as f:
            nr = int(f.read().strip())
        with open(os.path.join(path, "free_hugepages")) as f:
            free = int(f.read().strip())
        return nr, free
    except OSError:
        return 0, 0


def round_to_huge_pages(nbytes: int, page_bytes: int) -> int:
    if nbytes <= 0 or page_bytes <= 0:
        return max(nbytes, 0)
    return ((nbytes + page_bytes - 1) // page_bytes) * page_bytes


def parse_cpulist(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.replace("\n", "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(part))
    return cpus


def node_cpu_set(node: int) -> set[int]:
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            return parse_cpulist(f.read())
    except OSError:
        return set()


class numa_alloc_scope:
    """Bind the calling thread's mempolicy + affinity to ``node`` for mmap.

    Hugetlb pages are taken from the allocating thread's node. mbind(MOVE)
    cannot migrate 1G pages, so the policy has to be set *before* mmap.
    """

    def __init__(self, node: int):
        self.node = node
        self._prev_affinity = None

    def __enter__(self):
        cpus = node_cpu_set(self.node)
        if cpus:
            self._prev_affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(0, cpus)
        nodemask = ctypes.c_ulong(1 << self.node)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        rc = libc.syscall(
            ctypes.c_long(_SYS_SET_MEMPOLICY),
            ctypes.c_int(_MPOL_BIND),
            ctypes.byref(nodemask),
            ctypes.c_ulong(max(64, self.node + 1)),
        )
        if rc != 0:
            err = ctypes.get_errno()
            if self._prev_affinity is not None:
                os.sched_setaffinity(0, self._prev_affinity)
            raise EngramNumaError(
                f"set_mempolicy(BIND, node={self.node}) failed errno={err}"
            )
        return self

    def __exit__(self, *exc):
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        libc.syscall(
            ctypes.c_long(_SYS_SET_MEMPOLICY),
            ctypes.c_int(_MPOL_DEFAULT),
            ctypes.c_void_p(0),
            ctypes.c_ulong(0),
        )
        if self._prev_affinity is not None:
            os.sched_setaffinity(0, self._prev_affinity)
        return False


def smaps_huge_kb(addr: int) -> tuple[int, int, int]:
    """Return (rss_kB, thp_kB, hugetlb_kB) for the VMA containing ``addr``."""
    rss = thp = hugetlb = 0
    kernel_page_kb = 0
    inside = False
    try:
        with open("/proc/self/smaps") as f:
            for line in f:
                m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
                if m:
                    if inside:
                        break
                    inside = int(m.group(1), 16) <= addr < int(m.group(2), 16)
                    continue
                if not inside:
                    continue
                key, _, rest = line.partition(":")
                if key == "Rss":
                    rss = int(rest.split()[0])
                elif key in ("AnonHugePages", "ShmemPmdMapped", "FilePmdMapped"):
                    thp += int(rest.split()[0])
                elif key in ("Private_Hugetlb", "Shared_Hugetlb"):
                    hugetlb += int(rest.split()[0])
                elif key == "KernelPageSize":
                    kernel_page_kb = int(rest.split()[0])
    except OSError:
        pass
    if hugetlb == 0 and kernel_page_kb >= 1048576 and rss:
        hugetlb = rss
    return rss, thp, hugetlb


def mmap_hugetlb(
    nbytes: int,
    *,
    node: int,
    page_bytes: int = GIB,
    shared: bool = False,
    fd: Optional[int] = None,
    name: str = "sglang_engram",
) -> tuple[mmap.mmap, int, int]:
    """Map ``nbytes`` rounded up to ``page_bytes`` from the hugetlb pool.

    Returns ``(mm, fd, map_bytes)``. ``fd`` is -1 for a private anonymous map.
    Fails loud if that node does not have enough free huge pages.
    """
    if nbytes <= 0:
        raise ValueError("mmap_hugetlb nbytes must be > 0")
    if page_bytes not in (2 * 1024 * 1024, GIB):
        raise ValueError(f"unsupported hugetlb page {page_bytes}")
    map_bytes = round_to_huge_pages(nbytes, page_bytes)
    pages = map_bytes // page_bytes
    pagesize_kb = page_bytes // 1024
    if fd is None:
        nr, free = read_node_hugepages(node, pagesize_kb)
        if free < pages:
            raise EngramNumaError(
                f"need {pages} x {pagesize_kb}kB hugetlb pages on node {node} for "
                f"{nbytes / GIB:.1f} GiB ({map_bytes / GIB:.1f} GiB rounded), but "
                f"only {free} free (nr={nr}). Engram must use the boot pool; "
                f"do not silently fall back to 4 KiB."
            )
    huge_flag = MAP_HUGE_1GB if page_bytes == GIB else MAP_HUGE_2MB
    created_fd = False
    if shared:
        if fd is None:
            mfd_huge = MFD_HUGE_1GB if page_bytes == GIB else MFD_HUGE_2MB
            fd = memfd_create(name, MFD_HUGETLB | mfd_huge)
            created_fd = True
            try:
                os.ftruncate(fd, map_bytes)
            except OSError:
                os.close(fd)
                raise
        flags = mmap.MAP_SHARED | MAP_POPULATE
        fileno = fd
    else:
        fd = -1
        flags = (
            mmap.MAP_PRIVATE
            | MAP_ANONYMOUS
            | MAP_HUGETLB
            | huge_flag
            | MAP_POPULATE
        )
        fileno = -1
    try:
        with numa_alloc_scope(node):
            mm = mmap.mmap(
                fileno,
                map_bytes,
                flags=flags,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
    except OSError as e:
        if created_fd:
            os.close(fd)
        raise EngramNumaError(
            f"mmap hugetlb {map_bytes / GIB:.1f} GiB on node {node} failed: {e}"
        ) from e
    return mm, (-1 if not shared else int(fd)), map_bytes


def mmap_numa_thp(
    nbytes: int,
    *,
    node: int,
    populate: bool = True,
) -> mmap.mmap:
    """Anonymous node-local mapping. Uses THP (2 MiB), never the 1G hugetlb pool.

    The range is bound to ``node`` before its first touch, so its pages are
    allocated there and never migrated. A failed bind or populate raises
    ``EngramNumaError`` and unmaps the range, so the caller can try another node.
    """
    if nbytes <= 0:
        raise ValueError("mmap_numa_thp nbytes must be > 0")
    page = mmap.PAGESIZE
    map_bytes = ((nbytes + page - 1) // page) * page
    mm = mmap.mmap(
        -1,
        map_bytes,
        flags=mmap.MAP_PRIVATE | MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    try:
        ptr = ctypes.addressof((ctypes.c_char * 1).from_buffer(mm))
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        libc.madvise(ctypes.c_void_p(ptr), ctypes.c_size_t(map_bytes), MADV_HUGEPAGE)
        bind_buffer_to_numa_node(ptr, map_bytes, node, move_existing=False)
        if populate:
            # The range already has its policy; the scope makes the node's own CPUs zero it.
            with numa_alloc_scope(node):
                _populate_bound_mapping(mm, node=node)
    except BaseException:
        mm.close()
        raise
    return mm


def _populate_bound_mapping(mm: mmap.mmap, *, node: int) -> None:
    if not _has_madv_populate_write():
        # Kernels before 5.14: touching the pages cannot report a full node.
        np.frombuffer(mm, dtype=np.uint8)[:: mmap.PAGESIZE] = 0
        return
    try:
        mm.madvise(_MADV_POPULATE_WRITE)
    except OSError as e:
        raise EngramNumaError(
            f"populating {len(mm)} bytes on NUMA node {node} failed: {e}"
        ) from e
