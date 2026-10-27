"""Physical-capacity checks for new host mappings governed by task NUMA policy."""

import ctypes
import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MPOL_BIND = 2
_MPOL_F_MEMS_ALLOWED = 4
_MPOL_F_STATIC_NODES = 1 << 15
_MPOL_F_RELATIVE_NODES = 1 << 14
_MPOL_MODE_FLAGS = _MPOL_F_STATIC_NODES | _MPOL_F_RELATIVE_NODES


@functools.cache
def _libnuma():
    lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
    lib.numa_num_possible_nodes.restype = ctypes.c_int
    lib.get_mempolicy.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    lib.get_mempolicy.restype = ctypes.c_int
    return lib


def _get_numa_policy():
    lib = _libnuma()
    maxnode = lib.numa_num_possible_nodes()
    if maxnode <= 0:
        raise OSError("Cannot determine the NUMA nodemask size")
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    mask_type = ctypes.c_ulong * ((maxnode + bits - 1) // bits)

    def query(flags):
        mode, mask = ctypes.c_int(), mask_type()
        if lib.get_mempolicy(ctypes.byref(mode), mask, maxnode, None, flags):
            raise OSError(ctypes.get_errno(), "get_mempolicy failed")
        nodes = {i for i in range(maxnode) if mask[i // bits] & (1 << (i % bits))}
        return mode.value, nodes

    mode, policy_nodes = query(0)
    _, allowed_nodes = query(_MPOL_F_MEMS_ALLOWED)
    return mode, policy_nodes, allowed_nodes


def _parse_nodes(value):
    nodes = set()
    for entry in value.strip().split(","):
        first, _, last = entry.partition("-")
        nodes.update(range(int(first), int(last or first) + 1))
    return nodes


def _numa_memory_capacity(
    sysfs_root=Path("/sys/devices/system/node"), *, use_task_policy=True
):
    """Return (nodes, total bytes), or None when there is no known restriction.

    This is total physical capacity, not an estimate of reclaimable/free RAM.
    Preferred and interleave policies can fall back outside their nodemask;
    only MPOL_BIND adds a hard restriction to the cpuset's allowed nodes.
    """
    try:
        memory_nodes = _parse_nodes((sysfs_root / "has_memory").read_text())
        mode, policy_nodes, allowed_nodes = _get_numa_policy()
        allowed_nodes = allowed_nodes & memory_nodes
        nodes = allowed_nodes
        # Unknown flags retain the cpuset ceiling without interpreting the
        # policy mask. For example, NUMA balancing can return a saved user mask
        # whose dynamic rebinding cannot be recovered from these two queries.
        if use_task_policy and mode & ~_MPOL_MODE_FLAGS == _MPOL_BIND:
            if mode & _MPOL_F_RELATIVE_NODES and allowed_nodes:
                # get_mempolicy returns the original relative mask for this flag.
                # Match Linux's nodes_fold + nodes_onto mapping into the cpuset.
                ordered = sorted(allowed_nodes)
                policy_nodes = {ordered[i % len(ordered)] for i in policy_nodes}
            nodes = policy_nodes & allowed_nodes
            if not nodes and mode & _MPOL_F_STATIC_NODES:
                # Linux falls back to local allocation after a disjoint rebind.
                nodes = allowed_nodes
        if not nodes or nodes == memory_nodes:
            return None
        total = 0
        for node in sorted(nodes):
            fields = (sysfs_root / f"node{node}/meminfo").read_text().split()
            index = fields.index("MemTotal:")
            if fields[index + 2] != "kB":
                raise ValueError("Unexpected NUMA MemTotal unit")
            node_bytes = int(fields[index + 1]) * 1024
            if node_bytes <= 0:
                raise ValueError("Invalid NUMA MemTotal")
            total += node_bytes
        return sorted(nodes), total
    except (OSError, AttributeError, ValueError, IndexError) as exc:
        # Preserve existing allocation behavior where NUMA queries are denied
        # (e.g. seccomp), libnuma is absent, or sysfs is unavailable.
        logger.debug("Cannot check NUMA host-memory capacity: %s", exc)
        return None


def check_numa_memory_capacity(
    requested_bytes: int, *, use_task_policy: bool = True
) -> None:
    """Reject a new mapping that cannot fit in its permitted NUMA nodes.

    A request below this ceiling may still exceed currently available memory.
    This check does not reserve memory or replace HiCache's host/cgroup budget.
    For a filesystem mapping with an unknown mount policy, use_task_policy=False
    checks only the cpuset ceiling: the mount policy can override task binding.
    """
    capacity = _numa_memory_capacity(use_task_policy=use_task_policy)
    if capacity is None:
        return
    nodes, total = capacity
    if requested_bytes > total:
        raise ValueError(
            f"Host mapping requests {requested_bytes / 1024**3:.2f} GiB, "
            f"but permitted NUMA nodes {nodes} contain only "
            f"{total / 1024**3:.2f} GiB total RAM. "
            "Reduce the host-cache size or change the NUMA memory binding."
        )
