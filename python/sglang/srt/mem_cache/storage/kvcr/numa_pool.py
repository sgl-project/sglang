"""Opt-in ownership-scoped NUMA allocation for HiCache and KVCR DRAM.

Only NEW mappings are bound. Never change a task policy or migrate old pages.
The ordinary allocator remains the default when host_pool_numa_node is absent.
"""

import ctypes
import ctypes.util
import json
import logging
import math
import mmap
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)
HOST_PROOF_ATTR = "_sglang_owned_numa_host_proof"
MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


def node_option(value, name):
    if value is not None and (type(value) is not int or not 0 <= value < 65536):
        raise ValueError(f"{name} must be a nonnegative integer NUMA node or null")
    return value


def load_config(raw):
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, str):
        raise ValueError(
            "KVCR host allocator extra config must be a JSON string or @file"
        )
    if raw.startswith("@"):
        path = Path(raw[1:])
        if not raw[1:]:
            raise ValueError("KVCR host allocator config path is empty")
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text())
        elif path.suffix.lower() == ".toml":
            import tomllib

            with path.open("rb") as stream:
                value = tomllib.load(stream)
        elif path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            value = yaml.safe_load(path.read_text())
        else:
            raise ValueError("Unsupported KVCR host allocator config file extension")
    else:
        value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("KVCR host allocator extra config must contain an object")
    return value


def prove_range(maps, numa, smaps, address, length, node, shared):
    """A fully resident uniform covering VMA proves this exact owned subrange."""
    if address <= 0 or length <= 0 or address % mmap.PAGESIZE:
        raise RuntimeError("Invalid owned NUMA mapping range")
    rows = []
    for line in maps.splitlines():
        fields = line.split(None, 5)
        lo, hi = (int(v, 16) for v in fields[0].split("-"))
        if lo <= address and address + length <= hi:
            rows.append((lo, hi, fields, line))
    if len(rows) != 1:
        raise RuntimeError("Owned NUMA range has no unique complete covering VMA")
    lo, hi, fields, line = rows[0]
    if fields[1] != ("rw-s" if shared else "rw-p"):
        raise RuntimeError("Owned NUMA mapping sharing mode differs")
    if shared:
        if (
            len(fields) != 6
            or fields[5] != "/dev/zero (deleted)"
            or int(fields[4]) <= 0
        ):
            raise RuntimeError("Owned host mapping is not new shared anonymous memory")
    elif len(fields) != 5 or fields[4] != "0":
        raise RuntimeError("Owned local DRAM mapping is not private anonymous memory")
    policy = [row for row in numa.splitlines() if int(row.split()[0], 16) == lo]
    if len(policy) != 1 or policy[0].split()[1] != f"bind:{node}":
        raise RuntimeError(
            "Requested strict owned-mapping NUMA policy was not installed"
        )
    counts = {int(k): int(v) for k, v in re.findall(r"\bN(\d+)=(\d+)", policy[0])}
    if (
        counts.get(node) != (hi - lo) // mmap.PAGESIZE
        or sum(counts.values()) != (hi - lo) // mmap.PAGESIZE
    ):
        raise RuntimeError("Owned NUMA mapping is incomplete or has wrong-node pages")
    header = re.search(r"(?m)^" + re.escape(fields[0]) + r" .*$", smaps)
    if header is None:
        raise RuntimeError("Owned NUMA mapping has no smaps evidence")
    segment = smaps[header.end() :]
    following = re.search(r"(?m)^[0-9a-f]+-[0-9a-f]+ ", segment)
    if following:
        segment = segment[: following.start()]
    backing = {}
    for key in ("Size", "Rss", "AnonHugePages", "KernelPageSize", "ShmemPmdMapped"):
        match = re.search(r"(?m)^" + key + r":\s*(\d+) kB", segment)
        if match is None:
            raise RuntimeError("Incomplete owned NUMA page-backing evidence")
        backing[key] = int(match[1])
    if backing["Size"] != (hi - lo) // 1024 or backing["Rss"] != backing["Size"]:
        raise RuntimeError("Owned NUMA mapping is not fully resident")
    return dict(
        address=address,
        length=length,
        end=address + length,
        node=node,
        covering_start=lo,
        covering_end=hi,
        maps_line=line,
        numa_line=policy[0],
        node_pages=counts,
        smaps=backing,
        shared=shared,
        all_covering_pages_resident_on_requested_node=True,
    )


def snapshot_range(address, length, node, shared):
    return prove_range(
        *(
            Path("/proc/self/" + name).read_text()
            for name in ("maps", "numa_maps", "smaps")
        ),
        address,
        length,
        node,
        shared,
    )


def bind_new_mapping_before_use(allocation, node, *, shared):
    """Call only for a newly created owned, unfaulted mapping; flags=0."""
    node = node_option(node, "owned_mapping_numa_node")
    if node is None:
        raise ValueError("An explicit NUMA node is required for strict binding")
    node_path = Path(f"/sys/devices/system/node/node{node}")
    if not node_path.is_dir():
        raise RuntimeError(f"Requested NUMA node {node} is not present")
    address = ctypes.addressof(ctypes.c_char.from_buffer(allocation))
    length = len(allocation)
    if address % mmap.PAGESIZE or length <= 0 or length % mmap.PAGESIZE:
        raise ValueError("Owned NUMA mapping must be page-aligned")
    name = ctypes.util.find_library("numa")
    if name is None:
        raise RuntimeError("Strict owned-mapping NUMA placement requires libnuma")
    library = ctypes.CDLL(name, use_errno=True)
    library.mbind.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_uint,
    ]
    library.mbind.restype = ctypes.c_long
    word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
    mask = (ctypes.c_ulong * (node // word_bits + 1))()
    mask[node // word_bits] = 1 << (node % word_bits)
    ctypes.set_errno(0)
    if library.mbind(address, length, 2, mask, len(mask) * word_bits + 1, 0) != 0:
        error = ctypes.get_errno()
        raise RuntimeError(f"Owned-mapping mbind failed for node {node}: errno {error}")
    # A late bind is not a substitute: prefault only AFTER installing the policy.
    allocation.madvise(MADV_POPULATE_WRITE)
    return snapshot_range(address, length, node, shared)


def allocate_bound_host(dims, dtype, device, node):
    import torch
    from sglang.srt.environ import envs

    if device != "cpu":
        raise ValueError("KVCR NUMA host allocator only accepts CPU tensors")
    if envs.SGLANG_HUGEPAGE_SIZE.get():
        raise ValueError(
            "Explicit hugeTLB host allocation is not qualified with KVCR NUMA placement"
        )
    elements = math.prod(dims)
    if elements <= 0:
        raise ValueError("KVCR NUMA host allocation must contain at least one element")
    count = elements * torch.empty([], dtype=dtype).element_size()
    length = math.ceil(count / mmap.PAGESIZE) * mmap.PAGESIZE
    allocation = mmap.mmap(
        -1,
        length,
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    try:
        proof = bind_new_mapping_before_use(allocation, node, shared=True)
        tensor = torch.frombuffer(allocation, dtype=dtype, count=elements).reshape(dims)
    except BaseException:
        allocation.close()
        raise
    setattr(tensor, HOST_PROOF_ATTR, proof)
    return tensor


def allocator_for_config(raw):
    from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator

    config = load_config(raw)
    node = node_option(config.get("host_pool_numa_node"), "host_pool_numa_node")
    if node is None:
        return HostTensorAllocator()

    class BoundHostTensorAllocator(HostTensorAllocator):
        def allocate(self, dims, dtype, device):
            self.dtype, self.dims = dtype, dims
            return allocate_bound_host(dims, dtype, device, node)

    return BoundHostTensorAllocator()


def audit_registered_host(tensor, registrations):
    """Join actual cudaHostRegister intervals to the owned allocation proof."""
    proof = getattr(tensor, HOST_PROOF_ATTR)
    size = tensor.numel() * tensor.element_size()
    address = tensor.data_ptr()
    if address != proof["address"] or size > proof["length"]:
        raise RuntimeError(
            "CUDA-registered tensor does not match its owned NUMA allocation"
        )
    cursor = address
    for pointer, length in registrations:
        if pointer != cursor or length <= 0:
            raise RuntimeError(
                "CUDA host registrations do not completely cover the owned tensor"
            )
        cursor += length
    if cursor != address + size:
        raise RuntimeError(
            "CUDA host registrations have incomplete or excessive owned-tensor coverage"
        )
    fresh = snapshot_range(address, proof["length"], proof["node"], True)
    stable = (
        "address",
        "length",
        "end",
        "node",
        "covering_start",
        "covering_end",
        "maps_line",
        "node_pages",
        "shared",
    )
    if any(fresh[key] != proof[key] for key in stable):
        raise RuntimeError("Owned host placement changed during CUDA registration")
    logger.info(
        "KVCR owned host NUMA registration %s",
        json.dumps(
            dict(
                pid=os.getpid(),
                tensor_bytes=size,
                registration_ranges=registrations,
                placement=fresh,
                task_policy_changed=False,
                pages_migrated=False,
            )
        ),
    )


def bind_new_private_pool(allocation, node):
    proof = bind_new_mapping_before_use(allocation, node, shared=False)
    logger.info(
        "KVCR owned local DRAM NUMA placement %s",
        json.dumps(
            dict(
                pid=os.getpid(),
                placement=proof,
                task_policy_changed=False,
                pages_migrated=False,
            )
        ),
    )
    return proof
