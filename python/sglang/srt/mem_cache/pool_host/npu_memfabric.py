"""NPU host DRAM mapped into device VA space for the AIV IO path.

torch pin_memory buffers are only reachable by the SDMA engine; AIV kernels
(e.g. kv_exchange) can only de-reference host VAs that were mapped into the
device VA space first.  SGLANG_HICACHE_HOST_MEM_BACKEND selects the mapping
provider:

- ``memfabric``: the MemFabric offload entity (DRAM_MAP_HOST_VA, see
  acc_offload_local_dram_entry.cpp) allocates and registers host DRAM
  (offload.empty + offload.host_register).  The HiCache host pool is
  allocated through memfabric_hybrid.offload.empty AND the L2<->L1 IO
  uses the AIV kv_exchange kernel.
- ``acl``: standard CANN runtime registration (aclrtHostRegister with
  ACL_HOST_REGISTER_MAPPED, see the kernel-side
  sgl_kernel_npu.host_memory).  Plain pageable host tensors are allocated
  with torch.empty and registered so the AIV kv_exchange kernel can
  resolve their device pointers. The kernel-side auto routing sends every
  transfer to the AIV operator.

ascendc_io_enabled() gates the AIV IO path and is true for both backends.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_MEMFABRIC_GB = 1024**3
_memfabric_state = {
    "offload": None,
    "initialized": False,
    "reserved_bytes": 0,
    "allocated_bytes": 0,
    "device_id": None,
}
_acl_state = {
    "allocated_bytes": 0,
    "device_id": None,
}


def host_memory_backend() -> str | None:
    """Return 'memfabric' | 'acl' | None from SGLANG_HICACHE_HOST_MEM_BACKEND."""
    return os.environ.get("SGLANG_HICACHE_HOST_MEM_BACKEND", "").lower() or None


def memfabric_host_memory_enabled() -> bool:
    """MemFabric offload entity backs the host pool (offload.empty blocks)."""
    return host_memory_backend() == "memfabric"


def acl_host_memory_enabled() -> bool:
    """aclrtHostRegister backs the host pool (pageable torch.empty blocks)."""
    return host_memory_backend() == "acl"


def ascendc_io_enabled() -> bool:
    """Use the AIV kv_exchange kernel for HiCache L2<->L1 IO.

    Both registered-host-memory backends map host DRAM into the device VA
    space, which is what the AIV kernel de-references, so either one
    enables this IO path.
    """
    return host_memory_backend() in ("memfabric", "acl")


def _get_memfabric_offload():
    if _memfabric_state["offload"] is None:
        try:
            from memfabric_hybrid import offload
        except ImportError as exc:
            raise ImportError(
                "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric requires "
                "the memfabric_hybrid package (provides the acc_offload host "
                "memory allocator). Install it or unset the env var."
            ) from exc
        _memfabric_state["offload"] = offload
    return _memfabric_state["offload"]


def _memfabric_gb_aligned(nbytes: int) -> int:
    """Round one host block up to whole GBs.

    The host-memory entry rounds every offload.empty/host_malloc block up to
    the GB granularity on A3 (vmm giant pages), so both the reserve budget
    and the per-block accounting must round the same way.
    """
    return (nbytes + _MEMFABRIC_GB - 1) // _MEMFABRIC_GB * _MEMFABRIC_GB


def ensure_memfabric_capacity(component_nbytes, device_id: int) -> None:
    """Initialize the MemFabric host-memory entry, budgeted per component.

    component_nbytes is the per-block byte size of every buffer the calling
    host pool is about to allocate (each becomes its own offload.empty block,
    GB-rounded by the host-memory entry; ultimately derived from
    --hicache-size / --hicache-ratio).  The entity is sized by the first
    declaration; later host pools in the same process must fit into what is
    left.

    The entity is initialized with OFFLOAD_FLAG_HOST_MEM -- the
    init/malloc/register-split pool, and the only pool implementing
    host_register/host_get_device_pointer for the AIV kv_exchange kernel.
    This flag combination is the one device-validated on the mf_aiv branch.
    """
    if isinstance(component_nbytes, int):
        component_nbytes = [component_nbytes]
    required_bytes = sum(_memfabric_gb_aligned(n) for n in component_nbytes)
    offload = _get_memfabric_offload()
    if not _memfabric_state["initialized"]:
        config = offload.OffloadConfig()
        config.device_id = device_id
        config.reserve_size = required_bytes
        config.scene = offload.Scene.LOCAL
        config.flags = offload.OFFLOAD_FLAG_HOST_MEM
        ret = offload.initialize(config)
        if ret != 0:
            raise RuntimeError(
                f"offload.initialize(OFFLOAD_FLAG_HOST_MEM) failed: ret={ret}, "
                f"device_id={device_id}, reserve_size={required_bytes}"
            )
        _memfabric_state.update(
            initialized=True, reserved_bytes=required_bytes, device_id=device_id
        )
        logger.info(
            "[HiCache] memfabric host memory initialized (host-mem entry): "
            "reserve=%.2fGB device=%d (per-component GB-rounded budget)",
            required_bytes / _MEMFABRIC_GB,
            device_id,
        )
    elif _memfabric_state["device_id"] != device_id:
        raise RuntimeError(
            f"memfabric host memory is already initialized for device "
            f"{_memfabric_state['device_id']}, got device {device_id}"
        )
    remaining = _memfabric_state["reserved_bytes"] - _memfabric_state["allocated_bytes"]
    if required_bytes > remaining:
        raise RuntimeError(
            f"memfabric host memory exhausted: need "
            f"{required_bytes / _MEMFABRIC_GB:.2f}GB, "
            f"only {remaining / _MEMFABRIC_GB:.2f}GB left of the "
            f"{_memfabric_state['reserved_bytes'] / _MEMFABRIC_GB:.2f}GB reserve "
            "(sized automatically from the first L2 host pool, i.e. from "
            "--hicache-size / --hicache-ratio). All host pools of the "
            "process share this reserve."
        )


def _register_mf_host_block(tensor: torch.Tensor) -> None:
    """Register an MF-allocated block with the acc_offload entry manager.

    offload.empty memory is not visible to get_dva/host_get_device_pointer
    until host_register maps it, and the AIV kv_exchange kernel dereferences
    the resulting DVA directly. Registration happens once, at allocation
    time; the kernel-side host_memory table resolves interior views by byte
    offset at transfer time.
    """
    try:
        from sgl_kernel_npu.host_memory import register_host_memory
    except ImportError as exc:
        raise ImportError(
            "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric requires sgl-kernel-npu "
            "with host_memory.register_host_memory (AIV kv_exchange needs "
            "host-registered MF blocks)"
        ) from exc
    register_host_memory(tensor)


def alloc_with_memfabric(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """Allocate host tensor backed by Memfabric-mapped DRAM (AIV-de-referencable).

    Each call allocates one block from the host-memory entry (GB-rounded on
    A3) and immediately registers it so the AIV kv_exchange kernel can
    resolve its DVA.  The Python-side accounting mirrors the GB rounding.
    """
    offload = _get_memfabric_offload()
    numel = 1
    for d in dims:
        numel *= d
    aligned_bytes = _memfabric_gb_aligned(numel * dtype.itemsize)
    remaining = _memfabric_state["reserved_bytes"] - _memfabric_state["allocated_bytes"]
    if aligned_bytes > remaining:
        raise RuntimeError(
            f"memfabric host memory exhausted: block needs "
            f"{aligned_bytes / _MEMFABRIC_GB:.2f}GB, "
            f"only {remaining / _MEMFABRIC_GB:.2f}GB left of the "
            f"{_memfabric_state['reserved_bytes'] / _MEMFABRIC_GB:.2f}GB reserve"
        )
    tensor = offload.empty(list(dims), dtype=dtype)
    _register_mf_host_block(tensor)
    _memfabric_state["allocated_bytes"] += aligned_bytes
    return tensor


# ---------------------------------------------------------------------------
# aclrt-registered host DRAM (SGLANG_HICACHE_HOST_MEM_BACKEND=acl)
#
# Standard CANN runtime registration: plain pageable host pages are mapped
# into the device VA space by aclrtHostRegister(ACL_HOST_REGISTER_MAPPED)
# inside the kernel-side host_memory module.  Unlike MemFabric there is no
# pre-reserved entity and no GB granularity, so capacity tracking is pure
# Python-side accounting. Transfers use the AIV kv_exchange operator.
# ---------------------------------------------------------------------------


def ensure_acl_capacity(component_nbytes, device_id: int) -> None:
    """Soft accounting for the aclrt-registered host pool.

    Same per-component list protocol as ensure_memfabric_capacity so host
    pools can declare their blocks either way.  This is declaration-only:
    the real accounting happens in alloc_with_acl as each block is
    registered, so the totals are not double-counted.
    """
    if isinstance(component_nbytes, int):
        component_nbytes = [component_nbytes]
    total_bytes = sum(component_nbytes)
    if _acl_state["device_id"] is None:
        _acl_state["device_id"] = device_id
        logger.info(
            "[HiCache] aclrt host memory path selected: %d blocks, %.2fGB declared, "
            "device=%d (pageable torch.empty + aclrtHostRegister)",
            len(component_nbytes),
            total_bytes / _MEMFABRIC_GB,
            device_id,
        )
    elif _acl_state["device_id"] != device_id:
        raise RuntimeError(
            f"aclrt host memory is already registered for device "
            f"{_acl_state['device_id']}, got device {device_id}"
        )


def _register_acl_host_block(tensor: torch.Tensor) -> None:
    """Map one torch-allocated host block into the device VA space.

    The aclrt path lives in the kernel-side sgl_kernel_npu.host_memory
    module next to the AIV kv_exchange DVA table, so the AIV operator
    resolves interior views of this block through the same registry.
    """
    try:
        from sgl_kernel_npu.host_memory import register_acl_host_memory
    except ImportError as exc:
        raise ImportError(
            "SGLANG_HICACHE_HOST_MEM_BACKEND=acl requires sgl-kernel-npu "
            "with host_memory.register_acl_host_memory (AIV kv_exchange "
            "needs aclrt-registered host blocks)"
        ) from exc
    register_acl_host_memory(tensor)


def alloc_with_acl(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """Allocate host tensor backed by aclrt-registered pages (AIV-de-referencable).

    Same allocator signature as alloc_with_memfabric so host pools swap the
    function pointer unchanged.  Each call allocates one pageable host block
    with torch.empty and immediately registers it (aclrtHostRegister with
    ACL_HOST_REGISTER_MAPPED), which pins and maps the range for NPU access;
    the kernel-side host_memory table resolves interior views by byte offset
    at transfer time.
    """
    numel = 1
    for d in dims:
        numel *= d
    nbytes = numel * dtype.itemsize
    tensor = torch.empty(list(dims), dtype=dtype)
    _register_acl_host_block(tensor)
    _acl_state["allocated_bytes"] += nbytes
    return tensor


# ---------------------------------------------------------------------------
# Sync-free H2D upload (NPU)
#
# A pageable .to(device) / torch.tensor(..., device=npu) completes with an
# aclrtStreamSynchronize that drains EVERYTHING queued on the current stream
# (profiler-confirmed on the HiCache load path).  When called on the default
# stream before entering the load/write stream, that sync stalls all queued
# compute; on the load stream it serializes the layer-group pipeline.  Stage
# through pinned memory instead: pinned + non_blocking=True is a genuinely
# async enqueue.  The pinned staging tensors are kept alive until their
# consumer copy retires, tracked via events (Event.query() is host-side and
# never synchronizes).
# ---------------------------------------------------------------------------
_pinned_inflight: list = []


def track_pinned_staging(pinned: torch.Tensor) -> None:
    """Keep a pinned staging tensor alive until its async consumer retires.

    Completed entries are dropped on each call so the list stays small.
    """
    done = torch.npu.Event()
    done.record()
    _pinned_inflight.append((pinned, done))
    _pinned_inflight[:] = [e for e in _pinned_inflight if not e[1].query()]


def to_device_no_sync(cpu_tensor: torch.Tensor, device) -> torch.Tensor:
    """Upload a CPU tensor to the NPU without synchronizing the stream.

    NPU-only helper (torch.npu.Event); callers are on the AscendC IO path.
    """
    pinned = cpu_tensor.pin_memory()
    out = torch.empty(pinned.shape, dtype=pinned.dtype, device=device)
    out.copy_(pinned, non_blocking=True)
    track_pinned_staging(pinned)
    return out
