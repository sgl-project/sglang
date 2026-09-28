"""Memfabric-mapped host DRAM and AscendC sparse-copy IO path (NPU only).

torch pin_memory buffers are only reachable by the SDMA engine; AIV kernels
(e.g. offload.sparse_copy) can only de-reference host VAs that were mapped
into the device VA space via the Memfabric offload entity (DRAM_MAP_HOST_VA,
see acc_offload_local_dram_entry.cpp).  SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
is the single switch for the whole feature: the HiCache host pool is
allocated through memfabric_hybrid.offload.empty AND the L2<->L1 IO uses
the AIV sparse-copy kernel (see ascendc_io_enabled).
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


def memfabric_host_memory_enabled() -> bool:
    """Single switch for the Memfabric host pool + AscendC IO path."""
    return os.environ.get("SGLANG_HICACHE_HOST_MEM_BACKEND", "").lower() == (
        "memfabric"
    )


def ascendc_io_enabled() -> bool:
    """Use the acc_offload AIV sparse-copy kernel for HiCache L2<->L1 IO.

    Rides on the single memfabric switch: SGLANG_HICACHE_HOST_MEM_BACKEND=
    memfabric enables both the host pool allocation and this IO path (the
    AIV kernel de-references host pool pointers, which requires
    Memfabric-mapped memory).
    """
    return memfabric_host_memory_enabled()


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


def ensure_memfabric_capacity(total_bytes: int, device_id: int) -> None:
    """Lazily initialize the Memfabric offload entity, sized by total_bytes.

    total_bytes is the combined size of all buffers the calling host pool is
    about to allocate (ultimately derived from --hicache-size /
    --hicache-ratio).  The entity is sized by the first declaration; later
    host pools in the same process must fit into what is left.

    The C++ side aligns the reservation up to whole GBs, so the physical
    reservation may be up to ~1GB larger than the value passed here.
    """
    offload = _get_memfabric_offload()
    if not _memfabric_state["initialized"]:
        config = offload.OffloadConfig()
        config.device_id = device_id
        config.reserve_size = total_bytes
        config.alloc_size = total_bytes
        config.flags = offload.OFFLOAD_FLAG_URMA_POOL
        config.scene = offload.Scene.LOCAL
        assert offload.initialize(config) == 0, "offload.initialize failed"
        _memfabric_state.update(
            initialized=True, reserved_bytes=total_bytes, device_id=device_id
        )
        logger.info(
            "[HiCache] memfabric host memory initialized: reserve=%.2fGB device=%d "
            "(physically reserved up to %dGB after C++-side GB alignment)",
            total_bytes / _MEMFABRIC_GB,
            device_id,
            (total_bytes + _MEMFABRIC_GB - 1) // _MEMFABRIC_GB,
        )
    remaining = _memfabric_state["reserved_bytes"] - _memfabric_state["allocated_bytes"]
    if total_bytes > remaining:
        raise RuntimeError(
            f"memfabric host memory exhausted: need "
            f"{total_bytes / _MEMFABRIC_GB:.2f}GB, "
            f"only {remaining / _MEMFABRIC_GB:.2f}GB left of the "
            f"{_memfabric_state['reserved_bytes'] / _MEMFABRIC_GB:.2f}GB reserve "
            "(sized automatically from the first L2 host pool, i.e. from "
            "--hicache-size / --hicache-ratio). All host pools of the "
            "process share this reserve."
        )


def alloc_with_memfabric(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """Allocate host tensor backed by Memfabric-mapped DRAM (AIV-de-referencable)."""
    offload = _get_memfabric_offload()
    numel = 1
    for d in dims:
        numel *= d
    tensor = offload.empty(list(dims), dtype=dtype)
    _memfabric_state["allocated_bytes"] += numel * dtype.itemsize
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
