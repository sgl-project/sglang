from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import alloc_mmap
from sglang.srt.runtime_context import get_memory

logger = logging.getLogger(__name__)

_CUDA_HOST_REGISTERED_RANGES_ATTR = "_sglang_cuda_host_registered_ranges"


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == "cpu", (
            f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        )
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class ShmHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == "cpu", (
            f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        )
        self.dtype = dtype
        self.dims = dims
        from sglang.srt.mem_cache.storage.mmap import alloc_shm

        tensor, fd, mm = alloc_shm(dims, dtype)
        self.fds.append(fd)
        self.mms.append(mm)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type() -> str:
    """The host-allocator kind the published HiCache configuration asks for."""

    backend = get_memory().hicache_storage_backend
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = get_memory().hicache_storage_backend_extra_config
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_host_register(
    buffer: torch.Tensor, registration_granularity_bytes: int | None = None
) -> None:
    # Avoid oversized cudaHostRegister calls on large host pools.
    cudart = torch.cuda.cudart()
    base = buffer.data_ptr()
    total = buffer.numel() * buffer.element_size()
    chunk_limit_bytes = (
        max(envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB.get(), 1) * 1024**3
    )
    # Preserve the legacy single-call behavior unless the caller provides a
    # copy granularity. Splitting an unknown page-first layout at an arbitrary
    # byte offset can make one cudaMemcpyBatchAsync span two registrations.
    chunk_bytes = total
    if registration_granularity_bytes is not None:
        if registration_granularity_bytes <= 0:
            raise ValueError(
                "registration_granularity_bytes must be positive, got "
                f"{registration_granularity_bytes}"
            )
        if registration_granularity_bytes > chunk_limit_bytes:
            raise ValueError(
                "Host registration granularity exceeds the configured chunk limit: "
                f"granularity={registration_granularity_bytes}, "
                f"chunk_limit={chunk_limit_bytes}"
            )
        chunk_bytes = (
            chunk_limit_bytes // registration_granularity_bytes
        ) * registration_granularity_bytes
    registered_ranges: list[tuple[int, int]] = []
    try:
        offset = 0
        while offset < total:
            size = min(chunk_bytes, total - offset)
            ptr = base + offset
            rc = int(cudart.cudaHostRegister(ptr, size, 0))
            if rc != 0:
                raise RuntimeError(
                    f"cudaHostRegister failed (rc={rc}, "
                    f"{cudart.cudaGetErrorString(rc)}) at offset={offset} size={size} "
                    f"(total={total}, chunk_limit={chunk_bytes}); host buffer is not "
                    f"pinned and device transfers may silently return stale data."
                )
            registered_ranges.append((ptr, size))
            offset += size

        # Keep the exact registration bases alive with the tensor. CUDA requires
        # cudaHostUnregister to receive each base pointer, not just the tensor's
        # original base once after several independent registrations.
        setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, registered_ranges)
    except Exception:
        remaining_ranges = _cuda_host_unregister_ranges(
            cudart, registered_ranges, operation="registration rollback"
        )
        if remaining_ranges:
            setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)
        raise


def _cuda_host_unregister_ranges(
    cudart, registered_ranges: list[tuple[int, int]], *, operation: str
) -> list[tuple[int, int]]:
    failed_ranges = []
    for ptr, size in reversed(registered_ranges):
        rc = int(cudart.cudaHostUnregister(ptr))
        if rc != 0:
            failed_ranges.append((ptr, size))
            logger.warning(
                "cudaHostUnregister failed during %s (rc=%d, %s) for ptr=%#x size=%d",
                operation,
                rc,
                cudart.cudaGetErrorString(rc),
                ptr,
                size,
            )
    failed_ranges.reverse()
    return failed_ranges


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    registered_ranges = getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, None)
    if registered_ranges is None:
        # Compatibility for buffers registered before range metadata was added.
        registered_ranges = [
            (buffer.data_ptr(), buffer.numel() * buffer.element_size())
        ]
    if not registered_ranges:
        return

    remaining_ranges = _cuda_host_unregister_ranges(
        cudart, registered_ranges, operation="host-pool destroy"
    )
    setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer, registration_granularity_bytes)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


# ---------------------------------------------------------------------------
# Memfabric-mapped host DRAM (acc_offload)
#
# torch pin_memory buffers are only reachable by the SDMA engine; AIV kernels
# (e.g. offload.sparse_copy) can only de-reference host VAs that were mapped
# into the device VA space via the Memfabric offload entity (DRAM_MAP_HOST_VA,
# see acc_offload_local_dram_entry.cpp).  SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
# is the single switch for the whole feature: the HiCache host pool is
# allocated through memfabric_hybrid.offload.empty AND the L2<->L1 IO uses
# the AIV sparse-copy kernel (see ascendc_io_enabled).
# ---------------------------------------------------------------------------
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
    """
    Allocate host tensor backed by Memfabric-mapped DRAM (AIV-de-referencable).
    """
    offload = _get_memfabric_offload()
    numel = 1
    for d in dims:
        numel *= d
    tensor = offload.empty(list(dims), dtype=dtype)
    _memfabric_state["allocated_bytes"] += numel * dtype.itemsize
    return tensor


def ascendc_io_enabled() -> bool:
    """Use the acc_offload AIV sparse-copy kernel for HiCache L2<->L1 IO.

    Rides on the single memfabric switch: SGLANG_HICACHE_HOST_MEM_BACKEND=
    memfabric enables both the host pool allocation and this IO path (the
    AIV kernel de-references host pool pointers, which requires
    Memfabric-mapped memory).
    """
    return memfabric_host_memory_enabled()


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


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
