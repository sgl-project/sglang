from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import torch

from sglang.srt.mem_cache.storage.mmap import alloc_mmap

logger = logging.getLogger(__name__)


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
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
        assert (
            device == "cpu"
        ), f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
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


def get_allocator_type(server_args) -> str:
    backend = getattr(server_args, "hicache_storage_backend", None)
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = getattr(
            server_args, "hicache_storage_backend_extra_config", None
        )
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_host_register(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    n_bytes = buffer.numel() * buffer.element_size()
    rc = cudart.cudaHostRegister(buffer.data_ptr(), n_bytes, 0)
    if int(rc) != 0:
        raise RuntimeError(
            f"cudaHostRegister failed (rc={int(rc)}, "
            f"{cudart.cudaGetErrorString(rc)}) for ptr={buffer.data_ptr():#x} "
            f"size={n_bytes}; host buffer is not pinned and device transfers "
            f"may silently return stale data."
        )


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostUnregister(buffer.data_ptr())
    if int(rc) != 0:
        # Best-effort on shutdown: warn, don't raise -- a leak is reclaimed at exit.
        logger.warning(
            "cudaHostUnregister failed (rc=%d, %s) for ptr=%#x",
            int(rc),
            cudart.cudaGetErrorString(rc),
            buffer.data_ptr(),
        )


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


# ---------------------------------------------------------------------------
# hybm-mapped host DRAM (Memfabric acc_offload)
#
# torch pin_memory buffers are only reachable by the SDMA engine; AIV kernels
# (e.g. offload.sparse_copy) can only de-reference host VAs that were mapped
# into the device VA space via hybm (DRAM_MAP_HOST_VA, see
# acc_offload_local_dram_entry.cpp).  When SGLANG_HICACHE_HOST_MEM=hybm (or
# SGLANG_HICACHE_IO_ASCENDC=1, which implies it) the HiCache host pool is
# allocated through memfabric_hybrid.offload.empty so that both the legacy
# memcpy2d path and the AIV sparse-copy path can access it.
# ---------------------------------------------------------------------------
_HYBM_GB = 1024**3
_hybm_state = {
    "offload": None,
    "initialized": False,
    "reserved_bytes": 0,
    "allocated_bytes": 0,
    "device_id": None,
}


def hybm_host_memory_enabled() -> bool:
    if os.environ.get("SGLANG_HICACHE_HOST_MEM", "").lower() == "hybm":
        return True
    # The AscendC sparse-copy IO path de-references host pool pointers inside
    # the AIV kernel, which requires hybm-mapped memory.
    return os.environ.get("SGLANG_HICACHE_IO_ASCENDC", "").lower() in ("1", "true", "yes")


def _get_hybm_offload():
    if _hybm_state["offload"] is None:
        try:
            from memfabric_hybrid import offload
        except ImportError as exc:
            raise ImportError(
                "SGLANG_HICACHE_HOST_MEM=hybm / SGLANG_HICACHE_IO_ASCENDC=1 require "
                "the memfabric_hybrid package (provides the acc_offload hybm host "
                "memory allocator). Install it or unset the env vars."
            ) from exc
        _hybm_state["offload"] = offload
    return _hybm_state["offload"]


def ensure_hybm_capacity(total_bytes: int, device_id: int) -> None:
    """Lazily initialize the hybm offload entity with GB-aligned capacity.

    Must be called before the first alloc_with_hybm of a host pool, with the
    combined byte size of all buffers the pool is about to allocate.  For
    multiple host pools per process, size the first pool's reserve via
    SGLANG_HICACHE_HYBM_RESERVE_GB to cover the later ones as well.
    """
    offload = _get_hybm_offload()
    if not _hybm_state["initialized"]:
        env_gb = os.environ.get("SGLANG_HICACHE_HYBM_RESERVE_GB")
        reserve = int(env_gb) * _HYBM_GB if env_gb else 0
        reserve = max(reserve, total_bytes)
        reserve = ((reserve + _HYBM_GB - 1) // _HYBM_GB) * _HYBM_GB
        config = offload.OffloadConfig()
        config.device_id = device_id
        config.reserve_size = reserve
        config.alloc_size = reserve
        assert offload.initialize(config) == 0, "offload.initialize failed"
        _hybm_state.update(
            initialized=True, reserved_bytes=reserve, device_id=device_id
        )
        logger.info(
            "[HiCache] hybm host memory initialized: reserve=%dGB device=%d",
            reserve // _HYBM_GB,
            device_id,
        )
    remaining = _hybm_state["reserved_bytes"] - _hybm_state["allocated_bytes"]
    if total_bytes > remaining:
        raise RuntimeError(
            f"hybm host memory exhausted: need {total_bytes / _HYBM_GB:.2f}GB, "
            f"only {remaining / _HYBM_GB:.2f}GB left of the "
            f"{_hybm_state['reserved_bytes'] // _HYBM_GB}GB reserve. "
            "Size the reserve for ALL host pools of the process via "
            "SGLANG_HICACHE_HYBM_RESERVE_GB."
        )


def alloc_with_hybm(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate host tensor backed by hybm-mapped DRAM (AIV-de-referencable).
    """
    offload = _get_hybm_offload()
    numel = 1
    for d in dims:
        numel *= d
    tensor = offload.empty(list(dims), dtype=dtype)
    _hybm_state["allocated_bytes"] += numel * dtype.itemsize
    return tensor


def ascendc_io_enabled() -> bool:
    """Use the acc_offload AIV sparse-copy kernel for HiCache L2<->L1 IO."""
    return os.environ.get("SGLANG_HICACHE_IO_ASCENDC", "").lower() in (
        "1",
        "true",
        "yes",
    )


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
