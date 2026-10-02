from __future__ import annotations

import json
import logging
import mmap
import os
from collections import defaultdict
from functools import lru_cache
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import alloc_mmap
from sglang.srt.runtime_context import get_memory
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

_CUDA_HOST_REGISTERED_RANGES_ATTR = "_sglang_cuda_host_registered_ranges"
_OS_PAGE_BYTES = mmap.PAGESIZE


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
    elif allocator_type == "tensorcast":
        try:
            from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
                get_tensorcast_host_allocator_from_runtime,
            )

            return get_tensorcast_host_allocator_from_runtime()
        except ImportError:
            logger.warning(
                "TensorCast's tensor allocator requires tensorcast >= 0.1.1. Please install TensorCast by 'pip install tensorcast' or build from source by following https://tensorcast.ai/development/build-from-source/. Fallback to use default allocator"
            )
            return HostTensorAllocator()
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


def _host_register_chunk_limit_bytes() -> int:
    return max(envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB.get(), 1) * 1024**3


def _cuda_host_register(
    buffer: torch.Tensor, registration_granularity_bytes: int | None = None
) -> None:
    # Avoid oversized cudaHostRegister calls on large host pools.
    cudart = torch.cuda.cudart()
    base = buffer.data_ptr()
    total = buffer.numel() * buffer.element_size()
    chunk_limit_bytes = _host_register_chunk_limit_bytes()
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


@lru_cache(maxsize=1)
def _resolve_device_accessible_ptr_fn():
    try:
        from sgl_kernel.kvcacheio import get_device_accessible_ptr
    except ImportError:
        get_device_accessible_ptr = None
    else:
        if not hasattr(torch.ops.sgl_kernel, "get_device_accessible_ptr"):
            get_device_accessible_ptr = None

    if get_device_accessible_ptr is None:
        # CUDA's UVA makes host and device addresses equal; on HIP they differ.
        if _is_hip:
            raise ImportError(
                "sgl_kernel.kvcacheio.get_device_accessible_ptr is missing from the "
                "installed sglang-kernel. It is required on ROCm, where registered "
                "host memory carries a distinct device address. Rebuild sglang-kernel "
                "from python/sglang/kernels/aot (setup_rocm.py)."
            )
        logger.warning(
            "sgl_kernel.kvcacheio.get_device_accessible_ptr is missing from the "
            "installed sglang-kernel; using raw host addresses for kernel pointer "
            "tables. Build sglang-kernel from python/sglang/kernels/aot to enable it."
        )
    return get_device_accessible_ptr


def make_kernel_ptr_table(
    tensors: list[torch.Tensor],
    target_device: torch.device | str,
    *,
    host_memory_registered: bool,
) -> torch.Tensor:
    device = torch.device(target_device)
    get_device_accessible_ptr = (
        _resolve_device_accessible_ptr_fn()
        if host_memory_registered and device.type == "cuda"
        else None
    )
    if get_device_accessible_ptr is not None:
        if device.index is None:
            device_index = torch.cuda.current_device()
        else:
            device_index = device.index
        pointers = [
            get_device_accessible_ptr(tensor, device_index) for tensor in tensors
        ]
    else:
        pointers = [tensor.data_ptr() for tensor in tensors]
    return torch.tensor(
        pointers,
        dtype=torch.uint64,
        device=device,
    )


def _registered_page_segments(
    buffer: torch.Tensor, page_bytes: int
) -> tuple[Optional[tuple[tuple[int, int], ...]], str]:
    """Split ``buffer``'s pages by the host registration that holds them.

    Returns ``((first_page, end_page), ...)`` in order, or ``(None, reason)``.
    A registration must start on a page boundary and the registrations must
    tile the buffer, so that every page lies inside exactly one of them.
    """
    ranges = getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, None)
    if not ranges:
        return None, "the pool carries no host-registration ranges"
    base = buffer.data_ptr()
    total = buffer.numel() * buffer.element_size()
    if page_bytes <= 0 or total % page_bytes != 0:
        return (
            None,
            f"pool size {total} B is not a whole number of {page_bytes} B pages",
        )
    segments = []
    offset = 0
    for ptr, size in ranges:
        if offset >= total:
            break
        if ptr - base != offset:
            return None, (
                f"host registrations do not tile the pool (registration at offset "
                f"{ptr - base}, expected {offset})"
            )
        if ptr % _OS_PAGE_BYTES != 0:
            # Two registrations would share an OS page; keep to one per page.
            return None, (
                f"a host registration starts inside an OS page (offset {offset})"
            )
        end = min(offset + size, total)
        if end % page_bytes != 0:
            return None, (
                f"a host registration ends inside a page (offset {end}, "
                f"page {page_bytes} B)"
            )
        segments.append((offset // page_bytes, end // page_bytes))
        offset = end
    if offset < total:
        return None, f"host registrations cover {offset} of {total} B"
    return tuple(segments), ""


def direct_page_kernel_segments(
    buffer: Optional[torch.Tensor],
    *,
    page_bytes: int,
    item_bytes: int,
    pin_memory: bool,
    target_device,
    pool_name: str,
) -> Optional[tuple[tuple[int, int], ...]]:
    """Page segments for page_first_direct transfers of ``buffer`` via the gather kernel.

    The ``direct`` IO backend moves a page_first_direct pool one (page, layer)
    block per memcpy. On ROCm that is one hipMemcpyAsync of page_size rows per
    page and layer (the batched-memcpy path is compiled out), which is CPU-bound
    at a few GB/s. The AOT gather kernels copy the same blocks with one launch
    per layer, reading or writing the registered host pool directly.

    A kernel addresses the pool from one device-accessible base, which is only
    valid inside one host registration. A pool larger than
    SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB is registered in several page-aligned
    pieces, so the pages are returned per registration as
    ``((first_page, end_page), ...)`` and each piece gets its own launch.

    ``item_bytes`` is what one launch moves per (page, layer); the gather
    kernel needs a multiple of 8 bytes (a 1-token DSA indexer page is 132 B).

    Returns None when the per-page copy path must be used. Logs the path this
    pool takes, and why, once.
    """
    path = None
    if not _is_hip:
        reason = "the page gather kernel is used on ROCm only"
    elif buffer is None:
        reason = "the pool has no buffer"
    elif item_bytes % 8 != 0:
        reason = f"the {item_bytes} B (page, layer) block is not a multiple of 8 bytes"
    elif not pin_memory:
        reason = "the pool is not host-registered (pin_memory=False)"
    elif not torch.cuda.is_available() or torch.device(target_device).type != "cuda":
        reason = f"the device pool is not on a CUDA/HIP device ({target_device})"
    else:
        path, reason = _registered_page_segments(buffer, page_bytes)
    if path is None:
        logger.info(
            "HiCache %s host pool (page_first_direct): io_backend=direct uses "
            "per-page copies: %s.",
            pool_name,
            reason,
        )
    else:
        logger.info(
            "HiCache %s host pool (page_first_direct): io_backend=direct uses the "
            "page gather kernel over %d host registration(s) (%d pages of %d B).",
            pool_name,
            len(path),
            path[-1][1],
            page_bytes,
        )
    return path


def page_ids_of_tokens(indices: torch.Tensor, page_size: int) -> Optional[torch.Tensor]:
    """Page ids (int64, on the indices' device) of page-aligned token ``indices``.

    Returns None when ``indices`` do not cover whole pages in order, so the
    caller can keep its per-page copy path for them.
    """
    if indices.numel() % page_size != 0:
        return None
    firsts = indices.reshape(-1, page_size)
    if not indices.is_cuda:
        # Cheap on CPU: every row must be page_id * page_size + arange(page_size).
        expected = firsts[:, :1] + torch.arange(page_size, dtype=firsts.dtype)
        if not torch.equal(firsts, expected) or bool((firsts[:, 0] % page_size).any()):
            return None
    return (firsts[:, 0] // page_size).to(torch.int64)


class DirectPageIndices:
    """Per-registration (host, device) page ids for page_first_direct kernel transfers.

    ``get`` returns ``[(first_page, end_page, host_pages, device_pages), ...]``:
    one entry per host registration that the transfer touches, host page ids
    relative to ``first_page``, both on the device. None means the indices are
    not whole pages and the caller keeps its per-page copies (logged once).

    A load hands the same index tensors to every layer, so the entries
    converted for the most recent load are kept and reused by its other layers.
    """

    def __init__(self, page_size: int, device, segments, pool_name: str = ""):
        self.page_size = page_size
        self.device = device
        self.segments = tuple(segments)
        self.pool_name = pool_name
        self._last = None
        self._warned_fallback = False

    def _split(self, host_pages, device_pages):
        if host_pages.numel() == 0:
            return []
        if len(self.segments) == 1:
            first = self.segments[0][0]
            if first:
                host_pages = host_pages - first
            parts = [(first, self.segments[0][1], host_pages, device_pages)]
        else:
            # Several host registrations: route each page to the one holding it.
            # The direct backend keeps its indices on the CPU, so this is cheap.
            host_pages, device_pages = host_pages.cpu(), device_pages.cpu()
            parts = []
            for first, end in self.segments:
                mask = (host_pages >= first) & (host_pages < end)
                if bool(mask.any()):
                    parts.append(
                        (first, end, host_pages[mask] - first, device_pages[mask])
                    )
        return [
            (
                first,
                end,
                hp.to(self.device, non_blocking=True),
                dp.to(self.device, non_blocking=True),
            )
            for first, end, hp, dp in parts
        ]

    def get(self, host_indices, device_indices, *, reuse: bool):
        last = self._last if reuse else None
        if last is not None and last[0] is host_indices and last[1] is device_indices:
            return last[2]
        host_pages = page_ids_of_tokens(host_indices, self.page_size)
        device_pages = (
            None
            if host_pages is None
            else page_ids_of_tokens(device_indices, self.page_size)
        )
        if device_pages is None:
            if not self._warned_fallback:
                self._warned_fallback = True
                logger.warning(
                    "HiCache %s host pool: a page_first_direct transfer of %d/%d "
                    "host/device indices is not whole pages of %d tokens; such "
                    "transfers use per-page copies (logged once).",
                    self.pool_name,
                    host_indices.numel(),
                    device_indices.numel(),
                    self.page_size,
                )
            return None
        pages = self._split(host_pages, device_pages)
        if reuse:
            self._last = (host_indices, device_indices, pages)
        return pages


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
        "xpu": alloc_with_pin_memory,
    },
)
