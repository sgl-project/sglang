import ctypes
import logging
import math
import os
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.memory_pool_host import HostTensorAllocator

logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw != "" else default


class UMBPHostTensorAllocator(HostTensorAllocator):
    """Allocate the HiCache L2 host tensor from mori's UMBPHostMemAllocator."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import mori.umbp as umbp_mod
        except ImportError as exc:
            raise RuntimeError(
                "mori.umbp is not available. Build mori with BUILD_UMBP=ON "
                "or fall back to the default torch host allocator."
            ) from exc

        self._mod = umbp_mod
        self._allocator = umbp_mod.UMBPHostMemAllocator()

        self._use_hugepage = _bool_env("SGLANG_HICACHE_HOST_HUGEPAGE", True)
        self._hugepage_size = _int_env(
            "SGLANG_HICACHE_HOST_HUGEPAGE_SIZE", 2 * 1024 * 1024
        )
        self._numa_node = _int_env("SGLANG_HICACHE_HOST_NUMA_NODE", -1)
        self._prefault = _bool_env("SGLANG_HICACHE_HOST_PREFAULT", True)
        self._handle: Optional[Any] = None

    def allocate(
        self, dims: tuple, dtype: torch.dtype, device: str = "cpu"
    ) -> torch.Tensor:
        if device != "cpu":
            raise ValueError(
                "UMBPHostTensorAllocator only supports CPU host memory, "
                f"got device={device}"
            )

        self.dims = dims
        self.dtype = dtype

        element_size = torch.empty((), dtype=dtype).element_size()
        nbytes = math.prod(int(dim) for dim in dims) * element_size

        requested_backing = (
            self._mod.UMBPHostBufferBacking.AnonymousHugetlb
            if self._use_hugepage
            else self._mod.UMBPHostBufferBacking.Anonymous
        )

        handle = self._allocator.alloc(
            nbytes,
            requested_backing,
            self._hugepage_size,
            self._numa_node,
            self._prefault,
        )
        if not handle:
            raise RuntimeError(
                f"UMBPHostMemAllocator.alloc({nbytes} bytes) failed "
                f"(requested_backing={requested_backing}, "
                f"numa_node={self._numa_node})."
            )
        self._handle = handle

        c_array = (ctypes.c_byte * nbytes).from_address(handle.ptr)
        tensor = torch.frombuffer(c_array, dtype=torch.uint8, count=nbytes)

        if dtype != torch.uint8:
            tensor = tensor.view(dtype)

        logger.info(
            "UMBPHostTensorAllocator: allocated %.2f GB at 0x%x "
            "requested_backing=%s actual_backing=%s actual_alignment=%d "
            "mapped_size=%d numa_node=%d",
            nbytes / 1e9,
            handle.ptr,
            requested_backing,
            handle.actual_backing,
            handle.actual_alignment,
            handle.mapped_size,
            self._numa_node,
        )
        if (
            self._use_hugepage
            and handle.actual_backing == self._mod.UMBPHostBufferBacking.Anonymous
        ):
            logger.warning(
                "UMBPHostTensorAllocator: requested AnonymousHugetlb backing "
                "but kernel demoted to Anonymous (4 KiB pages). Check "
                "vm.nr_hugepages and HugePages_Free in /proc/meminfo. "
                "Performance and AINIC MR-size benefits will not apply."
            )

        return tensor.view(dims)

    def __del__(self) -> None:
        try:
            handle = getattr(self, "_handle", None)
            allocator = getattr(self, "_allocator", None)
            if handle is not None and allocator is not None:
                allocator.free(handle)
                self._handle = None
        except Exception:
            pass
