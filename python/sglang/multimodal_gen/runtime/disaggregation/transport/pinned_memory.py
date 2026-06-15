# SPDX-License-Identifier: Apache-2.0
"""CUDA host-registration helpers for shared-memory transfer buffers."""

from __future__ import annotations

import logging
import mmap
from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.utils import warn_or_raise

logger = logging.getLogger(__name__)


def _cuda_status_code(result) -> int:
    if isinstance(result, tuple):
        result = result[0]
    if hasattr(result, "value"):
        return int(result.value)
    if result is None:
        return 0
    return int(result)


def _align_host_region(ptr: int, size: int) -> tuple[int, int]:
    page_size = mmap.PAGESIZE
    aligned_ptr = ptr - (ptr % page_size)
    end = ptr + size
    aligned_end = ((end + page_size - 1) // page_size) * page_size
    return aligned_ptr, aligned_end - aligned_ptr


@dataclass
class PinnedHostMemoryRegistration:
    """Tracks a cudaHostRegister registration for one process-local mapping."""

    ptr: int
    size: int
    aligned_ptr: int
    aligned_size: int
    registered: bool = False
    status: str = "disabled"
    error: str | None = None

    def unregister(self) -> None:
        if not self.registered:
            return
        try:
            result = torch.cuda.cudart().cudaHostUnregister(self.aligned_ptr)
            status_code = _cuda_status_code(result)
            if status_code != 0:
                logger.warning(
                    "cudaHostUnregister failed for ptr=%s size=%s: cuda error %s",
                    self.aligned_ptr,
                    self.aligned_size,
                    status_code,
                )
        except Exception:
            logger.exception(
                "cudaHostUnregister raised for ptr=%s size=%s",
                self.aligned_ptr,
                self.aligned_size,
            )
        finally:
            self.registered = False
            self.status = "unregistered"


def register_pinned_host_memory(
    ptr: int,
    size: int,
    *,
    enabled: bool,
    strict: bool = False,
) -> PinnedHostMemoryRegistration:
    """Register a process-local host memory mapping as CUDA pinned memory.

    `multiprocessing.shared_memory` gives each process its own virtual address
    mapping, so cudaHostRegister must be called in every process that wants CUDA
    H2D/D2H copies to treat the mapping as page-locked.
    """

    aligned_ptr, aligned_size = _align_host_region(int(ptr), int(size))
    registration = PinnedHostMemoryRegistration(
        ptr=int(ptr),
        size=int(size),
        aligned_ptr=aligned_ptr,
        aligned_size=aligned_size,
    )
    if not enabled:
        return registration
    if size <= 0:
        registration.status = "skipped_empty"
        return registration
    if not torch.cuda.is_available():
        registration.status = "cuda_unavailable"
        registration.error = "CUDA is not available"
        warn_or_raise(
            logger,
            strict,
            "CUDA host registration requested but CUDA is unavailable; using unpinned shared-memory buffer.",
        )
        return registration

    try:
        cudart = torch.cuda.cudart()
        flags = getattr(cudart, "cudaHostRegisterPortable", 1)
        result = cudart.cudaHostRegister(aligned_ptr, aligned_size, flags)
        status_code = _cuda_status_code(result)
        if status_code == 0:
            registration.registered = True
            registration.status = "pinned"
            return registration
        registration.status = "failed"
        registration.error = f"cudaHostRegister returned cuda error {status_code}"
    except Exception as exc:
        registration.status = "failed"
        registration.error = str(exc)

    if strict:
        warn_or_raise(
            logger,
            True,
            "Failed to pin shared-memory transfer buffer: %s",
            registration.error or registration.status,
        )
    logger.warning(
        "Failed to pin shared-memory transfer buffer; using unpinned memory: %s",
        registration.error or registration.status,
    )
    return registration
