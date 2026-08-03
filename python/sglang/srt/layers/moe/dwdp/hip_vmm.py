"""ROCm-only HIP virtual-memory primitives for DWDP.

The native operators are loaded lazily from ``sgl_kernel`` so importing this
module is safe on CUDA and CPU-only installations. POSIX file descriptors have
explicit ownership:

* :func:`export_fd` returns a new descriptor owned by the caller, which must
  eventually call ``os.close``.
* :func:`import_fd` keeps the caller's descriptor open. It imports a temporary
  duplicate and closes that duplicate after HIP has acquired its own dma-buf
  reference. The returned HIP allocation handle must be released separately.

Tensors returned by :func:`tensor_from_ptr` do not own the VMM mapping. They
must be destroyed (and outstanding GPU work synchronized) before unmapping.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import torch

__all__ = [
    "HipVmmCleanupError",
    "HipVmmUnavailableError",
    "VmmMapping",
    "align_down",
    "align_up",
    "create_local_handle",
    "create_shareable_handle",
    "export_fd",
    "extension_availability",
    "free_va",
    "get_allocation_granularity",
    "import_fd",
    "is_supported",
    "map_handle",
    "map_handles",
    "release_handle",
    "reserve_va",
    "set_access",
    "tensor_from_ptr",
    "unmap_va",
]

_REQUIRED_OPS = (
    "hip_vmm_is_supported",
    "hip_vmm_get_allocation_granularity",
    "hip_vmm_create",
    "hip_vmm_release",
    "hip_vmm_address_reserve",
    "hip_vmm_address_free",
    "hip_vmm_map",
    "hip_vmm_unmap",
    "hip_vmm_set_access",
    "hip_vmm_export_fd",
    "hip_vmm_import_fd",
    "hip_vmm_tensor_from_address",
)

# ROCm 7.x associates hipMemSetAccess with the physical allocation. Calling it
# again through another VA alias returns hipErrorInvalidValue.
ACCESS_IS_ALLOCATION_SCOPED = True


class HipVmmUnavailableError(RuntimeError):
    """Raised when the ROCm ``sgl_kernel`` VMM operators cannot be loaded."""


class HipVmmCleanupError(RuntimeError):
    """Raised after all possible cleanup actions were attempted."""

    def __init__(self, context: str, errors: Sequence[BaseException]):
        self.context = context
        self.errors = tuple(errors)
        details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"{context}: {details}")


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    return (value // alignment) * alignment


@functools.lru_cache(maxsize=1)
def _ops():
    if torch.version.hip is None:
        raise HipVmmUnavailableError("HIP VMM requires a ROCm build of PyTorch")

    try:
        import sgl_kernel  # noqa: F401
    except Exception as error:
        raise HipVmmUnavailableError(
            f"failed to load the ROCm sgl_kernel extension: {error}"
        ) from error

    namespace = torch.ops.sgl_kernel
    missing = [name for name in _REQUIRED_OPS if not hasattr(namespace, name)]
    if missing:
        raise HipVmmUnavailableError(
            "the loaded sgl_kernel extension has no HIP VMM operators: "
            + ", ".join(missing)
        )
    return namespace


def extension_availability() -> Tuple[bool, str]:
    """Return whether the ROCm extension and all HIP VMM operators are usable."""

    try:
        _ops()
    except HipVmmUnavailableError as error:
        return False, str(error)
    return True, ""


def is_supported(device_id: int) -> bool:
    """Return the HIP device's VMM capability attribute."""

    return bool(_ops().hip_vmm_is_supported(device_id))


@functools.lru_cache(maxsize=None)
def get_allocation_granularity(
    device_id: int, *, shareable: bool = True, recommended: bool = True
) -> int:
    return int(
        _ops().hip_vmm_get_allocation_granularity(device_id, shareable, recommended)
    )


def create_shareable_handle(size: int, device_id: int) -> int:
    """Create a HIP allocation exportable as a POSIX file descriptor."""

    return int(_ops().hip_vmm_create(size, device_id, True))


def create_local_handle(size: int, device_id: int) -> int:
    """Create a non-exportable HIP allocation."""

    return int(_ops().hip_vmm_create(size, device_id, False))


def release_handle(handle: int) -> None:
    if handle != 0:
        _ops().hip_vmm_release(handle)


def reserve_va(size: int, alignment: int = 0, requested_address: int = 0) -> int:
    return int(_ops().hip_vmm_address_reserve(size, alignment, requested_address))


def free_va(address: int, size: int) -> None:
    if address != 0:
        _ops().hip_vmm_address_free(address, size)


def map_handle(address: int, size: int, handle: int, offset: int = 0) -> None:
    _ops().hip_vmm_map(address, size, handle, offset)


def unmap_va(address: int, size: int) -> None:
    if address != 0:
        _ops().hip_vmm_unmap(address, size)


def set_access(address: int, size: int, device_id: int) -> None:
    _ops().hip_vmm_set_access(address, size, device_id)


def export_fd(handle: int) -> int:
    """Export ``handle`` to a new caller-owned POSIX file descriptor."""

    return int(_ops().hip_vmm_export_fd(handle))


def import_fd(fd: int) -> int:
    """Import ``fd`` without consuming or closing the caller-owned descriptor."""

    if fd < 0:
        raise ValueError(f"fd must be non-negative, got {fd}")
    duplicate = os.dup(fd)
    try:
        return int(_ops().hip_vmm_import_fd(duplicate))
    finally:
        os.close(duplicate)


def tensor_from_ptr(
    ptr: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device_id: int,
) -> torch.Tensor:
    """Wrap mapped external VA in a non-owning, contiguous HIP tensor."""

    return _ops().hip_vmm_tensor_from_address(ptr, list(shape), dtype, device_id)


@dataclass
class VmmMapping:
    """Own a VA reservation and its mapped regions, but not allocation handles."""

    address: int
    size: int
    _mapped_regions: list[tuple[int, int]] = field(default_factory=list)

    @property
    def mapped_regions(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._mapped_regions)

    @property
    def closed(self) -> bool:
        return self.address == 0

    def close(self) -> None:
        """Unmap every region in reverse order, then free the reservation."""

        if self.closed:
            return

        errors: list[BaseException] = []
        for index in range(len(self._mapped_regions) - 1, -1, -1):
            region_address, region_size = self._mapped_regions[index]
            try:
                unmap_va(region_address, region_size)
            except BaseException as error:
                errors.append(error)
            else:
                del self._mapped_regions[index]

        if not self._mapped_regions:
            try:
                free_va(self.address, self.size)
            except BaseException as error:
                errors.append(error)
            else:
                self.address = 0

        if errors:
            raise HipVmmCleanupError("HIP VMM mapping cleanup failed", errors)

    def __enter__(self) -> VmmMapping:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is None:
                raise
            raise HipVmmCleanupError(
                "HIP VMM cleanup failed while handling another exception",
                (exc_value, cleanup_error),
            ) from exc_value
        return False


def map_handles(
    handles: Sequence[int],
    sizes: Sequence[int],
    device_id: int,
    *,
    alignment: int = 0,
) -> VmmMapping:
    """Map handles contiguously and roll back every completed step on failure."""

    if not handles:
        raise ValueError("at least one allocation handle is required")
    if len(handles) != len(sizes):
        raise ValueError(
            f"handles and sizes must have equal length, got {len(handles)} and {len(sizes)}"
        )
    if any(size <= 0 for size in sizes):
        raise ValueError(f"all mapping sizes must be positive, got {list(sizes)}")

    total_size = sum(sizes)
    if total_size > (1 << 63) - 1:
        raise OverflowError(f"total mapping size exceeds int64: {total_size}")

    address = reserve_va(total_size, alignment)
    mapping = VmmMapping(address=address, size=total_size)
    try:
        offset = 0
        for handle, size in zip(handles, sizes):
            region_address = address + offset
            map_handle(region_address, size, handle)
            mapping._mapped_regions.append((region_address, size))
            offset += size
        set_access(address, total_size, device_id)
    except BaseException as setup_error:
        try:
            mapping.close()
        except BaseException as cleanup_error:
            raise HipVmmCleanupError(
                "HIP VMM setup failed and rollback was incomplete",
                (setup_error, cleanup_error),
            ) from setup_error
        raise
    return mapping
