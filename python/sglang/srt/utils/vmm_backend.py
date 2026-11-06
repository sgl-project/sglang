"""One device-bound facade over a driver's VMM and handle-IPC primitives.

Drivers offer the same primitives (reserve VA, create physical memory, map physical into VA) but spell them differently and disagree on who owns an exported fd, so callers use get_vmm_backend(device_id) and each platform names its backend in get_vmm_backend_class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache

import torch
from torch.distributed import ProcessGroup

from sglang.srt.platforms import current_platform
from sglang.srt.utils.cuda_vmm_utils import (
    VmmReservation,
    _get_cuda_driver,
    check_drv,
    export_shareable_handles,
    get_device_granularity,
    import_peer_handle,
    make_device_allocation_prop,
    tensor_from_pointer,
)
from sglang.srt.utils.vmm_common import (
    Reservation,
    exchange_exported_handles,
    visible_physical_device_ids,
)


class VmmBackend(ABC):
    """VMM and handle-IPC operations bound to one device; granularity is the page every reservation, offset and mapping size aligns to."""

    def __init__(self, device_id: int) -> None:
        self.device_id = int(device_id)
        self.granularity = self._impl_granularity()

    @abstractmethod
    def make_reservation(
        self, size: int, *, exportable: bool, alignment: int = 0
    ) -> Reservation:
        """Reserve VA for size bytes; exportable decides whether memory mapped into it can be shared with peers."""
        raise NotImplementedError

    @abstractmethod
    def release_handle(self, handle: int) -> None:
        """Release one physical-memory handle whose mappings are all gone."""
        raise NotImplementedError

    @abstractmethod
    def tensor_from_pointer(
        self,
        pointer: int,
        nbytes: int,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.uint8,
    ) -> torch.Tensor:
        """Wrap a mapped VA range in a tensor with non-owning storage."""
        raise NotImplementedError

    def copy_tensor_to_pointer(self, pointer: int, source: torch.Tensor) -> None:
        """Copy source into a mapped VA range in its logical shape, so a strided source is densified rather than truncated."""
        destination = self.tensor_from_pointer(
            pointer,
            source.numel() * source.element_size(),
            shape=tuple(source.shape),
            dtype=source.dtype,
        )
        destination.copy_(source)

    @abstractmethod
    def export_handles(
        self, handles: list[int], group: ProcessGroup, rank: int
    ) -> tuple[list[bytes], list[int], bool]:
        """Export handles as (fabric_handles, posix_fds, use_fabric); exactly one of the two lists is populated."""
        raise NotImplementedError

    @abstractmethod
    def import_handle(
        self,
        fabric_handle: bytes | None,
        fd: int | None,
        *,
        use_fabric: bool,
        peer_rank: int,
        size: int,
    ) -> int:
        """Import one peer handle; size is the exporter's page-aligned size, for drivers whose fd does not carry it."""
        raise NotImplementedError

    def exchange_handles(
        self, handles: list[int], group: ProcessGroup, rank: int, world_size: int
    ) -> tuple[bool, dict[tuple[int, int], bytes | int]]:
        """Export handles and collect every peer's as (use_fabric, {(peer_rank, idx): fabric handle or fd}); the caller closes received fds."""
        fabric_handles, local_fds, use_fabric = self.export_handles(
            handles, group, rank
        )
        return exchange_exported_handles(
            fabric_handles,
            local_fds,
            use_fabric,
            group,
            rank,
            world_size,
            close_local_fds=self.owns_exported_fds(),
        )

    def physical_device_ids(self) -> list[str]:
        """Physical ids of the devices this process sees, by local ordinal."""
        return visible_physical_device_ids(
            self.visible_devices_env,
            torch.get_device_module(self.torch_device).device_count(),
        )

    @abstractmethod
    def owns_exported_fds(self) -> bool:
        """Whether the caller must close the fds from export_handles."""
        raise NotImplementedError

    @abstractmethod
    def can_access_peer(self, peer_device_id: int) -> bool:
        """Whether this device can map memory owned by peer_device_id."""
        raise NotImplementedError

    @abstractmethod
    def supports_aliased_mappings(self) -> bool:
        """Whether one physical object may sit in two live VA ranges; False sends DWDP down the remap path."""
        raise NotImplementedError

    def synchronize(self) -> None:
        torch.get_device_module(self.torch_device).synchronize(self.device_id)

    def empty_cache(self) -> None:
        torch.get_device_module(self.torch_device).empty_cache()

    @property
    @abstractmethod
    def torch_device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def visible_devices_env(self) -> str:
        """Environment variable that maps physical devices to this process's ordinals."""
        raise NotImplementedError

    @abstractmethod
    def _impl_granularity(self) -> int:
        raise NotImplementedError


class CudaVmmBackend(VmmBackend):
    @property
    def torch_device(self) -> torch.device:
        return torch.device("cuda", self.device_id)

    @property
    def visible_devices_env(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    def _impl_granularity(self) -> int:
        return get_device_granularity(self.device_id)

    def make_reservation(
        self, size: int, *, exportable: bool, alignment: int = 0
    ) -> Reservation:
        prop = make_device_allocation_prop(
            self.device_id, handle_types="auto" if exportable else None
        )
        return VmmReservation(size, prop, self.device_id, alignment=alignment)

    def release_handle(self, handle: int) -> None:
        check_drv(_get_cuda_driver().cuMemRelease(handle), "cuMemRelease")

    def tensor_from_pointer(
        self,
        pointer: int,
        nbytes: int,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.uint8,
    ) -> torch.Tensor:
        return tensor_from_pointer(
            pointer, nbytes, shape=shape, dtype=dtype, device_id=self.device_id
        )

    def export_handles(
        self, handles: list[int], group: ProcessGroup, rank: int
    ) -> tuple[list[bytes], list[int], bool]:
        return export_shareable_handles(handles, group, rank)

    def import_handle(
        self,
        fabric_handle: bytes | None,
        fd: int | None,
        *,
        use_fabric: bool,
        peer_rank: int,
        size: int,
    ) -> int:
        # size is unused: cuMemImportFromShareableHandle reads it from the handle
        return int(
            import_peer_handle(
                fabric_handle, fd, use_fabric=use_fabric, peer_rank=peer_rank
            )
        )

    def owns_exported_fds(self) -> bool:
        return True

    def can_access_peer(self, peer_device_id: int) -> bool:
        return torch.cuda.can_device_access_peer(self.device_id, int(peer_device_id))

    def supports_aliased_mappings(self) -> bool:
        return True


@cache
def get_vmm_backend(device_id: int) -> VmmBackend:
    """The VMM backend the current platform provides for device_id, built once per device."""
    backend_class = current_platform.get_vmm_backend_class()
    if backend_class is None:
        raise RuntimeError(f"{current_platform.device_name} has no VMM backend")
    return backend_class(device_id)


__all__ = [
    "CudaVmmBackend",
    "VmmBackend",
    "get_vmm_backend",
]
