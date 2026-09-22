"""One device-bound facade over the CUDA and Level Zero VMM/IPC backends.

Both drivers offer the same three primitives (reserve VA, create physical
memory, map physical into VA) but spell them differently and disagree on who
owns an exported fd, so callers use ``get_vmm_backend(device_id)`` instead of
importing a driver module.
"""

from __future__ import annotations

from functools import cache
from typing import List, Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from sglang.srt.utils.common import is_xpu


class VmmBackend:
    """VMM + handle-IPC operations bound to one device.

    ``granularity`` is the page every reservation, offset and mapping size aligns
    to. Read it: CUDA reports it per device, Level Zero per allocation size.
    """

    def __init__(self, device_id: int) -> None:
        self.device_id = int(device_id)
        self.granularity = self._impl_granularity()

    # ===== Reservations and physical memory =====

    def make_reservation(self, size: int, *, exportable: bool, alignment: int = 0):
        """Reserve VA for ``size`` bytes. ``exportable`` decides whether physical
        memory created through ``reservation.map`` can be shared with peers."""
        raise NotImplementedError

    def release_handle(self, handle: int) -> None:
        """Release one physical-memory handle. All its mappings must be gone."""
        raise NotImplementedError

    def tensor_from_pointer(
        self,
        pointer: int,
        nbytes: int,
        *,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: torch.dtype = torch.uint8,
    ) -> torch.Tensor:
        """Wrap a mapped VA range in a tensor with non-owning storage."""
        raise NotImplementedError

    def copy_tensor_to_pointer(self, pointer: int, source: torch.Tensor) -> None:
        """Copy ``source`` into a mapped VA range, densely and in row-major order.

        A tensor view, not a byte memcpy: the destination takes ``source``'s
        logical shape, so a strided source is densified rather than truncated.
        """
        destination = self.tensor_from_pointer(
            pointer,
            source.numel() * source.element_size(),
            shape=tuple(source.shape),
            dtype=source.dtype,
        )
        destination.copy_(source)

    # ===== Handle exchange =====

    def export_handles(
        self, handles: List[int], group: ProcessGroup, rank: int
    ) -> Tuple[List[bytes], List[int], bool]:
        """Export handles as ``(fabric_handles, posix_fds, use_fabric)``; exactly
        one of the two lists is populated."""
        raise NotImplementedError

    def import_handle(
        self,
        fabric_handle,
        fd: Optional[int],
        *,
        use_fabric: bool,
        peer_rank: int,
        size: int,
    ) -> int:
        """Import one peer handle, returning a handle owned by this process.

        ``size`` must be the exporter's page-aligned size; a Level Zero opaque fd
        does not carry it, CUDA's shareable handles do.
        """
        raise NotImplementedError

    def owns_exported_fds(self) -> bool:
        """Whether the caller must close the fds from ``export_handles``.

        CUDA hands out a fresh fd; Level Zero's is "Owned by the driver; must not
        be closed directly by the application" (ze_api.h 1.18).
        """
        raise NotImplementedError

    def supports_aliased_mappings(self) -> bool:
        """Whether one physical object may sit in two live VA ranges at once.

        Level Zero returns success from the second zeVirtualMemMap but any access
        then fails with DEVICE_LOST, so pages are remapped, not aliased.
        """
        raise NotImplementedError

    # ===== Device control =====

    def synchronize(self) -> None:
        torch.get_device_module(self.torch_device).synchronize(self.device_id)

    def empty_cache(self) -> None:
        torch.get_device_module(self.torch_device).empty_cache()

    @property
    def torch_device(self) -> torch.device:
        raise NotImplementedError

    def _impl_granularity(self) -> int:
        raise NotImplementedError


class CudaVmmBackend(VmmBackend):
    @property
    def torch_device(self) -> torch.device:
        return torch.device("cuda", self.device_id)

    def _impl_granularity(self) -> int:
        from sglang.srt.utils.cuda_vmm_utils import get_device_granularity

        return get_device_granularity(self.device_id)

    def make_reservation(self, size: int, *, exportable: bool, alignment: int = 0):
        from sglang.srt.utils.cuda_vmm_utils import (
            VmmReservation,
            make_device_allocation_prop,
        )

        prop = make_device_allocation_prop(
            self.device_id, handle_types="auto" if exportable else None
        )
        return VmmReservation(size, prop, self.device_id, alignment=alignment)

    def release_handle(self, handle: int) -> None:
        from cuda.bindings import driver as cuda

        from sglang.srt.utils.cuda_vmm_utils import check_drv

        check_drv(cuda.cuMemRelease(handle), "cuMemRelease")

    def tensor_from_pointer(
        self,
        pointer: int,
        nbytes: int,
        *,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: torch.dtype = torch.uint8,
    ) -> torch.Tensor:
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        return tensor_from_pointer(
            pointer, nbytes, shape=shape, dtype=dtype, device_id=self.device_id
        )

    def export_handles(
        self, handles: List[int], group: ProcessGroup, rank: int
    ) -> Tuple[List[bytes], List[int], bool]:
        from sglang.srt.utils.cuda_vmm_utils import export_shareable_handles

        return export_shareable_handles(handles, group, rank)

    def import_handle(
        self,
        fabric_handle,
        fd: Optional[int],
        *,
        use_fabric: bool,
        peer_rank: int,
        size: int,
    ) -> int:
        # cuMemImportFromShareableHandle reads the size out of the handle itself.
        del size
        from sglang.srt.utils.cuda_vmm_utils import import_peer_handle

        return int(
            import_peer_handle(
                fabric_handle, fd, use_fabric=use_fabric, peer_rank=peer_rank
            )
        )

    def owns_exported_fds(self) -> bool:
        return True

    def supports_aliased_mappings(self) -> bool:
        return True


class XpuVmmBackend(VmmBackend):
    @property
    def torch_device(self) -> torch.device:
        return torch.device("xpu", self.device_id)

    def _impl_granularity(self) -> int:
        from sglang.srt.utils.xpu_vmm_utils import get_device_granularity

        return get_device_granularity(self.device_id)

    def make_reservation(self, size: int, *, exportable: bool, alignment: int = 0):
        from sglang.srt.utils.xpu_vmm_utils import (
            VmmReservation,
            make_device_allocation_prop,
        )

        prop = make_device_allocation_prop(
            self.device_id, handle_types="auto" if exportable else None
        )
        return VmmReservation(size, prop, self.device_id, alignment=alignment)

    def release_handle(self, handle: int) -> None:
        from sglang.srt.utils.xpu_vmm_utils import release_physical_mem

        release_physical_mem(handle, self.device_id)

    def tensor_from_pointer(
        self,
        pointer: int,
        nbytes: int,
        *,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: torch.dtype = torch.uint8,
    ) -> torch.Tensor:
        from sglang.srt.utils.xpu_vmm_utils import tensor_from_pointer

        return tensor_from_pointer(
            pointer, nbytes, shape=shape, dtype=dtype, device_id=self.device_id
        )

    def export_handles(
        self, handles: List[int], group: ProcessGroup, rank: int
    ) -> Tuple[List[bytes], List[int], bool]:
        from sglang.srt.utils.xpu_vmm_utils import export_shareable_handles

        return export_shareable_handles(handles, group, rank, device_id=self.device_id)

    def import_handle(
        self,
        fabric_handle,
        fd: Optional[int],
        *,
        use_fabric: bool,
        peer_rank: int,
        size: int,
    ) -> int:
        from sglang.srt.utils.xpu_vmm_utils import import_peer_handle

        return import_peer_handle(
            fabric_handle,
            fd,
            use_fabric=use_fabric,
            peer_rank=peer_rank,
            device_id=self.device_id,
            size=size,
        )

    def owns_exported_fds(self) -> bool:
        return False

    def supports_aliased_mappings(self) -> bool:
        return False


@cache
def get_vmm_backend(device_id: int) -> VmmBackend:
    """The VMM backend for ``device_id`` on the current accelerator. Cached so
    granularity and the L0 context are resolved once per device."""
    if is_xpu():
        return XpuVmmBackend(device_id)
    return CudaVmmBackend(device_id)


__all__ = [
    "VmmBackend",
    "get_vmm_backend",
]
