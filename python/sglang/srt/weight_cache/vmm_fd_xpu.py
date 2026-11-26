# SPDX-License-Identifier: Apache-2.0
"""The ``vmm_fd`` weight cache transport on Intel XPU, over Level Zero.

The XPU half of the contract in ``transport.VmmFdTransportBase``: physical
memory from ``zePhysicalMemCreate``, mapped into a ``zeVirtualMemReserve`` range
and shared as the dma-buf fd ``zePhysicalMemGetProperties`` hands out. The driver
calls themselves are ``utils/xpu_vmm_utils``; this module is the weight-cache
policy on top of them, plus the one torch call that turns a mapped address into
a tensor.

XPU needs this transport rather than ``torch_ipc``: torch-xpu has no storage
reducer, so pickling an XPU tensor to another process fails outright
(``_share_fd_: only available on CPU``) until pytorch/pytorch#191725 lands.

Four Level Zero facts shape the code; the first three are measured on Arc Pro
B60 (device 0xE211, driver 1.15.39122+11) and documented in ``xpu_vmm_utils``:

- One physical handle carries one live mapping, so an allocation is mapped once
  and the slots inside it are addressed as offsets from that mapping. Packing
  already works that way, so no slot needs a mapping of its own.
- Device memory is committed at ``zeVirtualMemMap``, not at
  ``zePhysicalMemCreate``: a create of 256 GiB on a 24 GiB card succeeds. The
  map answers ERROR_OUT_OF_DEVICE_MEMORY instead, which is where the capacity
  is really checked, so a failure there is reported as the capacity problem it
  is.
- ``zeVirtualMemQueryPageSize`` returns a page size that grows with the
  requested size, and the reservation's -- queried for the whole span -- bounds
  every map offset inside it. One alignment, probed past that growth, therefore
  covers the reservation and every allocation in it.
- The importer holds its own physical handle for the dma-buf, so the buffer
  outlives the daemon: an engine keeps serving after the daemon exits, as on
  CUDA.
"""

from __future__ import annotations

import logging
import os
from typing import List, Sequence

import torch

from sglang.srt.utils import xpu_vmm_utils as xpu_vmm

from .transport import VmmFdTransportBase, _prefix_offsets
from .vmm_layout import SLOT_ALIGNMENT

logger = logging.getLogger(__name__)

# Size the allocation alignment is probed at. zeVirtualMemQueryPageSize answers
# 64 KiB below 2 MiB and 2 MiB from there up to 1 TiB on Arc Pro B60, so a probe
# at the top of that range yields the largest page size the driver asks for, and
# one alignment serves the reservation and every allocation in it. Over-aligning
# an allocation costs at most one page of padding per allocation.
_GRANULARITY_PROBE_BYTES = 1 << 40


def tensor_from_pointer(pointer: int, nbytes: int, *, device_id: int) -> torch.Tensor:
    """A uint8 tensor over ``nbytes`` at ``pointer`` on XPU ``device_id``.

    The storage is non-owning, as on CUDA: freeing the tensor does not unmap the
    pages, and the caller controls their lifetime. Kept here rather than in
    ``xpu_vmm_utils`` so that module stays a torch-free ctypes layer.
    """
    device = torch.device("xpu", device_id)
    storage = torch._C._construct_storage_from_data_pointer(pointer, device, nbytes)
    return torch.empty(0, dtype=torch.uint8, device=device).set_(storage, 0, (nbytes,))


def allocation_granularity(device_index: int) -> int:
    """Alignment for the reservation and for every allocation inside it."""
    granularity = xpu_vmm.query_page_size(
        _GRANULARITY_PROBE_BYTES, device_index=device_index
    )
    if granularity % SLOT_ALIGNMENT:
        raise RuntimeError(
            f"[weight_cache] vmm_fd: the Level Zero page size {granularity} is "
            f"not a multiple of the {SLOT_ALIGNMENT}-byte slot alignment, so a "
            f"packed tensor could not keep the alignment it was loaded with."
        )
    return granularity


def check_device_indices() -> None:
    """Refuse a mismatch between the torch and Level Zero device lists.

    Allocations are addressed by the torch device index, and ``xpu_vmm_utils``
    indexes the driver's own list. ZE_AFFINITY_MASK can shorten one without the
    other, which would silently put the weights on another card.
    """
    torch_count = torch.xpu.device_count()
    level_zero_count = xpu_vmm.device_count()
    if torch_count != level_zero_count:
        raise RuntimeError(
            f"[weight_cache] vmm_fd: torch sees {torch_count} XPU device(s) and "
            f"the Level Zero driver {level_zero_count}, so a torch device index "
            f"does not name the same card in both. Unset ZE_AFFINITY_MASK, or "
            f"set it so both lists match."
        )


class XpuVmmFdTransportBackend(VmmFdTransportBase):
    """``vmm_fd`` over Level Zero virtual memory on Intel XPU."""

    device_type = "xpu"

    @classmethod
    def for_export(cls, device_id: int) -> XpuVmmFdTransportBackend:
        """Daemon-side backend for ``device_id``.

        Probes the driver first -- create one exportable page, hand out its
        dma-buf fd and map it -- so a driver that cannot share physical memory
        fails before the model is loaded rather than after.
        """
        check_device_indices()
        reason = xpu_vmm.vmm_unsupported_reason(device_index=device_id)
        if reason is not None:
            raise RuntimeError(
                f"[weight_cache] vmm_fd needs Level Zero virtual memory with "
                f"dma-buf export, which is unavailable on XPU device "
                f"{device_id}: {reason}"
            )
        return cls(
            device=torch.device("xpu", device_id),
            granularity=allocation_granularity(device_id),
        )

    @property
    def _device_index(self) -> int:
        return self._device.index

    def _current_device(self) -> torch.device:
        """The device an engine maps the daemon's allocations into."""
        return torch.device("xpu", torch.xpu.current_device())

    @property
    def _reservation_base(self) -> int:
        return self._reservation.ptr

    def _empty_cache(self) -> None:
        torch.xpu.empty_cache()

    def _synchronize(self) -> None:
        torch.xpu.synchronize(self._device)

    def _free_device_bytes(self) -> int:
        free_bytes, _ = torch.xpu.mem_get_info(self._device_index)
        return int(free_bytes)

    def _reserve(self, total_bytes: int) -> xpu_vmm.VirtualReservation:
        reservation = xpu_vmm.VirtualReservation(
            total_bytes, device_index=self._device_index
        )
        if self._granularity % reservation.page_size:
            reservation.close()
            raise RuntimeError(
                f"[weight_cache] vmm_fd: the Level Zero page size for a "
                f"{total_bytes}-byte reservation is {reservation.page_size}, "
                f"which does not divide the {self._granularity}-byte allocation "
                f"alignment, so the allocations would not land on its pages."
            )
        return reservation

    def _create_and_map(self, offset: int, size: int) -> xpu_vmm.PhysicalMemory:
        physical = xpu_vmm.PhysicalMemory.create(
            size, device_index=self._device_index, exportable=True
        )
        if physical.size != size:
            # create() pads to its own page size. The padded bytes would be
            # allocated but unmapped, and the fd would carry a buffer larger
            # than the size the engine imports it with.
            physical.close()
            raise RuntimeError(
                f"[weight_cache] vmm_fd: Level Zero padded a {size}-byte "
                f"allocation to {physical.size} bytes, so the "
                f"{self._granularity}-byte alignment is below its page size "
                f"{physical.page_size}."
            )
        try:
            self._reservation.map(physical, offset=offset, size=size)
        except xpu_vmm.ZeVmmError as exc:
            physical.close()
            raise RuntimeError(
                f"[weight_cache] vmm_fd: could not map a {size}-byte allocation "
                f"at offset {offset} on XPU device {self._device_index} with "
                f"{self._free_device_bytes()} bytes free. Level Zero commits "
                f"device memory here, not at zePhysicalMemCreate, so the "
                f"allocation succeeding said nothing about capacity: {exc}"
            ) from exc
        except BaseException:
            physical.close()
            raise
        return physical

    def _tensor_from_pointer(self, pointer: int, nbytes: int) -> torch.Tensor:
        return tensor_from_pointer(pointer, nbytes, device_id=self._device_index)

    def _export_fds(self) -> List[int]:
        """Export a fresh dma-buf fd per allocation; the caller closes them."""
        fds: List[int] = []
        try:
            for physical in self._handles:
                fds.append(physical.export_dma_buf())
        except BaseException:
            for fd in fds:
                os.close(fd)
            raise
        return fds

    def _map_imported(self, fds: Sequence[int], sizes: Sequence[int]) -> None:
        if self._reservation is not None:
            raise RuntimeError("vmm_fd backend already holds a mapping")
        check_device_indices()
        self._device = self._current_device()
        device_id = self._device_index
        if not fds:
            return
        granularity = allocation_granularity(device_id)
        misaligned = [size for size in sizes if size % granularity]
        if misaligned:
            raise RuntimeError(
                f"vmm_fd allocation sizes {misaligned} are not multiples of the "
                f"{granularity}-byte Level Zero page size on this device; the "
                f"daemon packed them for a device with a different page size."
            )

        offsets = _prefix_offsets(sizes)
        physicals: List[xpu_vmm.PhysicalMemory] = []
        reservation = None
        try:
            # The fds stay the caller's: from_dma_buf imports a dup of each.
            for fd, size in zip(fds, sizes):
                physicals.append(
                    xpu_vmm.PhysicalMemory.from_dma_buf(
                        fd, size, device_index=device_id
                    )
                )
            reservation = xpu_vmm.VirtualReservation(sum(sizes), device_index=device_id)
            for physical, offset, size in zip(physicals, offsets, sizes):
                reservation.map(physical, offset=offset, size=size)
        except BaseException:
            _unwind_import(reservation, physicals)
            raise

        # Unlike CUDA, where a mapping holds its own reference and the imported
        # handle can go, Level Zero keeps the dma-buf alive through the handle:
        # it is what lets these weights outlive the daemon, so it is retained
        # and destroyed in close(), after the mapping.
        self._reservation = reservation
        self._handles = physicals
        self._alloc_sizes = list(sizes)
        self._alloc_offsets = offsets
        logger.info(
            "[weight_cache] vmm_fd: mapped %d allocation(s), %.2f GiB, at 0x%x",
            len(sizes),
            sum(sizes) / (1 << 30),
            reservation.ptr,
        )

    def _release_handles(self, handles: Sequence[xpu_vmm.PhysicalMemory]) -> None:
        _close_physicals(handles)


def _close_physicals(physicals: Sequence[xpu_vmm.PhysicalMemory]) -> None:
    """Destroy every allocation, carrying on past a failure.

    An allocation that refuses to go -- one whose mapping is still live -- must
    not strand the rest, so the refusal is logged rather than raised.
    """
    for physical in physicals:
        try:
            physical.close()
        except xpu_vmm.ZeVmmError as exc:
            logger.warning("[weight_cache] vmm_fd: zePhysicalMemDestroy: %s", exc)


def _unwind_import(reservation, physicals: Sequence[xpu_vmm.PhysicalMemory]) -> None:
    """Drop a half-built import without displacing the failure that caused it."""
    try:
        if reservation is not None:
            reservation.close()
    except xpu_vmm.ZeVmmError as exc:
        logger.warning("[weight_cache] vmm_fd: unwinding a failed import: %s", exc)
    finally:
        _close_physicals(physicals)
