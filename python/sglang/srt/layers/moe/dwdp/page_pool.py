# Adapted from NVIDIA TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM)
"""Double-buffered pool of local VMM pages backing the remote regions of the composite VA."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from sglang.srt.utils.alignment import align_up

if TYPE_CHECKING:
    from sglang.srt.utils.cuda_vmm_utils import VmmReservation

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE_MULTIPLIER = 8


class PagePool:
    # local (non-fabric) handles avoid consuming NVLink routing table entries
    DEFAULT_PAGE_SIZE_MULTIPLIER = DEFAULT_PAGE_SIZE_MULTIPLIER

    def __init__(
        self,
        slot_sizes: List[int],
        device_id: int,
        granularity: Optional[int] = None,
        page_size: Optional[int] = None,
        *,
        allocate_page: Optional[Callable[[int], int]] = None,
        release_page: Optional[Callable[[int], None]] = None,
    ):
        self._device_id = device_id
        if allocate_page is None:
            from sglang.srt.utils.cuda_vmm_utils import (
                get_device_granularity,
                make_device_allocation_prop,
            )

            self._granularity = granularity or get_device_granularity(device_id)
            self._prop = make_device_allocation_prop(device_id, handle_types=None)
            allocate_page = self._allocate_cuda_page
        else:
            if not granularity or release_page is None:
                raise ValueError(
                    "Custom page allocation requires granularity and release"
                )
            self._granularity = granularity
        self._release_page = release_page or self._release_cuda_page

        if page_size is None:
            self._page_size = self.DEFAULT_PAGE_SIZE_MULTIPLIER * self._granularity
        else:
            self._page_size = page_size

        self._slot_sizes = list(slot_sizes)
        self._slot_pages = [
            align_up(sz, self._page_size) // self._page_size for sz in slot_sizes
        ]

        self._page_handles: List[List[int]] = []
        self._released = False

        for slot_idx, num_pages in enumerate(self._slot_pages):
            handles = []
            for _ in range(num_pages):
                handles.append(int(allocate_page(self._page_size)))
            self._page_handles.append(handles)
            logger.debug(
                f"PagePool slot {slot_idx}: {num_pages} pages × {self._page_size} B"
            )

    def _allocate_cuda_page(self, size: int) -> int:
        from sglang.srt.utils.cuda_vmm_utils import VmmReservation

        reservation = VmmReservation(
            size, self._prop, self._device_id, alignment=self._granularity
        )
        handle = reservation.map(0, size, retain_handle=True)
        reservation.close(release_handles=False)
        return int(handle)

    @staticmethod
    def _release_cuda_page(handle: int) -> None:
        from cuda.bindings import driver as cuda

        from sglang.srt.utils.cuda_vmm_utils import check_drv

        check_drv(cuda.cuMemRelease(handle), "cuMemRelease")

    @classmethod
    def create(
        cls,
        slot_sizes: List[int],
        device_id: int,
        page_size: Optional[int] = None,
    ) -> PagePool:
        return cls(slot_sizes, device_id, page_size=page_size)

    @property
    def page_size(self) -> int:
        return self._page_size

    def num_pages(self, slot: int) -> int:
        return self._slot_pages[slot]

    def slot_size(self, slot: int) -> int:
        return self._slot_sizes[slot]

    def map_pages(
        self,
        slot: int,
        reservation: VmmReservation,
        offset: int,
        size: int,
        page_offset: int = 0,
    ) -> None:
        aligned_size = align_up(size, self._page_size)
        num_pages_needed = aligned_size // self._page_size

        for i in range(num_pages_needed):
            handle = self._page_handles[slot][page_offset + i]
            reservation.map_existing(
                offset + i * self._page_size,
                self._page_size,
                handle,
            )

    def map_weight_pages(self, slot, reservation, layout, local_handle, page_offset=0):
        """Map the remote prefix, resident local shard, then remote suffix."""
        if layout.pre_size > 0:
            self.map_pages(
                slot=slot,
                reservation=reservation,
                offset=0,
                size=layout.pre_size,
                page_offset=page_offset,
            )
            page_offset += layout.pre_pages

        reservation.map_existing(layout.pre_size, layout.mnnvl_size, local_handle)

        if layout.post_size > 0:
            self.map_pages(
                slot=slot,
                reservation=reservation,
                offset=layout.pre_size + layout.mnnvl_size,
                size=layout.post_size,
                page_offset=page_offset,
            )
            page_offset += layout.post_pages
        return page_offset

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for handles in self._page_handles:
            for h in handles:
                self._release_page(h)
        self._page_handles = [[], []]


def compute_slot_sizes(
    layouts: Dict[int, Dict[str, PageAlignedLayout]],  # noqa: F821
    buffer_slot_assignments: Dict[int, int],
) -> List[int]:
    slot_sizes = [0, 0]
    for layer_idx, weight_layouts in layouts.items():
        slot = buffer_slot_assignments.get(layer_idx, layer_idx % 2)
        total = sum(lo.pre_size + lo.post_size for lo in weight_layouts.values())
        slot_sizes[slot] = max(slot_sizes[slot], total)
    return slot_sizes
