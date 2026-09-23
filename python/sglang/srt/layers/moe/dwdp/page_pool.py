# Adapted from NVIDIA TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM)
"""Double-buffered pool of local VMM pages backing the remote regions of the composite VA."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import msgspec

from sglang.srt.utils.vmm_backend import get_vmm_backend
from sglang.srt.utils.vmm_common import Reservation, align_up

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE_MULTIPLIER = 8


class PoolBinding(msgspec.Struct, frozen=True):
    """One run of pool pages inside a composite VA: ``num_pages`` pages starting at
    page ``page_offset`` of ``slot``, mapped at ``offset`` in ``reservation``."""

    slot: int
    reservation: Reservation
    offset: int
    num_pages: int
    page_offset: int


class PagePool:
    DEFAULT_PAGE_SIZE_MULTIPLIER = DEFAULT_PAGE_SIZE_MULTIPLIER

    def __init__(
        self,
        slot_sizes: List[int],
        device_id: int,
        granularity: Optional[int] = None,
        page_size: Optional[int] = None,
    ):
        self._device_id = device_id
        self._backend = get_vmm_backend(device_id)
        self._granularity = granularity or self._backend.granularity

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
                # pool pages stay local to this rank, so exporting them would spend
                # a scarce per-device peer-sharing resource for nothing
                reservation = self._backend.make_reservation(
                    self._page_size,
                    exportable=False,
                    alignment=self._granularity,
                )
                handle = reservation.map(0, self._page_size, retain_handle=True)
                reservation.close(release_handles=False)
                handles.append(int(handle))
            self._page_handles.append(handles)
            logger.debug(
                f"PagePool slot {slot_idx}: {num_pages} pages × {self._page_size} B"
            )

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

    def map_binding(self, binding: PoolBinding) -> None:
        for i in range(binding.num_pages):
            handle = self._page_handles[binding.slot][binding.page_offset + i]
            binding.reservation.map_existing(
                binding.offset + i * self._page_size,
                self._page_size,
                handle,
            )

    def unmap_binding(self, binding: PoolBinding) -> None:
        for i in range(binding.num_pages):
            binding.reservation.unmap_existing(
                binding.offset + i * self._page_size,
                self._page_size,
            )

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for handles in self._page_handles:
            for h in handles:
                self._backend.release_handle(h)
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
