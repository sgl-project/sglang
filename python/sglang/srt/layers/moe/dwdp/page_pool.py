"""Physical page pool for DWDP double-buffered remote weight regions."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.layers.moe.dwdp.common import align_up
from sglang.srt.layers.moe.dwdp.layout import PageAlignedLayout

logger = logging.getLogger(__name__)


class PagePool:
    DEFAULT_PAGE_SIZE_MULTIPLIER = 8

    def __init__(
        self,
        slot_sizes: List[int],
        device_id: int,
        granularity: Optional[int] = None,
        page_size: Optional[int] = None,
        vmm_ops: Any = None,
    ):
        if vmm_ops is None:
            from sglang.srt.layers.moe.dwdp import vmm as vmm_ops

        self._vmm_ops = vmm_ops
        self._device_id = device_id
        self._granularity = granularity or vmm_ops.get_allocation_granularity(device_id)
        self._page_size = (
            self.DEFAULT_PAGE_SIZE_MULTIPLIER * self._granularity
            if page_size is None
            else page_size
        )
        self._slot_sizes = list(slot_sizes)
        self._slot_pages = [
            align_up(size, self._page_size) // self._page_size for size in slot_sizes
        ]
        self._region_pool = getattr(
            self._vmm_ops,
            "ACCESS_IS_ALLOCATION_SCOPED",
            False,
        )

        self._page_handles: List[List[int]] = []
        self._region_handles: Dict[Tuple[int, int, int], int] = {}
        self._access_initialized: set[int] = set()
        self._released = False

        for slot_idx, num_pages in enumerate(self._slot_pages):
            handles = []
            if not self._region_pool:
                for _ in range(num_pages):
                    handles.append(
                        self._vmm_ops.create_local_handle(
                            self._page_size,
                            device_id,
                        )
                    )
            self._page_handles.append(handles)
            logger.debug(
                "PagePool slot %d: %d pages x %d B",
                slot_idx,
                num_pages,
                self._page_size,
            )

    @classmethod
    def create(
        cls,
        slot_sizes: List[int],
        device_id: int,
        page_size: Optional[int] = None,
        vmm_ops: Any = None,
    ) -> PagePool:
        return cls(
            slot_sizes,
            device_id,
            page_size=page_size,
            vmm_ops=vmm_ops,
        )

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
        va_start: int,
        size: int,
        page_offset: int = 0,
    ) -> List[Tuple[int, int]]:
        aligned_size = align_up(size, self._page_size)
        num_pages_needed = aligned_size // self._page_size

        if self._region_pool:
            # ROCm 7.x rejects non-zero hipMemMap offsets and has a low active
            # generic-allocation limit. Pool each stable pre/post region as one
            # allocation instead of creating one allocation per page.
            key = (slot, page_offset, aligned_size)
            handle = self._region_handles.get(key)
            is_new = handle is None
            if is_new:
                handle = self._vmm_ops.create_local_handle(
                    aligned_size,
                    self._device_id,
                )
                self._region_handles[key] = handle
                self._page_handles[slot].append(handle)

            self._vmm_ops.map_handle(
                va_start,
                aligned_size,
                handle,
                offset=0,
            )
            if handle not in self._access_initialized:
                try:
                    self._vmm_ops.set_access(
                        va_start,
                        aligned_size,
                        self._device_id,
                    )
                except Exception as error:
                    self._vmm_ops.unmap_va(va_start, aligned_size)
                    if is_new:
                        self._page_handles[slot].remove(handle)
                        self._region_handles.pop(key, None)
                        self._vmm_ops.release_handle(handle)
                    raise RuntimeError(
                        f"Failed to set region-pool VMM access: key={key}, "
                        f"handle={handle:#x}, va={va_start:#x}, "
                        f"size={aligned_size}"
                    ) from error
                self._access_initialized.add(handle)
            return [(va_start, aligned_size)]

        mappings = []
        for index in range(num_pages_needed):
            va = va_start + index * self._page_size
            handle = self._page_handles[slot][page_offset + index]
            self._vmm_ops.map_handle(
                va,
                self._page_size,
                handle,
                offset=0,
            )
            try:
                self._vmm_ops.set_access(
                    va,
                    self._page_size,
                    self._device_id,
                )
            except Exception:
                self._vmm_ops.unmap_va(va, self._page_size)
                raise
            mappings.append((va, self._page_size))
        return mappings

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for handles in self._page_handles:
            for handle in handles:
                self._vmm_ops.release_handle(handle)
        self._page_handles = [[], []]
        self._region_handles.clear()
        self._access_initialized.clear()


def compute_slot_sizes(
    layouts: Dict[int, Dict[str, PageAlignedLayout]],
    buffer_slot_assignments: Dict[int, int],
) -> List[int]:
    slot_sizes = [0, 0]
    for layer_idx, weight_layouts in layouts.items():
        slot = buffer_slot_assignments.get(layer_idx, layer_idx % 2)
        total = sum(
            layout.pre_size + layout.post_size for layout in weight_layouts.values()
        )
        slot_sizes[slot] = max(slot_sizes[slot], total)
    return slot_sizes
