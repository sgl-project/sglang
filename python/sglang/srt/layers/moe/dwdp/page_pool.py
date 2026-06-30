"""Page pool for DWDP remote double buffer.

Two pools (slot 0, slot 1) of local VMM handles for the pre/post remote
regions of the composite VA.  Pages are reused across layers in the same
slot via double buffering.

Ported from TensorRT-LLM ``_torch/modules/dwdp/page_pool.py``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from sglang.srt.layers.moe.dwdp.vmm import (
    align_up,
    create_local_handle,
    get_allocation_granularity,
    map_handle,
    release_handle,
)

logger = logging.getLogger(__name__)

# Default pool-page multiplier (each handle = 8 * granularity ≈ 16 MB on GB200).
DEFAULT_PAGE_SIZE_MULTIPLIER = 8


class PagePool:
    """Pool of local (non-fabric) page handles for remote double buffer.

    Each pool handle is ``page_size`` bytes.  Using local handles (not fabric)
    avoids consuming NVLink routing table entries.
    """

    __slots__ = (
        "_device_id",
        "_granularity",
        "_page_size",
        "_slot_sizes",
        "_slot_pages",
        "_page_handles",
        "_released",
    )

    DEFAULT_PAGE_SIZE_MULTIPLIER = DEFAULT_PAGE_SIZE_MULTIPLIER

    def __init__(
        self,
        slot_sizes: List[int],
        device_id: int,
        granularity: Optional[int] = None,
        page_size: Optional[int] = None,
    ):
        self._device_id = device_id
        self._granularity = granularity or get_allocation_granularity(device_id)

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
                h = create_local_handle(self._page_size, device_id)
                handles.append(h)
            self._page_handles.append(handles)
            logger.debug(
                f"[PagePool] slot {slot_idx}: {num_pages} pages × {self._page_size} B"
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

    def map_pages(
        self,
        slot: int,
        va_start: int,
        size: int,
        page_offset: int = 0,
    ) -> List[Tuple[int, int]]:
        """Map pool pages into a VA region.  Does NOT call set_access."""
        aligned_size = align_up(size, self._page_size)
        num_pages_needed = aligned_size // self._page_size

        mappings = []
        for i in range(num_pages_needed):
            va = va_start + i * self._page_size
            handle = self._page_handles[slot][page_offset + i]
            map_handle(va, self._page_size, handle, offset=0)
            mappings.append((va, self._page_size))
        return mappings

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for handles in self._page_handles:
            for h in handles:
                try:
                    release_handle(h)
                except Exception:
                    pass
        self._page_handles = [[], []]

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def compute_slot_sizes(
    layouts: Dict[int, Dict[str, PageAlignedLayout]],  # noqa: F821
    buffer_slot_assignments: Dict[int, int],
) -> List[int]:
    """Compute required [slot0_size, slot1_size] from layouts."""
    slot_sizes = [0, 0]
    for layer_idx, weight_layouts in layouts.items():
        slot = buffer_slot_assignments.get(layer_idx, layer_idx % 2)
        total = sum(l.pre_size + l.post_size for l in weight_layouts.values())
        slot_sizes[slot] = max(slot_sizes[slot], total)
    return slot_sizes
