# SPDX-License-Identifier: Apache-2.0
"""Buddy-system memory allocator for TransferTensorBuffer."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class Block:
    offset: int  # byte offset from pool start
    size: int
    allocated: bool = False
    request_id: str | None = None


class BuddyAllocator:
    """Power-of-2 buddy-system allocator for pinned memory."""

    def __init__(self, pool_size: int, min_block_size: int = 1 << 20):
        if min_block_size <= 0 or (min_block_size & (min_block_size - 1)) != 0:
            raise ValueError(
                f"min_block_size must be a power of 2, got {min_block_size}"
            )

        self._min_block_size = min_block_size
        self._pool_size = self._next_power_of_2(max(pool_size, min_block_size))
        self._lock = threading.Lock()

        # Free lists indexed by order: order 0 = min_block_size, order 1 = 2*min_block_size, ...
        self._max_order = self._size_to_order(self._pool_size)
        self._free_lists: list[list[int]] = [[] for _ in range(self._max_order + 1)]

        self._blocks: dict[int, Block] = {}

        root = Block(offset=0, size=self._pool_size)
        self._blocks[0] = root
        self._free_lists[self._max_order].append(0)

        self._allocated_bytes = 0
        self._num_allocations = 0

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def allocate(self, size: int, request_id: str | None = None) -> int | None:
        """Allocate a block of at least `size` bytes. Returns offset or None."""
        if size <= 0:
            raise ValueError(f"Allocation size must be positive, got {size}")

        alloc_size = max(self._next_power_of_2(size), self._min_block_size)
        target_order = self._size_to_order(alloc_size)

        if target_order > self._max_order:
            logger.warning(
                "Requested size %d exceeds pool size %d", size, self._pool_size
            )
            return None

        with self._lock:
            return self._allocate_locked(target_order, request_id)

    def free(self, offset: int) -> bool:
        """Free the block at the given offset and coalesce with buddy if possible."""
        with self._lock:
            return self._free_locked(offset)

    def get_block_info(self, offset: int) -> Block | None:
        with self._lock:
            return self._blocks.get(offset)

    def get_stats(self) -> dict:
        with self._lock:
            free_blocks_by_order = {}
            for order, offsets in enumerate(self._free_lists):
                if offsets:
                    block_size = self._min_block_size << order
                    free_blocks_by_order[block_size] = len(offsets)

            return {
                "pool_size": self._pool_size,
                "min_block_size": self._min_block_size,
                "allocated_bytes": self._allocated_bytes,
                "free_bytes": self._pool_size - self._allocated_bytes,
                "num_allocations": self._num_allocations,
                "num_blocks": len(self._blocks),
                "free_blocks_by_size": free_blocks_by_order,
            }

    def count_free_slots(self, slot_size: int) -> int:
        """Count how many allocations of the given size can fit."""
        if slot_size <= 0:
            return 0
        alloc_size = max(self._next_power_of_2(slot_size), self._min_block_size)

        with self._lock:
            count = 0
            for order in range(self._size_to_order(alloc_size), self._max_order + 1):
                for _ in self._free_lists[order]:
                    block_size = self._min_block_size << order
                    count += block_size // alloc_size
            return count

    # --- Internal (caller must hold self._lock) ---

    def _allocate_locked(self, target_order: int, request_id: str | None) -> int | None:
        found_order = -1
        for order in range(target_order, self._max_order + 1):
            if self._free_lists[order]:
                found_order = order
                break

        if found_order < 0:
            return None

        offset = self._free_lists[found_order].pop(0)
        block = self._blocks[offset]

        # Split down to target_order
        while found_order > target_order:
            found_order -= 1
            buddy_size = self._min_block_size << found_order
            buddy_offset = offset + buddy_size

            buddy = Block(offset=buddy_offset, size=buddy_size)
            self._blocks[buddy_offset] = buddy
            self._free_lists[found_order].append(buddy_offset)

            block.size = buddy_size

        block.allocated = True
        block.request_id = request_id
        self._allocated_bytes += block.size
        self._num_allocations += 1

        return offset

    def _free_locked(self, offset: int) -> bool:
        block = self._blocks.get(offset)
        if block is None or not block.allocated:
            return False

        block.allocated = False
        block.request_id = None
        self._allocated_bytes -= block.size
        self._num_allocations -= 1

        self._coalesce(block)
        return True

    def _coalesce(self, block: Block) -> None:
        """Recursively merge with buddy if both are free."""
        while block.size < self._pool_size:
            buddy_offset = block.offset ^ block.size
            buddy = self._blocks.get(buddy_offset)

            if buddy is None or buddy.allocated or buddy.size != block.size:
                break

            order = self._size_to_order(buddy.size)
            self._free_lists[order].remove(buddy_offset)

            if buddy_offset < block.offset:
                del self._blocks[block.offset]
                buddy.size *= 2
                block = buddy
            else:
                del self._blocks[buddy_offset]
                block.size *= 2

        order = self._size_to_order(block.size)
        self._free_lists[order].append(block.offset)

    def _size_to_order(self, size: int) -> int:
        order = 0
        s = self._min_block_size
        while s < size:
            s <<= 1
            order += 1
        return order

    @staticmethod
    @lru_cache(maxsize=256)
    def _next_power_of_2(n: int) -> int:
        if n <= 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        return n + 1
