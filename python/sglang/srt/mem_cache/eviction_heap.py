from __future__ import annotations

import heapq
from typing import Callable, Generic, Hashable, Optional, TypeVar

T = TypeVar("T", bound=Hashable)


class LazyEvictionHeap(Generic[T]):
    """A bounded eviction queue with lazy deletion and priority validation.

    Additions and priority increases need no eager heap maintenance. Call
    ``refresh`` after a priority decrease so older versions lose immediately.
    Removed nodes and obsolete versions are discarded when reached.
    """

    def __init__(
        self,
        *,
        priority: Callable[[T], object],
        candidate: Callable[[T], bool],
        active_count: Callable[[], int],
        minimum_compaction_size: int = 1024,
    ):
        self._priority = priority
        self._candidate = candidate
        self._active_count = active_count
        self._minimum_compaction_size = minimum_compaction_size
        self._heap: list[tuple[object, int, T]] = []
        self._queued: set[T] = set()
        self._versions: dict[T, int] = {}

    def __len__(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._queued.clear()
        self._versions.clear()

    def add(self, node: T) -> None:
        if node in self._queued:
            return
        self._queued.add(node)
        version = self._versions.get(node, 0)
        self._versions[node] = version
        heapq.heappush(self._heap, (self._priority(node), version, node))

    def refresh(self, node: T) -> None:
        if node not in self._queued:
            if self._candidate(node):
                self.add(node)
            return
        version = self._versions[node] + 1
        self._versions[node] = version
        heapq.heappush(self._heap, (self._priority(node), version, node))

    def _compact_if_needed(self) -> None:
        if len(self._heap) <= max(
            self._minimum_compaction_size, 2 * self._active_count()
        ):
            return
        self._queued = {node for node in self._queued if self._candidate(node)}
        self._versions = {node: 0 for node in self._queued}
        self._heap = [(self._priority(node), 0, node) for node in self._queued]
        heapq.heapify(self._heap)

    def pop(self) -> Optional[T]:
        self._compact_if_needed()
        while self._heap:
            queued_priority, version, node = heapq.heappop(self._heap)
            if self._versions.get(node) != version:
                continue
            self._queued.remove(node)
            if not self._candidate(node):
                continue
            current_priority = self._priority(node)
            if queued_priority != current_priority:
                self.add(node)
                continue
            return node
        return None
