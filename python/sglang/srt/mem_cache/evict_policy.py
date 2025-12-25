from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(
        self, node: "TreeNode", now_time: Optional[float] = None
    ) -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode", now_time: Optional[float] = None) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(
        self, node: "TreeNode", now_time: Optional[float] = None
    ) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode", now_time: Optional[float] = None) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode", now_time: Optional[float] = None) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode", now_time: Optional[float] = None) -> float:
        return -node.creation_time


class AdapterMixStrategy(EvictionStrategy):

    def __init__(self):
        self.cold_count = 0
        self.hot_count = 0
        self.last_update_time = 0
        self.time_threshold = 180
        self.min_time_threshold = 30
        self.max_time_threshold = 600

    def get_priority(
        self, node: "TreeNode", now_time: Optional[float] = None
    ) -> Tuple[int, float]:
        if now_time is None:
            now_time = time.monotonic()
        self.update_threshold(now_time)

        if now_time - node.last_access_time > self.time_threshold:
            self.cold_count += 1
            return (-1, node.last_access_time)
        self.hot_count += 1
        return (node.hit_count, node.last_access_time)

    def update_threshold(self, now_time: float):

        if now_time - self.last_update_time < 60:
            return
        if self.last_update_time == 0:
            self.last_update_time = now_time
            return
        cold_count = self.cold_count
        hot_count = self.hot_count

        self.hot_count = 0
        self.cold_count = 0
        self.last_update_time = now_time

        if cold_count > hot_count:
            if hot_count == 0 or cold_count / hot_count >= 3:
                self.time_threshold = max(
                    self.min_time_threshold, self.time_threshold - 30
                )
            return

        if hot_count > cold_count:
            if cold_count == 0 or hot_count / cold_count >= 3:
                self.time_threshold = min(
                    self.max_time_threshold, self.time_threshold + 30
                )

        return


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(
        self, node: "TreeNode", now_time: Optional[float] = None
    ) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)
