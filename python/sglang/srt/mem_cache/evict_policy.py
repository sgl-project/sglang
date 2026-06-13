from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction with optional time-based decay.

    Priority is clamped to [0, 99] (100 discrete levels). If a node has
    retention_duration > 0 and its priority > 0, the effective priority
    decays to 0 once elapsed time since last access exceeds
    retention_duration. This gives soft retention semantics: active
    conversations auto-extend via last_access_time refresh on cache hits.
    """

    MIN_PRIORITY = 0
    MAX_PRIORITY = 99

    @staticmethod
    def clamp_priority(priority: int) -> int:
        return max(
            PriorityStrategy.MIN_PRIORITY,
            min(PriorityStrategy.MAX_PRIORITY, priority),
        )

    @staticmethod
    def _retention_rank(retention_duration: Optional[float]) -> float:
        # 0/None means "never decay", which should dominate finite durations.
        if retention_duration is None or retention_duration <= 0:
            return float("inf")
        return retention_duration

    @classmethod
    def pick_stronger_policy(
        cls,
        current_priority: Optional[int],
        current_retention_duration: Optional[float],
        new_priority: Optional[int],
        new_retention_duration: Optional[float],
    ) -> Tuple[int, float]:
        current_priority = cls.clamp_priority(current_priority or 0)
        new_priority = cls.clamp_priority(new_priority or 0)

        current_rank = (
            current_priority,
            cls._retention_rank(current_retention_duration),
        )
        new_rank = (new_priority, cls._retention_rank(new_retention_duration))

        if new_rank > current_rank:
            return new_priority, new_retention_duration or 0.0
        return current_priority, current_retention_duration or 0.0

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        priority = self.clamp_priority(node.priority)
        if node.retention_duration > 0 and priority > 0:
            elapsed = time.monotonic() - node.last_access_time
            if elapsed >= node.retention_duration:
                priority = 0
        return (priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)
