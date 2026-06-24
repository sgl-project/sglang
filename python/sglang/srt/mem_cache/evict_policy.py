from __future__ import annotations
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

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
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)

class AgentAwareStrategy(EvictionStrategy):
    """Agent-aware eviction.

    Lower tuple values are evicted first.

    This policy is intentionally conservative:
    - no hard pinning
    - no prefix matching changes
    - no cross-request orchestration
    - falls back to LRU when agent metadata is absent
    """

    HIGH_REUSE_HINTS = {"high", "keep", "reuse"}
    LOW_REUSE_HINTS = {"low"}

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        meta = getattr(node, "agent_meta", None)
        bucket = 1

        if meta:
            now = time.monotonic()
            ttl_deadline = meta.get("cache_ttl_deadline")
            reuse_hint = meta.get("reuse_hint")

            active_ttl = ttl_deadline is not None and ttl_deadline > now
            expired_ttl = ttl_deadline is not None and ttl_deadline <= now

            if expired_ttl:
                bucket = 0
            elif reuse_hint in self.LOW_REUSE_HINTS:
                bucket = 0

            if reuse_hint in self.HIGH_REUSE_HINTS:
                bucket = 2

            # Active TTL is the strongest soft-retention signal.
            # It must override low reuse_hint.
            if active_ttl:
                bucket = 2

        return (bucket, node.last_access_time)

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
