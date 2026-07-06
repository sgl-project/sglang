from __future__ import annotations

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


class HitsPerTokenStrategy(EvictionStrategy):
    """Hits-per-token (HPT): size-aware frequency eviction.

    ``priority = (hit_count + 1) / size``  (smaller value is evicted first)

    LRU/LFU/SLRU rank nodes by recency and/or frequency only. HPT additionally
    accounts for a node's *size* (its KV length in tokens): among prefixes with
    comparable reuse it evicts the largest first, freeing more KV per eviction and
    retaining more small, frequently-reused prefixes. The score is the node's reuse
    count per KV token it occupies. On workloads with heterogeneous prefix sizes and
    deep prefix reuse under KV pressure this raises the radix-cache hit rate (and,
    when prefill is a meaningful share of the work, end-to-end throughput).

    Related to the classic Greedy-Dual-Size-Frequency (GDSF) family, but simpler:
    it is the bare frequency-over-size ratio, with no aging/inflation clock and no
    cost term.
    """

    def get_priority(self, node: TreeNode) -> float:
        size = len(node.value) if node.value is not None else 1
        return (node.hit_count + 1) / max(size, 1)
