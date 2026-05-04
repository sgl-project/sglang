from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
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


class AgentAwareEvictionStrategy(EvictionStrategy):
    """Agent-aware eviction strategy that considers workflow DAG topology.

    Eviction priority is a tuple of four components (all ascending — lower = evict first):

    1. TTL protection: nodes with unexpired TTL are protected (sorted last).
    2. Steps-to-execution: nodes closer to being reused are protected.
       -1 (unknown) is treated as a neutral default.
    3. Workflow association: nodes associated with active workflows are protected.
    4. LRU tiebreaker: among equal nodes, least-recently-used is evicted first.

    This strategy is designed to be used with the ``--radix-eviction-policy agent_aware``
    server argument and requires ``--enable-agent-awareness`` to populate node metadata.
    Without agent metadata, it degrades gracefully to LRU.
    """

    def get_priority(self, node: "TreeNode") -> Tuple:
        now = time.monotonic()

        # 1. TTL protection: 0 = expired/no TTL (evict first), 1 = active TTL
        ttl_active = 1 if node.ttl_expire_time > now else 0

        # 2. Steps-to-execution: lower = further from use = evict first.
        #    -1 (unknown) maps to a large neutral value so it sits between
        #    "will never be used" (inf) and "about to be used" (small int).
        ste = node.steps_to_execution
        if ste < 0:
            ste_score = 1000  # Unknown — neutral
        else:
            ste_score = ste  # 0 = about to be used (protect), large = far away

        # 3. Workflow association count — more associations = more likely to be reused
        wf_count = len(node.workflow_ids)

        # 4. LRU tiebreaker
        lru = node.last_access_time

        # Tuple comparison: Python compares element-by-element, ascending.
        # Lower values are evicted first in a min-heap.
        return (ttl_active, ste_score, wf_count, lru)
