from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass

    def get_cascade_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        """Priority for nodes that became evictable because their children were evicted.

        In a radix tree, such nodes are shared prefixes — not conversation tails.
        Subclasses may override this to handle cascade nodes differently from
        direct eviction candidates. Default: same as get_priority.
        """
        return self.get_priority(node)


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


class TLRUStrategy(EvictionStrategy):
    """Tail-Optimized LRU: evicts TEL-safe blocks first, then falls back to LRU.

    A block is TEL-safe if evicting it won't cause the next turn's TTFT to exceed
    the latency threshold xi. See arXiv:2510.15152 for details.

    Args:
        xi: SLA latency threshold in tokens. Conversations whose next turn would
            still be under this threshold after eviction are TEL-safe.
        q_hat: Estimated next prompt length in tokens.
    """

    def __init__(self, xi: int, q_hat: int):
        self.xi = xi
        self.q_hat = q_hat

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        L = node.cumulative_tokens
        B = max(0, L + self.q_hat - self.xi)
        node_start = L - len(node.value)

        is_tel_safe = node_start >= B

        # (0, time) for TEL-safe → evicted first; (1, time) for protected → evicted second
        # Within each group, standard LRU ordering (older time = evicted first)
        return (0 if is_tel_safe else 1, node.last_access_time)

    def get_cascade_priority(self, node: "TreeNode") -> Tuple[int, float]:
        """Cascade nodes are shared prefixes whose children were evicted — not conversation tails.

        The paper's T-LRU (Algorithm 1) only marks conversation-level tail blocks as TEL-safe.
        A shared prefix node's cumulative_tokens underestimates the actual conversation history
        length (L), which would incorrectly classify it as TEL-safe. We fall back to protected/LRU
        priority to avoid evicting shared prefixes before actual conversation tails.
        """
        return (1, node.last_access_time)
