from __future__ import annotations

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


class CLOCKStrategy(EvictionStrategy):
    """Second-chance (CLOCK) approximate-LRU eviction.

    Each node carries a boolean ``referenced`` flag that is set to ``True``
    on every cache hit. When the CLOCK hand visits a node it checks that flag:
    * referenced=True  -> clear the flag and skip (give it a second chance).
    * referenced=False -> evict immediately.
    """

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # (0, t) sorts before (1, t): unreferenced nodes evicted first.
        ref_bit = 1 if getattr(node, "referenced", False) else 0
        return (ref_bit, node.last_access_time)
