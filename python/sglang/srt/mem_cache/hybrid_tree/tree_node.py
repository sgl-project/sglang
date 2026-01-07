"""
TreeNode structure for Hybrid Radix Tree.
"""

from collections import defaultdict
from typing import Dict, Optional

import torch
from numpy import float64

from sglang.srt.mem_cache.radix_cache import RadixKey


class ComponentData:
    """Data stored for each component at a tree node."""

    __slots__ = ["value", "lock_ref"]

    def __init__(self):
        self.value: Optional[torch.Tensor] = None
        self.lock_ref: int = 0

    def is_tombstone(self) -> bool:
        """Check if this component data represents a tombstone node."""
        return self.value is None


class TreeNode:
    """Shared tree node structure for all cache components."""

    counter = 0
    last_access_time_counter_float = float64(1.0)
    swa_uuid_counter = 1

    def __init__(self, id: Optional[int] = None):
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

        self.children: Dict = defaultdict(TreeNode)
        self.parent: Optional[TreeNode] = None
        self.key: RadixKey = None
        self.component_data: Dict[str, ComponentData] = {}
        self.last_access_time = get_last_access_time()
        self.swa_uuid: Optional[int] = None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


def gen_swa_uuid() -> int:
    """Generate unique SWA UUID for lock boundary marking."""
    ret = TreeNode.swa_uuid_counter
    TreeNode.swa_uuid_counter += 1
    return ret
