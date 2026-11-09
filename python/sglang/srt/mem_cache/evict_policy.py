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


class TLRUStrategy(EvictionStrategy):
    """Tail-Optimized LRU (Zhang et al., arXiv:2510.15152).

    A conversation with history length L whose next prompt is expected to add
    Q_hat tokens only has to keep L + Q_hat - threshold tokens cached to hold its
    next prefill under the TTFT budget; tokens past that budget cannot improve
    tail latency and are "TEL-safe", i.e. free to evict. Such nodes are reported
    as infinitely old, which is the implementation the paper suggests: the
    existing eviction driver then drains them before anything else (the paper's
    phase 1) and continues in plain recency order once they run out (phase 2),
    so neither eviction loop needs to know about T-LRU.

    threshold and next_prompt_estimate are token counts, whereas the paper states
    both in blocks; multiply the paper's values by page_size to convert.
    """

    def __init__(self, threshold: int = 0, next_prompt_estimate: int = 0):
        self.threshold = threshold
        self.next_prompt_estimate = next_prompt_estimate

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # node._tlru_history_len is the branch's high-water depth, i.e. the
        # paper's L, and deliberately does not shrink when the tail is trimmed.
        # Deriving L from what is still resident instead would leave the
        # shortened conversation over budget on the next pass too, and T-LRU
        # would walk it down to nothing rather than stopping after
        # (threshold - Q_hat) tokens.
        budget = max(
            node._tlru_history_len + self.next_prompt_estimate - self.threshold, 0
        )
        cached_without_this_node = node._tlru_cached_prefix_len - len(node.key)
        tel_safe = cached_without_this_node >= budget
        return (-1 if tel_safe else 0, node.last_access_time)


class GDSStrategy(EvictionStrategy):
    """GreedyDual-Size over recurrent-state checkpoints (FlashState,
    https://www.alphaxiv.org/abs/2611.flashstate).

    A checkpoint's priority is its stamp, the eviction watermark at its last use,
    plus the prefill it costs to rebuild from the nearest shallower checkpoint.
    Full KV is not ranked here: it stays in recency order, which is why
    get_priority matches LRUStrategy.
    """

    # Defaults are FlashState's GLM-5.3-Flash fit, in seconds; only the ratio
    # d/e affects the order, and other models may want their own.
    def __init__(self, d: float = 1.862e-5, e: float = 3.33e-6):
        if d < 0 or e < 0 or (d == 0 and e == 0):
            raise ValueError(
                "--radix-eviction-policy gds requires non-negative 'd' and 'e' "
                f"with at least one positive, got d={d}, e={e}."
            )
        self.d = d
        self.e = e

    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time

    def restore_cost(self, start: int, end: int) -> float:
        """Seconds of full-model prefill over tokens [start, end)."""
        return self.d * (end - start) + self.e * (end**2 - start**2) / 2e5


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
