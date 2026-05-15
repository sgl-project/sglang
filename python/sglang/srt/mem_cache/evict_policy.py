from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, Union

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(
        self, node: "TreeNode", now: Optional[float] = None
    ) -> Union[float, Tuple]:
        """Compute the eviction priority of ``node``.

        ``now`` is an optional wall-clock snapshot (``time.monotonic()``) that
        lets callers amortise clock reads across a whole eviction scan. Most
        strategies don't need it (their priority is a pure function of the
        node's own bookkeeping fields) and may ignore the argument. Strategies
        that *do* need a current time reference (e.g. ``SLRUStrategy`` for lazy
        decay) should fall back to ``time.monotonic()`` when ``now`` is None so
        the signature stays ergonomic for standalone calls in tests.
        """

    def on_hit(self, node: "TreeNode", now: Optional[float] = None) -> None:
        """Hook invoked when a radix-tree node is reused by a new insert.

        Default behaviour is the classical LFU counter bump: `hit_count += 1`.
        Strategies that need a different accounting (e.g. SLRU with debounce)
        can override this without touching the radix-tree hot path.
        """
        node.hit_count += 1


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode", now: Optional[float] = None) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(
        self, node: "TreeNode", now: Optional[float] = None
    ) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode", now: Optional[float] = None) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode", now: Optional[float] = None) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode", now: Optional[float] = None) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(
        self, node: "TreeNode", now: Optional[float] = None
    ) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    """Segmented LRU with write-side debounce and read-side lazy decay.

    ``debounce_sec`` prevents near-simultaneous hits from promoting a
    one-shot prefix. ``decay_sec`` lazily lowers the effective hit count during
    eviction so stale protected nodes can fall back to probationary without a
    background sweeper.
    """

    def __init__(
        self,
        protected_threshold: int = 2,
        debounce_sec: float = 0.1,
        decay_sec: float = 60.0,
    ):
        # Keep zero debounce valid for legacy-equivalent hit accounting, and
        # guard decay against division by zero.
        self.protected_threshold = protected_threshold
        self.debounce_sec = max(0.0, debounce_sec)
        self.decay_sec = max(1e-3, decay_sec)
        self._optimization_enabled = envs.SGLANG_ENABLE_SLRU_OPTIMIZATION.get()

    def on_hit(self, node: "TreeNode", now: Optional[float] = None) -> None:
        if not self._optimization_enabled:
            node.hit_count += 1
            return

        if now is None:
            now = time.monotonic()

        # Debounce suppresses repeated hits in a burst, but the first observed
        # access is the baseline count for this node.
        if node.hit_count == 0:
            node.hit_count = 1
            node.last_accessed_timestamp = now
            return

        # Protected nodes refresh recency but do not accumulate unbounded heat.
        if node.hit_count >= self.protected_threshold:
            node.last_accessed_timestamp = now
            return

        # Probationary tier: debounce successive hits so a burst of concurrent
        # requests cannot single-handedly promote a node.
        if now - node.last_accessed_timestamp >= self.debounce_sec:
            node.hit_count += 1
            node.last_accessed_timestamp = now

    def _effective_hit_count(self, node: "TreeNode", now: float) -> int:
        # Lazy decay: every `decay_sec` period halves the effective hit count.
        # During eviction we use this decayed score for tiering:
        # newer nodes keep more heat, very stale nodes naturally drop to 0.
        hit_count = node.hit_count
        age = now - node.last_accessed_timestamp
        if age <= 0:
            return hit_count

        halvings = int(age // self.decay_sec)

        # Saturate on huge ages rather than attempting a >> bit_length() shift
        if halvings >= hit_count.bit_length():
            return 0
        return hit_count >> halvings

    def get_priority(
        self, node: "TreeNode", now: Optional[float] = None
    ) -> Tuple[int, float]:
        # Lower priority is evicted earlier: probationary before protected,
        # then LRU within the same segment.
        if not self._optimization_enabled:
            is_protected = 1 if node.hit_count >= self.protected_threshold else 0
            return (is_protected, node.last_access_time)

        if now is None:
            now = time.monotonic()
        eff = self._effective_hit_count(node, now)
        is_protected = 1 if eff >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)
