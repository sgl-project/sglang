"""Unit tests for ``sglang.srt.mem_cache.evict_policy``.

These tests cover the eviction priority strategies used by the KV-cache
radix tree. They are pure-Python and run without launching a server or
loading any model weights, so they execute in seconds.

Each strategy implements ``get_priority(node) -> Union[float, Tuple]`` where
a *smaller* value means the node is evicted *earlier*.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)


def make_node(
    last_access_time: float = 0.0,
    hit_count: int = 0,
    creation_time: float = 0.0,
    priority: int = 0,
):
    """Build a lightweight stand-in for a radix ``TreeNode``."""
    return SimpleNamespace(
        last_access_time=last_access_time,
        hit_count=hit_count,
        creation_time=creation_time,
        priority=priority,
    )


def eviction_order(nodes, strategy: EvictionStrategy):
    """Return ``nodes`` sorted by eviction priority (first = evicted first)."""
    return sorted(nodes, key=strategy.get_priority)


class TestLRUStrategy(unittest.TestCase):
    def test_returns_last_access_time(self):
        node = make_node(last_access_time=42.0)
        assert LRUStrategy().get_priority(node) == 42.0

    def test_evicts_least_recently_used_first(self):
        old = make_node(last_access_time=1.0)
        recent = make_node(last_access_time=10.0)
        order = eviction_order([recent, old], LRUStrategy())
        assert order == [old, recent]


class TestFIFOStrategy(unittest.TestCase):
    def test_returns_creation_time(self):
        node = make_node(creation_time=7.0)
        assert FIFOStrategy().get_priority(node) == 7.0

    def test_evicts_oldest_created_first(self):
        older = make_node(creation_time=1.0)
        newer = make_node(creation_time=5.0)
        order = eviction_order([newer, older], FIFOStrategy())
        assert order == [older, newer]


class TestLFUStrategy(unittest.TestCase):
    def test_returns_hit_count_then_access_time(self):
        node = make_node(hit_count=3, last_access_time=9.0)
        assert LFUStrategy().get_priority(node) == (3, 9.0)

    def test_evicts_lowest_hit_count_first(self):
        cold = make_node(hit_count=1, last_access_time=100.0)
        hot = make_node(hit_count=5, last_access_time=1.0)
        order = eviction_order([hot, cold], LFUStrategy())
        # Hit count dominates: the rarely-used node goes first despite being older.
        assert order == [cold, hot]

    def test_breaks_hit_count_tie_by_access_time(self):
        a = make_node(hit_count=2, last_access_time=1.0)
        b = make_node(hit_count=2, last_access_time=2.0)
        order = eviction_order([b, a], LFUStrategy())
        assert order == [a, b]


class TestMRUStrategy(unittest.TestCase):
    def test_returns_negative_last_access_time(self):
        node = make_node(last_access_time=4.0)
        assert MRUStrategy().get_priority(node) == -4.0

    def test_evicts_most_recently_used_first(self):
        old = make_node(last_access_time=1.0)
        recent = make_node(last_access_time=10.0)
        order = eviction_order([old, recent], MRUStrategy())
        assert order == [recent, old]


class TestFILOStrategy(unittest.TestCase):
    def test_returns_negative_creation_time(self):
        node = make_node(creation_time=3.0)
        assert FILOStrategy().get_priority(node) == -3.0

    def test_evicts_most_recently_created_first(self):
        older = make_node(creation_time=1.0)
        newer = make_node(creation_time=5.0)
        order = eviction_order([older, newer], FILOStrategy())
        assert order == [newer, older]


class TestPriorityStrategy(unittest.TestCase):
    def test_returns_priority_then_access_time(self):
        node = make_node(priority=2, last_access_time=8.0)
        assert PriorityStrategy().get_priority(node) == (2, 8.0)

    def test_evicts_lowest_priority_first(self):
        low = make_node(priority=0, last_access_time=100.0)
        high = make_node(priority=9, last_access_time=1.0)
        order = eviction_order([high, low], PriorityStrategy())
        assert order == [low, high]

    def test_breaks_priority_tie_by_access_time(self):
        a = make_node(priority=1, last_access_time=1.0)
        b = make_node(priority=1, last_access_time=2.0)
        order = eviction_order([b, a], PriorityStrategy())
        assert order == [a, b]


class TestSLRUStrategy(unittest.TestCase):
    def test_default_protected_threshold_is_two(self):
        assert SLRUStrategy().protected_threshold == 2

    def test_probationary_before_protected(self):
        # hit_count below threshold -> segment 0 (evicted first)
        probationary = make_node(hit_count=1, last_access_time=100.0)
        # hit_count at/above threshold -> segment 1 (protected)
        protected = make_node(hit_count=2, last_access_time=1.0)
        order = eviction_order([protected, probationary], SLRUStrategy())
        assert order == [probationary, protected]

    def test_within_segment_uses_lru(self):
        a = make_node(hit_count=0, last_access_time=1.0)
        b = make_node(hit_count=0, last_access_time=2.0)
        order = eviction_order([b, a], SLRUStrategy())
        assert order == [a, b]

    def test_custom_threshold(self):
        strategy = SLRUStrategy(protected_threshold=5)
        low = make_node(hit_count=4, last_access_time=100.0)
        high = make_node(hit_count=5, last_access_time=1.0)
        order = eviction_order([high, low], strategy)
        assert order == [low, high]
