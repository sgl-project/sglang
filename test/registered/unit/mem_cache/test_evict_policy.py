"""Unit tests for evict_policy.py"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.environ import envs
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


def _make_slru_optimized(**kwargs):
    """Construct an SLRUStrategy with the optimization gate forced ON.

    ``SLRUStrategy`` snapshots ``SGLANG_ENABLE_SLRU_OPTIMIZATION`` at
    construction time, so tests that exercise the optimized code path
    toggle the env var inside a context manager and build a fresh
    instance. Tests that exercise the legacy (gate-off) path just call
    ``SLRUStrategy(...)`` directly; default env is off.
    """
    with envs.SGLANG_ENABLE_SLRU_OPTIMIZATION.override(True):
        return SLRUStrategy(**kwargs)


def _make_node(**kwargs):
    node = MagicMock()
    node.last_access_time = kwargs.get("last_access_time", 0.0)
    node.hit_count = kwargs.get("hit_count", 0)
    node.creation_time = kwargs.get("creation_time", 0.0)
    node.priority = kwargs.get("priority", 0)
    node.last_accessed_timestamp = kwargs.get(
        "last_accessed_timestamp", node.last_access_time
    )
    return node


class TestLRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = LRUStrategy()

    def test_priority_is_last_access_time(self):
        node = _make_node(last_access_time=42.0)
        self.assertEqual(self.strategy.get_priority(node), 42.0)

    def test_older_access_evicted_first(self):
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestLFUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = LFUStrategy()

    def test_priority_is_hit_count_and_time(self):
        node = _make_node(hit_count=5, last_access_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), (5, 3.0))

    def test_lower_hit_count_evicted_first(self):
        cold = _make_node(hit_count=1, last_access_time=10.0)
        hot = _make_node(hit_count=100, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(cold), self.strategy.get_priority(hot)
        )

    def test_same_hit_count_older_access_evicted_first(self):
        old = _make_node(hit_count=3, last_access_time=1.0)
        new = _make_node(hit_count=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestFIFOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FIFOStrategy()

    def test_priority_is_creation_time(self):
        node = _make_node(creation_time=7.0)
        self.assertEqual(self.strategy.get_priority(node), 7.0)

    def test_earlier_created_evicted_first(self):
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(first), self.strategy.get_priority(second)
        )


class TestMRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MRUStrategy()

    def test_priority_is_negated_access_time(self):
        node = _make_node(last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), -5.0)

    def test_most_recently_used_evicted_first(self):
        """MRU evicts the most recently accessed node first (lowest priority value)."""
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(new), self.strategy.get_priority(old)
        )


class TestFILOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FILOStrategy()

    def test_priority_is_negated_creation_time(self):
        node = _make_node(creation_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), -3.0)

    def test_last_created_evicted_first(self):
        """FILO evicts the most recently created node first."""
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(second), self.strategy.get_priority(first)
        )


class TestPriorityStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = PriorityStrategy()

    def test_priority_is_tuple(self):
        node = _make_node(priority=2, last_access_time=4.0)
        self.assertEqual(self.strategy.get_priority(node), (2, 4.0))

    def test_lower_priority_evicted_first(self):
        low = _make_node(priority=1, last_access_time=10.0)
        high = _make_node(priority=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(low), self.strategy.get_priority(high)
        )

    def test_same_priority_older_access_evicted_first(self):
        old = _make_node(priority=3, last_access_time=1.0)
        new = _make_node(priority=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestSLRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = SLRUStrategy(protected_threshold=2)

    def test_probationary_segment(self):
        node = _make_node(hit_count=1, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))

    def test_protected_segment(self):
        node = _make_node(hit_count=2, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_highly_accessed_is_protected(self):
        node = _make_node(hit_count=100, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_probationary_evicted_before_protected(self):
        prob = _make_node(hit_count=1, last_access_time=10.0)
        prot = _make_node(hit_count=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(prob), self.strategy.get_priority(prot)
        )

    def test_same_segment_older_access_evicted_first(self):
        old = _make_node(hit_count=0, last_access_time=1.0)
        new = _make_node(hit_count=0, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )

    def test_custom_threshold(self):
        strategy = SLRUStrategy(protected_threshold=5)
        below = _make_node(hit_count=4, last_access_time=1.0)
        at = _make_node(hit_count=5, last_access_time=1.0)
        self.assertEqual(strategy.get_priority(below), (0, 1.0))
        self.assertEqual(strategy.get_priority(at), (1, 1.0))

    def test_default_threshold_is_2(self):
        default = SLRUStrategy()
        self.assertEqual(default.protected_threshold, 2)


class TestEvictionOrdering(unittest.TestCase):
    """Integration-style test: sort a list of nodes by eviction priority."""

    def test_lru_ordering(self):
        strategy = LRUStrategy()
        nodes = [
            _make_node(last_access_time=5.0),
            _make_node(last_access_time=1.0),
            _make_node(last_access_time=3.0),
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        times = [n.last_access_time for n in eviction_order]
        self.assertEqual(times, [1.0, 3.0, 5.0])

    def test_slru_ordering(self):
        strategy = SLRUStrategy(protected_threshold=2)
        nodes = [
            _make_node(hit_count=5, last_access_time=1.0),  # protected, old
            _make_node(hit_count=0, last_access_time=10.0),  # probationary, new
            _make_node(hit_count=0, last_access_time=2.0),  # probationary, old
            _make_node(hit_count=3, last_access_time=8.0),  # protected, new
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        expected = [
            (0, 2.0),  # probationary old
            (0, 10.0),  # probationary new
            (1, 1.0),  # protected old
            (1, 8.0),  # protected new
        ]
        actual = [strategy.get_priority(n) for n in eviction_order]
        self.assertEqual(actual, expected)


class TestEvictionStrategyDefaultOnHit(unittest.TestCase):
    """The default on_hit hook preserves legacy hit-count accounting.

    Non-SLRU policies inherit this hook and must not mutate the SLRU-specific
    ``last_accessed_timestamp`` field.
    """

    def _check_default_on_hit_is_pure_increment(self, strategy):
        node = _make_node(hit_count=7, last_accessed_timestamp=42.0)
        strategy.on_hit(node)
        self.assertEqual(node.hit_count, 8)
        # Default on_hit must not mutate the SLRU-specific timestamp.
        self.assertEqual(node.last_accessed_timestamp, 42.0)

    def test_lru_default_on_hit(self):
        self._check_default_on_hit_is_pure_increment(LRUStrategy())

    def test_lfu_default_on_hit(self):
        self._check_default_on_hit_is_pure_increment(LFUStrategy())

    def test_fifo_default_on_hit(self):
        self._check_default_on_hit_is_pure_increment(FIFOStrategy())

    def test_mru_default_on_hit(self):
        self._check_default_on_hit_is_pure_increment(MRUStrategy())

    def test_filo_default_on_hit(self):
        self._check_default_on_hit_is_pure_increment(FILOStrategy())

    def test_priority_default_on_hit(self):
        self._check_default_on_hit_is_pure_increment(PriorityStrategy())

    def test_slru_inherits_default_on_hit_in_pr1(self):
        # With the optimization gate off, SLRU keeps legacy accounting.
        self._check_default_on_hit_is_pure_increment(SLRUStrategy())


class TestBatchedNowSnapshot(unittest.TestCase):
    """All strategies accept a shared eviction-time snapshot.

    Non-time-based strategies ignore ``now`` and preserve their priorities.
    """

    def test_non_time_strategies_accept_now_without_side_effects(self):
        strategies = [
            LRUStrategy(),
            LFUStrategy(),
            FIFOStrategy(),
            MRUStrategy(),
            FILOStrategy(),
            PriorityStrategy(),
        ]
        for strategy in strategies:
            node = _make_node(
                hit_count=3,
                last_access_time=5.0,
                creation_time=1.0,
                priority=2,
            )
            without_now = strategy.get_priority(node)
            with_now = strategy.get_priority(node, now=999.0)
            self.assertEqual(
                without_now,
                with_now,
                f"{type(strategy).__name__} changed output when `now` was supplied",
            )

    def test_slru_accepts_now_kwarg_in_pr1(self):
        # With the optimization gate off, SLRU ignores ``now``.
        strategy = SLRUStrategy()
        node = _make_node(hit_count=5, last_access_time=3.0)
        self.assertEqual(strategy.get_priority(node), (1, 3.0))
        self.assertEqual(strategy.get_priority(node, now=1e18), (1, 3.0))


class TestEvictionStrategyBaseClass(unittest.TestCase):
    """Sanity checks on the ABC itself."""

    def test_cannot_instantiate_abstract_base(self):
        with self.assertRaises(TypeError):
            EvictionStrategy()  # abstract — no concrete get_priority

    def test_slru_has_optimization_knobs(self):
        strategy = SLRUStrategy()
        self.assertEqual(strategy.protected_threshold, 2)
        self.assertEqual(strategy.debounce_sec, 0.1)
        self.assertEqual(strategy.decay_sec, 60.0)


# -----------------------------------------------------------------------------
# Integration tests against RadixCache. These require the full sglang runtime
# (torch, allocators, kv_events), so we skip gracefully when imports fail —
# CI runs these in a full environment, while minimal dev boxes (no GPU toolchain)
# still get the strategy-layer coverage above.
# -----------------------------------------------------------------------------

try:
    import torch  # noqa: F401

    from sglang.srt.mem_cache import radix_cache as _radix_cache_module
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

    _RADIX_CACHE_AVAILABLE = True
    _RADIX_CACHE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover — depends on environment
    _RADIX_CACHE_AVAILABLE = False
    _RADIX_CACHE_IMPORT_ERROR = exc


class _MockAllocator:
    """Counts freed tokens; enough to satisfy ``evict()`` without a real pool."""

    def __init__(self):
        self.device = None
        self.freed_tokens = 0

    def free(self, value):
        # value is a torch.Tensor of token indices
        self.freed_tokens += int(value.numel())


@unittest.skipUnless(
    _RADIX_CACHE_AVAILABLE,
    f"RadixCache import unavailable: {_RADIX_CACHE_IMPORT_ERROR}",
)
class TestRadixCacheSLRUScaffolding(unittest.TestCase):
    """RadixCache contracts required by SLRU debounce and lazy decay."""

    def setUp(self):
        self.allocator = _MockAllocator()
        self.cache = RadixCache.create_simulated(
            mock_allocator=self.allocator, page_size=1
        )

    def test_tree_node_has_last_accessed_timestamp_field(self):
        # The SLRU timestamp is initialized from the node's access time.
        import torch

        key = RadixKey(token_ids=[1, 2, 3], extra_key=None)
        value = torch.tensor([10, 20, 30], dtype=torch.int64)
        self.cache.insert(InsertParams(key=key, value=value))
        # Find the leaf we just inserted.
        node = list(self.cache.evictable_leaves)[0]
        self.assertTrue(hasattr(node, "last_accessed_timestamp"))
        self.assertEqual(node.last_accessed_timestamp, node.last_access_time)

    def test_evict_snapshots_monotonic_once_per_scan(self):
        """The decisive scaling guarantee: when the eviction heap is built
        across N leaves, ``time.monotonic()`` must be called exactly ONCE
        (the sweep snapshot), not N times. This test inserts 32 distinct
        leaves and spies on ``radix_cache.time.monotonic`` during evict().
        If any future refactor forgets to thread ``now`` through, call
        count explodes and this test fires.
        """
        import torch

        # Insert 32 disjoint prefixes so each becomes its own evictable leaf.
        for i in range(32):
            key = RadixKey(token_ids=[1000 + i, 2000 + i], extra_key=None)
            value = torch.tensor([i, i + 1], dtype=torch.int64)
            self.cache.insert(InsertParams(key=key, value=value))
        self.assertGreaterEqual(len(self.cache.evictable_leaves), 32)

        real_monotonic = _radix_cache_module.time.monotonic
        call_log = []

        def spy():
            call_log.append(None)
            return real_monotonic()

        original = _radix_cache_module.time.monotonic
        _radix_cache_module.time.monotonic = spy
        try:
            self.cache.evict(EvictParams(num_tokens=4))
        finally:
            _radix_cache_module.time.monotonic = original

        # Exactly one sweep snapshot; time.perf_counter() is a different
        # symbol and does not show up in this counter.
        self.assertEqual(
            len(call_log),
            1,
            f"evict() must snapshot time.monotonic exactly once per sweep, "
            f"got {len(call_log)} calls",
        )

    def test_split_inherits_last_accessed_timestamp(self):
        """When ``_split_node`` creates an intermediate node, it must
        inherit ``last_accessed_timestamp`` from the child so lazy decay keeps
        the original age of the represented prefix.
        """
        import torch

        # Insert a long prefix, then a shorter prefix that forces a split
        # inside the first one.
        long_key = RadixKey(token_ids=[1, 2, 3, 4, 5, 6], extra_key=None)
        long_val = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int64)
        self.cache.insert(InsertParams(key=long_key, value=long_val))

        # Pin a known SLRU-timestamp on the inserted leaf. Only SLRU's
        # on_hit advances this field, so a subsequent insert on a
        # different prefix will NOT overwrite it.
        leaf = list(self.cache.evictable_leaves)[0]
        leaf.last_accessed_timestamp = 123.456

        # Now insert a diverging prefix that shares the first 3 tokens —
        # _split_node will cleave the original into a parent + child.
        divergent_key = RadixKey(token_ids=[1, 2, 3, 99, 100], extra_key=None)
        divergent_val = torch.tensor([10, 20, 30, 990, 1000], dtype=torch.int64)
        self.cache.insert(InsertParams(key=divergent_key, value=divergent_val))

        # Walk from root down the shared prefix to find the newly-split
        # intermediate. It must carry the child's SLRU timestamp, NOT a
        # fresh time.monotonic() from TreeNode.__init__.
        node = self.cache.root_node
        child_key = RadixKey(token_ids=[1, 2, 3], extra_key=None).child_key(
            self.cache.page_size
        )
        self.assertIn(child_key, node.children)
        split_node = node.children[child_key]
        self.assertEqual(split_node.last_accessed_timestamp, 123.456)


# -----------------------------------------------------------------------------
# Core SLRU optimization: debounce, cap, and lazy decay.
#
# Tests below exercise the *optimized* code path, activated by the env var
# ``SGLANG_ENABLE_SLRU_OPTIMIZATION``. They all construct a fresh
# ``SLRUStrategy`` inside the override so the gate snapshot is captured ON.
# -----------------------------------------------------------------------------


class TestSLRUOptimizationGateOff(unittest.TestCase):
    """With the gate OFF (default), the new SLRU must behave byte-for-byte
    like the historical two-tier SLRU. Every on-disk behavior that
    existing users depend on is locked here as a regression guard.
    """

    def test_on_hit_is_pure_increment_when_gate_off(self):
        strategy = SLRUStrategy()
        node = _make_node(hit_count=3, last_accessed_timestamp=100.0)
        # Even far inside a would-be debounce window, the gate-off path
        # must ignore debounce entirely and just bump the count.
        strategy.on_hit(node, now=100.0001)
        self.assertEqual(node.hit_count, 4)

    def test_get_priority_ignores_now_when_gate_off(self):
        strategy = SLRUStrategy()
        node = _make_node(
            hit_count=5,
            last_access_time=10.0,
            last_accessed_timestamp=0.0,
        )
        # A huge ``now`` would collapse eff -> 0 if the gate were ON.
        self.assertEqual(strategy.get_priority(node, now=1e18), (1, 10.0))
        self.assertEqual(strategy.get_priority(node), (1, 10.0))


class TestSLRUDebounce(unittest.TestCase):
    """Write-side debounce: hits inside ``debounce_sec`` do not count."""

    def test_burst_within_debounce_only_counts_once(self):
        # 50 concurrent hits in a 10ms span must count as 1 (burst-one-shot).
        strategy = _make_slru_optimized(debounce_sec=5.0)
        node = _make_node(hit_count=0, last_accessed_timestamp=0.0)
        base = 100.0
        strategy.on_hit(node, now=base)
        for i in range(1, 50):
            strategy.on_hit(node, now=base + i * 0.0002)  # 50 hits in 10ms
        self.assertEqual(node.hit_count, 1)
        # Timestamp anchors on the FIRST counted hit — so the 50th hit
        # arriving at base+0.0098 is still inside the 5s window.
        self.assertEqual(node.last_accessed_timestamp, base)

    def test_first_hit_counts_even_at_creation_timestamp(self):
        strategy = _make_slru_optimized(debounce_sec=5.0)
        node = _make_node(hit_count=0, last_accessed_timestamp=100.0)
        strategy.on_hit(node, now=100.0)
        self.assertEqual(node.hit_count, 1)
        self.assertEqual(node.last_accessed_timestamp, 100.0)

    def test_hits_outside_debounce_accumulate(self):
        strategy = _make_slru_optimized(debounce_sec=1.0)
        node = _make_node(hit_count=0, last_accessed_timestamp=0.0)
        # 5 hits well-spaced past the debounce window ⇒ counter ramps
        # up to the cap (protected_threshold default 2).
        for i in range(5):
            strategy.on_hit(node, now=10.0 + i * 2.0)
        # With default cap of 2 (protected_threshold), hit_count plateaus
        # at 2 — but the *timestamp* keeps advancing past the cap (Cap
        # semantics, tested separately below).
        self.assertEqual(node.hit_count, 2)

    def test_zero_debounce_behaves_like_plain_counter(self):
        # debounce_sec=0 means "always count" (legacy-equivalent hit
        # bumping). Useful for tests that want to isolate Cap from
        # Debounce.
        strategy = _make_slru_optimized(debounce_sec=0.0)
        node = _make_node(hit_count=0, last_accessed_timestamp=0.0)
        for i in range(5):
            strategy.on_hit(node, now=1.0 + i * 0.001)
        # Hits 1 and 2 accumulate to hit_count=2 (threshold); subsequent
        # hits fall into the cap branch and don't bump.
        self.assertEqual(node.hit_count, 2)


class TestSLRUCap(unittest.TestCase):
    """Write-side cap: ``hit_count`` is bounded at ``protected_threshold``."""

    def test_hit_count_is_capped_at_threshold(self):
        strategy = _make_slru_optimized(debounce_sec=0.0, protected_threshold=2)
        node = _make_node(hit_count=0, last_accessed_timestamp=0.0)
        for i in range(1000):
            strategy.on_hit(node, now=1.0 + i * 0.001)
        # 1000 hits ⇒ hit_count saturates at threshold, no overflow.
        self.assertEqual(node.hit_count, 2)

    def test_cap_still_refreshes_timestamp(self):
        # A Protected node that keeps receiving hits must keep getting
        # its ``last_accessed_timestamp`` refreshed; otherwise lazy
        # decay would kick it out even while it's actively being used.
        strategy = _make_slru_optimized(debounce_sec=0.0, protected_threshold=2)
        node = _make_node(hit_count=2, last_accessed_timestamp=0.0)
        strategy.on_hit(node, now=555.0)
        self.assertEqual(node.hit_count, 2)  # capped
        self.assertEqual(node.last_accessed_timestamp, 555.0)  # refreshed


class TestSLRULazyDecay(unittest.TestCase):
    """Read-side lazy decay: ``get_priority`` synthesizes the effective
    hit count at scan time without mutating the node. Cold Protected
    nodes silently drop back to Probationary.
    """

    def test_cold_node_drops_back_to_probationary(self):
        strategy = _make_slru_optimized(decay_sec=1.0, protected_threshold=2)
        node = _make_node(
            hit_count=2,  # capped, in Protected
            last_access_time=100.0,
            last_accessed_timestamp=100.0,
        )
        # After 10s with decay_sec=1s, halvings=10 >> bit_length(2)=2 ⇒
        # eff=0, which is Probationary.
        priority = strategy.get_priority(node, now=110.0)
        self.assertEqual(priority, (0, 100.0))
        # Crucially: the node's hit_count is NOT mutated by this read.
        self.assertEqual(node.hit_count, 2)
        self.assertEqual(node.last_accessed_timestamp, 100.0)

    def test_recent_protected_node_stays_protected(self):
        strategy = _make_slru_optimized(decay_sec=60.0, protected_threshold=2)
        node = _make_node(
            hit_count=2,
            last_access_time=100.0,
            last_accessed_timestamp=100.0,
        )
        # 1s < 60s ⇒ 0 halvings ⇒ eff=2, still Protected.
        priority = strategy.get_priority(node, now=101.0)
        self.assertEqual(priority, (1, 100.0))

    def test_effective_hit_count_math_is_precise(self):
        strategy = _make_slru_optimized(decay_sec=1.0, protected_threshold=1)
        # hit_count=16, age=3s, decay=1s ⇒ halvings=3 ⇒ 16>>3=2
        node = _make_node(
            hit_count=16,
            last_access_time=0.0,
            last_accessed_timestamp=0.0,
        )
        self.assertEqual(strategy._effective_hit_count(node, 3.0), 2)
        # age=5s ⇒ halvings=5 >= bit_length(16)=5 ⇒ eff=0
        self.assertEqual(strategy._effective_hit_count(node, 5.0), 0)

    def test_decay_handles_huge_age_without_overflow(self):
        strategy = _make_slru_optimized(decay_sec=0.001, protected_threshold=2)
        node = _make_node(
            hit_count=2,
            last_access_time=0.0,
            last_accessed_timestamp=0.0,
        )
        # 1e10 / 0.001 = 1e13 halvings — no negative shift, no crash,
        # just saturates to 0.
        self.assertEqual(strategy._effective_hit_count(node, 1e10), 0)

    def test_zero_hit_count_fast_path(self):
        strategy = _make_slru_optimized(decay_sec=60.0, protected_threshold=2)
        node = _make_node(
            hit_count=0,
            last_access_time=0.0,
            last_accessed_timestamp=0.0,
        )
        # Never-hit node always has eff=0; short-circuits before any math.
        self.assertEqual(strategy._effective_hit_count(node, 999.0), 0)
        self.assertEqual(strategy.get_priority(node, now=999.0), (0, 0.0))


class TestBatchedNowSnapshotSLRU(unittest.TestCase):
    """Contract: when the caller provides ``now``, SLRU must use it
    instead of reading the wall clock. A refactor that accidentally
    re-reads ``time.monotonic()`` inside ``get_priority`` would silently
    degrade eviction-scan performance from O(1) to O(N) clock reads —
    this test makes that regression impossible to ship.
    """

    def test_slru_uses_caller_supplied_now_without_reading_clock(self):
        strategy = _make_slru_optimized(decay_sec=1.0, protected_threshold=2)
        node = _make_node(
            hit_count=2,
            last_access_time=0.0,
            last_accessed_timestamp=0.0,
        )

        import sglang.srt.mem_cache.evict_policy as ep_module

        original_monotonic = ep_module.time.monotonic
        try:
            # Any call to time.monotonic() inside on_hit/get_priority
            # when `now` is supplied MUST be considered a bug.
            def exploding_clock():
                raise AssertionError(
                    "time.monotonic() was called despite explicit now="
                )

            ep_module.time.monotonic = exploding_clock
            # These must not raise.
            strategy.on_hit(node, now=500.0)
            priority = strategy.get_priority(node, now=500.5)
            self.assertIsInstance(priority, tuple)
        finally:
            ep_module.time.monotonic = original_monotonic


# -----------------------------------------------------------------------------
# Three-way scenarios: LRU vs legacy SLRU vs optimized SLRU.
#
# Legacy SLRU is the gate-off path; optimized SLRU is constructed with the
# environment gate enabled.
# -----------------------------------------------------------------------------


class TestSLRUScenarios(unittest.TestCase):
    """Locks eviction decisions for LRU, legacy SLRU, and optimized SLRU.

    A. One-shot flood (scan attack) — SLRU family beats LRU.
    B. Concurrent burst on a one-shot prefix — debounce protects real hot data.
    C. Stale protected prefix vs newcomer — lazy decay admits the newcomer.
    """

    def test_scenario_a_one_shot_flood_lru_loses_slru_wins(self):
        """HOT prefix hit 10× (last at t=10) vs one-shot SHOT at t=11.
        LRU orders by age and picks HOT first (wrong). Both SLRUs keep
        HOT in Protected while SHOT sits in Probationary."""
        hot = _make_node(
            hit_count=10, last_access_time=10.0, last_accessed_timestamp=10.0
        )
        shot = _make_node(
            hit_count=1, last_access_time=11.0, last_accessed_timestamp=11.0
        )

        # LRU: older = evicted first → HOT loses (known weakness).
        lru = LRUStrategy()
        self.assertLess(lru.get_priority(hot), lru.get_priority(shot))

        # Naive SLRU (gate off): HOT is Protected, SHOT is Probationary,
        # so SHOT (lower tier) is evicted first.
        naive = SLRUStrategy()
        self.assertLess(naive.get_priority(shot), naive.get_priority(hot))

        # Optimized SLRU: same decision — SHOT first, HOT stays.
        new = _make_slru_optimized()
        self.assertLess(
            new.get_priority(shot, now=11.1),
            new.get_priority(hot, now=11.1),
        )

    def test_scenario_b_concurrent_burst_fools_legacy_only(self):
        """VIP hit 10× with 1s spacing (last at t=10) vs a 50-concurrent
        BURST on a throwaway prefix that lands in a 10ms window at t=11.

        * LRU: BURST is newer than VIP ⇒ VIP evicted first (wrong).
        * Legacy SLRU: BURST racks up hit_count=50 >> threshold, lands
          in Protected alongside VIP, and since VIP is older in
          Protected, VIP is evicted first (wrong — Protected got
          polluted by a one-shot).
        * Optimized SLRU: debounce compresses BURST to hit_count=1, so
          BURST stays in Probationary and is evicted first (right).
        """
        # Step 1: construct end states.
        # VIP has ten 1s-spaced real hits; reaches the cap at threshold=2
        # and keeps refreshing timestamp.
        # BURST has 50 hits within 10ms; in legacy SLRU it would be
        # hit_count=50, while optimized SLRU holds it at 1.

        # --- LRU ---
        lru_vip = _make_node(hit_count=10, last_access_time=10.0)
        lru_burst = _make_node(hit_count=50, last_access_time=11.0)
        lru = LRUStrategy()
        # Older = evicted first. VIP loses.
        self.assertLess(lru.get_priority(lru_vip), lru.get_priority(lru_burst))

        # --- Naive SLRU ---
        naive_vip = _make_node(hit_count=10, last_access_time=10.0)
        naive_burst = _make_node(hit_count=50, last_access_time=11.0)
        naive = SLRUStrategy()
        # Both in Protected (hit_count >= 2); within Protected, older
        # VIP is evicted first; debounce avoids this protected-tier pollution.
        self.assertLess(naive.get_priority(naive_vip), naive.get_priority(naive_burst))

        # --- Optimized SLRU ---
        strategy = _make_slru_optimized(
            debounce_sec=5.0, decay_sec=3600.0, protected_threshold=2
        )
        pr2_vip = _make_node(
            hit_count=0,
            last_access_time=0.0,
            last_accessed_timestamp=0.0,
        )
        pr2_burst = _make_node(
            hit_count=0,
            last_access_time=0.0,
            last_accessed_timestamp=0.0,
        )
        # VIP: ten 1s-spaced hits, reaches cap.
        for i in range(10):
            t = 1.0 + i * 1.0
            strategy.on_hit(pr2_vip, now=t)
            pr2_vip.last_access_time = t
        # BURST: 50 hits within 10ms at t=11.
        base = 11.0
        for i in range(50):
            t = base + i * 0.0002
            strategy.on_hit(pr2_burst, now=t)
            pr2_burst.last_access_time = t
        # VIP at cap (Protected). BURST's debounce held it at 1 hit
        # (Probationary).
        self.assertEqual(pr2_vip.hit_count, 2)
        self.assertEqual(pr2_burst.hit_count, 1)
        # At eviction (now=11.1), BURST is lower tier → evicted first.
        now = 11.1
        self.assertLess(
            strategy.get_priority(pr2_burst, now=now),
            strategy.get_priority(pr2_vip, now=now),
        )

    def test_scenario_c_stale_protected_starves_newcomer_in_legacy_only(
        self,
    ):
        """OLD hit heavily ages ago (hit_count=10, last access t=0);
        NEW is a newcomer with one hit at t=999. At eviction time
        (now=1000, decay_sec=60):

        * LRU: OLD is older ⇒ evicted first.
        * Legacy SLRU: OLD is Protected, NEW is Probationary ⇒ NEW
          evicted first.
        * Optimized SLRU: lazy decay reduces OLD's eff to 0, demoting it to
          Probationary; within Probationary, older OLD goes first
          and NEW stays.
        """
        now = 1000.0

        # --- LRU ---
        self.assertLess(
            LRUStrategy().get_priority(_make_node(last_access_time=0.0)),
            LRUStrategy().get_priority(_make_node(last_access_time=999.0)),
        )

        # --- Naive SLRU ---
        naive = SLRUStrategy()
        naive_old = _make_node(hit_count=10, last_access_time=0.0)
        naive_new = _make_node(hit_count=1, last_access_time=999.0)
        # NEW loses because OLD never leaves Protected.
        self.assertLess(naive.get_priority(naive_new), naive.get_priority(naive_old))

        # --- Optimized SLRU ---
        strategy = _make_slru_optimized(decay_sec=60.0, protected_threshold=2)
        pr2_old = _make_node(
            hit_count=2,  # hit the cap long ago
            last_access_time=0.0,
            last_accessed_timestamp=0.0,
        )
        pr2_new = _make_node(
            hit_count=1,
            last_access_time=999.0,
            last_accessed_timestamp=999.0,
        )
        # At now=1000s with decay=60s, OLD gets halvings=16 >>
        # bit_length(2)=2 ⇒ eff=0 (Probationary). NEW has hit_count=1 <
        # threshold=2 ⇒ also Probationary. Within Probationary, older
        # OLD is evicted first — newcomer stays.
        self.assertLess(
            strategy.get_priority(pr2_old, now=now),
            strategy.get_priority(pr2_new, now=now),
        )


if __name__ == "__main__":
    unittest.main()
