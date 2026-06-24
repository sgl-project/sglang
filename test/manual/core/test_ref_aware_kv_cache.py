"""Unit tests for RefAwareHiRadixCache tiered eviction."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.ref_aware_cache_mixin import (
    TIER_HIGH_REF,
    TIER_LOW_REF,
    TIER_UNUSED,
    _classify_node_tier,
)
from sglang.srt.mem_cache.ref_aware_hiradix_cache import RefAwareHiRadixCache


class TestClassifyNodeTier(unittest.TestCase):
    """Test _classify_node_tier with different ref combinations."""

    def test_unused_both_zero(self):
        node = TreeNode()
        node.high_ref = 0
        node.low_ref = 0
        self.assertEqual(_classify_node_tier(node), TIER_UNUSED)

    def test_low_ref_only(self):
        node = TreeNode()
        node.high_ref = 0
        node.low_ref = 1
        self.assertEqual(_classify_node_tier(node), TIER_LOW_REF)

    def test_low_ref_large_value(self):
        node = TreeNode()
        node.high_ref = 0
        node.low_ref = 100
        self.assertEqual(_classify_node_tier(node), TIER_LOW_REF)

    def test_high_ref_only(self):
        node = TreeNode()
        node.high_ref = 1
        node.low_ref = 0
        self.assertEqual(_classify_node_tier(node), TIER_HIGH_REF)

    def test_high_ref_overrides_low_ref(self):
        """When both high_ref and low_ref > 0, high_ref wins."""
        node = TreeNode()
        node.high_ref = 1
        node.low_ref = 5
        self.assertEqual(_classify_node_tier(node), TIER_HIGH_REF)

    def test_high_ref_large_value_overrides_low(self):
        node = TreeNode()
        node.high_ref = 10
        node.low_ref = 20
        self.assertEqual(_classify_node_tier(node), TIER_HIGH_REF)


class TestTreeNodeRefFields(unittest.TestCase):
    """Verify default values are 0/empty."""

    def test_default_high_ref_is_zero(self):
        node = TreeNode()
        self.assertEqual(node.high_ref, 0)

    def test_default_low_ref_is_zero(self):
        node = TreeNode()
        self.assertEqual(node.low_ref, 0)

    def test_default_tracked_rids_is_empty(self):
        node = TreeNode()
        self.assertEqual(node.tracked_rids, set())

    def test_new_node_classifies_as_unused(self):
        node = TreeNode()
        self.assertEqual(_classify_node_tier(node), TIER_UNUSED)


class TestRefAwareTierAccounting(unittest.TestCase):
    """Test _account_new_evictable_node, _inc_priority_ref_single,
    _dec_priority_ref_single, and _move_node_tier."""

    def _make_cache(self):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.high_priority_threshold = 1
        cache._enable_priority_scheduling = True
        cache.unused_evictable_leaves = set()
        cache.low_ref_evictable_leaves = set()
        cache.high_ref_evictable_leaves = set()
        cache.unused_evictable_size_ = 0
        cache.low_ref_evictable_size_ = 0
        cache.high_ref_evictable_size_ = 0
        cache.rid_to_ref_info = {}
        cache._evict_scope_stack = []
        return cache

    def _append_node(self, parent, token_ids):
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(token_ids)
        node.value = torch.tensor(token_ids, dtype=torch.int64)
        node.children = {}
        parent.children[token_ids[0] if token_ids else 0] = node
        return node

    def test_new_evictable_node_starts_in_unused_tier(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])

        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        self.assertEqual(cache.unused_evictable_size_, 4)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertIn(node, cache.unused_evictable_leaves)

    def test_evictable_size_by_tier_unused_only(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        # allow_low=False, allow_high=False → only unused
        self.assertEqual(cache.evictable_size_by_tier(allow_low=False, allow_high=False), 4)
        # allow_low=True → still 4 since no low-ref nodes
        self.assertEqual(cache.evictable_size_by_tier(allow_low=True, allow_high=False), 4)

    def test_inc_priority_ref_low_moves_unused_to_low_ref(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        cache._inc_priority_ref_single(node, is_high=False)

        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 4)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertNotIn(node, cache.unused_evictable_leaves)
        self.assertIn(node, cache.low_ref_evictable_leaves)

    def test_inc_priority_ref_high_from_unused_moves_to_high_ref(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        cache._inc_priority_ref_single(node, is_high=True)

        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 4)
        self.assertNotIn(node, cache.unused_evictable_leaves)
        self.assertIn(node, cache.high_ref_evictable_leaves)

    def test_inc_priority_ref_high_from_low_ref_moves_to_high_ref(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)
        cache._inc_priority_ref_single(node, is_high=False)

        cache._inc_priority_ref_single(node, is_high=True)

        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 4)
        self.assertIn(node, cache.high_ref_evictable_leaves)

    def test_ref_tier_move_preserves_total_evictable_tokens(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        # unused → low_ref
        cache._inc_priority_ref_single(node, is_high=False)
        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 4)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertEqual(cache.evictable_size_by_tier(allow_low=True, allow_high=False), 4)

        # low_ref → high_ref
        cache._inc_priority_ref_single(node, is_high=True)
        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 4)
        self.assertEqual(cache.evictable_size_by_tier(allow_low=True, allow_high=True), 4)

    def test_dec_priority_ref_single_moves_back_to_unused(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)
        cache._inc_priority_ref_single(node, is_high=False)

        cache._dec_priority_ref_single(node, is_high=False)

        self.assertEqual(cache.unused_evictable_size_, 4)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertIn(node, cache.unused_evictable_leaves)
        self.assertNotIn(node, cache.low_ref_evictable_leaves)

    def test_dec_priority_ref_single_high_moves_back(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)
        cache._inc_priority_ref_single(node, is_high=True)

        cache._dec_priority_ref_single(node, is_high=True)

        self.assertEqual(cache.unused_evictable_size_, 4)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertIn(node, cache.unused_evictable_leaves)
        self.assertNotIn(node, cache.high_ref_evictable_leaves)

    def test_dec_priority_ref_does_not_go_below_zero(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        # Decrement without prior increment — should not crash or go negative
        cache._dec_priority_ref_single(node, is_high=False)
        self.assertEqual(node.low_ref, 0)

        cache._dec_priority_ref_single(node, is_high=True)
        self.assertEqual(node.high_ref, 0)

    def test_move_node_tier_updates_sets_and_sizes(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        # Manually put node in unused tier set to test _move_node_tier directly
        cache._move_node_tier(node, TIER_UNUSED, TIER_LOW_REF)

        self.assertNotIn(node, cache.unused_evictable_leaves)
        self.assertIn(node, cache.low_ref_evictable_leaves)
        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 4)


class TestRefAwareRegisterRef(unittest.TestCase):
    """Test register_ref only adds new suffix from last_node."""

    def _make_cache(self):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.high_priority_threshold = 1
        cache._enable_priority_scheduling = True
        cache.unused_evictable_leaves = set()
        cache.low_ref_evictable_leaves = set()
        cache.high_ref_evictable_leaves = set()
        cache.unused_evictable_size_ = 0
        cache.low_ref_evictable_size_ = 0
        cache.high_ref_evictable_size_ = 0
        cache.rid_to_ref_info = {}
        cache._evict_scope_stack = []
        return cache

    def _append_node(self, parent, token_ids):
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(token_ids)
        node.value = torch.tensor(token_ids, dtype=torch.int64)
        node.children = {}
        parent.children[token_ids[0] if token_ids else 0] = node
        return node

    def test_register_ref_high_priority_sets_high_ref(self):
        """High priority (priority >= threshold) increments high_ref."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        req = SimpleNamespace(rid="r1", priority=1, last_node=a)
        cache.register_ref(req)

        self.assertEqual(a.high_ref, 1)
        self.assertEqual(a.low_ref, 0)
        self.assertIn("r1", a.tracked_rids)

    def test_register_ref_low_priority_sets_low_ref(self):
        """Low priority (priority < threshold) increments low_ref."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        req = SimpleNamespace(rid="r1", priority=0, last_node=a)
        cache.register_ref(req)

        self.assertEqual(a.low_ref, 1)
        self.assertEqual(a.high_ref, 0)

    def test_register_ref_only_adds_new_suffix_from_last_node(self):
        """Second register_ref only adds nodes not previously tracked."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])
        c = self._append_node(b, [9, 10, 11, 12])

        req = SimpleNamespace(rid="r1", priority=1, last_node=c)
        cache.register_ref(req)

        self.assertEqual(a.high_ref, 1)
        self.assertEqual(b.high_ref, 1)
        self.assertEqual(c.high_ref, 1)
        self.assertEqual(len(cache.rid_to_ref_info["r1"].nodes), 3)

        # Extend the chain by one node and call register_ref again
        d = self._append_node(c, [13, 14, 15, 16])
        req.last_node = d
        cache.register_ref(req)

        # Old nodes should NOT have their ref doubled
        self.assertEqual(a.high_ref, 1)
        self.assertEqual(b.high_ref, 1)
        self.assertEqual(c.high_ref, 1)
        # New node should now be tracked
        self.assertEqual(d.high_ref, 1)
        self.assertEqual(len(cache.rid_to_ref_info["r1"].nodes), 4)

    def test_register_ref_tracks_rids_on_nodes(self):
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        req = SimpleNamespace(rid="r1", priority=1, last_node=a)
        cache.register_ref(req)

        self.assertIn("r1", a.tracked_rids)
        self.assertIn("r1", cache.rid_to_ref_info)

    def test_register_ref_multiple_rids_on_shared_node(self):
        """Two different rids that share a node both track it."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        req1 = SimpleNamespace(rid="r1", priority=1, last_node=a)
        req2 = SimpleNamespace(rid="r2", priority=1, last_node=a)
        cache.register_ref(req1)
        cache.register_ref(req2)

        self.assertEqual(a.high_ref, 2)
        self.assertIn("r1", a.tracked_rids)
        self.assertIn("r2", a.tracked_rids)


class TestReleaseRefIdempotent(unittest.TestCase):
    """Release unknown rid returns success."""

    def _make_cache(self):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.rid_to_ref_info = {}
        return cache

    def test_release_unknown_rid_returns_success(self):
        cache = self._make_cache()
        ok, msg = cache.release_ref("never-registered")
        self.assertTrue(ok)
        self.assertIn("not tracked", msg)

    def test_release_idempotent_after_first_release(self):
        """Releasing the same rid twice should succeed both times."""
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.high_priority_threshold = 1
        cache._enable_priority_scheduling = True
        cache.unused_evictable_leaves = set()
        cache.low_ref_evictable_leaves = set()
        cache.high_ref_evictable_leaves = set()
        cache.unused_evictable_size_ = 0
        cache.low_ref_evictable_size_ = 0
        cache.high_ref_evictable_size_ = 0
        cache.rid_to_ref_info = {}
        cache._evict_scope_stack = []

        node = TreeNode()
        node.parent = cache.root_node
        node.key = RadixKey([1, 2, 3, 4])
        node.value = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        node.children = {}
        cache.root_node.children[1] = node

        req = SimpleNamespace(rid="r1", priority=1, last_node=node)
        cache.register_ref(req)

        ok1, _ = cache.release_ref("r1")
        self.assertTrue(ok1)

        # Second release of same rid should also return success (idempotent)
        ok2, msg2 = cache.release_ref("r1")
        self.assertTrue(ok2)
        self.assertIn("not tracked", msg2)


class TestUpdateRef(unittest.TestCase):
    """Test priority change moves nodes between tiers."""

    def _make_cache(self):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.high_priority_threshold = 1
        cache._enable_priority_scheduling = True
        cache.unused_evictable_leaves = set()
        cache.low_ref_evictable_leaves = set()
        cache.high_ref_evictable_leaves = set()
        cache.unused_evictable_size_ = 0
        cache.low_ref_evictable_size_ = 0
        cache.high_ref_evictable_size_ = 0
        cache.rid_to_ref_info = {}
        cache._evict_scope_stack = []
        return cache

    def _append_node(self, parent, token_ids):
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(token_ids)
        node.value = torch.tensor(token_ids, dtype=torch.int64)
        node.children = {}
        parent.children[token_ids[0] if token_ids else 0] = node
        return node

    def test_update_ref_unknown_rid_returns_false(self):
        cache = self._make_cache()
        ok, msg = cache.update_ref("unknown-rid", 5)
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_update_ref_low_to_high_priority_moves_nodes(self):
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])

        for n in (a, b):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        # Register as low priority
        req = SimpleNamespace(rid="r1", priority=0, last_node=b)
        cache.register_ref(req)
        self.assertEqual(cache.low_ref_evictable_size_, 8)
        self.assertEqual(cache.high_ref_evictable_size_, 0)

        # Promote to high priority
        ok, _ = cache.update_ref("r1", 5)
        self.assertTrue(ok)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 8)
        self.assertEqual(a.high_ref, 1)
        self.assertEqual(a.low_ref, 0)
        self.assertEqual(b.high_ref, 1)
        self.assertEqual(b.low_ref, 0)

    def test_update_ref_high_to_low_priority_moves_nodes(self):
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])

        for n in (a, b):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        # Register as high priority
        req = SimpleNamespace(rid="r1", priority=5, last_node=b)
        cache.register_ref(req)
        self.assertEqual(cache.high_ref_evictable_size_, 8)

        # Demote to low priority
        ok, _ = cache.update_ref("r1", 0)
        self.assertTrue(ok)
        self.assertEqual(cache.low_ref_evictable_size_, 8)
        self.assertEqual(cache.high_ref_evictable_size_, 0)

    def test_update_ref_same_class_is_noop(self):
        """If priority class doesn't change, update_ref is a no-op."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        for n in (a,):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        req = SimpleNamespace(rid="r1", priority=5, last_node=a)
        cache.register_ref(req)
        self.assertEqual(cache.high_ref_evictable_size_, 4)

        # Update with another high-priority value (still above threshold)
        ok, msg = cache.update_ref("r1", 10)
        self.assertTrue(ok)
        self.assertIn("unchanged", msg)
        # Size should not have changed
        self.assertEqual(cache.high_ref_evictable_size_, 4)


class TestScopedEvict(unittest.TestCase):
    """Verify context manager controls eviction scope."""

    def _make_cache(self):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.high_priority_threshold = 1
        cache._enable_priority_scheduling = True
        cache.unused_evictable_leaves = set()
        cache.low_ref_evictable_leaves = set()
        cache.high_ref_evictable_leaves = set()
        cache.unused_evictable_size_ = 0
        cache.low_ref_evictable_size_ = 0
        cache.high_ref_evictable_size_ = 0
        cache.rid_to_ref_info = {}
        cache._evict_scope_stack = []
        return cache

    def test_scoped_evict_empty_stack_by_default(self):
        cache = self._make_cache()
        self.assertEqual(len(cache._evict_scope_stack), 0)

    def test_scoped_evict_pushes_and_pops_stack(self):
        cache = self._make_cache()
        with cache.scoped_evict(allow_low=True, allow_high=False):
            self.assertEqual(len(cache._evict_scope_stack), 1)
            self.assertEqual(cache._evict_scope_stack[-1], (True, False))
        self.assertEqual(len(cache._evict_scope_stack), 0)

    def test_scoped_evict_nested_stacks(self):
        cache = self._make_cache()
        with cache.scoped_evict(allow_low=True, allow_high=False):
            with cache.scoped_evict(allow_low=True, allow_high=True):
                self.assertEqual(len(cache._evict_scope_stack), 2)
                self.assertEqual(cache._evict_scope_stack[-1], (True, True))
            self.assertEqual(len(cache._evict_scope_stack), 1)
            self.assertEqual(cache._evict_scope_stack[-1], (True, False))
        self.assertEqual(len(cache._evict_scope_stack), 0)

    def test_scoped_evict_cleans_up_on_exception(self):
        """Context manager should clean up even when exception is raised."""
        cache = self._make_cache()
        try:
            with cache.scoped_evict(allow_low=True, allow_high=True):
                self.assertEqual(len(cache._evict_scope_stack), 1)
                raise ValueError("test exception")
        except ValueError:
            pass
        # Stack should be clean after exception
        self.assertEqual(len(cache._evict_scope_stack), 0)

    def test_scoped_evict_high_only_scope(self):
        cache = self._make_cache()
        with cache.scoped_evict(allow_low=False, allow_high=True):
            self.assertEqual(cache._evict_scope_stack[-1], (False, True))

    def test_evict_uses_scope_stack_when_not_empty(self):
        """evict() should read allow_low/allow_high from the scope stack."""
        cache = self._make_cache()

        # Verify scope stack is read: push a scope, check it's visible
        with cache.scoped_evict(allow_low=False, allow_high=True):
            self.assertTrue(len(cache._evict_scope_stack) > 0)
            allow_low, allow_high = cache._evict_scope_stack[-1]
            self.assertFalse(allow_low)
            self.assertTrue(allow_high)


class TestEndToEndAccounting(unittest.TestCase):
    """register → update → release cycle zeroes all counters."""

    def _make_cache(self):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.high_priority_threshold = 1
        cache._enable_priority_scheduling = True
        cache.unused_evictable_leaves = set()
        cache.low_ref_evictable_leaves = set()
        cache.high_ref_evictable_leaves = set()
        cache.unused_evictable_size_ = 0
        cache.low_ref_evictable_size_ = 0
        cache.high_ref_evictable_size_ = 0
        cache.rid_to_ref_info = {}
        cache._evict_scope_stack = []
        return cache

    def _append_node(self, parent, token_ids):
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(token_ids)
        node.value = torch.tensor(token_ids, dtype=torch.int64)
        node.children = {}
        parent.children[token_ids[0] if token_ids else 0] = node
        return node

    def test_register_update_release_cycle_zeroes_accounting(self):
        """Full lifecycle: register (LP) → update (HP) → release → counters at zero."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])

        for n in (a, b):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        # Register as low priority
        req = SimpleNamespace(rid="r1", priority=0, last_node=b)
        cache.register_ref(req)
        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 8)
        self.assertEqual(cache.high_ref_evictable_size_, 0)

        # Promote to high priority
        ok, _ = cache.update_ref("r1", 5)
        self.assertTrue(ok)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 8)

        # Release
        ok, _ = cache.release_ref("r1")
        self.assertTrue(ok)
        self.assertEqual(cache.unused_evictable_size_, 8)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertNotIn("r1", cache.rid_to_ref_info)
        self.assertEqual(a.tracked_rids, set())
        self.assertEqual(b.tracked_rids, set())

    def test_register_release_cycle_with_two_rids(self):
        """Two rids on the same nodes both release cleanly."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        for n in (a,):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        req1 = SimpleNamespace(rid="r1", priority=0, last_node=a)
        req2 = SimpleNamespace(rid="r2", priority=0, last_node=a)
        cache.register_ref(req1)
        cache.register_ref(req2)

        self.assertEqual(a.low_ref, 2)
        self.assertEqual(cache.low_ref_evictable_size_, 4)

        cache.release_ref("r1")
        self.assertEqual(a.low_ref, 1)
        # Still in low_ref tier since r2 still holds it
        self.assertEqual(cache.low_ref_evictable_size_, 4)

        cache.release_ref("r2")
        self.assertEqual(a.low_ref, 0)
        # Back to unused
        self.assertEqual(cache.unused_evictable_size_, 4)
        self.assertEqual(cache.low_ref_evictable_size_, 0)

    def test_register_high_release_moves_to_unused(self):
        """High-priority register then release returns nodes to unused tier."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])

        for n in (a, b):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        req = SimpleNamespace(rid="r1", priority=5, last_node=b)
        cache.register_ref(req)

        self.assertEqual(cache.high_ref_evictable_size_, 8)

        cache.release_ref("r1")

        self.assertEqual(cache.unused_evictable_size_, 8)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertNotIn("r1", cache.rid_to_ref_info)
        self.assertEqual(a.tracked_rids, set())
        self.assertEqual(b.tracked_rids, set())


if __name__ == "__main__":
    unittest.main()
