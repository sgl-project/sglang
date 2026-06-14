"""Unit tests for lora/eviction_policy.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest

from sglang.srt.lora.eviction_policy import (
    FIFOEvictionPolicy,
    LRUEvictionPolicy,
    get_eviction_policy,
)
from sglang.test.test_utils import CustomTestCase


# ---------------------------------------------------------------------------
# LRUEvictionPolicy
# ---------------------------------------------------------------------------
class TestLRUEvictionPolicyInit(CustomTestCase):
    """Verify initial state of a fresh LRU policy."""

    def test_initial_state(self):
        policy = LRUEvictionPolicy()
        self.assertEqual(len(policy.access_order), 0)
        self.assertEqual(policy.total_accesses, 0)
        self.assertEqual(policy.eviction_count, 0)


class TestLRUMarkUsed(CustomTestCase):
    """Tests for LRUEvictionPolicy.mark_used()."""

    def setUp(self):
        self.policy = LRUEvictionPolicy()

    def test_mark_used_adds_uid(self):
        self.policy.mark_used("adapter_a")
        self.assertIn("adapter_a", self.policy.access_order)
        self.assertEqual(self.policy.total_accesses, 1)

    def test_mark_used_none_is_ignored(self):
        """None uid (base model) should not be tracked."""
        self.policy.mark_used(None)
        self.assertEqual(len(self.policy.access_order), 0)
        self.assertEqual(self.policy.total_accesses, 0)

    def test_mark_used_updates_access_order(self):
        """Re-marking a uid moves it to the end (most recent)."""
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.mark_used("a")  # re-access moves "a" to end
        keys = list(self.policy.access_order.keys())
        self.assertEqual(keys, ["b", "a"])

    def test_mark_used_increments_total_accesses(self):
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.mark_used("a")
        self.assertEqual(self.policy.total_accesses, 3)

    def test_mark_used_updates_timestamp(self):
        """Each mark_used call should record a monotonically increasing time."""
        self.policy.mark_used("a")
        t1 = self.policy.access_order["a"]
        self.policy.mark_used("b")
        self.policy.mark_used("a")
        t2 = self.policy.access_order["a"]
        self.assertGreater(t2, t1)


class TestLRUSelectVictim(CustomTestCase):
    """Tests for LRUEvictionPolicy.select_victim()."""

    def setUp(self):
        self.policy = LRUEvictionPolicy()

    def test_selects_least_recently_used(self):
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.mark_used("c")
        # "a" is the oldest
        victim = self.policy.select_victim({"a", "b", "c"})
        self.assertEqual(victim, "a")

    def test_selects_lru_among_candidates_subset(self):
        """Only candidates are considered — the global LRU may not be a candidate."""
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.mark_used("c")
        # "a" is LRU but not a candidate; "b" should be selected
        victim = self.policy.select_victim({"b", "c"})
        self.assertEqual(victim, "b")

    def test_respects_reaccess_order(self):
        """After re-accessing, the eviction order should change."""
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.mark_used("c")
        self.policy.mark_used("a")  # "a" is now most recent
        victim = self.policy.select_victim({"a", "b", "c"})
        self.assertEqual(victim, "b")

    def test_selects_none_when_only_candidate(self):
        """When None is the only candidate and no tracked uid matches."""
        self.policy.mark_used("a")
        victim = self.policy.select_victim({None})
        self.assertIsNone(victim)

    def test_selects_none_when_no_tracked_match(self):
        """None in candidates falls back when no tracked uid is a candidate."""
        self.policy.mark_used("a")
        victim = self.policy.select_victim({None, "x"})
        # "x" is not tracked; None is the fallback
        self.assertIsNone(victim)

    def test_increments_eviction_count(self):
        self.policy.mark_used("a")
        self.policy.select_victim({"a"})
        self.assertEqual(self.policy.eviction_count, 1)
        self.policy.select_victim({"a"})
        self.assertEqual(self.policy.eviction_count, 2)

    def test_assert_on_empty_candidates_with_no_match(self):
        """Should assert when candidates is non-empty but no uid can be selected."""
        self.policy.mark_used("a")
        with self.assertRaises(AssertionError):
            self.policy.select_victim({"x", "y"})

    def test_single_candidate(self):
        self.policy.mark_used("only")
        victim = self.policy.select_victim({"only"})
        self.assertEqual(victim, "only")


class TestLRURemove(CustomTestCase):
    """Tests for LRUEvictionPolicy.remove()."""

    def setUp(self):
        self.policy = LRUEvictionPolicy()

    def test_remove_existing(self):
        self.policy.mark_used("a")
        self.policy.remove("a")
        self.assertNotIn("a", self.policy.access_order)

    def test_remove_nonexistent_is_noop(self):
        """Removing a uid that was never tracked should not raise."""
        self.policy.remove("ghost")  # should not raise

    def test_remove_none_is_noop(self):
        """Removing None should be a no-op."""
        self.policy.mark_used("a")
        self.policy.remove(None)
        self.assertIn("a", self.policy.access_order)

    def test_remove_then_select_skips_removed(self):
        """After removing the LRU entry, the next oldest should be selected."""
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.mark_used("c")
        self.policy.remove("a")
        victim = self.policy.select_victim({"a", "b", "c"})
        self.assertEqual(victim, "b")


# ---------------------------------------------------------------------------
# FIFOEvictionPolicy
# ---------------------------------------------------------------------------
class TestFIFOEvictionPolicyInit(CustomTestCase):
    """Verify initial state of a fresh FIFO policy."""

    def test_initial_state(self):
        policy = FIFOEvictionPolicy()
        self.assertEqual(len(policy.insertion_order), 0)
        self.assertEqual(policy.eviction_count, 0)


class TestFIFOMarkUsed(CustomTestCase):
    """Tests for FIFOEvictionPolicy.mark_used()."""

    def setUp(self):
        self.policy = FIFOEvictionPolicy()

    def test_mark_used_adds_uid(self):
        self.policy.mark_used("adapter_a")
        self.assertIn("adapter_a", self.policy.insertion_order)

    def test_mark_used_none_is_ignored(self):
        self.policy.mark_used(None)
        self.assertEqual(len(self.policy.insertion_order), 0)

    def test_mark_used_preserves_insertion_order(self):
        """Re-marking a uid should NOT change its insertion position (FIFO semantics)."""
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.mark_used("a")  # re-access — should NOT move "a"
        keys = list(self.policy.insertion_order.keys())
        self.assertEqual(keys, ["a", "b"])

    def test_mark_used_does_not_overwrite_existing(self):
        """Calling mark_used on an existing uid should not duplicate it."""
        self.policy.mark_used("a")
        self.policy.mark_used("a")
        self.assertEqual(len(self.policy.insertion_order), 1)


class TestFIFOSelectVictim(CustomTestCase):
    """Tests for FIFOEvictionPolicy.select_victim()."""

    def setUp(self):
        self.policy = FIFOEvictionPolicy()

    def test_selects_first_inserted(self):
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.mark_used("c")
        victim = self.policy.select_victim({"a", "b", "c"})
        self.assertEqual(victim, "a")

    def test_selects_fifo_among_candidates_subset(self):
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.mark_used("c")
        victim = self.policy.select_victim({"b", "c"})
        self.assertEqual(victim, "b")

    def test_reaccess_does_not_change_fifo_order(self):
        """Unlike LRU, re-accessing should NOT affect eviction order."""
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.mark_used("a")  # re-access — no effect on FIFO order
        victim = self.policy.select_victim({"a", "b"})
        self.assertEqual(victim, "a")

    def test_selects_none_when_only_candidate(self):
        self.policy.mark_used("a")
        victim = self.policy.select_victim({None})
        self.assertIsNone(victim)

    def test_selects_none_when_no_tracked_match(self):
        self.policy.mark_used("a")
        victim = self.policy.select_victim({None, "x"})
        self.assertIsNone(victim)

    def test_increments_eviction_count(self):
        self.policy.mark_used("a")
        self.policy.select_victim({"a"})
        self.assertEqual(self.policy.eviction_count, 1)

    def test_assert_on_empty_candidates_with_no_match(self):
        self.policy.mark_used("a")
        with self.assertRaises(AssertionError):
            self.policy.select_victim({"x", "y"})

    def test_single_candidate(self):
        self.policy.mark_used("only")
        victim = self.policy.select_victim({"only"})
        self.assertEqual(victim, "only")


class TestFIFORemove(CustomTestCase):
    """Tests for FIFOEvictionPolicy.remove()."""

    def setUp(self):
        self.policy = FIFOEvictionPolicy()

    def test_remove_existing(self):
        self.policy.mark_used("a")
        self.policy.remove("a")
        self.assertNotIn("a", self.policy.insertion_order)

    def test_remove_nonexistent_is_noop(self):
        self.policy.remove("ghost")

    def test_remove_none_is_noop(self):
        self.policy.mark_used("a")
        self.policy.remove(None)
        self.assertIn("a", self.policy.insertion_order)

    def test_remove_then_select_skips_removed(self):
        self.policy.mark_used("a")
        self.policy.mark_used("b")
        self.policy.remove("a")
        victim = self.policy.select_victim({"a", "b"})
        self.assertEqual(victim, "b")


# ---------------------------------------------------------------------------
# get_eviction_policy factory
# ---------------------------------------------------------------------------
class TestGetEvictionPolicy(CustomTestCase):
    """Tests for the get_eviction_policy() factory function."""

    def test_returns_lru_instance(self):
        policy = get_eviction_policy("lru")
        self.assertIsInstance(policy, LRUEvictionPolicy)

    def test_returns_fifo_instance(self):
        policy = get_eviction_policy("fifo")
        self.assertIsInstance(policy, FIFOEvictionPolicy)

    def test_unknown_policy_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            get_eviction_policy("random")
        self.assertIn("random", str(ctx.exception))

    def test_each_call_returns_fresh_instance(self):
        p1 = get_eviction_policy("lru")
        p2 = get_eviction_policy("lru")
        self.assertIsNot(p1, p2)


# ---------------------------------------------------------------------------
# Cross-policy behavioral comparison
# ---------------------------------------------------------------------------
class TestLRUVsFIFOBehavior(CustomTestCase):
    """Verify the key behavioral difference between LRU and FIFO."""

    def test_reaccess_changes_lru_victim_but_not_fifo(self):
        """LRU should change victim after re-access; FIFO should not."""
        lru = LRUEvictionPolicy()
        fifo = FIFOEvictionPolicy()

        for policy in (lru, fifo):
            policy.mark_used("a")
            policy.mark_used("b")
            policy.mark_used("c")

        # Re-access "a" — this should move it to the end in LRU only
        lru.mark_used("a")
        fifo.mark_used("a")

        candidates = {"a", "b", "c"}
        self.assertEqual(lru.select_victim(candidates), "b")
        self.assertEqual(fifo.select_victim(candidates), "a")


# ---------------------------------------------------------------------------
# Stress / many-adapters scenario
# ---------------------------------------------------------------------------
class TestManyAdapters(CustomTestCase):
    """Stress-test eviction ordering with many adapters."""

    def test_lru_evicts_in_access_order(self):
        policy = LRUEvictionPolicy()
        uids = [f"adapter_{i}" for i in range(100)]
        for uid in uids:
            policy.mark_used(uid)
        # Oldest should be evicted first
        all_set = set(uids)
        for expected in uids:
            victim = policy.select_victim(all_set)
            self.assertEqual(victim, expected)
            policy.remove(victim)
            all_set.discard(victim)

    def test_fifo_evicts_in_insertion_order(self):
        policy = FIFOEvictionPolicy()
        uids = [f"adapter_{i}" for i in range(100)]
        for uid in uids:
            policy.mark_used(uid)
        all_set = set(uids)
        for expected in uids:
            victim = policy.select_victim(all_set)
            self.assertEqual(victim, expected)
            policy.remove(victim)
            all_set.discard(victim)


if __name__ == "__main__":
    unittest.main()
