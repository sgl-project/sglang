"""Unit tests for KVFlow workflow-aware eviction (arXiv:2507.07400).

Tests KVFlowAgentManager priority logic, KVFlowEvictionStrategy ordering,
and the _steps_to_priority mapping — all without GPU or SGLang runtime deps.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.evict_policy import KVFlowEvictionStrategy
from sglang.srt.mem_cache.kvflow_agent_manager import (
    KVFlowAgentManager,
    _steps_to_priority,
)
from sglang.test.test_utils import CustomTestCase


def _make_node(parent=None, kvflow_priority: int = 0, last_access_time: float = 0.0):
    node = MagicMock()
    node.parent = parent
    node.kvflow_priority = kvflow_priority
    node.last_access_time = last_access_time
    return node


def _make_chain(length: int):
    """Build a root→…→leaf chain of `length` nodes; return (root, leaf)."""
    root = _make_node(parent=None)
    cur = root
    for _ in range(length - 1):
        child = _make_node(parent=cur)
        cur = child
    return root, cur  # (root, leaf)


class TestStepsToPriority(CustomTestCase):
    def test_steps_one_gets_max_protection(self):
        # steps=1 (next agent) → hold_step, maximum priority
        self.assertEqual(_steps_to_priority(1, hold_step=4), 4)

    def test_steps_boundary_just_before_hold(self):
        # steps=hold_step-1 gets priority 2
        self.assertEqual(_steps_to_priority(3, hold_step=4), 2)

    def test_steps_at_hold_gets_minimum_protection(self):
        # steps=hold_step → 1, minimal but still non-zero
        self.assertEqual(_steps_to_priority(4, hold_step=4), 1)

    def test_steps_beyond_hold_also_get_minimum(self):
        # steps > hold_step → same 1 (far future, low protection)
        self.assertEqual(_steps_to_priority(10, hold_step=4), 1)

    def test_priority_strictly_decreases_with_steps(self):
        # Closer to execution → higher priority
        priorities = [_steps_to_priority(s, hold_step=4) for s in [1, 2, 3, 4, 5]]
        self.assertEqual(priorities, sorted(priorities, reverse=True))

    def test_custom_hold_step(self):
        self.assertEqual(_steps_to_priority(1, hold_step=8), 8)
        self.assertEqual(_steps_to_priority(7, hold_step=8), 2)
        self.assertEqual(_steps_to_priority(8, hold_step=8), 1)


class TestKVFlowEvictionStrategy(CustomTestCase):
    def setUp(self):
        self.strategy = KVFlowEvictionStrategy()

    def _node(self, kvflow_priority, last_access_time):
        node = MagicMock()
        node.kvflow_priority = kvflow_priority
        node.last_access_time = last_access_time
        return node

    def test_returns_kvflow_priority_then_lru(self):
        node = self._node(kvflow_priority=3, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (3, 5.0))

    def test_lower_kvflow_priority_evicted_first(self):
        protected = self._node(kvflow_priority=4, last_access_time=1.0)
        unprotected = self._node(kvflow_priority=0, last_access_time=10.0)
        # unprotected (priority=0) has smaller tuple → evicted first
        self.assertLess(
            self.strategy.get_priority(unprotected),
            self.strategy.get_priority(protected),
        )

    def test_same_priority_older_access_evicted_first(self):
        old = self._node(kvflow_priority=2, last_access_time=1.0)
        new = self._node(kvflow_priority=2, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old),
            self.strategy.get_priority(new),
        )

    def test_eviction_ordering_matches_expected_sequence(self):
        nodes = [
            self._node(kvflow_priority=0, last_access_time=5.0),  # evict 1st (low prio)
            self._node(kvflow_priority=4, last_access_time=1.0),  # evict 4th (high prio, but old)
            self._node(kvflow_priority=1, last_access_time=2.0),  # evict 2nd
            self._node(kvflow_priority=4, last_access_time=8.0),  # evict 5th (high prio, new)
            self._node(kvflow_priority=1, last_access_time=7.0),  # evict 3rd
        ]
        eviction_order = sorted(nodes, key=self.strategy.get_priority)
        expected_keys = [(0, 5.0), (1, 2.0), (1, 7.0), (4, 1.0), (4, 8.0)]
        actual_keys = [self.strategy.get_priority(n) for n in eviction_order]
        self.assertEqual(actual_keys, expected_keys)


class TestKVFlowAgentManagerRegisterLeaf(CustomTestCase):
    def test_register_adds_leaf_to_agent(self):
        mgr = KVFlowAgentManager(hold_step=4)
        leaf = _make_node()
        mgr.register_leaf("agent_A", leaf)
        self.assertIn(leaf, mgr._agent_to_leaves["agent_A"])

    def test_register_none_leaf_is_ignored(self):
        mgr = KVFlowAgentManager(hold_step=4)
        mgr.register_leaf("agent_A", None)
        self.assertNotIn("agent_A", mgr._agent_to_leaves)

    def test_register_multiple_leaves_same_agent(self):
        mgr = KVFlowAgentManager(hold_step=4)
        leaf1 = _make_node()
        leaf2 = _make_node()
        mgr.register_leaf("agent_A", leaf1)
        mgr.register_leaf("agent_A", leaf2)
        self.assertEqual(mgr._agent_to_leaves["agent_A"], {leaf1, leaf2})

    def test_register_different_agents(self):
        mgr = KVFlowAgentManager(hold_step=4)
        leaf_a = _make_node()
        leaf_b = _make_node()
        mgr.register_leaf("agent_A", leaf_a)
        mgr.register_leaf("agent_B", leaf_b)
        self.assertIn(leaf_a, mgr._agent_to_leaves["agent_A"])
        self.assertIn(leaf_b, mgr._agent_to_leaves["agent_B"])


class TestKVFlowAgentManagerUpdate(CustomTestCase):
    def _setup_chain(self, agent_id: str, chain_len: int, mgr: KVFlowAgentManager):
        """Create a chain and register its leaf with mgr. Return (root, leaf)."""
        root, leaf = _make_chain(chain_len)
        mgr.register_leaf(agent_id, leaf)
        return root, leaf

    def test_update_sets_priority_on_path(self):
        mgr = KVFlowAgentManager(hold_step=4)
        root, leaf = self._setup_chain("agent_A", 4, mgr)

        mgr.update({"agent_A": 1})  # steps=1 → priority=4

        self.assertEqual(leaf.kvflow_priority, 4)
        self.assertEqual(leaf.parent.kvflow_priority, 4)
        self.assertEqual(leaf.parent.parent.kvflow_priority, 4)
        self.assertEqual(root.kvflow_priority, 0)  # root excluded from walk

    def test_steps_zero_skipped_no_protection(self):
        mgr = KVFlowAgentManager(hold_step=4)
        _, leaf = self._setup_chain("agent_A", 3, mgr)

        mgr.update({"agent_A": 0})  # steps=0 = currently running, skip

        self.assertEqual(leaf.kvflow_priority, 0)
        self.assertEqual(leaf.parent.kvflow_priority, 0)

    def test_stale_agent_removed_from_tracking(self):
        mgr = KVFlowAgentManager(hold_step=4)
        _, leaf = self._setup_chain("agent_A", 3, mgr)

        # First update registers agent_A
        mgr.update({"agent_A": 1})
        self.assertIn("agent_A", mgr._agent_to_leaves)

        # Second update omits agent_A → stale, should be removed
        mgr.update({"agent_B": 2})
        self.assertNotIn("agent_A", mgr._agent_to_leaves)

    def test_stale_agent_priorities_reset(self):
        mgr = KVFlowAgentManager(hold_step=4)
        _, leaf = self._setup_chain("agent_A", 3, mgr)

        mgr.update({"agent_A": 1})
        self.assertEqual(leaf.kvflow_priority, 4)

        # Next update drops agent_A — its nodes must be reset to 0
        mgr.update({})
        self.assertEqual(leaf.kvflow_priority, 0)
        self.assertEqual(leaf.parent.kvflow_priority, 0)

    def test_shared_ancestor_gets_best_priority(self):
        """Two agents sharing a prefix node → node gets the higher priority."""
        mgr = KVFlowAgentManager(hold_step=4)

        # Build: root → shared → leaf_A
        #                      → leaf_B
        # shared must have a parent so _walk_path visits it (walk stops before parent=None)
        root = _make_node(parent=None)
        shared = _make_node(parent=root)
        leaf_a = _make_node(parent=shared)
        leaf_b = _make_node(parent=shared)

        mgr.register_leaf("agent_A", leaf_a)
        mgr.register_leaf("agent_B", leaf_b)

        # agent_A: steps=1 (priority=4), agent_B: steps=3 (priority=2)
        mgr.update({"agent_A": 1, "agent_B": 3})

        # shared is on both paths → should get max(4, 2) = 4
        self.assertEqual(shared.kvflow_priority, 4)
        self.assertEqual(leaf_a.kvflow_priority, 4)
        self.assertEqual(leaf_b.kvflow_priority, 2)
        self.assertEqual(root.kvflow_priority, 0)  # root excluded from walk

    def test_full_refresh_on_each_update(self):
        """Priorities are fully recomputed from scratch each update call."""
        mgr = KVFlowAgentManager(hold_step=4)
        _, leaf = self._setup_chain("agent_A", 3, mgr)

        # Round 1: steps=1 → priority=4
        mgr.update({"agent_A": 1})
        self.assertEqual(leaf.kvflow_priority, 4)

        # Round 2: steps=3 → priority=2 (not additive — should overwrite to 2)
        mgr.update({"agent_A": 3})
        self.assertEqual(leaf.kvflow_priority, 2)
        self.assertEqual(leaf.parent.kvflow_priority, 2)

    def test_far_future_agent_gets_minimal_protection(self):
        mgr = KVFlowAgentManager(hold_step=4)
        _, leaf = self._setup_chain("agent_A", 3, mgr)

        mgr.update({"agent_A": 10})  # steps=10 >> hold_step=4 → priority=1

        self.assertEqual(leaf.kvflow_priority, 1)

    def test_empty_update_resets_all(self):
        mgr = KVFlowAgentManager(hold_step=4)
        _, leaf = self._setup_chain("agent_A", 3, mgr)
        mgr.update({"agent_A": 1})
        self.assertEqual(leaf.kvflow_priority, 4)

        mgr.update({})
        self.assertEqual(leaf.kvflow_priority, 0)
        self.assertFalse(mgr._agent_to_leaves)

    def test_unknown_agent_in_update_is_a_noop(self):
        """agent in update but never registered has no leaves → no-op, no crash."""
        mgr = KVFlowAgentManager(hold_step=4)
        mgr.update({"ghost_agent": 1})  # should not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
