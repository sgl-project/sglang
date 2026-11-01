"""CPU-only unit tests for the Mamba host eviction order."""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest
from collections import defaultdict

import torch

from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedLRUList, UnifiedTreeNode

MAMBA = ComponentType.MAMBA


class _FakeTreeCore:
    tree_components = (ComponentType.FULL, ComponentType.MAMBA)
    enable_session_radix_cache = False

    def __init__(self):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.evictable_host_leaves = set()
        self.host_lru_lists = {
            MAMBA: UnifiedLRUList(MAMBA, self.tree_components, use_host_ptr=True)
        }
        self.evicted = []

    def _evict_component_and_detach_lru(
        self, node, component, *, target, tracker, device_frees, host_frees, **_
    ):
        # Host tombstone: free the slot and leave the host LRU.
        cd = node.component_data[MAMBA]
        host_frees[MAMBA].append(cd.host_value)
        cd.host_value = None
        self.host_lru_lists[MAMBA].remove_node(node)
        tracker[MAMBA] += 1
        self.evicted.append(node)

    def _cascade_evict(self, *args, **kwargs):
        pass

    def _update_evictable_leaf_sets(self, node):
        pass


def _component(core):
    component = object.__new__(MambaComponent)
    component.tree_core = core
    return component


def _child(core, parent, key):
    node = UnifiedTreeNode(core.tree_components)
    node.parent = parent
    parent.children[key] = node
    return node


def _chain(core, length):
    nodes, parent = [], core.root_node
    for index in range(length):
        parent = _child(core, parent, index)
        nodes.append(parent)
    return nodes


def _seed_host_lru(core, nodes):
    """Give `nodes` host Mamba slots 10, 11, ... in LRU order (first is least recent)."""
    for index, node in enumerate(nodes):
        node.component_data[MAMBA].host_value = torch.tensor([10 + index])
        core.host_lru_lists[MAMBA].insert_mru(node)


def _drive(core, num_tokens):
    tracker = defaultdict(int)
    host_frees = defaultdict(list)
    _component(core).drive_host_eviction(
        num_tokens, tracker, defaultdict(list), host_frees, []
    )
    freed = [value.item() for value in host_frees[MAMBA]]
    assert tracker[MAMBA] == len(freed)
    return freed


def _has_host_state(node):
    return node.component_data[MAMBA].host_value is not None


class TestMambaHostEvictionOrder(unittest.TestCase):
    def test_pass_through_checkpoint_goes_before_an_older_tail(self):
        core = _FakeTreeCore()
        a, tail = _chain(core, 2)
        # The tail is least recent; plain LRU order would evict it first.
        _seed_host_lru(core, [tail, a])

        self.assertEqual(_drive(core, 1), [11])
        self.assertFalse(_has_host_state(a))
        self.assertTrue(_has_host_state(tail))
        self.assertTrue(core.host_lru_lists[MAMBA].in_list(tail))

    def test_fork_is_kept_until_the_fallback_pass(self):
        core = _FakeTreeCore()
        fork, left = _chain(core, 2)
        passing = _child(core, fork, "right")
        right = _child(core, passing, "leaf")
        # The fork is least recent but shared by both branches.
        _seed_host_lru(core, [fork, left, right, passing])

        self.assertEqual(_drive(core, 1), [13])
        self.assertFalse(_has_host_state(passing))
        for node in (fork, left, right):
            self.assertTrue(_has_host_state(node))

    def test_state_whose_child_has_none_is_kept(self):
        core = _FakeTreeCore()
        p, a, _stateless, tail = _chain(core, 4)
        # `a` is the deepest state of every hit that ends inside `_stateless`.
        _seed_host_lru(core, [a, tail, p])

        self.assertEqual(_drive(core, 1), [12])
        self.assertTrue(_has_host_state(a))
        self.assertTrue(_has_host_state(tail))

    def test_falls_back_to_lru_order_once_no_pass_through_is_left(self):
        core = _FakeTreeCore()
        a, b, tail = _chain(core, 3)
        _seed_host_lru(core, [tail, a, b])

        # a and b are pass-through checkpoints; the tail goes in the fallback pass.
        self.assertEqual(_drive(core, 3), [11, 12, 10])
        self.assertEqual(core.evicted, [a, b, tail])
        for node in (a, b, tail):
            self.assertFalse(core.host_lru_lists[MAMBA].in_list(node))


if __name__ == "__main__":
    unittest.main()
