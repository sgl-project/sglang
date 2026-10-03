"""CPU-only unit tests for the per-path SWA window cap."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from collections import defaultdict

import torch

from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedLRUList, UnifiedTreeNode


class _FakeTreeCore:
    tree_components = (ComponentType.FULL, ComponentType.SWA)

    def __init__(self):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.evictable_device_leaves = set()
        self.component_evictable_size_ = {ComponentType.SWA: 0}
        self.component_protected_size_ = {ComponentType.SWA: 0}
        self.lru_lists = {
            ComponentType.SWA: UnifiedLRUList(ComponentType.SWA, self.tree_components)
        }
        self.host_lru_lists = {
            ComponentType.SWA: UnifiedLRUList(
                ComponentType.SWA, self.tree_components, use_host_ptr=True
            )
        }
        self.evicted = []
        self._is_tracking_unbacked_tokens = False
        self._tracked_unbacked_tokens = 0

    def _evict_component_and_detach_lru(self, node, component, *args, **kwargs):
        self.evicted.append(node)
        return UnifiedTreeCore._evict_component_and_detach_lru(
            self, node, component, *args, **kwargs
        )

    def _cascade_evict(self, node, component, tracker, device_frees, host_frees):
        pass


class _FakeUnifiedCache:
    tree_components = _FakeTreeCore.tree_components


def _build_chain(cap, length=3):
    core = _FakeTreeCore()
    component = object.__new__(SWAComponent)
    component.cache = _FakeUnifiedCache()
    component.tree_core = core
    component.swa_max_states_per_path = cap

    nodes = []
    parent = core.root_node
    for index in range(length):
        node = UnifiedTreeNode(core.tree_components)
        node.parent = parent
        node.component_data[ComponentType.FULL].value = torch.tensor([100 + index])
        node.component_data[ComponentType.SWA].value = torch.tensor([index])
        parent.children[index] = node
        core.component_evictable_size_[ComponentType.SWA] += 1
        core.lru_lists[ComponentType.SWA].insert_mru(node)
        nodes.append(node)
        parent = node
    return component, nodes, core


def _trim(component, tail):
    device_frees = defaultdict(list)
    component._evict_excess_path_states(tail, device_frees, defaultdict(list))
    return device_frees


class TestSWAPathWindowCap(unittest.TestCase):
    def test_removes_shallow_windows_but_keeps_full_kv(self):
        component, nodes, core = _build_chain(cap=1)

        device_frees = _trim(component, nodes[-1])

        self.assertEqual(core.evicted, [nodes[0], nodes[1]])
        swa = [node.component_data[ComponentType.SWA].value for node in nodes]
        self.assertEqual([value is None for value in swa], [True, True, False])
        # SWA slots are freed through their FULL indices, never the SWA ones.
        self.assertEqual(
            [value.item() for value in device_frees[ComponentType.SWA]], [100, 101]
        )
        self.assertTrue(
            all(
                node.component_data[ComponentType.FULL].value is not None
                for node in nodes
            )
        )
        self.assertEqual(core.component_evictable_size_[ComponentType.SWA], 1)

    def test_cap_is_soft_for_fork_and_locked_nodes(self):
        component, nodes, core = _build_chain(cap=1, length=4)
        nodes[0].children["fork"] = UnifiedTreeNode(core.tree_components)
        nodes[1].component_data[ComponentType.SWA].lock_ref = 1

        _trim(component, nodes[-1])

        self.assertEqual(core.evicted, [nodes[2]])

    def test_negative_one_disables_cap(self):
        component, nodes, core = _build_chain(cap=-1)

        self.assertEqual(dict(_trim(component, nodes[-1])), {})
        self.assertEqual(core.evicted, [])


if __name__ == "__main__":
    unittest.main()
