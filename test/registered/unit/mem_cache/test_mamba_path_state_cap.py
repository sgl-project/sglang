"""CPU-only unit tests for the per-path Mamba checkpoint cap."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import argparse
from collections import defaultdict
import unittest

import torch

from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedLRUList, UnifiedTreeNode
from sglang.srt.server_args import ServerArgs


class _FakeTreeCore:
    tree_components = (ComponentType.FULL, ComponentType.MAMBA)

    def __init__(self):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.evictable_device_leaves = set()
        self.component_evictable_size_ = {ComponentType.MAMBA: 0}
        self.component_protected_size_ = {ComponentType.MAMBA: 0}
        self.lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(
                ComponentType.MAMBA, self.tree_components
            )
        }
        self.host_lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(
                ComponentType.MAMBA, self.tree_components, use_host_ptr=True
            )
        }
        self.evicted = []
        self.cascaded = []

    def _evict_component_and_detach_lru(self, node, component, *args, **kwargs):
        self.evicted.append(node)
        return UnifiedTreeCore._evict_component_and_detach_lru(
            self, node, component, *args, **kwargs
        )

    def _cascade_evict(self, node, component, tracker, device_frees, host_frees):
        self.cascaded.append(node)


class _FakeUnifiedCache:
    tree_components = _FakeTreeCore.tree_components


def _build_unified_chain(cap, length=3):
    cache = _FakeUnifiedCache()
    core = _FakeTreeCore()
    component = object.__new__(MambaComponent)
    component.cache = cache
    component.tree_core = core
    component.mamba_max_states_per_path = cap

    nodes = []
    parent = core.root_node
    for index in range(length):
        node = UnifiedTreeNode(core.tree_components)
        node.parent = parent
        node.component_data[ComponentType.FULL].value = torch.tensor([100 + index])
        node.component_data[ComponentType.MAMBA].value = torch.tensor([index])
        parent.children[index] = node
        core.component_evictable_size_[ComponentType.MAMBA] += 1
        core.lru_lists[ComponentType.MAMBA].insert_mru(node)
        nodes.append(node)
        parent = node
    return component, nodes, core, cache


class TestMambaPathStateCap(unittest.TestCase):
    def test_server_arg_defaults_to_unlimited(self):
        self.assertEqual(
            ServerArgs(model_path="dummy").mamba_max_states_per_path,
            -1,
        )

    def test_server_arg_cli(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(
            ["--model-path", "dummy", "--mamba-max-states-per-path", "3"]
        )

        self.assertEqual(args.mamba_max_states_per_path, 3)

    def test_server_arg_rejects_zero_and_values_below_negative_one(self):
        for value in (0, -2):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "must be -1 \\(unlimited\\) or a positive integer",
            ):
                ServerArgs(
                    model_path="dummy",
                    mamba_max_states_per_path=value,
                )

    def test_unified_cache_removes_only_shallow_mamba_state(self):
        component, nodes, core, cache = _build_unified_chain(cap=2)

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        component._evict_excess_path_states(nodes[-1], device_frees, host_frees)

        self.assertEqual(core.evicted, [nodes[0]])
        self.assertEqual(core.cascaded, [nodes[0]])
        self.assertIsNone(nodes[0].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[-1].component_data[ComponentType.MAMBA].value)
        self.assertEqual(
            [v.item() for v in device_frees[ComponentType.MAMBA]],
            [0],
        )
        self.assertEqual(core.component_evictable_size_[ComponentType.MAMBA], 2)
        self.assertFalse(core.lru_lists[ComponentType.MAMBA].in_list(nodes[0]))
        self.assertTrue(
            all(
                node.component_data[ComponentType.FULL].value is not None
                for node in nodes
            )
        )

    def test_unified_cache_cap_is_soft_for_fork_and_locked_nodes(self):
        component, nodes, core, cache = _build_unified_chain(cap=1, length=4)
        fork_child = UnifiedTreeNode(core.tree_components)
        fork_child.parent = nodes[0]
        nodes[0].children["fork"] = fork_child
        nodes[1].component_data[ComponentType.MAMBA].lock_ref = 1

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        component._evict_excess_path_states(nodes[-1], device_frees, host_frees)

        self.assertEqual(core.evicted, [nodes[2]])
        self.assertIsNotNone(nodes[0].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[1].component_data[ComponentType.MAMBA].value)
        self.assertIsNone(nodes[2].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[3].component_data[ComponentType.MAMBA].value)

    def test_unified_cache_preserves_existing_host_backup(self):
        component, nodes, core, cache = _build_unified_chain(cap=2)
        mamba_data = nodes[0].component_data[ComponentType.MAMBA]
        mamba_data.host_value = torch.tensor([10])

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        component._evict_excess_path_states(nodes[-1], device_frees, host_frees)

        self.assertIsNone(mamba_data.value)
        self.assertIsNotNone(mamba_data.host_value)
        self.assertTrue(core.host_lru_lists[ComponentType.MAMBA].in_list(nodes[0]))

    def test_unified_cache_negative_one_disables_cap(self):
        component, nodes, core, cache = _build_unified_chain(cap=-1)

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        component._evict_excess_path_states(nodes[-1], device_frees, host_frees)

        self.assertEqual(dict(device_frees), {})
        self.assertEqual(core.evicted, [])
        self.assertTrue(
            all(
                node.component_data[ComponentType.MAMBA].value is not None
                for node in nodes
            )
        )


if __name__ == "__main__":
    unittest.main()
