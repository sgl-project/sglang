"""CPU-only unit tests for the per-path Mamba checkpoint cap."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import argparse
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedLRUList,
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.srt.server_args import ServerArgs


class _RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free(self, value):
        self.freed.extend(value.tolist())


class _FakeUnifiedCache:
    tree_components = (ComponentType.FULL, ComponentType.MAMBA)

    def __init__(self):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.evictable_device_leaves = set()
        self.req_to_token_pool = SimpleNamespace(mamba_allocator=_RecordingAllocator())
        self.component_evictable_size_ = {ComponentType.MAMBA: 0}
        self.component_protected_size_ = {ComponentType.MAMBA: 0}
        self.lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(
                ComponentType.MAMBA, self.tree_components
            )
        }
        self.host_lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(
                ComponentType.MAMBA, self.tree_components
            )
        }
        self.evicted = []
        self.cascaded = []

    def _evict_component_and_detach_lru(self, node, component, **kwargs):
        self.evicted.append(node)
        return UnifiedRadixCache._evict_component_and_detach_lru(
            self, node, component, **kwargs
        )

    def _cascade_evict(self, node, component, tracker):
        self.cascaded.append(node)


def _build_unified_chain(cap, length=3):
    cache = _FakeUnifiedCache()
    component = object.__new__(MambaComponent)
    component.cache = cache
    component.mamba_max_states_per_path = cap

    nodes = []
    parent = cache.root_node
    for index in range(length):
        node = UnifiedTreeNode(cache.tree_components)
        node.parent = parent
        node.component_data[ComponentType.FULL].value = torch.tensor([100 + index])
        node.component_data[ComponentType.MAMBA].value = torch.tensor([index])
        parent.children[index] = node
        cache.component_evictable_size_[ComponentType.MAMBA] += 1
        cache.lru_lists[ComponentType.MAMBA].insert_mru(node)
        nodes.append(node)
        parent = node
    return component, nodes, cache


class TestMambaPathStateCap(unittest.TestCase):
    def test_server_arg_is_opt_in(self):
        self.assertEqual(
            ServerArgs(model_path="dummy").mamba_max_states_per_path,
            0,
        )

    def test_server_arg_cli(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(
            ["--model-path", "dummy", "--mamba-max-states-per-path", "3"]
        )

        self.assertEqual(args.mamba_max_states_per_path, 3)

    def test_unified_cache_removes_only_shallow_mamba_state(self):
        component, nodes, cache = _build_unified_chain(cap=2)

        component._enforce_path_state_cap(nodes[-1])

        self.assertEqual(cache.evicted, [nodes[0]])
        self.assertEqual(cache.cascaded, [nodes[0]])
        self.assertIsNone(nodes[0].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[-1].component_data[ComponentType.MAMBA].value)
        self.assertEqual(
            cache.req_to_token_pool.mamba_allocator.freed,
            [0],
        )
        self.assertEqual(cache.component_evictable_size_[ComponentType.MAMBA], 2)
        self.assertFalse(cache.lru_lists[ComponentType.MAMBA].in_list(nodes[0]))
        self.assertTrue(
            all(
                node.component_data[ComponentType.FULL].value is not None
                for node in nodes
            )
        )

    def test_unified_cache_cap_is_soft_for_fork_and_locked_nodes(self):
        component, nodes, cache = _build_unified_chain(cap=1, length=4)
        fork_child = UnifiedTreeNode(cache.tree_components)
        fork_child.parent = nodes[0]
        nodes[0].children["fork"] = fork_child
        nodes[1].component_data[ComponentType.MAMBA].lock_ref = 1

        component._enforce_path_state_cap(nodes[-1])

        self.assertEqual(cache.evicted, [nodes[2]])
        self.assertIsNotNone(nodes[0].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[1].component_data[ComponentType.MAMBA].value)
        self.assertIsNone(nodes[2].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[3].component_data[ComponentType.MAMBA].value)

    def test_unified_cache_preserves_existing_host_backup(self):
        component, nodes, cache = _build_unified_chain(cap=2)
        mamba_data = nodes[0].component_data[ComponentType.MAMBA]
        mamba_data.host_value = torch.tensor([10])

        component._enforce_path_state_cap(nodes[-1])

        self.assertIsNone(mamba_data.value)
        self.assertIsNotNone(mamba_data.host_value)
        self.assertTrue(cache.host_lru_lists[ComponentType.MAMBA].in_list(nodes[0]))

    def test_unified_cache_zero_disables_cap(self):
        component, nodes, cache = _build_unified_chain(cap=0)

        component._enforce_path_state_cap(nodes[-1])

        self.assertEqual(cache.evicted, [])
        self.assertTrue(
            all(
                node.component_data[ComponentType.MAMBA].value is not None
                for node in nodes
            )
        )


if __name__ == "__main__":
    unittest.main()
