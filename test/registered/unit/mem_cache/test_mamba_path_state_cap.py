"""CPU-only unit tests for the per-path Mamba checkpoint cap."""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import argparse
import unittest
from collections import defaultdict
from unittest import mock

import torch
from test_unified_radix_cache_unittest import CacheConfig, UnifiedRadixCacheSuite

from sglang.srt.mem_cache.unified_cache.cache_action import MambaEvictExcessPathStates
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedLRUList, UnifiedTreeNode
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


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


@unittest.skipUnless(torch.cuda.is_available(), "mamba pool fixtures need CUDA")
class TestMambaPathCapWriteThroughOrdering(CustomTestCase):
    """CI-active write-through/path-cap ordering regressions (the unified radix
    cache unittest module is temporarily gated off on trunk)."""

    cfg = CacheConfig(components=(ComponentType.FULL, ComponentType.MAMBA))
    _rid = 0
    # Borrow the fixture helpers without inheriting the full gated suite.
    _make_req = UnifiedRadixCacheSuite._make_req
    _alloc = UnifiedRadixCacheSuite._alloc
    _insert = UnifiedRadixCacheSuite._insert
    _init_hicache = UnifiedRadixCacheSuite._init_hicache
    _build_hicache_fixture = UnifiedRadixCacheSuite._build_hicache_fixture

    def test_write_through_backup_survives_mamba_path_cap(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        cache.write_through_threshold = 2
        cache.components[ComponentType.MAMBA].mamba_max_states_per_path = 1

        # The first insert creates the ancestor with a mamba state (hit_count 1).
        self._insert(cache, allocator, req_to_token_pool, [1, 2])
        ancestor = next(iter(cache.root_node.children.values()))
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].value)

        # The extending insert crosses the ancestor's write-through threshold in
        # the same walk whose commit runs the path-cap eviction; the cap must
        # leave the pending-backup node's device state for the deferred BackupKV.
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        cache.writing_check(write_back=True)

        self.assertTrue(ancestor.backuped)
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].host_value)

    def test_write_through_backup_chain_survives_mamba_path_cap(self):
        """A failed backup leaves an unbacked ancestor inside a later deferred
        backup chain; the cap walk must spare the whole chain, not just its tip."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        cache.write_through_threshold = 3
        mamba_comp = cache.components[ComponentType.MAMBA]

        self._insert(cache, allocator, req_to_token_pool, [1, 2])
        ancestor = next(iter(cache.root_node.children.values()))
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        middle = next(iter(ancestor.children.values()))

        # The ancestor crosses the threshold here; a host-exhaustion failure
        # leaves it unbacked with hit_count past the bar and its state intact.
        with mock.patch.object(cache, "_execute_kv_backup", return_value=None):
            self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4, 5, 6])
        self.assertFalse(ancestor.backuped)
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].value)

        # The middle node crosses next, so the deferred chain is
        # [ancestor, middle]; the extending insert adopts a new leaf state,
        # firing the now-enabled cap walk before the chain executes — it must
        # not evict either chain node's device state.
        mamba_comp.mamba_max_states_per_path = 1
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4, 5, 6, 7, 8])
        cache.writing_check(write_back=True)

        self.assertTrue(ancestor.backuped)
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].host_value)
        self.assertTrue(middle.backuped)
        self.assertIsNotNone(middle.component_data[ComponentType.MAMBA].host_value)

    def test_backup_retry_after_mamba_cap_skips_tombstoned_state(self):
        """A backup that fails before the cap and retries via the leaf action
        rebuilds its spec post-cap: KV backs up, the tombstoned mamba arm stays gone."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        cache.write_through_threshold = 1
        mamba_comp = cache.components[ComponentType.MAMBA]

        # A failed write-through leaves the ancestor unbacked with device state.
        with mock.patch.object(cache, "_execute_kv_backup", return_value=None):
            self._insert(cache, allocator, req_to_token_pool, [1, 2])
        ancestor = next(iter(cache.root_node.children.values()))
        self.assertFalse(ancestor.backuped)
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].value)

        # The walk backup fails again, the cap tombstones the unlocked
        # ancestor's mamba state, then the leaf-action retry succeeds.
        mamba_comp.mamba_max_states_per_path = 1
        real_backup = cache._execute_kv_backup
        attempts = []

        def fail_once(*args, **kwargs):
            attempts.append(args)
            if len(attempts) == 1:
                return None
            return real_backup(*args, **kwargs)

        with mock.patch.object(cache, "_execute_kv_backup", side_effect=fail_once):
            self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        leaf = next(iter(ancestor.children.values()))
        cache.writing_check(write_back=True)

        # Post-cap spec rebuild: no resurrection of the tombstoned mamba state.
        self.assertTrue(ancestor.backuped)
        ancestor_cd = ancestor.component_data[ComponentType.MAMBA]
        self.assertIsNone(ancestor_cd.value)
        self.assertIsNone(ancestor_cd.host_value)
        self.assertTrue(leaf.backuped)
        self.assertIsNotNone(leaf.component_data[ComponentType.MAMBA].host_value)
        cache.sanity_check()

    def test_cap_walk_failure_still_drains_collected_frees(self):
        """A cap walk that raises mid-eviction must still free the tombstoned
        slots it already collected (the pre-split inline frees could not leak)."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        mamba_comp = cache.components[ComponentType.MAMBA]

        self._insert(cache, allocator, req_to_token_pool, [1, 2])
        ancestor = next(iter(cache.root_node.children.values()))
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        leaf = next(iter(ancestor.children.values()))
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].value)

        mamba_comp.mamba_max_states_per_path = 1
        available = req_to_token_pool.mamba_allocator.available_size()
        with mock.patch.object(
            cache.tree_core, "_cascade_evict", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                mamba_comp.apply_component_action(MambaEvictExcessPathStates(leaf.id))

        self.assertIsNone(ancestor.component_data[ComponentType.MAMBA].value)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), available + 1
        )


if __name__ == "__main__":
    unittest.main()
