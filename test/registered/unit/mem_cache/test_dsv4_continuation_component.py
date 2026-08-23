import unittest
from collections import defaultdict
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    InsertResult,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
    FreeComponentHostSlot,
)
from sglang.srt.mem_cache.unified_cache.components.dsv4_continuation_component import (
    DSV4ContinuationComponent,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4ContinuationComponent(unittest.TestCase):
    component_types = (ComponentType.FULL, ComponentType.DSV4_CONTINUATION)

    def setUp(self):
        self.component = object.__new__(DSV4ContinuationComponent)
        self.component.pool = MagicMock()
        self.component.tree_core = MagicMock()

    def _node(self):
        return UnifiedTreeNode(tree_components=self.component_types)

    def test_split_keeps_continuation_only_on_suffix(self):
        parent = self._node()
        child = self._node()
        child_data = child.component_data[ComponentType.DSV4_CONTINUATION]
        child_data.value = torch.tensor([7])
        child_data.lock_ref = 2
        child_data.host_value = torch.tensor([11])
        child_data.host_lock_ref = 3
        child_data.session_ref = 1

        self.component.redistribute_on_node_split(parent, child)

        parent_data = parent.component_data[ComponentType.DSV4_CONTINUATION]
        self.assertIsNone(parent_data.value)
        self.assertEqual(parent_data.lock_ref, 0)
        self.assertIsNone(parent_data.host_value)
        self.assertEqual(parent_data.host_lock_ref, 0)
        self.assertEqual(parent_data.session_ref, 0)
        self.assertTrue(torch.equal(child_data.value, torch.tensor([7])))
        self.assertTrue(torch.equal(child_data.host_value, torch.tensor([11])))

    def test_reset_releases_all_continuation_slots(self):
        self.component._session_leaves = {"session": {self._node()}}

        self.component.reset_session_state()

        self.component.pool.clear.assert_called_once_with()
        self.assertEqual(dict(self.component._session_leaves), {})

    def test_session_coverage_refreshes_lru_partitions_at_boundaries(self):
        old_ancestor = self._node()
        leaf = self._node()
        self.component._refresh_session_partition = MagicMock()
        old_ancestor.component_data[ComponentType.DSV4_CONTINUATION].session_ref = 1

        self.component._advance_session_coverage("session", leaf, old_ancestor=None)
        self.component._advance_session_coverage(
            "session", leaf, old_ancestor=old_ancestor
        )
        self.component._recede_session_coverage("session", leaf, fallback=old_ancestor)

        leaf_data = leaf.component_data[ComponentType.DSV4_CONTINUATION]
        ancestor_data = old_ancestor.component_data[ComponentType.DSV4_CONTINUATION]
        self.assertEqual(leaf_data.session_ref, 1)
        self.assertEqual(ancestor_data.session_ref, 1)
        self.assertEqual(
            self.component._refresh_session_partition.call_args_list,
            [
                unittest.mock.call(leaf),
                unittest.mock.call(old_ancestor),
                unittest.mock.call(old_ancestor),
            ],
        )

    def test_device_eviction_returns_unbacked_internal_node_for_backup(self):
        node = self._node()
        node.id = 13
        data = node.component_data[ComponentType.DSV4_CONTINUATION]
        data.value = torch.tensor([5])
        data.host_value = None
        lru = MagicMock()
        lru.in_list.side_effect = lambda candidate: candidate is node
        lru.get_prev_no_lock.return_value = None
        self.component.tree_core.lru_lists = {ComponentType.DSV4_CONTINUATION: lru}
        self.component._evict_device_request_cnt = 1
        self.component._evict_device_cursor = node

        result = self.component._evict_device_next_node(
            defaultdict(int), defaultdict(list), defaultdict(list)
        )

        self.assertEqual(result, 13)
        self.component.tree_core._evict_component_and_detach_lru.assert_not_called()

    def test_device_eviction_allows_host_backed_slot(self):
        node = self._node()
        data = node.component_data[ComponentType.DSV4_CONTINUATION]
        data.value = torch.tensor([5])
        data.host_value = torch.tensor([9])
        lru = MagicMock()
        lru.in_list.side_effect = lambda candidate: candidate is node
        lru.get_prev_no_lock.return_value = None
        self.component.tree_core.lru_lists = {ComponentType.DSV4_CONTINUATION: lru}
        self.component._evict_device_request_cnt = 1
        self.component._evict_device_cursor = node
        tracker = defaultdict(int)
        device_frees = defaultdict(list)
        host_frees = defaultdict(list)

        result = self.component._evict_device_next_node(
            tracker, device_frees, host_frees
        )

        self.assertIsNone(result)
        self.component.tree_core._evict_component_and_detach_lru.assert_called_once_with(
            node,
            self.component,
            target=EvictLayer.DEVICE,
            tracker=tracker,
            device_frees=device_frees,
            host_frees=host_frees,
        )

    def test_device_leaf_eviction_returns_unbacked_node_to_driver(self):
        node = self._node()
        node.id = 23
        data = node.component_data[ComponentType.DSV4_CONTINUATION]
        data.value = torch.tensor([5])
        data.host_value = None
        lru = MagicMock()
        lru.in_list.side_effect = lambda candidate: candidate is node
        lru.get_prev_no_lock.return_value = None
        self.component.tree_core.lru_lists = {ComponentType.DSV4_CONTINUATION: lru}
        self.component.tree_core.evictable_device_leaves = {node}
        self.component.tree_core.enable_session_radix_cache = False
        self.component._evict_device_request_cnt = 1
        self.component._evict_device_cursor = node

        result = self.component._evict_device_next_node(
            defaultdict(int), defaultdict(list), defaultdict(list)
        )

        self.assertEqual(result, 23)
        self.component.tree_core._evict_component_and_detach_lru.assert_not_called()

    def test_full_backed_leaf_waits_for_continuation_backup(self):
        node = self._node()
        node.id = 41
        core = MagicMock()
        core.node_by_id.return_value = node
        core._is_device_leaf.return_value = True
        core._needs_incremental_component_backup.return_value = True
        backup_action = object()
        core._build_backup_kv_action.return_value = backup_action

        result = UnifiedTreeCore.evict_device_leaf(core, node.id, is_write_back=True)

        self.assertIs(result.backup_kv, backup_action)
        core._demote.assert_not_called()

    def test_failed_incremental_backup_keeps_backed_leaf_resident(self):
        node = self._node()
        node.id = 42
        node.component_data[ComponentType.FULL].host_value = torch.tensor([9])
        core = MagicMock()
        core.node_by_id.return_value = node
        core._is_device_leaf.return_value = True

        result = UnifiedTreeCore.drop_subtree_no_host(core, node.id)

        self.assertFalse(result.is_dropped)
        core._release_all_component_layers.assert_not_called()
        core._delete_unbacked_device_leaf.assert_not_called()

    def test_component_only_eviction_backs_up_then_preserves_full(self):
        node = self._node()
        node.id = 43
        full_data = node.component_data[ComponentType.FULL]
        full_data.value = torch.tensor([1, 2])
        continuation_data = node.component_data[ComponentType.DSV4_CONTINUATION]
        continuation_data.value = torch.tensor([5])

        core = MagicMock()
        core.node_by_id.return_value = node
        core.components_by_type = {ComponentType.DSV4_CONTINUATION: self.component}
        core.component_evictable_size_ = {ComponentType.DSV4_CONTINUATION: 1}
        core.lru_lists = {ComponentType.DSV4_CONTINUATION: MagicMock()}
        core.host_lru_lists = {ComponentType.DSV4_CONTINUATION: MagicMock()}
        core._build_backup_kv_action.return_value = backup_action = object()
        core._evict_component_and_detach_lru.side_effect = (
            lambda *args, **kwargs: UnifiedTreeCore._evict_component_and_detach_lru(
                core, *args, **kwargs
            )
        )
        self.component.tree_core = core

        backup_result = UnifiedTreeCore.evict_component_device_node(
            core,
            node.id,
            ComponentType.DSV4_CONTINUATION,
            require_host_backup=True,
        )

        self.assertIs(backup_result.backup_kv, backup_action)
        self.assertTrue(torch.equal(continuation_data.value, torch.tensor([5])))
        self.assertTrue(torch.equal(full_data.value, torch.tensor([1, 2])))

        continuation_data.host_value = torch.tensor([9])
        evict_result = UnifiedTreeCore.evict_component_device_node(
            core,
            node.id,
            ComponentType.DSV4_CONTINUATION,
            require_host_backup=True,
        )

        self.assertIsNone(evict_result.backup_kv)
        self.assertIsNone(continuation_data.value)
        self.assertTrue(torch.equal(continuation_data.host_value, torch.tensor([9])))
        self.assertTrue(torch.equal(full_data.value, torch.tensor([1, 2])))
        self.assertEqual(evict_result.tracker[ComponentType.DSV4_CONTINUATION], 1)
        self.assertTrue(
            torch.equal(
                evict_result.device_frees[ComponentType.DSV4_CONTINUATION][0],
                torch.tensor([5]),
            )
        )
        evict_result.device_frees.clear()

    def test_write_back_cascade_preserves_unbacked_continuation(self):
        node = self._node()
        node.component_data[ComponentType.DSV4_CONTINUATION].value = torch.tensor([5])
        trigger = MagicMock()
        trigger.component_type = ComponentType.SWA
        trigger.eviction_priority.return_value = 1
        continuation = MagicMock()
        continuation.component_type = ComponentType.DSV4_CONTINUATION
        continuation.eviction_priority.return_value = -1
        continuation.node_has_component_data.return_value = True
        continuation.needs_incremental_backup.return_value = True
        core = MagicMock()
        core.components = (trigger, continuation)
        core.evictable_device_leaves = set()
        core.evictable_host_leaves = set()
        core.is_write_back = True

        UnifiedTreeCore._cascade_evict(
            core,
            node,
            trigger,
            defaultdict(int),
            defaultdict(list),
            defaultdict(list),
        )

        core._evict_component_and_detach_lru.assert_not_called()
        continuation.needs_incremental_backup.assert_called_once_with(node)

    def test_free_action_returns_slot_to_continuation_pool(self):
        slots = torch.tensor([2, 4])
        action = FreeComponentDeviceSlot(
            [slots], component_type=ComponentType.DSV4_CONTINUATION
        )

        self.component.apply_component_action(action)

        self.component.pool.free.assert_called_once_with(slots)

    def test_duplicate_insert_releases_new_slot(self):
        slots = torch.tensor([3])
        params = InsertParams(dsv4_continuation_value=slots)
        result = InsertResult(prefix_len=0, dsv4_continuation_exist=True)

        self.component.cleanup_after_caching_req(
            MagicMock(),
            is_finished=False,
            insert_result=result,
            insert_params=params,
        )

        self.component.pool.free.assert_called_once_with(slots)

    def test_skipped_insert_releases_new_slot(self):
        slots = torch.tensor([6])
        params = InsertParams(dsv4_continuation_value=slots)

        self.component.cleanup_after_caching_req(
            MagicMock(), is_finished=False, insert_params=params
        )

        self.component.pool.free.assert_called_once_with(slots)

    def test_disabled_insert_releases_req_owned_slot(self):
        slots = torch.tensor([8])
        req = MagicMock()
        req.dsv4_continuation_value = slots
        req.dsv4_continuation_endpoint = 2048

        self.component.cleanup_after_caching_req(
            req, is_finished=False, insert_params=None
        )

        self.component.pool.free.assert_called_once_with(slots)
        self.assertIsNone(req.dsv4_continuation_value)
        self.assertIsNone(req.dsv4_continuation_endpoint)

    def test_host_only_match_sets_load_back_marker(self):
        node = self._node()
        node.component_data[ComponentType.DSV4_CONTINUATION].host_value = torch.tensor(
            [3]
        )
        result = MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
        )

        result = self.component.finalize_match_result_in_tree_core(
            result, MagicMock(), [], 0
        )

        self.assertTrue(result.dsv4_continuation_host_hit)

    def test_backup_and_load_back_transfer_lifecycle(self):
        node = self._node()
        node.id = 17
        data = node.component_data[ComponentType.DSV4_CONTINUATION]
        data.value = torch.tensor([5])

        backup = self.component.build_hicache_transfers(
            node, CacheTransferPhase.BACKUP_HOST
        )

        self.assertEqual(backup[0].name, PoolName.DSV4_CONTINUATION)
        self.assertTrue(torch.equal(backup[0].device_indices, torch.tensor([5])))
        backup[0].host_indices = torch.tensor([2])
        self.component.commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            backup,
            cache_actions=[],
        )
        self.assertTrue(torch.equal(data.host_value, torch.tensor([2])))

        data.value = None
        load = self.component.build_hicache_transfers(
            node, CacheTransferPhase.LOAD_BACK
        )
        self.assertEqual(load[0].nodes_to_load, [17])
        load[0].device_indices = torch.tensor([9])
        device_lru = MagicMock()
        host_lru = MagicMock()
        host_lru.in_list.return_value = True
        self.component.tree_core.lru_lists = {
            ComponentType.DSV4_CONTINUATION: device_lru
        }
        self.component.tree_core.host_lru_lists = {
            ComponentType.DSV4_CONTINUATION: host_lru
        }
        self.component.tree_core.component_evictable_size_ = {
            ComponentType.DSV4_CONTINUATION: 0
        }

        self.component.commit_hicache_transfer(
            node,
            CacheTransferPhase.LOAD_BACK,
            load,
            cache_actions=[],
        )

        self.assertTrue(torch.equal(data.value, torch.tensor([9])))
        host_lru.remove_node.assert_called_once_with(node)
        device_lru.insert_mru.assert_called_once_with(node)
        self.assertEqual(
            self.component.tree_core.component_evictable_size_[
                ComponentType.DSV4_CONTINUATION
            ],
            1,
        )

    def test_host_free_action_is_deferred_to_controller(self):
        self.component.cache = MagicMock()
        host_indices = torch.tensor([4])

        self.component.apply_component_action(
            FreeComponentHostSlot(
                [host_indices], component_type=ComponentType.DSV4_CONTINUATION
            )
        )

        call = self.component.cache.cache_controller.append_host_mem_release.call_args
        transfer = call.kwargs["extra_pools"][0]
        self.assertEqual(transfer.name, PoolName.DSV4_CONTINUATION)
        self.assertTrue(torch.equal(transfer.host_indices, host_indices))

    def test_aux_only_backup_counts_as_write_back_success(self):
        cache = MagicMock()
        cache.buffer_pipeline = None
        node_id = 51
        device_slot = torch.tensor([7])
        host_slot = torch.tensor([3])
        transfer = MagicMock()
        transfer.name = PoolName.DSV4_CONTINUATION
        transfer.device_indices = device_slot
        transfer.host_indices = None
        cache.tree_core.build_backup_spec.return_value = (
            torch.empty(0, dtype=torch.int64),
            {ComponentType.DSV4_CONTINUATION: [transfer]},
        )
        cache._build_backup_sidecar.return_value = []
        cache._execute_kv_backup.return_value = torch.empty(0, dtype=torch.int64)

        def commit_backup(_node_id, _host_indices, comp_xfers):
            comp_xfers[ComponentType.DSV4_CONTINUATION][0].host_indices = host_slot

        cache.tree_core.commit_backup.side_effect = commit_backup
        cache._track_write_through_node = MagicMock()

        from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        written = UnifiedRadixCache._execute_and_commit_kv_backup(
            cache, BackupKV([node_id]), write_back=True
        )

        self.assertEqual(written, 1)
        cache.tree_core.commit_backup.assert_called_once()

    def test_internal_pressure_backs_up_then_frees_only_continuation(self):
        cache = MagicMock()
        cache.tree_components = (ComponentType.DSV4_CONTINUATION,)
        cache.is_write_back = True
        cache.tree_core = MagicMock()
        cache._evict_device_next_node.side_effect = [61, None]
        backup = MagicMock()

        calls = 0

        def evict_component_node(node_id, component_type, tracker):
            nonlocal calls
            calls += 1
            self.assertEqual(node_id, 61)
            self.assertEqual(component_type, ComponentType.DSV4_CONTINUATION)
            if calls == 1:
                return backup
            tracker[component_type] += 1
            return None

        cache._evict_component_device_node.side_effect = evict_component_node
        cache._execute_and_commit_kv_backup.return_value = 1

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        tracker = defaultdict(int)
        UnifiedRadixCache._evict_components(
            cache,
            {ComponentType.DSV4_CONTINUATION: 1},
            tracker,
        )

        self.assertEqual(tracker[ComponentType.DSV4_CONTINUATION], 1)
        self.assertEqual(calls, 2)
        cache.writing_check.assert_called_once_with(write_back=True)
        cache._evict_device_leaf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
