from __future__ import annotations

import unittest
from array import array
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.cache_controller import StorageOperation
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import (
    StorageCheckpointRegistry,
    StorageCheckpointState,
    StoragePrefetchState,
    StoragePrefetchTracker,
    StorageWriteTracker,
)
from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeNode:
    def __init__(
        self,
        node_id: int,
        page_hashes: list[str],
        parent: _FakeNode | None,
    ) -> None:
        self.id = node_id
        self.hash_value = page_hashes
        self.parent = parent
        self.backuped = True
        self.write_through_pending_id = None
        self.mamba_value = object()
        self.mamba_host_value = object()
        self.key = RadixKey(array("q"), None)

    @property
    def mamba_backuped(self) -> bool:
        return self.mamba_host_value is not None


class _FakeCheckpointCache:
    def __init__(self) -> None:
        self.root_node = _FakeNode(0, [], None)
        self.parent_node = _FakeNode(1, ["page-a"], self.root_node)
        self.target_node = _FakeNode(2, ["page-b"], self.parent_node)
        self.enable_storage = True
        self.cache_controller = SimpleNamespace(
            storage_write_tracker=StorageWriteTracker()
        )
        self.storage_checkpoint_registry = StorageCheckpointRegistry()
        self.ongoing_write_through = {}
        self.host_backups: list[_FakeNode] = []
        self.mamba_backups: list[_FakeNode] = []
        self.storage_writes: list[tuple[int, _FakeNode]] = []

    def synchronize_storage_prefetch_flag(self, flag: bool) -> bool:
        return flag

    def match_prefix(self, _params):
        return SimpleNamespace(last_device_node=self.target_node)

    def write_backup(self, node: _FakeNode) -> int:
        if node.parent is not self.root_node and not node.parent.backuped:
            self.write_backup(node.parent)
        node.backuped = True
        node.write_through_pending_id = node.id
        self.ongoing_write_through[node.id] = node
        self.host_backups.append(node)
        return len(node.hash_value)

    def write_backup_storage(self, node: _FakeNode) -> int:
        operation_id = 100 + len(self.storage_writes)
        self.cache_controller.storage_write_tracker.register(
            operation_id, node.hash_value
        )
        self.storage_writes.append((operation_id, node))
        return operation_id

    def _write_mamba_backup(self, node: _FakeNode) -> bool:
        node.mamba_host_value = object()
        self.ongoing_write_through[node.id] = node
        self.mamba_backups.append(node)
        return True

    def _write_missing_aux_backup(self, _node: _FakeNode) -> None:
        return None


class TestCacheStorageCheckpoint(unittest.TestCase):
    def test_all_cache_variants_publish_tp_min_durability(self):
        drain_methods = (
            UnifiedRadixCache._drain_storage_control_queues_impl,
            HiRadixCache._drain_storage_control_queues_impl,
            HiMambaRadixCache._drain_storage_control_queues_impl,
        )

        for drain_method in drain_methods:
            for remote_durable_pages, expected_state in (
                (2, StorageCheckpointState.READY),
                (1, StorageCheckpointState.FAILED),
            ):
                with self.subTest(
                    cache=drain_method.__qualname__,
                    remote_durable_pages=remote_durable_pages,
                ):
                    write_tracker = StorageWriteTracker()
                    checkpoint_registry = StorageCheckpointRegistry()
                    operation = StorageOperation(
                        torch.tensor([0, 1]),
                        [10, 11],
                        hash_value=["page-a", "page-b"],
                    )
                    operation.completed_tokens = 2
                    write_tracker.register(operation.id, operation.hash_value)
                    checkpoint_registry.create(
                        "hicache:request-a", operation.hash_value
                    )
                    cache_controller = SimpleNamespace(
                        prefetch_revoke_queue=Queue(),
                        prefetch_hit_queue=Queue(),
                        ack_backup_queue=Queue(),
                        host_mem_release_queue=Queue(),
                        extra_host_mem_release_queues={},
                        storage_write_tracker=write_tracker,
                        backup_skip=False,
                    )
                    cache_controller.ack_backup_queue.put(operation)
                    cache = SimpleNamespace(
                        cache_controller=cache_controller,
                        storage_checkpoint_registry=checkpoint_registry,
                        ongoing_backup={},
                        page_size=1,
                        tp_world_size=2,
                        tp_group=None,
                        enable_storage_metrics=False,
                        storage_metrics_collector=None,
                    )

                    def reduce_durable_pages(tensor, _op):
                        tensor[0] = min(int(tensor[0].item()), remote_durable_pages)

                    if (
                        drain_method
                        is UnifiedRadixCache._drain_storage_control_queues_impl
                    ):
                        cache._all_reduce_attn_groups = reduce_durable_pages
                        drain_method(
                            cache,
                            n_revoke=0,
                            n_storage_hit=0,
                            n_backup=1,
                            n_release=0,
                            extra_release_counts={},
                            log_metrics=False,
                        )
                    elif (
                        drain_method is HiRadixCache._drain_storage_control_queues_impl
                    ):
                        cache._all_reduce_attn_groups = reduce_durable_pages
                        drain_method(
                            cache,
                            n_revoke=0,
                            n_storage_hit=0,
                            n_backup=1,
                            n_release=0,
                            log_metrics=False,
                            synchronize_durability=True,
                        )
                    else:
                        with patch(
                            "torch.distributed.all_reduce",
                            side_effect=lambda tensor, **_kwargs: reduce_durable_pages(
                                tensor, None
                            ),
                        ):
                            drain_method(
                                cache,
                                n_revoke=0,
                                n_storage_hit=0,
                                n_backup=1,
                                n_release=0,
                                log_metrics=False,
                                synchronize_durability=True,
                            )

                    self.assertEqual(
                        checkpoint_registry.get_state("hicache:request-a"),
                        expected_state,
                    )

    def test_unified_checkpoint_stages_missing_auxiliary_pool(self):
        class _FakeComponent:
            component_type = ComponentType.MAMBA

            def build_hicache_transfers(self, _node, phase):
                assert phase is CacheTransferPhase.BACKUP_HOST
                return [
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        device_indices=torch.tensor([7], dtype=torch.int64),
                    )
                ]

            def commit_hicache_transfer(self, node, _phase, transfers):
                node.component_data[self.component_type].host_value = transfers[
                    0
                ].host_indices

        class _FakeController:
            def write(self, device_indices, node_id, extra_pools):
                self.device_indices = device_indices
                self.node_id = node_id
                extra_pools[0].host_indices = torch.tensor([11], dtype=torch.int64)
                return torch.empty((0,), dtype=torch.int64)

        class _FakeCache:
            def __init__(self):
                component = _FakeComponent()
                self.cache_controller = _FakeController()
                self._components_tuple = (component,)
                self.components = {component.component_type: component}
                self.tracked = None

            def inc_lock_ref(self, _node):
                return SimpleNamespace(to_dec_params=lambda: "lock-params")

            def _track_write_through_node(self, node, lock_params):
                self.tracked = (node, lock_params)

        cache = _FakeCache()
        node = SimpleNamespace(
            id=3,
            component_data={
                BASE_COMPONENT_TYPE: SimpleNamespace(
                    value=torch.tensor([1, 2], dtype=torch.int64),
                    host_value=torch.tensor([4, 5], dtype=torch.int64),
                ),
                ComponentType.MAMBA: SimpleNamespace(
                    value=torch.tensor([7], dtype=torch.int64),
                    host_value=None,
                ),
            },
        )

        started = UnifiedRadixCache._write_missing_aux_backup(cache, node)

        self.assertTrue(started)
        self.assertEqual(cache.cache_controller.device_indices.numel(), 0)
        self.assertTrue(
            torch.equal(
                node.component_data[ComponentType.MAMBA].host_value,
                torch.tensor([11], dtype=torch.int64),
            )
        )
        self.assertEqual(cache.tracked, (node, "lock-params"))

    def test_mamba_pending_full_backup_follows_split_fragments(self):
        cache = _FakeCheckpointCache()
        original_node = cache.target_node
        prefix_node = _FakeNode(3, ["page-b-prefix"], cache.parent_node)

        HiMambaRadixCache._track_write_through_node(
            cache,
            original_node,
            release_lock=True,
            includes_full_kv=True,
        )
        HiMambaRadixCache._replace_pending_write_through_node(
            cache, original_node, [prefix_node, original_node]
        )

        pending = cache.ongoing_write_through[original_node.id]
        self.assertEqual(pending.publish_nodes, [prefix_node, original_node])
        self.assertEqual(prefix_node.write_through_pending_id, original_node.id)
        self.assertEqual(original_node.write_through_pending_id, original_node.id)

    def test_mamba_checkpoint_stages_missing_target_state(self):
        cache = _FakeCheckpointCache()
        cache.target_node.mamba_host_value = None

        HiMambaRadixCache._create_storage_checkpoint(
            cache,
            "hicache:request-a",
            RadixKey(array("q", [1, 2]), None),
        )

        self.assertEqual(cache.mamba_backups, [cache.target_node])
        self.assertEqual(
            [node for _, node in cache.storage_writes], [cache.parent_node]
        )

    def test_all_cache_variants_force_root_to_leaf_host_backup(self):
        cache_methods = (
            UnifiedRadixCache._create_storage_checkpoint,
            HiRadixCache._create_storage_checkpoint,
            HiMambaRadixCache._create_storage_checkpoint,
        )
        radix_key = RadixKey(array("q", [1, 2]), None)

        for create_checkpoint in cache_methods:
            with self.subTest(cache=create_checkpoint.__qualname__):
                cache = _FakeCheckpointCache()
                cache.parent_node.backuped = False
                cache.target_node.backuped = False

                create_checkpoint(cache, "hicache:request-a", radix_key)

                self.assertEqual(
                    cache.host_backups, [cache.parent_node, cache.target_node]
                )
                self.assertEqual(
                    cache.storage_checkpoint_registry.get_state("hicache:request-a"),
                    StorageCheckpointState.PENDING,
                )

    def test_all_cache_variants_fill_parent_host_gap(self):
        cache_methods = (
            UnifiedRadixCache._create_storage_checkpoint,
            HiRadixCache._create_storage_checkpoint,
            HiMambaRadixCache._create_storage_checkpoint,
        )
        radix_key = RadixKey(array("q", [1, 2]), None)

        for create_checkpoint in cache_methods:
            with self.subTest(cache=create_checkpoint.__qualname__):
                cache = _FakeCheckpointCache()
                cache.parent_node.backuped = False

                create_checkpoint(cache, "hicache:request-a", radix_key)

                self.assertEqual(cache.host_backups, [cache.parent_node])
                self.assertEqual(
                    [node for _, node in cache.storage_writes], [cache.target_node]
                )
                self.assertEqual(
                    cache.storage_checkpoint_registry.get_state("hicache:request-a"),
                    StorageCheckpointState.PENDING,
                )

    def test_all_cache_variants_join_pending_writes_and_finish_checkpoint(self):
        cache_methods = (
            UnifiedRadixCache._create_storage_checkpoint,
            HiRadixCache._create_storage_checkpoint,
            HiMambaRadixCache._create_storage_checkpoint,
        )
        radix_key = RadixKey(array("q", [1, 2]), None)

        for create_checkpoint in cache_methods:
            with self.subTest(cache=create_checkpoint.__qualname__):
                cache = _FakeCheckpointCache()
                write_tracker = cache.cache_controller.storage_write_tracker
                write_tracker.register(1, cache.parent_node.hash_value)

                create_checkpoint(cache, "hicache:request-a", radix_key)

                self.assertEqual(
                    cache.storage_checkpoint_registry.get_state("hicache:request-a"),
                    StorageCheckpointState.PENDING,
                )
                self.assertEqual(
                    [node for _, node in cache.storage_writes], [cache.target_node]
                )

                pending_completion = write_tracker.complete(1, durable_pages=1)
                self.assertIsNotNone(pending_completion)
                cache.storage_checkpoint_registry.record_write_completion(
                    pending_completion
                )
                for operation_id, _node in cache.storage_writes:
                    completion = write_tracker.complete(operation_id, durable_pages=1)
                    self.assertIsNotNone(completion)
                    cache.storage_checkpoint_registry.record_write_completion(
                        completion
                    )

                self.assertEqual(
                    cache.storage_checkpoint_registry.get_state("hicache:request-a"),
                    StorageCheckpointState.READY,
                )

    def test_all_cache_variants_reuse_acknowledged_durability(self):
        cache_methods = (
            UnifiedRadixCache._create_storage_checkpoint,
            HiRadixCache._create_storage_checkpoint,
            HiMambaRadixCache._create_storage_checkpoint,
        )
        radix_key = RadixKey(array("q", [1, 2]), None)

        for create_checkpoint in cache_methods:
            with self.subTest(cache=create_checkpoint.__qualname__):
                cache = _FakeCheckpointCache()
                cache.cache_controller.storage_write_tracker.mark_durable(
                    ["page-a", "page-b"]
                )

                create_checkpoint(cache, "hicache:request-a", radix_key)

                self.assertEqual(
                    cache.storage_checkpoint_registry.get_state("hicache:request-a"),
                    StorageCheckpointState.READY,
                )
                self.assertEqual(cache.storage_writes, [])

    def test_all_cache_variants_defer_while_matching_write_is_pending(self):
        prefetch_methods = (
            UnifiedRadixCache.prefetch_from_storage,
            HiRadixCache.prefetch_from_storage,
            HiMambaRadixCache.prefetch_from_storage,
        )

        for prefetch_from_storage in prefetch_methods:
            with self.subTest(cache=prefetch_from_storage.__qualname__):
                cache = _FakeCheckpointCache()
                cache.page_size = 1
                cache.prefetch_threshold = 1
                cache.is_eagle = False
                cache.tp_world_size = 1
                cache.storage_prefetch_tracker = StoragePrefetchTracker()
                cache.cache_controller.has_pending_storage_write = (
                    lambda _tokens, _last_hash: True
                )
                cache.cache_controller.prefetch_rate_limited = lambda: False
                cache._all_reduce_attn_groups = lambda _value, _op: None

                state = prefetch_from_storage(
                    cache,
                    "request-a",
                    cache.root_node,
                    [1, 2],
                    None,
                    None,
                    None,
                )

                self.assertEqual(state, StoragePrefetchState.DEFERRED)
                self.assertEqual(
                    cache.storage_prefetch_tracker.get("request-a"),
                    StoragePrefetchState.DEFERRED,
                )

    def test_all_cache_variants_defer_when_prefetch_is_rate_limited(self):
        prefetch_methods = (
            UnifiedRadixCache.prefetch_from_storage,
            HiRadixCache.prefetch_from_storage,
            HiMambaRadixCache.prefetch_from_storage,
        )

        for prefetch_from_storage in prefetch_methods:
            with self.subTest(cache=prefetch_from_storage.__qualname__):
                cache = _FakeCheckpointCache()
                cache.page_size = 1
                cache.prefetch_threshold = 1
                cache.is_eagle = False
                cache.tp_world_size = 1
                cache.storage_prefetch_tracker = StoragePrefetchTracker()
                cache.cache_controller.has_pending_storage_write = (
                    lambda _tokens, _last_hash: False
                )
                cache.cache_controller.prefetch_rate_limited = lambda: True
                cache._all_reduce_attn_groups = lambda _value, _op: None

                state = prefetch_from_storage(
                    cache,
                    "request-a",
                    cache.root_node,
                    [1, 2],
                    None,
                    None,
                    None,
                )

                self.assertEqual(state, StoragePrefetchState.DEFERRED)

    def test_all_cache_variants_wait_for_checkpoint_dependency(self):
        prefetch_methods = (
            UnifiedRadixCache.prefetch_from_storage,
            HiRadixCache.prefetch_from_storage,
            HiMambaRadixCache.prefetch_from_storage,
        )

        for prefetch_from_storage in prefetch_methods:
            with self.subTest(cache=prefetch_from_storage.__qualname__):
                cache = _FakeCheckpointCache()
                cache.page_size = 1
                cache.prefetch_threshold = 1
                cache.is_eagle = False
                cache.tp_world_size = 1
                cache.storage_prefetch_tracker = StoragePrefetchTracker()
                cache.cache_controller.has_pending_storage_write = (
                    lambda _tokens, _last_hash: False
                )
                cache.cache_controller.prefetch_rate_limited = lambda: False
                cache._all_reduce_attn_groups = lambda _value, _op: None
                cache.storage_checkpoint_registry.reserve("hicache:prior-request")

                state = prefetch_from_storage(
                    cache,
                    "request-a",
                    cache.root_node,
                    [1, 2],
                    None,
                    None,
                    "hicache:prior-request",
                )
                self.assertEqual(state, StoragePrefetchState.DEFERRED)

                cache.storage_checkpoint_registry.fail("hicache:prior-request")
                state = prefetch_from_storage(
                    cache,
                    "request-a",
                    cache.root_node,
                    [1, 2],
                    None,
                    None,
                    "hicache:prior-request",
                )
                self.assertEqual(state, StoragePrefetchState.FAILED)


if __name__ == "__main__":
    unittest.main()
