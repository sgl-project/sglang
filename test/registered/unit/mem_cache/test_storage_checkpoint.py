from __future__ import annotations

import unittest
from array import array
from collections import deque
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
    CheckpointRetryResource,
    L3StagingLease,
    L3StagingManager,
    PendingStorageCheckpoint,
    StorageCheckpointRegistry,
    StorageCheckpointRetryQueues,
    StorageCheckpointState,
    StoragePrefetchState,
    StoragePrefetchTracker,
    StorageWriteTracker,
    bounded_host_eviction_scan,
    get_host_eviction_scan_budget,
    l3_staging_allocation,
)
from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    FullComponent,
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
        self.key = RadixKey(array("q", [node_id]) if page_hashes else array("q"), None)
        self.component_data = [
            SimpleNamespace(
                value=torch.arange(len(page_hashes), dtype=torch.int64),
                host_value=(
                    torch.arange(len(page_hashes), dtype=torch.int64)
                    if self.backuped
                    else None
                ),
                lock_ref=0,
                host_lock_ref=0,
            )
            for _ in range(3)
        ]

    @property
    def mamba_backuped(self) -> bool:
        return self.mamba_host_value is not None


class _StagingHostPool:
    def __init__(self, size: int, page_size: int) -> None:
        self.size = size
        self.page_size = page_size
        self.free_slots = torch.arange(size, dtype=torch.int64)
        self.allocated: set[int] = set()

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, need_size: int) -> torch.Tensor | None:
        if need_size > len(self.free_slots):
            return None
        indices = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        values = set(indices.tolist())
        assert not values.intersection(self.allocated)
        self.allocated.update(values)
        return indices

    def free(self, indices: torch.Tensor) -> int:
        values = set(indices.tolist())
        assert values.issubset(self.allocated)
        self.allocated.difference_update(values)
        self.free_slots = torch.cat((self.free_slots, indices))
        return len(indices)


def _new_staging_cache(
    manager: L3StagingManager, mem_pool_host: object
) -> UnifiedRadixCache:
    cache = object.__new__(UnifiedRadixCache)
    cache.cache_controller = SimpleNamespace(
        l3_staging=manager,
        mem_pool_host=mem_pool_host,
    )
    cache.l3_staging_pending_leases = deque()
    cache.l3_staging_pending_keys = set()
    cache.l3_staging_pending_tokens = {}
    cache.l3_staging_pending_by_key = {}
    cache.l3_staging_pending_token_total = 0
    cache.l3_staging_reclaim_eligible = 0
    cache.storage_state_change_resources = set()
    cache.ongoing_load_back = {}
    cache.components = {}
    cache._require_storage_prefetch_rank_agreement = lambda _phase, _values: None
    cache.synchronize_storage_prefetch_min_values = lambda values: values
    return cache


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
        self.storage_pending_checkpoints = {}
        self.storage_checkpoint_generation = 0
        self.storage_checkpoint_owners_by_operation = {}
        self.storage_checkpoint_retries = StorageCheckpointRetryQueues()
        self.storage_state_change_resources = set()
        self.storage_prefetch_ownership = {}
        self.ongoing_prefetch = {}
        self.storage_checkpoint_dependency_results = {}
        self.enable_storage_metrics = False
        self.storage_metrics_collector = None
        self.storage_prefetch_admission_host_locks = {}
        self.storage_prefetch_admission_device_locks = {}
        self.l3_staging_pending_leases = deque()
        self.l3_staging_pending_keys = set()
        self.l3_staging_pending_tokens = {}
        self.l3_staging_pending_by_key = {}
        self.l3_staging_pending_token_total = 0
        self.l3_staging_reclaim_eligible = 0
        self.storage_radix_tree_generation = 0
        self._components_tuple = ()
        self.components = {}
        self.page_size = 1
        self.cache_controller.l3_staging = SimpleNamespace(
            quotas={},
            reserve_tokens=0,
            shortfall_tokens=0,
            usage=lambda: {},
        )
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.host_backups: list[_FakeNode] = []
        self.mamba_backups: list[_FakeNode] = []
        self.storage_writes: list[tuple[int, _FakeNode]] = []

    def synchronize_storage_prefetch_flag(self, flag: bool) -> bool:
        return flag

    def match_prefix(self, _params):
        return SimpleNamespace(
            last_device_node=self.target_node,
            best_match_node=self.target_node,
        )

    def inc_lock_ref(self, _node):
        return SimpleNamespace(to_dec_params=lambda: None)

    def dec_lock_ref(self, _node, _params=None):
        return None

    def inc_host_lock_ref(self, _node):
        return SimpleNamespace(to_dec_params=lambda: None)

    def dec_host_lock_ref(self, _node, _params=None):
        return None

    def write_backup(self, node: _FakeNode, **_kwargs) -> int:
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

    def _write_missing_aux_backup(self, _node: _FakeNode, **_kwargs) -> None:
        return None

    _align_checkpoint_swa_boundary = UnifiedRadixCache._align_checkpoint_swa_boundary
    _capture_checkpoint_staging = UnifiedRadixCache._capture_checkpoint_staging
    _checkpoint_path = UnifiedRadixCache._checkpoint_path
    _checkpoint_path_hashes = staticmethod(UnifiedRadixCache._checkpoint_path_hashes)
    _checkpoint_staging_leases = UnifiedRadixCache._checkpoint_staging_leases
    _checkpoint_swa_prefix_tokens = UnifiedRadixCache._checkpoint_swa_prefix_tokens
    _checkpoint_skip_components = UnifiedRadixCache._checkpoint_skip_components
    _drive_storage_checkpoint = UnifiedRadixCache._drive_storage_checkpoint
    _enqueue_staging_leases = UnifiedRadixCache._enqueue_staging_leases
    _l3_staging_leases = UnifiedRadixCache._l3_staging_leases
    _l3_staging_capacity_state = UnifiedRadixCache._l3_staging_capacity_state
    _pin_storage_checkpoint_path = UnifiedRadixCache._pin_storage_checkpoint_path
    _queue_storage_checkpoint_retry = UnifiedRadixCache._queue_storage_checkpoint_retry
    _reclaim_l3_staging_headroom = UnifiedRadixCache._reclaim_l3_staging_headroom
    _release_storage_checkpoint = UnifiedRadixCache._release_storage_checkpoint
    _reset_checkpoint_path_scan = staticmethod(
        UnifiedRadixCache._reset_checkpoint_path_scan
    )
    _scan_storage_checkpoint_path = UnifiedRadixCache._scan_storage_checkpoint_path
    _staging_lease_key = staticmethod(UnifiedRadixCache._staging_lease_key)
    get_storage_checkpoint_dependency_state = (
        UnifiedRadixCache.get_storage_checkpoint_dependency_state
    )
    record_storage_checkpoint_dependency_result = (
        UnifiedRadixCache.record_storage_checkpoint_dependency_result
    )
    retry_storage_checkpoint = UnifiedRadixCache.retry_storage_checkpoint

    def _require_storage_prefetch_rank_agreement(self, _phase, _values):
        return None


class TestCacheStorageCheckpoint(unittest.TestCase):
    def test_unified_checkpoint_pins_host_resident_prefix(self):
        cache = _FakeCheckpointCache()

        UnifiedRadixCache._create_storage_checkpoint(
            cache,
            "hicache:request-a",
            RadixKey(array("q", [1, 2]), None),
        )

        pending = cache.storage_pending_checkpoints["hicache:request-a"]
        self.assertIs(pending.host_pin[0], cache.target_node)
        self.assertIsNone(pending.device_pin)

    def test_unified_checkpoint_pins_deferred_device_source(self):
        cache = _FakeCheckpointCache()
        cache.parent_node.backuped = False
        cache.target_node.backuped = False

        UnifiedRadixCache._create_storage_checkpoint(
            cache,
            "hicache:request-a",
            RadixKey(array("q", [1, 2]), None),
        )

        pending = cache.storage_pending_checkpoints["hicache:request-a"]
        self.assertIs(pending.device_pin[0], cache.target_node)
        self.assertIsNone(pending.host_pin)

    def test_releasing_checkpoint_pin_wakes_capacity_retries(self):
        cache = _FakeCheckpointCache()
        UnifiedRadixCache._create_storage_checkpoint(
            cache,
            "hicache:request-a",
            RadixKey(array("q", [1, 2]), None),
        )
        cache.storage_state_change_resources.clear()

        cache._release_storage_checkpoint("hicache:request-a", release_staging=True)

        self.assertEqual(
            cache.storage_state_change_resources,
            {
                CheckpointRetryResource.HOST,
                CheckpointRetryResource.AUXILIARY,
            },
        )

    def test_storage_state_changes_are_unioned_across_ranks(self):
        cache = SimpleNamespace()

        def add_remote_auxiliary_change(flags, _op):
            flags[1] = 1

        cache._all_reduce_attn_groups = add_remote_auxiliary_change

        resources = UnifiedRadixCache._synchronize_storage_state_change_resources(
            cache, {CheckpointRetryResource.HOST}
        )

        self.assertEqual(
            resources,
            {
                CheckpointRetryResource.HOST,
                CheckpointRetryResource.AUXILIARY,
            },
        )

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
                        storage_state_change_resources=set(),
                        storage_prefetch_aborted_requests=set(),
                        storage_checkpoint_owners_by_operation={},
                        storage_pending_checkpoints={},
                        l3_staging_pending_leases=deque(),
                        l3_staging_reclaim_eligible=0,
                    )
                    cache._require_storage_prefetch_rank_agreement = (
                        lambda _phase, _values: None
                    )
                    cache._backup_sequence_fingerprint = (
                        UnifiedRadixCache._backup_sequence_fingerprint
                    )
                    cache.synchronize_storage_prefetch_flag = lambda value: value
                    cache._synchronize_storage_state_change_resources = (
                        lambda resources: resources
                    )
                    cache._enqueue_staging_leases = lambda _leases: None
                    cache._l3_staging_leases = lambda *_args, **_kwargs: ()
                    cache._reclaim_l3_staging_headroom = lambda: 0
                    cache._drive_storage_checkpoint_retries = lambda _resources: None

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

            def _l3_staging_capacity_state(self, *_args):
                return None

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

                if create_checkpoint is UnifiedRadixCache._create_storage_checkpoint:
                    cache.parent_node.write_through_pending_id = None
                    cache.cache_controller.storage_write_tracker.mark_durable(
                        cache.parent_node.hash_value
                    )
                    cache.retry_storage_checkpoint("hicache:request-a")

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

                if create_checkpoint is UnifiedRadixCache._create_storage_checkpoint:
                    cache.parent_node.write_through_pending_id = None
                    cache.cache_controller.storage_write_tracker.mark_durable(
                        cache.parent_node.hash_value
                    )
                    cache.retry_storage_checkpoint("hicache:request-a")

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
                expected_initial_writes = (
                    []
                    if create_checkpoint is UnifiedRadixCache._create_storage_checkpoint
                    else [cache.target_node]
                )
                self.assertEqual(
                    [node for _, node in cache.storage_writes],
                    expected_initial_writes,
                )

                pending_completion = write_tracker.complete(1, durable_pages=1)
                self.assertIsNotNone(pending_completion)
                cache.storage_checkpoint_registry.record_write_completion(
                    pending_completion
                )
                if create_checkpoint is UnifiedRadixCache._create_storage_checkpoint:
                    cache.retry_storage_checkpoint("hicache:request-a")
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


class TestStoragePrefetchAdmissionPins(unittest.TestCase):
    def test_host_pin_replacement_acquires_before_release(self):
        first_node = object()
        second_node = object()
        events = []
        cache = SimpleNamespace(
            storage_prefetch_admission_host_locks={},
            storage_prefetch_admission_device_locks={},
            cache_controller=SimpleNamespace(prefetch_tokens_occupied=10),
            storage_state_change_resources=set(),
        )

        def acquire(node):
            events.append(("acquire", node))
            return SimpleNamespace(to_dec_params=lambda: f"lock-{id(node)}")

        cache.inc_host_lock_ref = acquire
        cache.dec_host_lock_ref = lambda node, _params: events.append(("release", node))
        cache._staging_lease_key = UnifiedRadixCache._staging_lease_key
        cache._enqueue_staging_leases = lambda _leases: None

        UnifiedRadixCache.pin_storage_prefetch_admission(
            cache, "request-a", first_node, 4
        )
        UnifiedRadixCache.pin_storage_prefetch_admission(
            cache, "request-a", second_node, 6
        )
        UnifiedRadixCache.release_storage_prefetch_admission(cache, "request-a")

        self.assertEqual(
            events,
            [
                ("acquire", first_node),
                ("acquire", second_node),
                ("release", first_node),
                ("release", second_node),
            ],
        )
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)

    def test_device_pin_replacement_acquires_before_release(self):
        first_node = object()
        second_node = object()
        events = []
        cache = SimpleNamespace(
            storage_prefetch_admission_host_locks={},
            storage_prefetch_admission_device_locks={},
            cache_controller=None,
            storage_state_change_resources=set(),
        )

        def acquire(node):
            events.append(("acquire", node))
            return SimpleNamespace(to_dec_params=lambda: f"lock-{id(node)}")

        cache.inc_lock_ref = acquire
        cache.dec_lock_ref = lambda node, _params: events.append(("release", node))
        cache._enqueue_staging_leases = lambda _leases: None

        UnifiedRadixCache.pin_storage_prefetch_device_admission(
            cache, "request-a", first_node
        )
        UnifiedRadixCache.pin_storage_prefetch_device_admission(
            cache, "request-a", second_node
        )
        UnifiedRadixCache.release_storage_prefetch_admission(cache, "request-a")

        self.assertEqual(
            events,
            [
                ("acquire", first_node),
                ("acquire", second_node),
                ("release", first_node),
                ("release", second_node),
            ],
        )


class TestL3StagingReclaim(unittest.TestCase):
    def test_full_host_eviction_respects_scan_budget(self):
        ordered_nodes = [object() for _ in range(1_000)]
        nodes = set(ordered_nodes)
        priority_calls = []
        tracker = {BASE_COMPONENT_TYPE: 0}

        def evict(node, counts):
            nodes.remove(node)
            counts[BASE_COMPONENT_TYPE] += 1

        component = SimpleNamespace(
            component_type=BASE_COMPONENT_TYPE,
            cache=SimpleNamespace(
                evictable_host_leaves=nodes,
                evictable_host_scan_queue=dict.fromkeys(ordered_nodes),
                eviction_strategy=SimpleNamespace(
                    get_priority=lambda node: priority_calls.append(node) or 0
                ),
                _evict_host_leaf=evict,
            ),
        )

        with bounded_host_eviction_scan(256):
            FullComponent.drive_host_eviction(component, 1_000, tracker)

        self.assertEqual(tracker[BASE_COMPONENT_TYPE], 256)
        self.assertEqual(len(priority_calls), 256)
        self.assertEqual(
            list(component.cache.evictable_host_scan_queue), ordered_nodes[256:]
        )

    def test_reclaim_caps_each_eviction_pass(self):
        pool = _StagingHostPool(size=10_000, page_size=10)
        manager = L3StagingManager()
        manager.install(pool, 0.8, lambda values: values)
        self.assertIsNotNone(pool.alloc(2_000))
        with l3_staging_allocation():
            staging_indices = pool.alloc(8_000)
        assert staging_indices is not None
        quota = next(iter(manager.quotas.values()))
        cache = _new_staging_cache(manager, pool)
        eviction_requests = []
        cache.evict_host = lambda request: eviction_requests.append(request) or 0
        cache._enqueue_staging_leases(
            (L3StagingLease(quota=quota, indices=staging_indices),)
        )

        shortfall = cache._reclaim_l3_staging_headroom()

        self.assertEqual(shortfall, 8_000)
        self.assertEqual(eviction_requests, [640])
        self.assertEqual(cache.l3_staging_reclaim_eligible, 0)

    def test_exhausted_eviction_scan_rearms_reclaim(self):
        pool = _StagingHostPool(size=10_000, page_size=10)
        manager = L3StagingManager()
        manager.install(pool, 0.8, lambda values: values)
        self.assertIsNotNone(pool.alloc(2_000))
        with l3_staging_allocation():
            staging_indices = pool.alloc(8_000)
        assert staging_indices is not None
        quota = next(iter(manager.quotas.values()))
        cache = _new_staging_cache(manager, pool)

        def exhaust_scan(_request: int) -> int:
            budget = get_host_eviction_scan_budget()
            assert budget is not None
            budget.remaining_nodes = 0
            return 0

        cache.evict_host = exhaust_scan
        cache._enqueue_staging_leases(
            (L3StagingLease(quota=quota, indices=staging_indices),)
        )
        cache.storage_state_change_resources.clear()

        shortfall = cache._reclaim_l3_staging_headroom()

        self.assertEqual(shortfall, 8_000)
        self.assertEqual(cache.l3_staging_reclaim_eligible, 1)
        self.assertEqual(
            cache.storage_state_change_resources,
            {CheckpointRetryResource.SCHEDULER},
        )

    def test_discard_preserves_full_and_swa_ordering(self):
        full_pool = _StagingHostPool(size=100, page_size=10)
        swa_pool = _StagingHostPool(size=50, page_size=10)
        full_entry = SimpleNamespace(
            name=PoolName.KV,
            host_pool=full_pool,
            is_primary_index_anchor=True,
            host_evict_fn=None,
        )
        swa_entry = SimpleNamespace(
            name=PoolName.SWA,
            host_pool=swa_pool,
            is_primary_index_anchor=False,
            host_evict_fn=lambda _request: 0,
        )
        pool_group = SimpleNamespace(
            entries=(full_entry, swa_entry),
            entry_map={PoolName.KV: full_entry, PoolName.SWA: swa_entry},
            anchor_entry=full_entry,
            size=full_pool.size,
            page_size=full_pool.page_size,
            alloc=full_pool.alloc,
            free=full_pool.free,
            available_size=full_pool.available_size,
        )
        manager = L3StagingManager()
        manager.install(pool_group, 0.2, lambda values: values)
        full_quota = manager.quota_for_pool(full_pool)
        swa_quota = manager.quota_for_pool(swa_pool)
        assert full_quota is not None and swa_quota is not None
        self.assertIsNotNone(full_pool.alloc(80))
        with l3_staging_allocation():
            full_staging = full_pool.alloc(20)
            swa_staging = swa_pool.alloc(10)
        assert full_staging is not None and swa_staging is not None

        root_node = SimpleNamespace()
        node = SimpleNamespace(
            id=1,
            parent=root_node,
            hash_value=("page-a", "page-b"),
            component_data=[
                SimpleNamespace(
                    value=object(), host_value=full_staging, host_lock_ref=0
                ),
                SimpleNamespace(
                    value=object(), host_value=swa_staging, host_lock_ref=0
                ),
                SimpleNamespace(value=None, host_value=None, host_lock_ref=0),
            ],
        )
        eviction_order = []

        class _Component:
            def __init__(
                self, component_type: ComponentType, pool: _StagingHostPool
            ) -> None:
                self.component_type = component_type
                self.pool = pool

            def evict_component(self, owner, target):
                self.assert_target(target)
                component_data = owner.component_data[self.component_type]
                host_value = component_data.host_value
                self.pool.free(host_value)
                component_data.host_value = None
                eviction_order.append(self.component_type)
                if self.component_type == BASE_COMPONENT_TYPE:
                    assert owner.component_data[ComponentType.SWA].host_value is None
                return 0, len(host_value)

            @staticmethod
            def assert_target(target):
                assert target is EvictLayer.HOST

        components = {
            BASE_COMPONENT_TYPE: _Component(BASE_COMPONENT_TYPE, full_pool),
            ComponentType.SWA: _Component(ComponentType.SWA, swa_pool),
        }
        cache = _new_staging_cache(manager, pool_group)
        cache.root_node = root_node
        cache.components = components
        cache.evict_host = lambda _request: 0
        cache._evict_component_and_detach_lru = lambda owner, component, target: (
            component.evict_component(owner, target)
        )
        cache._update_evictable_leaf_sets = lambda _node: None
        cache._enqueue_staging_leases(
            (
                L3StagingLease(
                    quota=full_quota,
                    indices=full_staging,
                    owner_node=node,
                    component_type=BASE_COMPONENT_TYPE,
                ),
                L3StagingLease(
                    quota=swa_quota,
                    indices=swa_staging,
                    owner_node=node,
                    component_type=ComponentType.SWA,
                ),
            )
        )

        self.assertEqual(cache._reclaim_l3_staging_headroom(), 30)
        self.assertEqual(cache._reclaim_l3_staging_headroom(), 0)
        self.assertEqual(eviction_order, [ComponentType.SWA, BASE_COMPONENT_TYPE])
        self.assertEqual(full_quota.available_tokens, 20)
        self.assertEqual(swa_quota.available_tokens, 10)

    def test_duplicate_lease_keeps_canonical_cursor_accounting(self):
        pool = _StagingHostPool(size=200, page_size=1)
        manager = L3StagingManager()
        manager.install(pool, 0.5, lambda values: values)
        quota = next(iter(manager.quotas.values()))
        owner = SimpleNamespace(id=7)
        indices = torch.arange(100, dtype=torch.int64)
        canonical = L3StagingLease(
            quota=quota,
            indices=indices,
            owner_node=owner,
            component_type=BASE_COMPONENT_TYPE,
            reclaim_index_cursor=64,
        )
        duplicate = L3StagingLease(
            quota=quota,
            indices=indices.clone(),
            owner_node=owner,
            component_type=BASE_COMPONENT_TYPE,
        )
        cache = _new_staging_cache(manager, pool)

        cache._enqueue_staging_leases((canonical,), schedule=False)
        cache._enqueue_staging_leases((duplicate,), schedule=False)
        self.assertEqual(list(cache.l3_staging_pending_leases), [canonical])
        self.assertEqual(cache.l3_staging_pending_token_total, 36)

        duplicate.reclaim_index_cursor = 80
        cache._enqueue_staging_leases((duplicate,), schedule=False)
        self.assertEqual(canonical.reclaim_index_cursor, 80)
        self.assertEqual(cache.l3_staging_pending_token_total, 20)


class TestStorageCheckpointRetryQueues(unittest.TestCase):
    def test_checkpoint_drive_only_scans_one_path_quantum(self):
        page_hashes = [f"page-{index}" for index in range(100)]
        tracker = StorageWriteTracker()
        tracker.mark_durable(page_hashes)
        path = [
            SimpleNamespace(
                hash_value=[page_hash],
                write_through_pending_id=None,
                backuped=True,
            )
            for page_hash in page_hashes
        ]
        checkpoint = PendingStorageCheckpoint(
            handle="hicache:request-a",
            generation=1,
            radix_key=RadixKey(array("q", range(100)), None),
            path=tuple(path),
            path_hashes=tuple(page_hashes),
        )
        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(storage_write_tracker=tracker),
            storage_state_change_resources=set(),
        )
        cache._require_storage_prefetch_rank_agreement = lambda _phase, _values: None

        blocked_reason = UnifiedRadixCache._drive_storage_checkpoint(
            cache, checkpoint, path
        )

        self.assertEqual(blocked_reason, "continuation")
        self.assertEqual(checkpoint.cursor_pages, 64)
        self.assertEqual(
            cache.storage_state_change_resources,
            {CheckpointRetryResource.SCHEDULER},
        )

    def test_scheduler_retry_work_is_quantum_bounded(self):
        retries = StorageCheckpointRetryQueues()
        pending = {}
        for sequence in range(20):
            handle = f"hicache:request-{sequence}"
            checkpoint = PendingStorageCheckpoint(
                handle=handle,
                generation=sequence,
                radix_key=None,
                path=(),
                path_hashes=(),
                blocked_resource=CheckpointRetryResource.SCHEDULER,
                blocked_generation=0,
                retry_queued=True,
                retry_ticket=1,
            )
            pending[handle] = checkpoint
            retries.queues[CheckpointRetryResource.SCHEDULER].append(
                (handle, sequence, 1)
            )
        driven = []
        cache = SimpleNamespace(
            storage_checkpoint_retries=retries,
            storage_pending_checkpoints=pending,
            storage_state_change_resources=set(),
        )
        cache._require_storage_prefetch_rank_agreement = lambda _phase, _values: None
        cache.retry_storage_checkpoint = driven.append

        UnifiedRadixCache._drive_storage_checkpoint_retries(
            cache, {CheckpointRetryResource.SCHEDULER}
        )

        self.assertEqual(len(driven), 8)
        self.assertEqual(len(retries.queues[CheckpointRetryResource.SCHEDULER]), 12)
        self.assertEqual(
            retries.continuation_resources,
            {CheckpointRetryResource.SCHEDULER},
        )

        UnifiedRadixCache._drive_storage_checkpoint_retries(
            cache, {CheckpointRetryResource.SCHEDULER}
        )
        UnifiedRadixCache._drive_storage_checkpoint_retries(
            cache, {CheckpointRetryResource.SCHEDULER}
        )

        self.assertEqual(len(driven), 20)
        self.assertFalse(retries.queues[CheckpointRetryResource.SCHEDULER])
        self.assertFalse(retries.continuation_resources)

    def test_checkpoint_path_scan_is_bounded(self):
        root = _FakeNode(0, [], None)
        path = []
        parent = root
        for index in range(70):
            node = _FakeNode(index + 1, [f"page-{index}"], parent)
            path.append(node)
            parent = node
        checkpoint = PendingStorageCheckpoint(
            handle="hicache:request-a",
            generation=1,
            radix_key=RadixKey(array("q", range(70)), None),
            path=tuple(path),
            path_hashes=tuple(f"page-{index}" for index in range(70)),
        )
        cache = SimpleNamespace(
            root_node=root,
            storage_radix_tree_generation=0,
            storage_state_change_resources=set(),
        )
        cache._reset_checkpoint_path_scan = (
            UnifiedRadixCache._reset_checkpoint_path_scan
        )
        cache._capture_checkpoint_staging = lambda _checkpoint, _nodes: None
        cache._require_storage_prefetch_rank_agreement = lambda _phase, _values: None
        cache._checkpoint_path_hashes = UnifiedRadixCache._checkpoint_path_hashes
        cache._pin_storage_checkpoint_path = lambda _checkpoint, _path: None

        first_path, first_drive_path, first_complete = (
            UnifiedRadixCache._scan_storage_checkpoint_path(cache, checkpoint)
        )
        second_path, second_drive_path, second_complete = (
            UnifiedRadixCache._scan_storage_checkpoint_path(cache, checkpoint)
        )

        self.assertFalse(first_complete)
        self.assertIsNone(first_path)
        self.assertIsNone(first_drive_path)
        self.assertTrue(second_complete)
        self.assertEqual(second_path, path)
        self.assertEqual(list(second_drive_path), path)


if __name__ == "__main__":
    unittest.main()
