"""Storage submission failures must return staging and release cache ownership."""

import unittest
from array import array
from collections import defaultdict
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import torch
from test_storage_prefetch_lifecycle import _staged_fixture

from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.srt.mem_cache.buffer_mode.pipeline import (
    _UnifiedBackupIntent,
    _UnifiedBufferBackupEntry,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import PrefetchOperation
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
)
from sglang.srt.mem_cache.unified_radix_cache import _OngoingPrefetch
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestStorageSubmissionRecovery(unittest.TestCase):
    def test_backup_submission_failure_releases_staging_and_content_refs(self):
        cache, pipeline, _ = _staged_fixture()
        cc = cache.cache_controller
        cc.write_storage = Mock(side_effect=RuntimeError("queue closed"))
        cc.mem_pool_host.entry_map[PoolName.SWA].host_pool.page_size = 2
        cache.enable_storage_metrics = True
        cache.storage_metrics_collector = Mock()
        cache.dec_lock_ref = Mock()
        full = torch.arange(4)
        swa = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(2))
        sidecar = PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=full,
            indices_from_pool=PoolName.KV,
        )
        snapshot = BufferBackupSnapshot(
            node_id=7,
            parent_node_id=0,
            parent_is_root=True,
            parent_last_hash=None,
            hash_values=["a", "b"],
            key=RadixKey(array("q", range(4))),
            prefix_keys=None,
        )
        lock = DecLockRefParams()
        pipeline.ongoing_write_through[7] = _UnifiedBufferBackupEntry(
            _UnifiedBackupIntent(snapshot), full, [swa, sidecar], lock, 6
        )
        pipeline.write_staged_tokens_ = 10
        pipeline.inflight_backup_node_ids = {7, 8}
        pipeline.inflight_backup_hashes = {"a": 2, "b": 1, "other": 1}

        with self.assertRaisesRegex(RuntimeError, "queue closed"):
            pipeline.finish_backup_ack(7)

        cache.dec_lock_ref.assert_called_once_with(7, lock)
        cc.mem_pool_host.free.assert_called_once_with(full)
        cc.mem_pool_host.entry_map[PoolName.SWA].host_pool.free.assert_called_once_with(
            swa.host_indices
        )
        self.assertEqual(pipeline.ongoing_write_through, {})
        self.assertEqual(pipeline.ongoing_backup, {})
        self.assertEqual(pipeline.write_staged_tokens_, 4)
        self.assertEqual(pipeline.inflight_backup_node_ids, {8})
        self.assertEqual(pipeline.inflight_backup_hashes, {"a": 1, "other": 1})
        cache.storage_metrics_collector.log_backup_dropped_tokens.assert_called_once_with(
            4
        )

    def _prefetch(self, mode, *, allocated=False):
        cache, pipeline, req = _staged_fixture()
        handle = req.cache_request_handle
        cc = cache.cache_controller
        cache.host_memory_mode = mode
        cache.ongoing_backup = {}
        cache.dec_host_lock_ref = Mock()
        cache.match_prefix = Mock(
            return_value=SimpleNamespace(device_indices=torch.arange(2))
        )
        cache.storage_existence_cache = Mock()
        cache.enable_storage_metrics = True
        cache.storage_metrics_collector = Mock()
        pipeline.staged_prefetches.clear()
        pipeline.anchor_lock_cap_tokens = 100
        pipeline.set_prefix_ctx(handle, [0, 1])
        key = RadixKey(array("q", range(2, 6)))
        swa = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(2))
        operation = PrefetchOperation(handle, key, pool_transfers=[swa])
        operation.storage_hit_count = 4
        operation.stats_requested_tokens = 4
        operation.hash_value = ["a", "b"]
        operation.all_hash_values = ["a", "b"]
        operation.storage_start = 2
        host_indices = torch.arange(4)
        operation.host_indices = host_indices if allocated else None
        operation.buffer_host_occupied_units = 4 if allocated else None
        cache.ongoing_prefetch[handle] = _OngoingPrefetch(
            1,
            key,
            operation.host_indices,
            operation,
            DecLockRefParams() if mode == "cache" else None,
            {ComponentType.SWA: [swa]},
        )
        if mode == "cache":
            cache.buffer_pipeline = None
        cc.prefetch_tokens_occupied = 11 + (4 if mode == "cache" or allocated else 0)
        cc.prefetch_hit_queue = Queue()
        cc.ack_prefetch_queue = Queue()
        cc.ack_backup_queue = Queue()
        cc.host_mem_release_queue = Queue()
        cc.extra_host_mem_release_queues = {}
        cc.prefetch_buffer = Mock()
        cc.allocate_storage_hit = Mock(return_value=(host_indices, 4))
        cc.can_fit_prefetch_host_buffers = Mock(return_value=True)
        cc.mem_pool_host.page_size = 2
        return cache, pipeline, handle, operation, swa, host_indices

    def test_prefetch_submission_failure_releases_all_ownership(self):
        for mode in ("buffer_only", "cache"):
            with self.subTest(mode=mode):
                cache, pipeline, handle, operation, swa, full = self._prefetch(mode)
                cc = cache.cache_controller
                cc.prefetch_buffer.put.side_effect = RuntimeError("queue closed")
                cc.prefetch_hit_queue.put(operation)
                info = cache.ongoing_prefetch[handle]
                swa_indices = swa.host_indices

                with self.assertRaisesRegex(RuntimeError, "queue closed"):
                    cache._drain_storage_control_queues_impl(1, 0, 0, 0, {}, False)

                cc.mem_pool_host.free.assert_called_once_with(full)
                cc.mem_pool_host.entry_map[
                    PoolName.SWA
                ].host_pool.free.assert_called_once_with(swa_indices)
                self.assertTrue(operation.is_terminated())
                self.assertIsNone(operation.host_indices)
                self.assertIsNone(operation.buffer_host_occupied_units)
                self.assertIsNone(swa.host_indices)
                self.assertEqual(cache.ongoing_prefetch, {})
                self.assertEqual(cache._storage_prefetch_hit_remaining_by_reqid, {})
                self.assertEqual(cc.prefetch_tokens_occupied, 11)
                self.assertEqual(pipeline.anchor_locks, {})
                self.assertEqual(pipeline.anchor_locked_tokens_, 0)
                if mode == "buffer_only":
                    self.assertNotIn(handle, pipeline._prefetch_prefix_ctx)
                    cache.tree_core.dec_full_pin.assert_called_once_with(1)
                    cache.dec_host_lock_ref.assert_not_called()
                else:
                    cache.dec_host_lock_ref.assert_called_once_with(
                        1, info.anchor_lock_params
                    )
                cache.storage_metrics_collector.log_storage_prefetch_unfulfilled_tokens.assert_called_once_with(
                    4, "dropped"
                )

    def test_cleanup_terminates_pending_unallocated_prefetch(self):
        cache, pipeline, handle, operation, swa, _ = self._prefetch("buffer_only")
        swa.host_indices = None

        StorageAttachment(cache)._release_pending_storage_ops()

        self.assertTrue(operation.is_terminated())
        self.assertEqual(cache.ongoing_prefetch, {})
        self.assertNotIn(handle, pipeline._prefetch_prefix_ctx)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 11)

    def test_cleanup_releases_only_scheduler_owned_sidecars(self):
        for done in (False, True):
            with self.subTest(pool_transfers_done=done):
                cache, _, _, operation, swa, full = self._prefetch(
                    "buffer_only", allocated=True
                )
                operation.pool_transfers_done = done
                operation.completed_tokens = 2
                cc = cache.cache_controller
                cc.append_host_mem_release = Mock()

                StorageAttachment(cache)._release_pending_storage_ops()

                self.assertEqual(cc.append_host_mem_release.call_count, 1)
                call = cc.append_host_mem_release.call_args
                torch.testing.assert_close(call.kwargs["host_indices"], full[:2])
                self.assertEqual(call.kwargs["extra_pools"], [swa] if done else None)
                self.assertEqual(cc.prefetch_tokens_occupied, 11)
                self.assertEqual(cache.ongoing_prefetch, {})


class TestHostDuplicateOrder(unittest.TestCase):
    def test_reclaim_order_is_independent_of_completion_order(self):
        for order in ((9, 3, 6), (6, 9, 3)):
            with self.subTest(order=order):
                core = UnifiedTreeCore.__new__(UnifiedTreeCore)
                core.full_host_duplicates = {
                    nid: SimpleNamespace(
                        id=nid,
                        component_data={
                            ComponentType.FULL: SimpleNamespace(
                                value=True, host_value=True
                            )
                        },
                    )
                    for nid in order
                }
                # Membership and release are isolated; the two-pass selection
                # and token budget execute in the real tree-core method.
                core.evictable_device_leaves = ()
                core._can_reclaim_full_host_duplicate = Mock(return_value=True)
                released = []

                def release(node, tracker, device_frees, host_frees):
                    released.append(node.id)
                    tracker[ComponentType.FULL] += 1

                core._release_full_host_duplicate = release
                for _ in range(2):
                    core._reclaim_full_host_duplicates(1, defaultdict(int), {}, {})
                self.assertEqual(released, [3, 6])
                self.assertEqual(list(core.full_host_duplicates), [9])


class TestUnifiedStorageBackendGate(unittest.TestCase):
    def test_backend_gate_is_scoped_to_unified_hicache(self):
        for unified, hicache, backend in (
            (True, True, "mooncake"),
            (True, True, "nixl"),
            (True, True, "file"),
            (True, True, None),
            (False, True, "mooncake"),
            (True, False, "nixl"),
        ):
            with self.subTest(unified=unified, hicache=hicache, backend=backend):
                args = ServerArgs(model_path="dummy")
                msgspec.Struct.__setattr__(args, "enable_unified_memory", unified)
                msgspec.Struct.__setattr__(args, "enable_hierarchical_cache", hicache)
                msgspec.Struct.__setattr__(args, "hicache_storage_backend", backend)
                with patch(
                    "sglang.srt.arg_groups.kv_cache_hook.attention_backends_of",
                    return_value=("triton", "triton"),
                ):
                    if unified and hicache and backend in {"mooncake", "nixl"}:
                        with self.assertRaisesRegex(ValueError, "storage backends"):
                            handle_unified_memory_pool(args)
                    else:
                        handle_unified_memory_pool(args)


if __name__ == "__main__":
    unittest.main()
