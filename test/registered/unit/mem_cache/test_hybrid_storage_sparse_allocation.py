"""CPU regressions for sparse hybrid checkpoints under host capacity pressure."""

import unittest
from array import array
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_CONTROLLER_MODULE = HybridCacheController.__module__
_PAGE_SIZE = 4
_PAGES = 32


def _operation(*, trailing=True, checkpoints=(16, 32), assume_stored=False):
    transfers = (
        [
            PoolTransfer(
                name=PoolName.SWA,
                keys=["placeholder"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]
        if trailing
        else None
    )
    operation = PrefetchOperation(
        CacheRequestHandle("sparse-capacity", 0),
        list(range(_PAGES * _PAGE_SIZE)),
        pool_transfers=transfers,
        assume_stored=assume_stored,
    )
    operation.hash_value = [f"page-{page}" for page in range(1, _PAGES + 1)]
    operation.storage_hit_count = _PAGES * _PAGE_SIZE
    operation.storage_start = 0
    operation.pool_storage_result.restorable_prefix_pages = (
        None if checkpoints is None else list(checkpoints)
    )
    return operation


def _controller(capacity_pages):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.page_size = _PAGE_SIZE
    controller.mem_pool_host = SimpleNamespace(
        anchor_entry=SimpleNamespace(host_pool=object()),
        available_size=Mock(return_value=capacity_pages * _PAGE_SIZE),
    )
    # Model only host capacity. Selection and retry order are the production method.
    controller.alloc_prefetch_host_buffers = Mock(
        side_effect=lambda operation, size: (
            torch.arange(size) if 0 < size <= capacity_pages * _PAGE_SIZE else None
        )
    )
    controller.can_fit_prefetch_host_buffers = Mock(
        side_effect=lambda operation, size, **kwargs: (
            size <= capacity_pages * _PAGE_SIZE
        )
    )
    controller.prefetch_hits_sync_groups = []
    controller._all_reduce = Mock()
    return controller


def _allocate(controller, operation, *, shared=False, allow_partial=True, min_pages=1):
    with patch(f"{_CONTROLLER_MODULE}.uses_shared_host_layout", return_value=shared):
        return controller.allocate_storage_hit(
            operation,
            operation.storage_hit_count,
            allow_partial=allow_partial,
            min_tokens=min_pages * _PAGE_SIZE,
            evict_host=Mock(),
        )


class TestSparseCheckpointAllocation(CustomTestCase):
    def test_capacity_between_checkpoints_selects_previous_checkpoint(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                controller = _controller(24)
                operation = _operation()
                indices, count = _allocate(controller, operation, shared=shared)

                self.assertEqual(count, 16 * _PAGE_SIZE)
                self.assertEqual(indices.numel(), count)
                attempted = [
                    args[1]
                    for args, _ in controller.alloc_prefetch_host_buffers.call_args_list
                ]
                self.assertEqual(attempted[-1], 16 * _PAGE_SIZE)
                self.assertNotIn(24 * _PAGE_SIZE, attempted)

    def test_no_checkpoint_fits_does_not_allocate_invalid_endpoint(self):
        for shared in (False, True):
            for checkpoints, capacity in (((16, 32), 8), ((), 24)):
                with self.subTest(shared=shared, checkpoints=checkpoints):
                    controller = _controller(capacity)
                    indices, count = _allocate(
                        controller, _operation(checkpoints=checkpoints), shared=shared
                    )
                    self.assertIsNone(indices)
                    self.assertEqual(count, 0)
                    calls = controller.alloc_prefetch_host_buffers.call_args_list
                    attempted = [args[1] for args, _ in calls]
                    self.assertTrue(
                        all(size == _PAGES * _PAGE_SIZE for size in attempted)
                    )

    def test_checkpoint_below_minimum_does_not_allocate(self):
        controller = _controller(24)
        indices, _ = _allocate(controller, _operation(), min_pages=17)
        self.assertIsNone(indices)
        attempted = [
            args[1] for args, _ in controller.alloc_prefetch_host_buffers.call_args_list
        ]
        self.assertNotIn(16 * _PAGE_SIZE, attempted)

    def test_capacity_at_checkpoint_is_unchanged(self):
        for shared in (False, True):
            for capacity in (16, 32):
                with self.subTest(shared=shared, capacity=capacity):
                    controller = _controller(capacity)
                    indices, count = _allocate(controller, _operation(), shared=shared)
                    self.assertEqual(count, capacity * _PAGE_SIZE)
                    self.assertEqual(indices.numel(), count)

    def test_disallow_partial_keeps_full_allocation_contract(self):
        controller = _controller(24)
        indices, count = _allocate(controller, _operation(), allow_partial=False)
        self.assertIsNone(indices)
        self.assertEqual(count, _PAGES * _PAGE_SIZE)

    def test_plain_kv_and_legacy_contiguous_results_keep_contiguous_cap(self):
        for trailing in (False, True):
            with self.subTest(trailing=trailing):
                controller = _controller(24)
                operation = _operation(trailing=trailing, checkpoints=None)
                indices, count = _allocate(controller, operation)
                self.assertEqual(count, 24 * _PAGE_SIZE)
                self.assertEqual(indices.numel(), count)

    def test_shared_allocation_realigns_entire_trailing_window(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = _PAGE_SIZE
        operation = _operation()
        transfer = operation.pool_transfers[0]
        transfer.keys = ["old-31", "old-32"]
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=SimpleNamespace(name=PoolName.KV, host_pool=object()),
            entry_map={
                PoolName.SWA: SimpleNamespace(host_pool=SimpleNamespace(page_size=1))
            },
        )
        kv, swa = torch.arange(16 * _PAGE_SIZE), torch.arange(2)
        controller._alloc_shared_host_requests_with_reclaim = Mock(
            return_value=[kv, swa]
        )
        with patch(f"{_CONTROLLER_MODULE}.uses_shared_host_layout", return_value=True):
            indices = controller.alloc_prefetch_host_buffers(operation, 16 * _PAGE_SIZE)

        self.assertIs(indices, kv)
        self.assertIs(transfer.host_indices, swa)
        self.assertEqual(transfer.keys, ["page-15", "page-16"])
        controller._alloc_shared_host_requests_with_reclaim.assert_called_once_with(
            [(PoolName.KV, 16 * _PAGE_SIZE), (PoolName.SWA, 2)]
        )


class TestSparseCheckpointQuery(CustomTestCase):
    def test_rank_intersection_is_retained_for_later_capacity_capping(self):
        local_sets = ((16, 24, 32), (16, 32))
        for rank, checkpoints in enumerate(local_sets):
            with self.subTest(rank=rank):
                controller = _controller(24)
                operation = _operation(checkpoints=checkpoints)
                operation.query_pool_hit_pages = {PoolName.KV: _PAGES}
                peer = torch.zeros(_PAGES + 1, dtype=torch.int)
                peer[0] = 1
                peer[list(local_sets[1 - rank])] = 1

                def reduce(tensor, op, groups):
                    self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                    tensor[: _PAGES + 1] = torch.minimum(tensor[: _PAGES + 1], peer)

                controller._all_reduce.side_effect = reduce
                count = controller._sync_prefetch_hit_query(
                    operation, _PAGES * _PAGE_SIZE
                )
                self.assertEqual(count, _PAGES * _PAGE_SIZE)
                self.assertEqual(
                    operation.pool_storage_result.restorable_prefix_pages, [16, 32]
                )
                indices, count = _allocate(controller, operation)
                self.assertEqual(count, 16 * _PAGE_SIZE)
                self.assertEqual(indices.numel(), count)

    def test_cancelled_rank_votes_zero_with_same_collective_shape(self):
        controller = _controller(24)
        operation = _operation()
        sizes = []
        controller._all_reduce.side_effect = lambda tensor, *_: sizes.append(
            tensor.numel()
        )
        self.assertEqual(
            controller._sync_prefetch_hit_query(operation, 32 * _PAGE_SIZE),
            32 * _PAGE_SIZE,
        )
        operation.mark_terminate()
        self.assertEqual(controller._sync_prefetch_hit_query(operation, 0), 0)
        self.assertEqual(operation.pool_storage_result.restorable_prefix_pages, [])
        self.assertEqual(sizes, [33 + len(PoolName)] * 2)

    def test_assumed_stored_trailing_state_proves_only_original_endpoint(self):
        controller = _controller(24)
        controller.storage_backend = Mock()
        operation = _operation(checkpoints=None, assume_stored=True)
        # Hashing is independent of checkpoint selection and uses a Linux-only
        # native extension. Exercise the real query with a fixed hash chain.
        with patch(
            f"{_CONTROLLER_MODULE}.get_storage_hash_str",
            return_value=operation.hash_value,
        ):
            hashes, count = controller._storage_hit_query(operation)

        self.assertEqual(count, _PAGES * _PAGE_SIZE)
        self.assertEqual(len(hashes), _PAGES)
        self.assertEqual(operation.query_pool_hit_pages, {})
        self.assertEqual(
            operation.pool_storage_result.restorable_prefix_pages, [_PAGES]
        )
        controller.storage_backend.batch_exists.assert_not_called()
        controller.storage_backend.batch_exists_v2.assert_not_called()
        self.assertEqual(_allocate(controller, operation), (None, 0))

    def test_assumed_stored_plain_kv_remains_contiguous_without_lookup(self):
        controller = _controller(24)
        controller.storage_backend = Mock()
        operation = _operation(trailing=False, checkpoints=None, assume_stored=True)
        with patch(
            f"{_CONTROLLER_MODULE}.get_storage_hash_str",
            return_value=operation.hash_value,
        ):
            _, count = controller._storage_hit_query(operation)
        self.assertEqual(count, _PAGES * _PAGE_SIZE)
        self.assertIsNone(operation.pool_storage_result.restorable_prefix_pages)
        indices, allocated = _allocate(controller, operation)
        self.assertEqual(allocated, 24 * _PAGE_SIZE)
        self.assertEqual(indices.numel(), allocated)
        controller.storage_backend.batch_exists.assert_not_called()
        controller.storage_backend.batch_exists_v2.assert_not_called()


class TestSparseCheckpointCacheDrain(CustomTestCase):
    def _cache(self, capacity):
        controller = _controller(capacity)
        controller.prefetch_hit_queue = Queue()
        controller.prefetch_buffer = Queue()
        controller.ack_prefetch_queue = Queue()
        controller.ack_backup_queue = Queue()
        controller.host_mem_release_queue = Queue()
        controller.extra_host_mem_release_queues = {}
        controller.append_host_mem_release = Mock()
        controller.prefetch_tokens_occupied = _PAGES * _PAGE_SIZE
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(page_size=_PAGE_SIZE)
        cache.cache_controller = controller
        cache.host_memory_mode = "cache"
        cache.buffer_pipeline = None
        cache.page_size = cache.prefetch_threshold = _PAGE_SIZE
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.evict_host = Mock()
        cache.storage_prefetch_retries = Mock()
        cache._record_storage_prefetch_hit = Mock()
        cache._account_prefetch_outcome = Mock()
        cache._finish_storage_prefetch = Mock()
        cache._resolve_storage_prefetch_tokens = Mock()
        cache._alloc_prefetch_aux_staging = Mock(return_value=True)
        cache.dec_host_lock_ref = Mock()
        operation = _operation()
        info = _OngoingPrefetch(
            17, RadixKey(array("q", operation.token_ids)), None, operation, object(), {}
        )
        cache.ongoing_prefetch = {operation.handle: info}
        controller.prefetch_hit_queue.put(operation)
        return cache, operation, info

    @staticmethod
    def _drain(cache):
        with patch(f"{_CONTROLLER_MODULE}.uses_shared_host_layout", return_value=False):
            cache._drain_storage_control_queues_impl(
                n_storage_hit=1,
                n_ack_prefetch=0,
                n_backup=0,
                n_release=0,
                extra_release_counts=None,
                log_metrics=False,
            )

    def test_no_legal_endpoint_retires_pending_prefetch_without_io(self):
        cache, operation, info = self._cache(8)
        self._drain(cache)
        self.assertTrue(cache.cache_controller.prefetch_buffer.empty())
        self.assertNotIn(operation.handle, cache.ongoing_prefetch)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
        cache.dec_host_lock_ref.assert_called_once_with(
            info.anchor_node_id, info.anchor_lock_params
        )
        cache._alloc_prefetch_aux_staging.assert_not_called()
        cache.cache_controller.append_host_mem_release.assert_called_once_with(
            host_indices=None, extra_pools=[]
        )

    def test_cache_drain_stages_auxiliary_pools_for_selected_endpoint(self):
        cache, operation, info = self._cache(24)
        self._drain(cache)
        cache._alloc_prefetch_aux_staging.assert_called_once_with(info, 16 * _PAGE_SIZE)
        self.assertIs(cache.cache_controller.prefetch_buffer.get_nowait(), operation)
        self.assertEqual(operation.storage_hit_count, 16 * _PAGE_SIZE)
        self.assertEqual(operation.hash_value[-1], "page-16")
        self.assertEqual(operation.host_indices.numel(), 16 * _PAGE_SIZE)
        cache.dec_host_lock_ref.assert_not_called()


if __name__ == "__main__":
    unittest.main()
