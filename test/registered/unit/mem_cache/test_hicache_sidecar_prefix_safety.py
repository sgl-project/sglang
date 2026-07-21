"""Tests for the DSA/NSA sidecar prefix-safety invariant (issue #30057).

HiCache L3 prefetch must never publish a KV prefix longer than an ALL_PAGES
sidecar pool (e.g. DSA/NSA INDEXER) actually covers.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    clamp_prefix_to_sidecar_coverage,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd")

PAGE_SIZE = 4
NUM_PAGES = 8


class TestUpdateExtraPoolHitPages(CustomTestCase):
    """update_extra_pool_hit_pages records both stats unconditionally: sum for
    TRAILING_PAGES consumers, contiguous prefix for ALL_PAGES consumers."""

    def test_mid_hole_sum_and_prefix_diverge(self):
        result = PoolTransferResult.empty()
        result.update_extra_pool_hit_pages(
            {PoolName.INDEXER: [True, False, True, True]}
        )
        self.assertEqual(result.extra_pool_hit_pages[PoolName.INDEXER], 3)
        self.assertEqual(result.extra_pool_hit_prefix[PoolName.INDEXER], 1)

    def test_all_true_sum_and_prefix_equal_full_length(self):
        result = PoolTransferResult.empty()
        result.update_extra_pool_hit_pages({PoolName.INDEXER: [True, True, True]})
        self.assertEqual(result.extra_pool_hit_pages[PoolName.INDEXER], 3)
        self.assertEqual(result.extra_pool_hit_prefix[PoolName.INDEXER], 3)

    def test_leading_miss_prefix_zero_sum_nonzero(self):
        result = PoolTransferResult.empty()
        result.update_extra_pool_hit_pages(
            {PoolName.INDEXER: [False, True, True, True]}
        )
        self.assertEqual(result.extra_pool_hit_pages[PoolName.INDEXER], 3)
        self.assertEqual(result.extra_pool_hit_prefix[PoolName.INDEXER], 0)

    def test_multiple_pools_tracked_independently(self):
        result = PoolTransferResult.empty()
        result.update_extra_pool_hit_pages(
            {
                PoolName.INDEXER: [True, False, True],
                PoolName.SWA: [True, True, True],
            }
        )
        self.assertEqual(result.extra_pool_hit_prefix[PoolName.INDEXER], 1)
        self.assertEqual(result.extra_pool_hit_prefix[PoolName.SWA], 3)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.INDEXER], 2)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.SWA], 3)


class TestClampPrefixToSidecarCoverage(CustomTestCase):
    def setUp(self):
        self.transfer = PoolTransfer(
            name=PoolName.INDEXER,
            hit_policy=PoolHitPolicy.ALL_PAGES,
            indices_from_pool=PoolName.KV,
        )

    def test_missing_sidecar_entry_clamps_to_zero(self):
        safe = clamp_prefix_to_sidecar_coverage(8, [self.transfer], {}, page_size=4)
        self.assertEqual(safe, 0)

    def test_partial_sidecar_clamps_down(self):
        safe = clamp_prefix_to_sidecar_coverage(
            8, [self.transfer], {PoolName.INDEXER: 1}, page_size=4
        )
        self.assertEqual(safe, 4)

    def test_full_sidecar_coverage_is_a_noop(self):
        safe = clamp_prefix_to_sidecar_coverage(
            8, [self.transfer], {PoolName.INDEXER: 2}, page_size=4
        )
        self.assertEqual(safe, 8)

    def test_trailing_pages_pool_is_never_clamped(self):
        swa_transfer = PoolTransfer(
            name=PoolName.SWA, hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        safe = clamp_prefix_to_sidecar_coverage(8, [swa_transfer], {}, page_size=4)
        self.assertEqual(safe, 8)

    def test_no_pool_transfers_is_a_noop(self):
        safe = clamp_prefix_to_sidecar_coverage(8, None, {}, page_size=4)
        self.assertEqual(safe, 8)


class TestPageTransferSkipsSidecarOnPartialCompletion(CustomTestCase):
    """Sidecar fetch is skipped when the KV batch completes partially."""

    def test_partial_kv_completion_skips_sidecar_fetch(self):
        controller = object.__new__(HybridCacheController)
        controller.page_size = PAGE_SIZE

        class _FakeBackend:
            def __init__(self):
                self.calls = []

            def batch_get_v2(self, transfers, extra_info=None):
                self.calls.append([(t.name, len(t.keys)) for t in transfers])
                return {t.name: [True] * len(t.keys) for t in transfers}

        backend = _FakeBackend()
        controller.storage_backend = backend

        indexer_transfer = PoolTransfer(
            name=PoolName.INDEXER,
            hit_policy=PoolHitPolicy.ALL_PAGES,
            indices_from_pool=PoolName.KV,
        )
        op = PrefetchOperation(
            request_id="req-0",
            host_indices=torch.arange(NUM_PAGES * PAGE_SIZE),
            token_ids=list(range(NUM_PAGES * PAGE_SIZE)),
            pool_transfers=[indexer_transfer],
        )
        op.hash_value = [f"hash-{i}" for i in range(NUM_PAGES)]

        completed_pages = 2

        def fake_kv_transfer(self, operation):
            operation.completed_tokens = completed_pages * PAGE_SIZE

        with mock.patch.object(
            BaseHiCacheController, "_page_transfer", fake_kv_transfer
        ):
            HybridCacheController._page_transfer(controller, op)

        self.assertEqual(backend.calls, [])
        self.assertTrue(op.pool_transfers_done)
        self.assertEqual(op.pool_storage_result.extra_pool_hit_pages, {})


def _make_indexer_operation(completed_pages, extra_pool_hit_prefix):
    indexer_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        hit_policy=PoolHitPolicy.ALL_PAGES,
        indices_from_pool=PoolName.KV,
    )
    operation = PrefetchOperation(
        request_id="req-0",
        host_indices=torch.arange(NUM_PAGES * PAGE_SIZE),
        token_ids=list(range(NUM_PAGES * PAGE_SIZE)),
        pool_transfers=[indexer_transfer],
    )
    operation.completed_tokens = completed_pages * PAGE_SIZE
    operation.pool_storage_result.extra_pool_hit_prefix.update(extra_pool_hit_prefix)
    return operation


class _FakePrefetchCacheController(HybridCacheController):
    # subclassed so isinstance(_, HybridCacheController) gates the clamp path
    def __init__(self):
        self.mem_pool_host = SimpleNamespace(free=lambda idx: None)
        self.prefetch_tokens_occupied = 100

    def terminate_prefetch(self, operation):
        return operation.completed_tokens, [f"hash-{i}" for i in range(NUM_PAGES)]

    def append_host_mem_release(self, host_indices):
        pass


class TestUnifiedRadixCacheCheckPrefetchProgressClamp(CustomTestCase):
    """check_prefetch_progress must clamp the published length by sidecar coverage."""

    def _run(self, extra_pool_hit_prefix):
        req_id = "req-0"
        operation = _make_indexer_operation(
            completed_pages=2, extra_pool_hit_prefix=extra_pool_hit_prefix
        )
        completed_tokens = operation.completed_tokens

        insert_calls = []

        def fake_insert_helper_host(
            last_host_node, fetched_key, written_indices, hash_value
        ):
            insert_calls.append(
                (len(fetched_key), len(written_indices), len(hash_value))
            )
            return SimpleNamespace(prefix_len=0)

        cache = object.__new__(UnifiedRadixCache)
        cache.page_size = PAGE_SIZE
        cache.tp_world_size = 1
        cache.enable_storage_metrics = False
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.cache_controller = _FakePrefetchCacheController()
        cache.can_terminate_prefetch = lambda op: True
        cache.dec_host_lock_ref = lambda node, params: None
        cache._insert_helper_host = fake_insert_helper_host
        cache.ongoing_prefetch = {
            req_id: (
                object(),  # last_host_node
                list(range(NUM_PAGES * PAGE_SIZE)),  # prefetch_key
                torch.arange(NUM_PAGES * PAGE_SIZE),  # host_indices
                operation,
                object(),  # anchor_lock_params
                {},  # comp_xfers
            )
        }

        result = cache.check_prefetch_progress(req_id)
        self.assertTrue(result)
        self.assertEqual(len(insert_calls), 1)
        return insert_calls[0], completed_tokens

    def test_missing_sidecar_clamps_insertion_to_zero(self):
        (fetched_len, written_len, hash_len), completed_tokens = self._run({})
        self.assertEqual(fetched_len, 0)
        self.assertEqual(written_len, 0)
        self.assertEqual(hash_len, 0)
        self.assertNotEqual(fetched_len, completed_tokens)

    def test_full_sidecar_coverage_publishes_full_kv_length(self):
        (fetched_len, written_len, hash_len), completed_tokens = self._run(
            {PoolName.INDEXER: 2}
        )
        self.assertEqual(fetched_len, completed_tokens)
        self.assertEqual(written_len, completed_tokens)
        self.assertEqual(hash_len, completed_tokens // PAGE_SIZE)


class TestHiRadixCacheCheckPrefetchProgressClamp(CustomTestCase):
    """Same clamp invariant as UnifiedRadixCache, on the HiRadixCache DSA path."""

    def _run(self, extra_pool_hit_prefix):
        req_id = "req-0"
        operation = _make_indexer_operation(
            completed_pages=2, extra_pool_hit_prefix=extra_pool_hit_prefix
        )
        completed_tokens = operation.completed_tokens

        insert_calls = []

        def fake_insert_helper_host(
            last_host_node, fetched_key, written_indices, hash_value
        ):
            insert_calls.append(
                (len(fetched_key), len(written_indices), len(hash_value))
            )
            return 0  # matched_length

        cache = object.__new__(HiRadixCache)
        cache.page_size = PAGE_SIZE
        cache.tp_world_size = 1
        cache.attn_cp_group = None
        cache.attn_tp_group = None
        cache.enable_storage_metrics = False
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.cache_controller = _FakePrefetchCacheController()
        cache.can_terminate_prefetch = lambda op: True
        cache._insert_helper_host = fake_insert_helper_host
        cache.ongoing_prefetch = {
            req_id: (
                SimpleNamespace(release_host=lambda: None),  # last_host_node
                list(range(NUM_PAGES * PAGE_SIZE)),  # prefetch_key
                torch.arange(NUM_PAGES * PAGE_SIZE),  # host_indices
                operation,
            )
        }

        result = cache.check_prefetch_progress(req_id)
        self.assertTrue(result)
        self.assertEqual(len(insert_calls), 1)
        return insert_calls[0], completed_tokens

    def test_missing_sidecar_clamps_insertion_to_zero(self):
        (fetched_len, written_len, hash_len), completed_tokens = self._run({})
        self.assertEqual(fetched_len, 0)
        self.assertEqual(written_len, 0)
        self.assertEqual(hash_len, 0)
        self.assertNotEqual(fetched_len, completed_tokens)

    def test_full_sidecar_coverage_publishes_full_kv_length(self):
        (fetched_len, written_len, hash_len), completed_tokens = self._run(
            {PoolName.INDEXER: 2}
        )
        self.assertEqual(fetched_len, completed_tokens)
        self.assertEqual(written_len, completed_tokens)
        self.assertEqual(hash_len, completed_tokens // PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
