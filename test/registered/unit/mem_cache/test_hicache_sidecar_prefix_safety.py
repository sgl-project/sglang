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
    hybrid_pools_fully_covered,
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


class TestHybridPoolsFullyCovered(CustomTestCase):
    """All-or-nothing check for sidecars that can't be truncated page by page
    (TRAILING_PAGES, or any pool not KV-derived, e.g. SWA/Mamba)."""

    def setUp(self):
        self.hash_value = [f"hash-{i}" for i in range(NUM_PAGES)]
        self.transfer = PoolTransfer(
            name=PoolName.SWA,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
            keys=["hash-6", "hash-7"],
        )

    def test_full_kv_and_full_sidecar_succeeds(self):
        covered = hybrid_pools_fully_covered(
            NUM_PAGES * PAGE_SIZE,
            self.hash_value,
            [self.transfer],
            {PoolName.SWA: 2},
            PAGE_SIZE,
        )
        self.assertTrue(covered)

    def test_partial_sidecar_fails(self):
        covered = hybrid_pools_fully_covered(
            NUM_PAGES * PAGE_SIZE,
            self.hash_value,
            [self.transfer],
            {PoolName.SWA: 1},
            PAGE_SIZE,
        )
        self.assertFalse(covered)

    def test_partial_kv_completion_fails_even_with_full_sidecar(self):
        covered = hybrid_pools_fully_covered(
            (NUM_PAGES - 1) * PAGE_SIZE,
            self.hash_value,
            [self.transfer],
            {PoolName.SWA: 2},
            PAGE_SIZE,
        )
        self.assertFalse(covered)


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def batch_get_v2(self, transfers, extra_info=None):
        self.calls.append([(t.name, len(t.keys)) for t in transfers])
        return {t.name: [True] * len(t.keys) for t in transfers}


def _run_page_transfer(completed_pages, terminated=False):
    controller = object.__new__(HybridCacheController)
    controller.page_size = PAGE_SIZE
    backend = _FakeBackend()
    controller.storage_backend = backend

    indexer_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        hit_policy=PoolHitPolicy.ALL_PAGES,
        indices_from_pool=PoolName.KV,
    )
    op = PrefetchOperation(
        request_id="req-0",
        token_ids=list(range(NUM_PAGES * PAGE_SIZE)),
        pool_transfers=[indexer_transfer],
    )
    op.host_indices = torch.arange(NUM_PAGES * PAGE_SIZE)
    op.hash_value = [f"hash-{i}" for i in range(NUM_PAGES)]
    if terminated:
        op.mark_terminate()

    def fake_kv_transfer(self, operation):
        operation.completed_tokens = completed_pages * PAGE_SIZE

    with mock.patch.object(BaseHiCacheController, "_page_transfer", fake_kv_transfer):
        HybridCacheController._page_transfer(controller, op)
    return backend, op


class TestPageTransferSkipsSidecarOnPartialCompletion(CustomTestCase):
    """Sidecar fetch is skipped when the KV batch completes partially."""

    def test_partial_kv_completion_skips_sidecar_fetch(self):
        backend, op = _run_page_transfer(completed_pages=2)
        self.assertEqual(backend.calls, [])
        self.assertTrue(op.pool_transfers_done)
        self.assertEqual(op.pool_storage_result.extra_pool_hit_pages, {})

    def test_full_kv_completion_fetches_sidecar(self):
        backend, op = _run_page_transfer(completed_pages=NUM_PAGES)
        self.assertEqual(len(backend.calls), 1)


class TestPageTransferSkipsSidecarOnTermination(CustomTestCase):
    """A terminated operation must not still trigger sidecar IO, even if KV
    happened to complete in full before the termination was observed."""

    def test_terminated_operation_skips_sidecar_fetch(self):
        backend, op = _run_page_transfer(completed_pages=NUM_PAGES, terminated=True)
        self.assertEqual(backend.calls, [])
        self.assertEqual(op.pool_storage_result.extra_pool_hit_pages, {})


def _make_indexer_operation(completed_pages, extra_pool_hit_prefix):
    indexer_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        hit_policy=PoolHitPolicy.ALL_PAGES,
        indices_from_pool=PoolName.KV,
    )
    operation = PrefetchOperation(
        request_id="req-0",
        token_ids=list(range(NUM_PAGES * PAGE_SIZE)),
        pool_transfers=[indexer_transfer],
    )
    operation.host_indices = torch.arange(NUM_PAGES * PAGE_SIZE)
    operation.completed_tokens = completed_pages * PAGE_SIZE
    operation.pool_storage_result.extra_pool_hit_prefix.update(extra_pool_hit_prefix)
    return operation


class _FakePrefetchCacheController(HybridCacheController):
    # subclassed so isinstance(_, HybridCacheController) gates the clamp path
    def __init__(self):
        self.mem_pool_host = SimpleNamespace(free=lambda idx: None)
        self.prefetch_tokens_occupied = 100
        self.release_calls = []

    def terminate_prefetch(self, operation):
        return operation.completed_tokens, [f"hash-{i}" for i in range(NUM_PAGES)]

    def append_host_mem_release(self, host_indices=None, extra_pools=None):
        self.release_calls.append((host_indices, extra_pools))


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


class TestUnifiedRadixCacheAllOrNothingDiscard(CustomTestCase):
    """A TRAILING_PAGES sidecar (e.g. SWA) can't be safely clamped page by
    page, so a shortfall must discard the whole prefetch result instead of
    silently publishing a truncated-but-unverified prefix."""

    def _run(self, completed_pages, swa_hit_pages):
        req_id = "req-0"
        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
            keys=["hash-6", "hash-7"],
        )
        operation = PrefetchOperation(
            request_id=req_id,
            token_ids=list(range(NUM_PAGES * PAGE_SIZE)),
            pool_transfers=[swa_transfer],
        )
        operation.host_indices = torch.arange(NUM_PAGES * PAGE_SIZE)
        operation.completed_tokens = completed_pages * PAGE_SIZE
        operation.pool_storage_result.extra_pool_hit_pages.update(
            {PoolName.SWA: swa_hit_pages}
        )

        insert_calls = []

        def fake_insert_helper_host(*args, **kwargs):
            insert_calls.append(args)
            return SimpleNamespace(prefix_len=0)

        cache = object.__new__(UnifiedRadixCache)
        cache.page_size = PAGE_SIZE
        cache.tp_world_size = 1
        cache.enable_storage_metrics = False
        cache.prefetch_loaded_tokens_by_reqid = {}
        controller = _FakePrefetchCacheController()
        cache.cache_controller = controller
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
        return insert_calls, controller, cache

    def test_partial_sidecar_discards_whole_prefetch(self):
        insert_calls, controller, cache = self._run(
            completed_pages=NUM_PAGES, swa_hit_pages=1
        )
        self.assertEqual(insert_calls, [])
        self.assertEqual(len(controller.release_calls), 1)
        self.assertNotIn("req-0", cache.ongoing_prefetch)
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid["req-0"], 0)

    def test_full_sidecar_coverage_still_inserts(self):
        insert_calls, controller, cache = self._run(
            completed_pages=NUM_PAGES, swa_hit_pages=2
        )
        self.assertEqual(len(insert_calls), 1)
        # no discard-specific release (extra_pools set) happened, only the
        # ordinary empty tail-release
        self.assertTrue(all(pools is None for _, pools in controller.release_calls))


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
