import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_recompute import (
    SWARecomputeBatchState,
    _recompute_commit_len,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSWARecompute(CustomTestCase):
    def test_commit_len_is_regular_swa_tail(self):
        cases = [
            (5376, 128, 256, 256),
            (96, 128, 16, 96),
            (0, 128, 256, 0),
        ]
        for recompute_len, window_size, page_size, expected in cases:
            with self.subTest(
                recompute_len=recompute_len,
                window_size=window_size,
                page_size=page_size,
            ):
                self.assertEqual(
                    _recompute_commit_len(recompute_len, window_size, page_size),
                    expected,
                )

    def test_cache_accounting_consumes_storage_suffix_first(self):
        state = SWARecomputeBatchState(
            extra_compute_lens=[50], logical_prefix_lens=[100]
        )
        self.assertEqual(
            state.split_cached_tokens_by_source(
                0,
                pre_len=50,
                host_hit_length=100,
                storage_hit_length=80,
            ),
            (0, 20, 30),
        )

    def test_finish_marks_shared_update_inactive(self):
        for finish in ("commit", "abort"):
            with self.subTest(finish=finish):
                state = SWARecomputeBatchState(
                    extra_compute_lens=[16], logical_prefix_lens=[32]
                )
                shared_reference = state

                getattr(state, finish)(None)

                self.assertFalse(shared_reference.is_pending)

    def test_commit_updates_req_kv_swa_eviction_boundary(self):
        allocator = mock.Mock(spec=SWATokenToKVPoolAllocator)
        allocator.page_size = 4
        req = SimpleNamespace(
            kv=SimpleNamespace(swa_evicted_seqlen=20),
            prefix_indices=torch.arange(16),
            extra_compute_prefix_len=8,
        )
        state = SWARecomputeBatchState(
            extra_compute_lens=[8],
            logical_prefix_lens=[16],
            txns=[
                SimpleNamespace(
                    req=req,
                    recompute_len=8,
                    full_indices=torch.arange(8),
                    fresh_swa_indices=torch.arange(8),
                )
            ],
        )
        tree_cache = SimpleNamespace(
            sliding_window_size=3,
            complete_swa_recompute_lock=mock.Mock(),
        )
        batch = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree_cache,
        )

        state.commit(batch)

        self.assertEqual(req.kv.swa_evicted_seqlen, 12)
        self.assertEqual(req.extra_compute_prefix_len, 0)
        tree_cache.complete_swa_recompute_lock.assert_called_once_with(req, 4)
        allocator.free_fresh_swa.assert_called_once()
        allocator.commit_fresh_swa_for_recompute_window.assert_called_once()
        self.assertFalse(state.is_pending)

    def test_forward_metadata_disables_prefill_cuda_graph(self):
        state = SWARecomputeBatchState(
            extra_compute_lens=[16], logical_prefix_lens=[32]
        )
        forward_batch = SimpleNamespace(allow_prefill_cuda_graph=True)

        state.apply_forward_metadata(forward_batch)

        self.assertFalse(forward_batch.allow_prefill_cuda_graph)
        self.assertEqual(forward_batch.swa_recompute_boundaries, [32])


if __name__ == "__main__":
    unittest.main()
