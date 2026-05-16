"""Unit tests for common.py — CPU-only tests"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.common import (
    alloc_token_slots,
    available_and_evictable_str,
    evict_from_tree_cache,
    get_last_loc_torch,
    write_cache_indices,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tree_cache(available=100, is_chunk=False):
    tc = MagicMock()
    tc.is_chunk_cache.return_value = is_chunk
    tc.available_and_evictable_str.return_value = "available: 0, evictable: 0"
    alloc = MagicMock()
    alloc.available_size.return_value = available
    tc.token_to_kv_pool_allocator = alloc
    return tc


# ---------------------------------------------------------------------------
# get_last_loc_torch (CPU)
# ---------------------------------------------------------------------------


class TestGetLastLocTorch(CustomTestCase):
    def _pool(self, rows, cols):
        return torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)

    def test_positive_prefix_returns_last_slot(self):
        pool = self._pool(4, 8)
        out = get_last_loc_torch(pool, torch.tensor([0]), torch.tensor([3]))
        self.assertEqual(out[0].item(), pool[0, 2].item())

    def test_zero_prefix_returns_minus_one(self):
        pool = self._pool(4, 8)
        out = get_last_loc_torch(pool, torch.tensor([1]), torch.tensor([0]))
        self.assertEqual(out[0].item(), -1)

    def test_batch_mixed_prefix_lens(self):
        pool = self._pool(4, 8)
        req = torch.tensor([0, 1, 2])
        pre = torch.tensor([0, 4, 2])
        out = get_last_loc_torch(pool, req, pre)
        self.assertEqual(out[0].item(), -1)
        self.assertEqual(out[1].item(), pool[1, 3].item())
        self.assertEqual(out[2].item(), pool[2, 1].item())

    def test_output_shape_matches_input(self):
        pool = self._pool(8, 16)
        req = torch.zeros(5, dtype=torch.int64)
        pre = torch.ones(5, dtype=torch.int64)
        self.assertEqual(get_last_loc_torch(pool, req, pre).shape, pre.shape)

    def test_all_zero_prefix_returns_all_minus_one(self):
        pool = self._pool(3, 8)
        req = torch.tensor([0, 1, 2])
        pre = torch.zeros(3, dtype=torch.int64)
        self.assertTrue((get_last_loc_torch(pool, req, pre) == -1).all())

    def test_single_token_prefix(self):
        pool = self._pool(2, 4)
        out = get_last_loc_torch(pool, torch.tensor([1]), torch.tensor([1]))
        self.assertEqual(out[0].item(), pool[1, 0].item())

    def test_different_req_pool_indices(self):
        pool = self._pool(4, 4)
        req = torch.tensor([0, 3])
        pre = torch.tensor([2, 2])
        out = get_last_loc_torch(pool, req, pre)
        self.assertEqual(out[0].item(), pool[0, 1].item())
        self.assertEqual(out[1].item(), pool[3, 1].item())

    def test_prefix_len_equals_context_len(self):
        pool = self._pool(2, 4)
        out = get_last_loc_torch(pool, torch.tensor([0]), torch.tensor([4]))
        self.assertEqual(out[0].item(), pool[0, 3].item())


# ---------------------------------------------------------------------------
# evict_from_tree_cache
# ---------------------------------------------------------------------------


class TestEvictFromTreeCache(CustomTestCase):
    def test_none_tree_cache_is_noop(self):
        evict_from_tree_cache(None, 10)

    def test_chunk_cache_is_noop(self):
        tc = _make_tree_cache(is_chunk=True)
        evict_from_tree_cache(tc, 10)
        tc.evict.assert_not_called()

    def test_no_eviction_when_space_sufficient(self):
        tc = _make_tree_cache(available=50)
        evict_from_tree_cache(tc, 10)
        tc.evict.assert_not_called()

    def test_evicts_when_space_insufficient(self):
        tc = _make_tree_cache(available=5)
        evict_from_tree_cache(tc, 10)
        tc.evict.assert_called_once()

    def test_evict_called_with_correct_num_tokens(self):
        tc = _make_tree_cache(available=5)
        evict_from_tree_cache(tc, 20)
        evict_params = tc.evict.call_args[0][0]
        self.assertEqual(evict_params.num_tokens, 20)

    def test_no_eviction_when_available_equals_requested(self):
        tc = _make_tree_cache(available=10)
        evict_from_tree_cache(tc, 10)
        tc.evict.assert_not_called()

    def test_evicts_when_one_less_than_needed(self):
        tc = _make_tree_cache(available=9)
        evict_from_tree_cache(tc, 10)
        tc.evict.assert_called_once()

    def test_swa_both_pools_sufficient_no_evict(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

        tc = MagicMock()
        tc.is_chunk_cache.return_value = False
        alloc = MagicMock(spec=SWATokenToKVPoolAllocator)
        alloc.full_available_size.return_value = 50
        alloc.swa_available_size.return_value = 50
        tc.token_to_kv_pool_allocator = alloc
        evict_from_tree_cache(tc, 10)
        tc.evict.assert_not_called()

    def test_swa_full_pool_short_triggers_evict(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

        tc = MagicMock()
        tc.is_chunk_cache.return_value = False
        alloc = MagicMock(spec=SWATokenToKVPoolAllocator)
        alloc.full_available_size.return_value = 5
        alloc.swa_available_size.return_value = 50
        tc.token_to_kv_pool_allocator = alloc
        evict_from_tree_cache(tc, 10)
        tc.evict.assert_called_once()

    def test_swa_swa_pool_short_triggers_evict(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

        tc = MagicMock()
        tc.is_chunk_cache.return_value = False
        alloc = MagicMock(spec=SWATokenToKVPoolAllocator)
        alloc.full_available_size.return_value = 50
        alloc.swa_available_size.return_value = 3
        tc.token_to_kv_pool_allocator = alloc
        evict_from_tree_cache(tc, 10)
        tc.evict.assert_called_once()

    def test_swa_evict_params_both_pools(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

        tc = MagicMock()
        tc.is_chunk_cache.return_value = False
        alloc = MagicMock(spec=SWATokenToKVPoolAllocator)
        alloc.full_available_size.return_value = 3
        alloc.swa_available_size.return_value = 6
        tc.token_to_kv_pool_allocator = alloc
        evict_from_tree_cache(tc, 10)
        tc.evict.assert_called_once()
        params = tc.evict.call_args[0][0]
        self.assertEqual(params.num_tokens, 7)
        self.assertEqual(params.swa_num_tokens, 4)


# ---------------------------------------------------------------------------
# alloc_token_slots
# ---------------------------------------------------------------------------


class TestAllocTokenSlots(CustomTestCase):
    def _make_tc(self, alloc_result):
        tc = MagicMock()
        tc.is_chunk_cache.return_value = True
        tc.available_and_evictable_str.return_value = "available: 0"
        alloc = MagicMock()
        alloc.alloc.return_value = alloc_result
        alloc.backup_state.return_value = ("fp", "rp")
        tc.token_to_kv_pool_allocator = alloc
        return tc

    def test_returns_allocated_tensor(self):
        expected = torch.tensor([1, 2, 3])
        tc = self._make_tc(expected)
        out = alloc_token_slots(tc, 3)
        self.assertTrue(torch.equal(out, expected))

    def test_oom_raises_runtime_error(self):
        tc = self._make_tc(None)
        with self.assertRaises(RuntimeError):
            alloc_token_slots(tc, 10)

    def test_backup_state_true_returns_tuple(self):
        expected = torch.tensor([1, 2, 3])
        tc = self._make_tc(expected)
        out, state = alloc_token_slots(tc, 3, backup_state=True)
        self.assertTrue(torch.equal(out, expected))
        self.assertEqual(state, ("fp", "rp"))

    def test_backup_state_false_returns_tensor_only(self):
        expected = torch.tensor([4, 5])
        tc = self._make_tc(expected)
        out = alloc_token_slots(tc, 2, backup_state=False)
        self.assertTrue(torch.equal(out, expected))

    def test_eviction_called_when_space_insufficient(self):
        tc = MagicMock()
        tc.is_chunk_cache.return_value = False
        alloc = MagicMock()
        alloc.available_size.return_value = 0
        alloc.alloc.return_value = torch.tensor([1])
        tc.token_to_kv_pool_allocator = alloc
        alloc_token_slots(tc, 1)
        tc.evict.assert_called_once()


# ---------------------------------------------------------------------------
# available_and_evictable_str
# ---------------------------------------------------------------------------


class TestAvailableAndEvictableStr(CustomTestCase):
    def test_delegates_to_tree_cache(self):
        tc = MagicMock()
        tc.available_and_evictable_str.return_value = "available: 10, evictable: 5"
        result = available_and_evictable_str(tc)
        tc.available_and_evictable_str.assert_called_once()
        self.assertEqual(result, "available: 10, evictable: 5")

    def test_returns_string(self):
        tc = MagicMock()
        tc.available_and_evictable_str.return_value = "x"
        self.assertIsInstance(available_and_evictable_str(tc), str)


# ---------------------------------------------------------------------------
# write_cache_indices — non-Triton path
# ---------------------------------------------------------------------------


class TestWriteCacheIndicesNonTriton(CustomTestCase):
    def _run(self, n_reqs=2, seq_len=4, prefix_len=2):
        extend_len = seq_len - prefix_len
        out_cache_loc = torch.arange(n_reqs * extend_len, dtype=torch.int32)
        req_pool_idx_cpu = torch.arange(n_reqs, dtype=torch.int64)
        prefix_lens_cpu = torch.full((n_reqs,), prefix_len, dtype=torch.int64)
        seq_lens_cpu = torch.full((n_reqs,), seq_len, dtype=torch.int64)
        extend_lens_cpu = torch.full((n_reqs,), extend_len, dtype=torch.int64)
        prefix_tensors = [
            torch.zeros(prefix_len, dtype=torch.int64) for _ in range(n_reqs)
        ]
        pool = MagicMock()

        with patch(
            "sglang.srt.mem_cache.common.support_triton", return_value=False
        ), patch("sglang.srt.mem_cache.common.get_global_server_args") as mock_args:
            mock_args.return_value.attention_backend = "torch_native"
            write_cache_indices(
                out_cache_loc,
                req_pool_idx_cpu,
                req_pool_idx_cpu,
                prefix_lens_cpu,
                prefix_lens_cpu,
                seq_lens_cpu,
                seq_lens_cpu,
                extend_lens_cpu,
                extend_lens_cpu,
                prefix_tensors,
                pool,
            )
        return pool

    def test_write_called_for_each_request(self):
        pool = self._run(n_reqs=3)
        self.assertEqual(pool.write.call_count, 6)

    def test_prefix_slice_written(self):
        pool = self._run(n_reqs=1, seq_len=4, prefix_len=2)
        first_call_key = pool.write.call_args_list[0][0][0]
        self.assertEqual(first_call_key, (0, slice(0, 2)))

    def test_extend_slice_written(self):
        pool = self._run(n_reqs=1, seq_len=4, prefix_len=2)
        second_call_key = pool.write.call_args_list[1][0][0]
        self.assertEqual(second_call_key, (0, slice(2, 4)))


# ---------------------------------------------------------------------------
# High-Level Operations
# ---------------------------------------------------------------------------


class TestCommonHighLevelFunctions(CustomTestCase):
    @patch("sglang.srt.mem_cache.common.get_global_server_args")
    def test_alloc_paged_token_slots_extend(self, mock_args):
        from sglang.srt.mem_cache.common import alloc_paged_token_slots_extend

        tc = MagicMock()
        tc.token_to_kv_pool_allocator.alloc_extend.return_value = torch.tensor([1])
        tc.token_to_kv_pool_allocator.page_size = 4

        out, state = alloc_paged_token_slots_extend(
            tc,
            torch.tensor([1]),
            torch.tensor([1]),
            torch.tensor([5]),
            torch.tensor([5]),
            torch.tensor([0]),
            4,
            backup_state=True,
        )
        self.assertIsNotNone(out)

        tc.token_to_kv_pool_allocator.alloc_extend.return_value = None
        with self.assertRaises(RuntimeError):
            alloc_paged_token_slots_extend(
                tc,
                torch.tensor([1]),
                torch.tensor([1]),
                torch.tensor([5]),
                torch.tensor([5]),
                torch.tensor([0]),
                4,
            )

    @patch("sglang.srt.mem_cache.common.get_global_server_args")
    def test_alloc_req_slots(self, mock_args):
        from sglang.srt.mem_cache.common import alloc_req_slots

        pool = MagicMock()
        pool.__class__ = HybridReqToTokenPool
        pool.mamba_pool.available_size.return_value = 0
        pool.alloc.return_value = [1]

        tc = MagicMock()
        tc.supports_mamba.return_value = True

        out = alloc_req_slots(pool, [MagicMock()], tc)
        self.assertEqual(out, [1])
        tc.evict.assert_called_once()

        pool.alloc.return_value = None
        with self.assertRaises(RuntimeError):
            alloc_req_slots(pool, [MagicMock()], tc)

    @patch("sglang.srt.mem_cache.common.get_global_server_args")
    def test_alloc_for_extend(self, mock_args):
        from sglang.srt.mem_cache.common import alloc_for_extend

        batch = MagicMock()
        batch.reqs = [MagicMock()]
        batch.reqs[0].prefix_indices = torch.tensor([1])
        batch.prefix_lens = [1]
        batch.extend_lens = [4]
        batch.device = "cpu"
        batch.seq_lens = torch.tensor([5])
        batch.seq_lens_cpu = torch.tensor([5])
        batch.extend_num_tokens = 4
        batch.tree_cache.page_size = 1

        with patch(
            "sglang.srt.mem_cache.common.alloc_token_slots",
            return_value=torch.tensor([1]),
        ), patch(
            "sglang.srt.mem_cache.common.alloc_req_slots", return_value=[1]
        ), patch(
            "sglang.srt.mem_cache.common.write_cache_indices"
        ):
            out, dev, req = alloc_for_extend(batch)
            self.assertIsNotNone(out)

        batch.tree_cache.page_size = 4
        with patch(
            "sglang.srt.mem_cache.common.alloc_paged_token_slots_extend",
            return_value=torch.tensor([1]),
        ), patch(
            "sglang.srt.mem_cache.common.alloc_req_slots", return_value=[1]
        ), patch(
            "sglang.srt.mem_cache.common.write_cache_indices"
        ):
            out, dev, req = alloc_for_extend(batch)
            self.assertIsNotNone(out)

    @patch("sglang.srt.mem_cache.common.get_global_server_args")
    def test_alloc_paged_token_slots_decode(self, mock_args):
        from sglang.srt.mem_cache.common import alloc_paged_token_slots_decode

        tc = MagicMock()
        tc.token_to_kv_pool_allocator.alloc_decode.return_value = torch.tensor([1])
        tc.token_to_kv_pool_allocator.page_size = 4

        out = alloc_paged_token_slots_decode(
            tc, torch.tensor([5]), torch.tensor([5]), torch.tensor([4])
        )
        self.assertIsNotNone(out)

        tc.token_to_kv_pool_allocator.alloc_decode.return_value = None
        with self.assertRaises(RuntimeError):
            alloc_paged_token_slots_decode(
                tc, torch.tensor([5]), torch.tensor([5]), torch.tensor([4])
            )

    @patch("sglang.srt.mem_cache.common.get_global_server_args")
    def test_alloc_for_decode(self, mock_args):
        from sglang.srt.mem_cache.common import alloc_for_decode

        batch = MagicMock()
        batch.seq_lens = torch.tensor([5])
        batch.seq_lens_cpu = torch.tensor([5])
        batch.tree_cache.page_size = 1
        batch.model_config.is_encoder_decoder = True
        batch.encoder_lens = torch.tensor([2])
        batch.req_pool_indices = torch.tensor([0])

        with patch(
            "sglang.srt.mem_cache.common.alloc_token_slots",
            return_value=torch.tensor([1]),
        ):
            out = alloc_for_decode(batch, 1)
            self.assertIsNotNone(out)

        batch.tree_cache.page_size = 4
        batch.model_config.is_encoder_decoder = False
        with patch(
            "sglang.srt.mem_cache.common.alloc_paged_token_slots_decode",
            return_value=torch.tensor([1]),
        ):
            out = alloc_for_decode(batch, 1)
            self.assertIsNotNone(out)

    @patch("sglang.srt.mem_cache.common.get_global_server_args")
    def test_release_kv_cache(self, mock_args):
        from sglang.srt.mem_cache.common import release_kv_cache

        # Branch 1: early alloc mamba state release (mamba_pool_idx is not None)
        req = MagicMock()
        tc = MagicMock()
        req.req_pool_idx = None
        tc.supports_mamba.return_value = True
        req.mamba_pool_idx = torch.tensor([1])
        release_kv_cache(req, tc)
        self.assertIsNone(req.mamba_pool_idx)

        # Branch 2: early alloc mamba state release (mamba_pool_idx is None)
        req = MagicMock()
        tc = MagicMock()
        req.req_pool_idx = None
        tc.supports_mamba.return_value = True
        req.mamba_pool_idx = None
        release_kv_cache(req, tc)

        # Branch 3: cache_finished_req sets req_pool_idx to None
        req = MagicMock()
        tc = MagicMock()
        req.req_pool_idx = 1

        def side_effect(r, is_insert):
            r.req_pool_idx = None

        tc.cache_finished_req.side_effect = side_effect
        release_kv_cache(req, tc)

        # Branch 4: normal release with overallocation & HybridReqToTokenPool
        req = MagicMock()
        tc = MagicMock()
        req.req_pool_idx = 1
        req.pop_overallocated_kv_cache.return_value = (2, 4)
        mock_args.return_value.page_size = 2
        mock_args.return_value.speculative_algorithm = "EAGLE"

        tc.req_to_token_pool = MagicMock()
        tc.req_to_token_pool.__class__ = HybridReqToTokenPool
        tc.req_to_token_pool.req_to_token = torch.arange(20).reshape(2, 10)

        tc.supports_mamba.return_value = False
        req.mamba_pool_idx = 1

        release_kv_cache(req, tc)

        tc.req_to_token_pool.free_mamba_cache.assert_called_with(req)
        tc.req_to_token_pool.free.assert_called_with(req)

        # Branch 5: normal release without overallocation
        req = MagicMock()
        tc = MagicMock()
        req.req_pool_idx = 1
        req.pop_overallocated_kv_cache.return_value = (2, 2)
        mock_args.return_value.page_size = 1
        mock_args.return_value.speculative_algorithm = None

        tc.req_to_token_pool = MagicMock()

        release_kv_cache(req, tc)
        tc.req_to_token_pool.free.assert_called_with(req)


if __name__ == "__main__":
    unittest.main()
