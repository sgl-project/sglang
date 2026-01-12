# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.beam_search_type import BeamSearchList
from sglang.srt.managers.schedule_batch_beam_search_mixin import (
    ReqBeamSearchMixin,
    ScheduleBatchBeamSearchMixin,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, suite="per-commit-1-gpu")
register_amd_ci(est_time=5, suite="per-commit-1-gpu-amd")


class TestPrepareForBeamSearchDecode(unittest.TestCase):
    """Test prepare_for_beam_search_decode method."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    @patch("sglang.srt.managers.schedule_batch_beam_search_mixin.alloc_for_decode")
    def test_prepare_for_beam_search_decode_different_beam_widths(
        self, mock_alloc_for_decode
    ):
        """Test preparation when requests have different beam_widths."""
        batch = Mock()
        batch.device = self.device

        req1 = Mock()
        req1.beam_width = 2
        req1.beam_list = Mock()
        req1.beam_list.last_tokens = torch.tensor([100, 200], device=self.device)

        req2 = Mock()
        req2.beam_width = 3
        req2.beam_list = Mock()
        req2.beam_list.last_tokens = torch.tensor([300, 400, 500], device=self.device)

        batch.reqs = [req1, req2]
        batch.seq_lens = torch.tensor([5, 5, 6, 6, 6], device=self.device)
        batch.seq_lens_cpu = torch.tensor([5, 5, 6, 6, 6], device=torch.device("cpu"))
        batch.orig_seq_lens = torch.tensor([5, 5, 6, 6, 6], device=self.device)

        mock_alloc_for_decode.return_value = torch.tensor(
            [10, 11, 12, 13, 14], device=self.device
        )

        ScheduleBatchBeamSearchMixin.prepare_for_beam_search_decode(batch)

        self.assertEqual(batch.forward_mode, ForwardMode.DECODE)
        self.assertTrue(
            torch.equal(
                batch.input_ids,
                torch.tensor([100, 200, 300, 400, 500], dtype=torch.int32),
            )
        )
        self.assertIsNone(batch.output_ids)
        batch._prepare_for_new_beam_search.assert_called_once()
        mock_alloc_for_decode.assert_called_once_with(batch, token_per_req=1)
        self.assertTrue(
            torch.equal(
                batch.out_cache_loc,
                torch.tensor([10, 11, 12, 13, 14], device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.seq_lens, torch.tensor([6, 6, 7, 7, 7], device=self.device)
            )
        )
        self.assertTrue(
            torch.equal(
                batch.seq_lens_cpu,
                torch.tensor([6, 6, 7, 7, 7], device=torch.device("cpu")),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.orig_seq_lens, torch.tensor([6, 6, 7, 7, 7], device=self.device)
            )
        )


class TestFilterBeamSearchBatch(unittest.TestCase):
    """Test filter_beam_search_batch method."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_filter_beam_search_batch_exclude_finished(self):
        """Test filtering by excluding finished requests."""
        batch = Mock()
        batch.device = self.device

        req1 = Mock()
        req1.beam_width = 2
        req1.beam_list = Mock()
        req1.beam_list.batch_slot_start_idx = 0
        req1.finished = Mock(return_value=False)

        req2 = Mock()
        req2.beam_width = 2
        req2.beam_list = Mock()
        req2.beam_list.batch_slot_start_idx = 2
        req2.finished = Mock(return_value=True)

        batch.reqs = [req1, req2]
        batch.req_pool_indices = torch.tensor([0, 1, 2, 3], device=self.device)
        batch.seq_lens = torch.tensor([7, 7, 8, 8], device=self.device)
        batch.orig_seq_lens = torch.tensor([7, 7, 8, 8], device=self.device)
        batch.has_stream = False
        batch.has_grammar = False

        # Mock req_to_token_pool to verify KV cache indexing
        req_to_token_pool = Mock()
        req_to_token_pool.req_to_token = torch.arange(40, device=self.device).reshape(
            4, 10
        )
        batch.req_to_token_pool = req_to_token_pool

        ScheduleBatchBeamSearchMixin.filter_beam_search_batch(batch)

        self.assertEqual(len(batch.reqs), 1)
        self.assertEqual(batch.reqs[0], req1)
        self.assertEqual(req1.beam_list.batch_slot_start_idx, 0)
        self.assertTrue(
            torch.equal(
                batch.req_pool_indices, torch.tensor([0, 1], device=self.device)
            )
        )
        self.assertTrue(
            torch.equal(batch.seq_lens, torch.tensor([7, 7], device=self.device))
        )
        self.assertTrue(
            torch.equal(
                batch.seq_lens_cpu,
                torch.tensor([7, 7], device=torch.device("cpu")),
            )
        )
        self.assertEqual(batch.seq_lens_sum, 14)
        self.assertTrue(
            torch.equal(batch.orig_seq_lens, torch.tensor([7, 7], device=self.device))
        )

        # Verify req_to_token is correctly accessible after filtering
        # req1 uses pool indices [0, 1], req2 (filtered out) used [2, 3]
        expected_req_to_token = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # pool index 0
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # pool index 1
            ],
            device=self.device,
        )
        actual_req_to_token = req_to_token_pool.req_to_token[batch.req_pool_indices]
        self.assertTrue(torch.equal(actual_req_to_token, expected_req_to_token))

    def test_filter_beam_search_batch_empty_result(self):
        """Test filtering when all requests are excluded."""
        batch = Mock()
        batch.device = self.device

        req1 = Mock()
        req1.finished = Mock(return_value=True)

        batch.reqs = [req1]

        ScheduleBatchBeamSearchMixin.filter_beam_search_batch(batch)

        self.assertEqual(len(batch.reqs), 0)

    def test_filter_beam_search_batch_no_change(self):
        """Test filtering when all requests are kept."""
        batch = Mock()
        batch.device = self.device

        req1 = Mock()
        req1.beam_width = 2
        req1.finished = Mock(return_value=False)

        req2 = Mock()
        req2.beam_width = 2
        req2.finished = Mock(return_value=False)

        batch.reqs = [req1, req2]
        original_reqs = batch.reqs.copy()

        ScheduleBatchBeamSearchMixin.filter_beam_search_batch(batch)

        self.assertEqual(batch.reqs, original_reqs)

    def test_filter_beam_search_batch_mixed_beam_widths(self):
        """Test filtering with mixed beam widths, prefill requests, and explicit keep_indices."""
        batch = Mock()
        batch.device = self.device

        req1 = Mock()
        req1.beam_width = 2
        req1.beam_list = Mock()
        req1.beam_list.batch_slot_start_idx = 0

        req2 = Mock()
        req2.beam_width = 1
        req2.beam_list = Mock()
        req2.beam_list.batch_slot_start_idx = -1

        req3 = Mock()
        req3.beam_width = 3
        req3.beam_list = Mock()
        req3.beam_list.batch_slot_start_idx = 3

        batch.reqs = [req1, req2, req3]
        batch.req_pool_indices = torch.tensor([0, 1, 2, 3, 4, 5], device=self.device)
        batch.seq_lens = torch.tensor([7, 7, 8, 9, 9, 9], device=self.device)
        batch.orig_seq_lens = torch.tensor([7, 7, 8, 9, 9, 9], device=self.device)
        batch.has_stream = False
        batch.has_grammar = False

        # Mock req_to_token_pool to verify KV cache indexing
        req_to_token_pool = Mock()
        req_to_token_pool.req_to_token = torch.arange(60, device=self.device).reshape(
            6, 10
        )
        batch.req_to_token_pool = req_to_token_pool

        ScheduleBatchBeamSearchMixin.filter_beam_search_batch(
            batch, keep_indices=[0, 2]
        )

        self.assertEqual(len(batch.reqs), 2)
        self.assertEqual(batch.reqs[0], req1)
        self.assertEqual(batch.reqs[1], req3)
        self.assertTrue(
            torch.equal(
                batch.req_pool_indices,
                torch.tensor([0, 1, 3, 4, 5], device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.seq_lens, torch.tensor([7, 7, 9, 9, 9], device=self.device)
            )
        )
        self.assertTrue(
            torch.equal(
                batch.seq_lens_cpu,
                torch.tensor([7, 7, 9, 9, 9], device=torch.device("cpu")),
            )
        )
        self.assertEqual(batch.seq_lens_sum, 41)
        self.assertTrue(
            torch.equal(
                batch.orig_seq_lens, torch.tensor([7, 7, 9, 9, 9], device=self.device)
            )
        )
        self.assertEqual(req1.beam_list.batch_slot_start_idx, 0)
        self.assertEqual(req3.beam_list.batch_slot_start_idx, 2)

        # Verify req_to_token is correctly accessible after filtering
        # The req_pool_indices should still correctly map to the original req_to_token rows
        # req1 uses pool indices [0, 1], req3 uses pool indices [3, 4, 5]
        expected_req_to_token = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # pool index 0
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # pool index 1
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],  # pool index 3
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],  # pool index 4
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],  # pool index 5
            ],
            device=self.device,
        )
        actual_req_to_token = req_to_token_pool.req_to_token[batch.req_pool_indices]
        self.assertTrue(torch.equal(actual_req_to_token, expected_req_to_token))


class TestPrepareForNewBeamSearch(unittest.TestCase):
    """Test _prepare_for_new_beam_search method."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_prepare_for_new_beam_search_single_request(self):
        """Test preparing new beam search for a single request."""
        batch = Mock()
        batch.device = self.device

        req = Mock()
        req.beam_width = 3
        req.beam_list = Mock()
        req.beam_list.batch_slot_start_idx = -1

        batch.reqs = [req]
        batch.req_pool_indices = torch.tensor([0], device=self.device)
        batch.seq_lens = torch.tensor([5], device=self.device)
        batch.orig_seq_lens = torch.tensor([5], device=self.device)

        req_to_token_pool = Mock()
        req_to_token_pool.req_to_token = torch.arange(50, device=self.device).reshape(
            5, 10
        )
        req_to_token_pool.alloc = Mock(return_value=[1, 2, 3])
        batch.req_to_token_pool = req_to_token_pool

        ScheduleBatchBeamSearchMixin._prepare_for_new_beam_search(batch)

        req_to_token_pool.alloc.assert_called_once_with(3)
        self.assertTrue(
            torch.equal(
                batch.req_pool_indices, torch.tensor([1, 2, 3], device=self.device)
            )
        )
        self.assertTrue(
            torch.equal(batch.seq_lens, torch.tensor([5, 5, 5], device=self.device))
        )
        self.assertTrue(
            torch.equal(
                batch.orig_seq_lens, torch.tensor([5, 5, 5], device=self.device)
            )
        )
        self.assertEqual(req.beam_list.batch_slot_start_idx, 0)

        # Verify req_to_token copying: the original KV cache at pool index 0
        # should be copied to the new beam indices [1, 2, 3]
        original_kvcache = req_to_token_pool.req_to_token[0, :5]  # [0, 1, 2, 3, 4]
        for beam_idx in [1, 2, 3]:
            copied_kvcache = req_to_token_pool.req_to_token[beam_idx, :5]
            self.assertTrue(
                torch.equal(copied_kvcache, original_kvcache),
                f"KV cache at beam index {beam_idx} should match original",
            )

    def test_prepare_for_new_beam_search_multiple_requests(self):
        """Test preparing new beam search for multiple requests."""
        batch = Mock()
        batch.device = self.device

        req1 = Mock()
        req1.beam_width = 2
        req1.beam_list = Mock()
        req1.beam_list.batch_slot_start_idx = 0

        req2 = Mock()
        req2.beam_width = 3
        req2.beam_list = Mock()
        req2.beam_list.batch_slot_start_idx = -1

        batch.reqs = [req1, req2]
        batch.req_pool_indices = torch.tensor([0, 1, 2], device=self.device)
        batch.seq_lens = torch.tensor([7, 7, 8], device=self.device)
        batch.orig_seq_lens = torch.tensor([7, 7, 8], device=self.device)

        req_to_token_pool = Mock()
        req_to_token_pool.req_to_token = torch.arange(60, device=self.device).reshape(
            6, 10
        )
        req_to_token_pool.alloc = Mock(return_value=[3, 4, 5])
        batch.req_to_token_pool = req_to_token_pool

        ScheduleBatchBeamSearchMixin._prepare_for_new_beam_search(batch)

        req_to_token_pool.alloc.assert_called_once_with(3)
        self.assertTrue(
            torch.equal(
                batch.req_pool_indices,
                torch.tensor([0, 1, 3, 4, 5], device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.seq_lens, torch.tensor([7, 7, 8, 8, 8], device=self.device)
            )
        )
        self.assertTrue(
            torch.equal(
                batch.orig_seq_lens, torch.tensor([7, 7, 8, 8, 8], device=self.device)
            )
        )
        self.assertEqual(req2.beam_list.batch_slot_start_idx, 2)

        # Verify req_to_token copying: req2's original KV cache at pool index 2
        # should be copied to the new beam indices [3, 4, 5]
        # Since the original slot 2 is not overwritten, we can read it directly
        original_kvcache_req2 = req_to_token_pool.req_to_token[2, :8]  # seq_len=8
        for beam_idx in [3, 4, 5]:
            copied_kvcache = req_to_token_pool.req_to_token[beam_idx, :8]
            self.assertTrue(
                torch.equal(copied_kvcache, original_kvcache_req2),
                f"KV cache at beam index {beam_idx} should match req2's original",
            )

    def test_prepare_for_new_beam_search_no_new_requests(self):
        """Test when there are no new beam search requests."""
        batch = Mock()
        batch.device = self.device

        req = Mock()
        req.beam_width = 2
        req.beam_list = Mock()
        req.beam_list.batch_slot_start_idx = 0

        batch.reqs = [req]
        batch.req_pool_indices = torch.tensor([0, 1], device=self.device)

        req_to_token_pool = Mock()
        req_to_token_pool.alloc = Mock()
        batch.req_to_token_pool = req_to_token_pool

        original_pool_indices = batch.req_pool_indices.clone()

        ScheduleBatchBeamSearchMixin._prepare_for_new_beam_search(batch)

        req_to_token_pool.alloc.assert_not_called()
        self.assertTrue(torch.equal(batch.req_pool_indices, original_pool_indices))

    def test_prepare_for_new_beam_search_out_of_memory(self):
        """Test handling out of memory error."""
        batch = Mock()
        batch.device = self.device

        req = Mock()
        req.beam_width = 3
        req.beam_list = Mock()
        req.beam_list.batch_slot_start_idx = -1

        batch.reqs = [req]
        batch.req_pool_indices = torch.tensor([0], device=self.device)

        req_to_token_pool = Mock()
        req_to_token_pool.alloc = Mock(return_value=None)
        batch.req_to_token_pool = req_to_token_pool

        with self.assertRaises(RuntimeError) as context:
            ScheduleBatchBeamSearchMixin._prepare_for_new_beam_search(batch)

        self.assertIn("Out of memory", str(context.exception))


class TestInitBeamSearchAttributes(unittest.TestCase):
    """Test _init_beam_search_attributes method."""

    def test_init_beam_search_attributes_enabled(self):
        """Test initialization when beam search is enabled."""
        req = Mock()

        sampling_params = Mock()
        sampling_params.use_beam_search = True
        sampling_params.n = 3

        ReqBeamSearchMixin._init_beam_search_attributes(req, sampling_params)

        self.assertTrue(req.is_beam_search)
        self.assertEqual(req.beam_width, 3)
        self.assertEqual(req.beam_candidates, 6)
        self.assertIsNotNone(req.beam_list)
        self.assertIsInstance(req.beam_list, BeamSearchList)
        self.assertIsNone(req._stop_token_ids_cache)

    def test_init_beam_search_attributes_disabled_n_equals_1(self):
        """Test initialization when n=1 (beam search disabled)."""
        req = Mock()

        sampling_params = Mock()
        sampling_params.use_beam_search = True
        sampling_params.n = 1

        ReqBeamSearchMixin._init_beam_search_attributes(req, sampling_params)

        self.assertFalse(req.is_beam_search)
        self.assertEqual(req.beam_width, 0)
        self.assertEqual(req.beam_candidates, 0)

    def test_init_beam_search_attributes_disabled_flag_false(self):
        """Test initialization when use_beam_search=False."""
        req = Mock()

        sampling_params = Mock()
        sampling_params.use_beam_search = False
        sampling_params.n = 3

        ReqBeamSearchMixin._init_beam_search_attributes(req, sampling_params)

        self.assertFalse(req.is_beam_search)
        self.assertEqual(req.beam_width, 0)
        self.assertEqual(req.beam_candidates, 0)


class TestStopTokenIds(unittest.TestCase):
    """Test stop_token_ids property."""

    def test_stop_token_ids_basic(self):
        """Test basic stop token IDs collection."""
        req = Mock()
        req._stop_token_ids_cache = None

        req.sampling_params = Mock()
        req.sampling_params.stop_token_ids = [100, 200]
        req.eos_token_ids = [300, 400]

        tokenizer = Mock()
        tokenizer.eos_token_id = 50256
        tokenizer.additional_stop_token_ids = [500, 600]
        req.tokenizer = tokenizer

        result = ReqBeamSearchMixin.stop_token_ids.fget(req)

        expected = {100, 200, 300, 400, 50256, 500, 600}
        self.assertEqual(result, expected)
        self.assertEqual(req._stop_token_ids_cache, expected)

    def test_stop_token_ids_cached(self):
        """Test that stop token IDs are cached."""
        req = Mock()
        cached_ids = {100, 200, 300}
        req._stop_token_ids_cache = cached_ids

        result = ReqBeamSearchMixin.stop_token_ids.fget(req)

        self.assertEqual(result, cached_ids)

    def test_stop_token_ids_empty_sources(self):
        """Test stop token IDs when all sources are empty."""
        req = Mock()
        req._stop_token_ids_cache = None

        req.sampling_params = Mock()
        req.sampling_params.stop_token_ids = None
        req.eos_token_ids = None

        tokenizer = Mock()
        tokenizer.eos_token_id = None
        tokenizer.additional_stop_token_ids = None
        req.tokenizer = tokenizer

        result = ReqBeamSearchMixin.stop_token_ids.fget(req)

        self.assertEqual(result, set())

    def test_stop_token_ids_no_tokenizer(self):
        """Test stop token IDs when tokenizer is None."""
        req = Mock()
        req._stop_token_ids_cache = None

        req.sampling_params = Mock()
        req.sampling_params.stop_token_ids = [100, 200]
        req.eos_token_ids = [300]
        req.tokenizer = None

        result = ReqBeamSearchMixin.stop_token_ids.fget(req)

        expected = {100, 200, 300}
        self.assertEqual(result, expected)

    def test_stop_token_ids_deduplication(self):
        """Test that duplicate token IDs are deduplicated."""
        req = Mock()
        req._stop_token_ids_cache = None

        req.sampling_params = Mock()
        req.sampling_params.stop_token_ids = [100, 200]
        req.eos_token_ids = [200, 300]

        tokenizer = Mock()
        tokenizer.eos_token_id = 300
        tokenizer.additional_stop_token_ids = [100, 400]
        req.tokenizer = tokenizer

        result = ReqBeamSearchMixin.stop_token_ids.fget(req)

        expected = {100, 200, 300, 400}
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
