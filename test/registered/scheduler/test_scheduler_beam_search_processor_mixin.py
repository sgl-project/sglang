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

from sglang.srt.managers.beam_search_type import BeamSearchList, BeamSearchSequence
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    FINISHED_MATCHED_REGEX,
)
from sglang.srt.managers.scheduler_beam_search_processor_mixin import (
    SchedulerBeamSearchProcessorMixin,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, suite="per-commit-1-gpu")
register_amd_ci(est_time=5, suite="per-commit-1-gpu-amd")


def create_mock_scheduler():
    """Create a mock scheduler object with necessary attributes for testing."""

    # Create a class that inherits from the mixin
    class MockScheduler(SchedulerBeamSearchProcessorMixin):
        pass

    scheduler = MockScheduler()
    scheduler.req_to_token_pool = Mock()
    scheduler.token_to_kv_pool_allocator = Mock()
    scheduler.tree_cache = Mock()
    scheduler.stream_output = Mock()
    scheduler.log_decode_stats = Mock()
    scheduler.attn_tp_rank = 0
    scheduler.server_args = Mock()
    scheduler.server_args.decode_log_interval = 10
    scheduler.num_generated_tokens = 0
    scheduler.forward_ct_decode = 0
    scheduler.current_scheduler_metrics_enabled = Mock(return_value=False)
    return scheduler


class TestProcessPrefillResult(unittest.TestCase):
    """Test process_beam_search_prefill_result method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_process_beam_search_prefill_result_single_req",
    )
    def test_process_prefill_result_basic(self, mock_process_prefill):
        """Test basic prefill result processing with one request."""
        device = torch.device("cpu")

        batch = Mock()
        batch.decoding_reqs = []
        batch.return_logprob = False
        req = Mock()
        req.is_retracted = False
        req.is_beam_search = True
        req.beam_width = 2
        req.finished = Mock(return_value=False)
        batch.reqs = [req]

        logits_output = Mock()
        logits_output.logprobs = torch.randn(1, 100, device=device)

        self.scheduler.process_beam_search_prefill_result(batch, logits_output)

        mock_process_prefill.assert_called_once()
        self.scheduler.tree_cache.cache_unfinished_req.assert_called_once_with(req)

    @patch("sglang.srt.managers.scheduler_beam_search_processor_mixin.release_kv_cache")
    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_process_beam_search_prefill_result_single_req",
    )
    def test_process_prefill_result_two_reqs(
        self, mock_process_prefill, mock_release_kv_cache
    ):
        """Test prefill result processing with two requests: one unfinished (cached) and one finished (released)."""
        device = torch.device("cpu")

        batch = Mock()
        batch.decoding_reqs = []
        batch.device = device
        batch.return_logprob = False

        req1 = Mock()
        req1.is_retracted = False
        req1.is_beam_search = True
        req1.beam_width = 2
        req1.finished = Mock(return_value=False)
        req2 = Mock()
        req2.is_retracted = False
        req2.is_beam_search = True
        req2.beam_width = 2
        req2.finished = Mock(return_value=True)
        batch.reqs = [req1, req2]

        logits_output = Mock()
        logits_output.logprobs = torch.randn(2, 100, device=device)

        SchedulerBeamSearchProcessorMixin.process_beam_search_prefill_result(
            self.scheduler, batch, logits_output
        )

        self.assertEqual(mock_process_prefill.call_count, len(batch.reqs))
        self.scheduler.tree_cache.cache_unfinished_req.assert_called_once_with(req1)
        mock_release_kv_cache.assert_called_once_with(req2, self.scheduler.tree_cache)

    @patch("sglang.srt.managers.scheduler_beam_search_processor_mixin.release_kv_cache")
    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_process_beam_search_prefill_result_single_req",
    )
    def test_process_prefill_result_retracted(
        self, mock_process_prefill, mock_release_kv_cache
    ):
        """Test prefill result processing when request is retracted (should skip processing)."""
        device = torch.device("cpu")

        batch = Mock()
        batch.decoding_reqs = []
        batch.device = device
        batch.return_logprob = False

        # Create retracted request
        req = Mock()
        req.is_retracted = True  # Key: request is retracted
        req.is_beam_search = True

        batch.reqs = [req]

        logits_output = Mock()
        logits_output.logprobs = torch.randn(1, 100, device=device)

        SchedulerBeamSearchProcessorMixin.process_beam_search_prefill_result(
            self.scheduler, batch, logits_output
        )

        mock_process_prefill.assert_not_called()
        self.scheduler.tree_cache.cache_unfinished_req.assert_not_called()
        mock_release_kv_cache.assert_not_called()


class TestProcessDecodeResult(unittest.TestCase):
    """Test process_beam_search_decode_result method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()
        # Mock token_to_kv_pool_allocator methods
        self.scheduler.token_to_kv_pool_allocator.free_group_begin = Mock()
        self.scheduler.token_to_kv_pool_allocator.free_group_end = Mock()

    @patch.object(SchedulerBeamSearchProcessorMixin, "_handle_beam_kv_cache")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_cache_finished_beam_search")
    @patch.object(
        SchedulerBeamSearchProcessorMixin, "_calculate_beam_score_with_eos_check"
    )
    @patch.object(SchedulerBeamSearchProcessorMixin, "_process_beam_search_expansion")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_extract_beam_topk_data")
    def test_process_decode_result_basic(
        self,
        mock_extract_topk,
        mock_process_expansion,
        mock_calculate_score,
        mock_cache_finished,
        mock_handle_kv,
    ):
        """Test basic decode result processing with one unfinished request."""
        device = torch.device("cpu")

        mock_extract_topk.return_value = (
            torch.tensor([[1, 2, 3]], device=device),
            torch.tensor([[-1.0, -2.0, -3.0]], device=device),
        )

        mock_process_expansion.return_value = None

        batch = Mock()
        batch.req_pool_indices = torch.tensor([0], device=device)
        batch.return_logprob = False

        req = Mock()
        req.is_retracted = False
        req.finished = Mock(return_value=False)
        req.beam_width = 2
        req.beam_candidates = 4

        batch.reqs = [req]

        result = Mock()
        result.can_run_cuda_graph = False

        SchedulerBeamSearchProcessorMixin.process_beam_search_decode_result(
            self.scheduler, batch, result
        )

        mock_extract_topk.assert_called_once()
        mock_process_expansion.assert_called_once()
        mock_calculate_score.assert_not_called()
        mock_cache_finished.assert_not_called()
        mock_handle_kv.assert_not_called()

    @patch.object(SchedulerBeamSearchProcessorMixin, "_handle_beam_kv_cache")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_cache_finished_beam_search")
    @patch.object(
        SchedulerBeamSearchProcessorMixin, "_calculate_beam_score_with_eos_check"
    )
    @patch.object(SchedulerBeamSearchProcessorMixin, "_process_beam_search_expansion")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_extract_beam_topk_data")
    def test_process_decode_result_finished_request_without_incomplete(
        self,
        mock_extract_topk,
        mock_process_expansion,
        mock_calculate_score,
        mock_cache_finished,
        mock_handle_kv,
    ):
        """Test decode result processing with a finished request."""
        device = torch.device("cpu")

        mock_extract_topk.return_value = (
            torch.tensor([[1, 2, 3]], device=device),
            torch.tensor([[-1.0, -2.0, -3.0]], device=device),
        )
        mock_process_expansion.return_value = None
        mock_calculate_score.return_value = -1.0

        batch = Mock()
        batch.req_pool_indices = torch.tensor([0], device=device)
        batch.return_logprob = False

        req = Mock()
        req.is_retracted = False
        req.finished = Mock(return_value=True)
        req.beam_width = 2
        req.beam_candidates = 4
        req.beam_list = Mock()
        req.beam_list.incomplete = []
        req.beam_list.completed = []
        req.stop_token_ids = set()

        batch.reqs = [req]

        result = Mock()
        result.can_run_cuda_graph = False

        SchedulerBeamSearchProcessorMixin.process_beam_search_decode_result(
            self.scheduler, batch, result
        )

        mock_calculate_score.assert_not_called()
        mock_cache_finished.assert_called_once()

    @patch.object(SchedulerBeamSearchProcessorMixin, "_handle_beam_kv_cache")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_cache_finished_beam_search")
    @patch.object(
        SchedulerBeamSearchProcessorMixin, "_calculate_beam_score_with_eos_check"
    )
    @patch.object(SchedulerBeamSearchProcessorMixin, "_process_beam_search_expansion")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_extract_beam_topk_data")
    def test_process_decode_result_finished_with_incomplete_beams(
        self,
        mock_extract_topk,
        mock_process_expansion,
        mock_calculate_score,
        mock_cache_finished,
        mock_handle_kv,
    ):
        """Test decode result processing with a finished request that has incomplete beams."""
        device = torch.device("cpu")

        mock_extract_topk.return_value = (
            torch.tensor([[1, 2, 3]], device=device),
            torch.tensor([[-1.0, -2.0, -3.0]], device=device),
        )
        mock_process_expansion.return_value = None
        mock_calculate_score.return_value = -1.0

        batch = Mock()
        batch.req_pool_indices = torch.tensor([0], device=device)
        batch.return_logprob = False

        req = Mock()
        req.is_retracted = False
        req.finished = Mock(return_value=True)
        req.beam_width = 2
        req.beam_candidates = 4
        req.beam_list = Mock()
        beam1 = Mock()
        beam1.beam_score = None
        beam2 = Mock()
        beam2.beam_score = None
        req.beam_list.incomplete = [beam1, beam2]
        req.beam_list.completed = []
        req.stop_token_ids = set()

        batch.reqs = [req]

        result = Mock()
        result.can_run_cuda_graph = False

        self.scheduler.process_beam_search_decode_result(batch, result)

        self.assertEqual(mock_calculate_score.call_count, 2)
        mock_cache_finished.assert_called_once()

    @patch.object(SchedulerBeamSearchProcessorMixin, "_handle_beam_kv_cache")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_cache_finished_beam_search")
    @patch.object(
        SchedulerBeamSearchProcessorMixin, "_calculate_beam_score_with_eos_check"
    )
    @patch.object(SchedulerBeamSearchProcessorMixin, "_process_beam_search_expansion")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_extract_beam_topk_data")
    def test_process_decode_result_with_kv_copy(
        self,
        mock_extract_topk,
        mock_process_expansion,
        mock_calculate_score,
        mock_cache_finished,
        mock_handle_kv,
    ):
        """Test decode result processing when KV cache copy is needed (last_batch_slot_indices is not None)."""
        device = torch.device("cpu")

        mock_extract_topk.return_value = (
            torch.tensor([[1, 2, 3]], device=device),
            torch.tensor([[-1.0, -2.0, -3.0]], device=device),
        )
        mock_last_batch_slot_indices = torch.tensor([0, 1], device=device)
        mock_process_expansion.return_value = mock_last_batch_slot_indices

        batch = Mock()
        batch.req_pool_indices = torch.tensor([0], device=device)
        batch.return_logprob = False

        req = Mock()
        req.is_retracted = False
        req.finished = Mock(return_value=False)
        req.beam_width = 2
        req.beam_candidates = 4

        batch.reqs = [req]

        result = Mock()
        result.can_run_cuda_graph = False

        self.scheduler.process_beam_search_decode_result(batch, result)

        mock_process_expansion.assert_called_once()
        mock_cache_finished.assert_not_called()
        mock_handle_kv.assert_called_once()

    @patch.object(SchedulerBeamSearchProcessorMixin, "_handle_beam_kv_cache")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_cache_finished_beam_search")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_process_beam_search_expansion")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_extract_beam_topk_data")
    def test_process_decode_result_retracted(
        self,
        mock_extract_topk,
        mock_process_expansion,
        mock_cache_finished,
        mock_handle_kv,
    ):
        """Test decode result processing when request is retracted (should skip processing)."""
        device = torch.device("cpu")

        mock_extract_topk.return_value = (
            torch.tensor([[1, 2, 3]], device=device),
            torch.tensor([[-1.0, -2.0, -3.0]], device=device),
        )

        batch = Mock()
        batch.req_pool_indices = torch.tensor([0], device=device)
        batch.return_logprob = False

        # Create retracted request
        req = Mock()
        req.is_retracted = True
        req.beam_width = 2

        batch.reqs = [req]

        result = Mock()
        result.can_run_cuda_graph = False

        self.scheduler.process_beam_search_decode_result(batch, result)

        mock_process_expansion.assert_not_called()


class TestSumBeamCompletionTokens(unittest.TestCase):
    """Test sum_beam_completion_tokens static method."""

    def test_sum_beam_completion_tokens(self):
        """Test calculation of total completion tokens."""
        req = Mock()
        req.beam_list = Mock()
        req.beam_list.completed = [
            BeamSearchSequence(tokens=[1, 2, 3]),
            BeamSearchSequence(tokens=[4, 5]),
            BeamSearchSequence(tokens=[6, 7, 8, 9]),
        ]

        total_tokens = SchedulerBeamSearchProcessorMixin.sum_beam_completion_tokens(req)
        self.assertEqual(total_tokens, 9)

    def test_sum_beam_completion_tokens_empty(self):
        """Test calculation with no completed beams."""
        req = Mock()
        req.beam_list = Mock()
        req.beam_list.completed = []

        total_tokens = SchedulerBeamSearchProcessorMixin.sum_beam_completion_tokens(req)
        self.assertEqual(total_tokens, 0)


class TestConvertBeamSequencesToOutput(unittest.TestCase):
    """Test convert_beam_sequences_to_output static method."""

    def test_convert_beam_sequences_to_output(self):
        """Test conversion of beam sequences to output format."""
        req = Mock()
        req.beam_list = Mock()
        req.beam_list.completed = [
            BeamSearchSequence(
                tokens=[1, 2, 3],
                cum_logprob=-5.0,
                beam_score=-1.67,
                finish_reason=FINISH_LENGTH(length=3),
            ),
            BeamSearchSequence(
                tokens=[4, 5],
                cum_logprob=-3.0,
                beam_score=-1.5,
                finish_reason=FINISH_MATCHED_TOKEN(matched=50256),
            ),
        ]

        output = SchedulerBeamSearchProcessorMixin.convert_beam_sequences_to_output(req)

        self.assertEqual(len(output.sequences), 2)

        seq1 = output.sequences[0]
        self.assertEqual(seq1.tokens, [1, 2, 3])
        self.assertEqual(seq1.cum_logprob, -5.0)
        self.assertIsNotNone(seq1.finish_reason)
        self.assertEqual(seq1.finish_reason["type"], "length")

        seq2 = output.sequences[1]
        self.assertEqual(seq2.tokens, [4, 5])
        self.assertEqual(seq2.cum_logprob, -3.0)
        self.assertIsNotNone(seq2.finish_reason)
        self.assertEqual(seq2.finish_reason["type"], "stop")


class TestInternalProcessPrefillResult(unittest.TestCase):
    """Test _process_prefill_result internal method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    @patch.object(SchedulerBeamSearchProcessorMixin, "_create_initial_beam_sequences")
    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_batch_check_prefill_generated_tokens_stop_conditions",
    )
    def test_internal_process_prefill_result_normal_path(
        self, mock_check_stop, mock_create_beams
    ):
        """Test normal path with sufficient unfinished candidates.

        This tests the case where there are enough unfinished candidates (>= beam_width),
        so _create_initial_beam_sequences is called.
        """
        device = torch.device("cpu")

        req = Mock()
        req.beam_width = 2
        req.beam_candidates = 4
        req.origin_input_ids = [1, 2, 3]
        req.sampling_params = Mock()
        req.sampling_params.max_new_tokens = 10
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = []
        req.sampling_params.stop_regex_strs = []
        req.sampling_params.length_penalty = 1.0
        req.stop_token_ids = set()
        req.beam_list = BeamSearchList()
        req.finished_reason = None
        req.tokenizer = Mock()

        batch = Mock()
        batch.device = device
        logprobs = torch.randn(100, device=device)

        # enough unfinished candidates (>= beam_width)
        def mock_check_stop_func(_, __, ___):
            finish_mask = torch.zeros(4, dtype=torch.bool, device=device)
            finish_mask[0] = True
            finish_mask[1:] = False
            return finish_mask

        mock_check_stop.side_effect = mock_check_stop_func

        self.scheduler._process_beam_search_prefill_result_single_req(
            req, batch, logprobs, device
        )

        mock_check_stop.assert_called_once()
        mock_create_beams.assert_called_once()

        call_args = mock_create_beams.call_args
        self.assertEqual(call_args[0][0], req)
        self.assertIsInstance(call_args[0][1], list)  # top_logprobs_val
        self.assertEqual(len(call_args[0][1]), req.beam_candidates)
        self.assertIsInstance(call_args[0][2], list)  # top_logprobs_idx
        self.assertEqual(len(call_args[0][2]), req.beam_candidates)
        self.assertIsInstance(call_args[0][3], list)  # finish_mask_cpu
        self.assertEqual(len(call_args[0][3]), req.beam_candidates)
        self.assertEqual(call_args[0][4], device)  # device

    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_create_completed_beams_for_insufficient_candidates",
    )
    def test_internal_process_prefill_result_finish_by_len_1(
        self, mock_create_completed
    ):
        """Test finish_by_len=True when max_new_tokens <= 1.

        When max_new_tokens <= 1, all candidates should be marked as finished,
        triggering the insufficient candidates path.
        """
        device = torch.device("cpu")

        req = Mock()
        req.beam_width = 2
        req.beam_candidates = 4
        req.origin_input_ids = [1, 2, 3]
        req.sampling_params = Mock()
        req.sampling_params.max_new_tokens = 1  # Should trigger finish_by_len
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = []
        req.sampling_params.stop_regex_strs = []
        req.sampling_params.length_penalty = 1.0
        req.stop_token_ids = set()
        req.beam_list = BeamSearchList()
        req.finished_reason = None
        req.tokenizer = Mock()

        batch = Mock()
        batch.device = device
        logprobs = torch.randn(100, device=device)

        self.scheduler._process_beam_search_prefill_result_single_req(
            req, batch, logprobs, device
        )

        mock_create_completed.assert_called_once()

        call_args = mock_create_completed.call_args
        self.assertEqual(call_args[0][0], req)
        self.assertIsInstance(call_args[0][1], list)  # top_logprobs_val
        self.assertIsInstance(call_args[0][2], list)  # top_logprobs_idx
        self.assertIsInstance(call_args[0][3], list)  # finish_mask_cpu
        self.assertTrue(all(call_args[0][3]))
        self.assertTrue(call_args[0][4])  # finish_by_len should be True

    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_batch_check_prefill_generated_tokens_stop_conditions",
    )
    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_create_completed_beams_for_insufficient_candidates",
    )
    def test_internal_process_prefill_result_insufficient_candidates(
        self, mock_create_completed, mock_check_stop
    ):
        """Test internal prefill processing when insufficient candidates (< beam_width).

        This tests the branch where (~finish_mask).sum().item() < req.beam_width,
        which triggers _create_completed_beams_for_insufficient_candidates.
        """
        device = torch.device("cpu")

        req = Mock()
        req.beam_width = 2
        req.beam_candidates = 4
        req.origin_input_ids = [1, 2, 3]
        req.sampling_params = Mock()
        req.sampling_params.max_new_tokens = 10
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = []
        req.sampling_params.stop_regex_strs = []
        req.sampling_params.length_penalty = 1.0
        req.stop_token_ids = set()
        req.beam_list = BeamSearchList()
        req.finished_reason = None
        req.tokenizer = Mock()

        batch = Mock()
        batch.device = device

        logprobs = torch.randn(100, device=device)

        def mock_check_stop_func(_, __, ___):
            finish_mask = torch.ones(4, dtype=torch.bool, device=device)
            finish_mask[3] = False
            return finish_mask

        mock_check_stop.side_effect = mock_check_stop_func

        self.scheduler._process_beam_search_prefill_result_single_req(
            req, batch, logprobs, device
        )

        mock_check_stop.assert_called_once()
        mock_create_completed.assert_called_once()
        call_args = mock_create_completed.call_args
        self.assertEqual(call_args[0][0], req)
        self.assertEqual(len(call_args[0][1]), req.beam_candidates)
        self.assertEqual(len(call_args[0][2]), req.beam_candidates)
        self.assertEqual(call_args[0][3], [True, True, True, False])
        self.assertEqual(call_args[0][4], False)


class TestBatchCheckPrefillStopConditions(unittest.TestCase):
    """Test _batch_check_prefill_generated_tokens_stop_conditions method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    def test_batch_check_with_stop_tokens(self):
        """Test batch checking with stop tokens."""
        device = torch.device("cpu")

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = []
        req.stop_token_ids = {50256, 50257}

        generated_token_ids = torch.tensor(
            [100, 50256, 200, 50257], dtype=torch.int64, device=device
        )

        result = self.scheduler._batch_check_prefill_generated_tokens_stop_conditions(
            req, generated_token_ids, device
        )

        # Tokens at indices 1 and 3 should be marked as finished
        expected = torch.tensor([False, True, False, True], dtype=torch.bool)
        self.assertTrue(torch.equal(result.cpu(), expected))

    def test_batch_check_with_stop_strings(self):
        """Test batch checking with stop strings."""
        device = torch.device("cpu")

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = ["STOP"]
        req.stop_token_ids = set()
        req.tokenizer = Mock()
        req.tokenizer.decode = Mock(
            side_effect=lambda tokens, **kwargs: (
                "STOP" if tokens[0] == 100 else "continue"
            )
        )

        generated_token_ids = torch.tensor(
            [100, 200, 300], dtype=torch.int64, device=device
        )

        result = self.scheduler._batch_check_prefill_generated_tokens_stop_conditions(
            req, generated_token_ids, device
        )

        self.assertTrue(result[0].item())
        self.assertFalse(result[1].item())
        self.assertFalse(result[2].item())


class TestCreateCompletedBeamsForInsufficientCandidates(unittest.TestCase):
    """Test _create_completed_beams_for_insufficient_candidates method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    def test_create_completed_beams_for_insufficient_candidates(self):
        """Test creating completed beams when candidates are insufficient."""
        req = Mock()
        req.beam_width = 2
        req.sampling_params = Mock()
        req.sampling_params.max_new_tokens = 10
        req.beam_list = Mock()
        req.finished_reason = None

        top_logprobs_val = [-1.0, -2.0, -3.0, -4.0]
        top_logprobs_idx = [100, 200, 300, 400]
        finish_mask_cpu = [True, True, True, False]
        finish_by_len = False

        self.scheduler._create_completed_beams_for_insufficient_candidates(
            req, top_logprobs_val, top_logprobs_idx, finish_mask_cpu, finish_by_len
        )

        self.assertEqual(len(req.beam_list.completed), 2)
        self.assertEqual(len(req.beam_list.incomplete), 0)

        # Check first beam (finished with stop token)
        first_beam = req.beam_list.completed[0]
        self.assertEqual(first_beam.tokens, [100])
        self.assertEqual(first_beam.cum_logprob, -1.0)
        self.assertIsInstance(first_beam.finish_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(first_beam.finish_reason.matched, 100)
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, 100)

    def test_create_completed_beams_with_finished_beam(self):
        """Test that completed beams have beam_score when one candidate is finished.

        This tests the branch in _create_initial_beam_sequences where is_finished=True,
        ensuring that beam_score is calculated for the finished beam.
        """
        device = torch.device("cpu")

        req = Mock()
        req.beam_width = 3
        req.beam_candidates = 6
        req.origin_input_ids = [1, 2, 3]
        req.sampling_params = Mock()
        req.sampling_params.max_new_tokens = 10
        req.beam_list = BeamSearchList()
        req.finished_reason = None

        top_logprobs_val = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        top_logprobs_idx = [100, 200, 300, 400, 500, 600]
        finish_mask_cpu = [
            True,
            False,
            False,
            False,
            False,
            False,
        ]

        self.scheduler._create_initial_beam_sequences(
            req, top_logprobs_val, top_logprobs_idx, finish_mask_cpu, device
        )

        self.assertEqual(len(req.beam_list.incomplete), 3)
        self.assertEqual(len(req.beam_list.completed), 1)

        completed_beam = req.beam_list.completed[0]
        self.assertIsNotNone(
            completed_beam.beam_score, "Completed beam should have beam_score"
        )

        # Verify other properties of completed beam
        self.assertEqual(completed_beam.tokens, [100])
        self.assertEqual(completed_beam.cum_logprob, -1.0)
        self.assertIsInstance(completed_beam.finish_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(completed_beam.finish_reason.matched, 100)

        # Verify incomplete beams don't have beam_score
        for i, beam in enumerate(req.beam_list.incomplete):
            self.assertIsNone(
                beam.beam_score, f"Incomplete beam {i} should not have beam_score"
            )
            self.assertIsNone(
                beam.finish_reason,
                f"Incomplete beam {i} should not have finish_reason",
            )

        # Verify incomplete beams have correct tokens
        self.assertEqual(req.beam_list.incomplete[0].tokens, [200])
        self.assertEqual(req.beam_list.incomplete[1].tokens, [300])
        self.assertEqual(req.beam_list.incomplete[2].tokens, [400])


class TestCreateInitialBeamSequences(unittest.TestCase):
    """Test _create_initial_beam_sequences method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    def test_create_initial_beam_sequences(self):
        """Test creating initial beam sequences from prefill."""
        device = torch.device("cpu")

        req = Mock()
        req.beam_width = 2
        req.beam_candidates = 4
        req.origin_input_ids = [1, 2, 3]
        req.beam_list = BeamSearchList()  # Initialize beam_list properly

        top_logprobs_val = [-1.0, -2.0, -3.0, -4.0, -5.0]
        top_logprobs_idx = [100, 200, 300, 400, 500]
        finish_mask_cpu = [False, False, False, False, False]

        self.scheduler._create_initial_beam_sequences(
            req, top_logprobs_val, top_logprobs_idx, finish_mask_cpu, device
        )

        self.assertEqual(len(req.beam_list.incomplete), 2)
        self.assertEqual(len(req.beam_list.completed), 0)
        self.assertEqual(req.beam_list.incomplete[0].tokens, [100])
        self.assertEqual(req.beam_list.incomplete[1].tokens, [200])
        self.assertEqual(len(req.beam_list.prompt_lens), 2)
        self.assertEqual(len(req.beam_list.last_tokens), 2)
        self.assertEqual(len(req.beam_list.cum_logprobs), 2)
        self.assertEqual(req.beam_list.prompt_lens[0], 3)
        self.assertEqual(req.beam_list.prompt_lens[1], 3)
        self.assertEqual(req.beam_list.last_tokens[0], 100)
        self.assertEqual(req.beam_list.last_tokens[1], 200)
        self.assertEqual(req.beam_list.cum_logprobs[0], -1.0)
        self.assertEqual(req.beam_list.cum_logprobs[1], -2.0)


class TestCheckBeamFinished(unittest.TestCase):
    """Test _check_beam_finished method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    def test_check_beam_finished_with_stop_token(self):
        """Test beam finish check with stop token."""
        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = []
        req.sampling_params.stop_regex_strs = []
        req.stop_token_ids = {50256, 50257}

        beam = BeamSearchSequence(
            tokens=[1, 2, 3, 50256],
            cum_logprob=-5.0,
        )

        is_finished = self.scheduler._check_beam_finished(req, beam)

        self.assertTrue(is_finished)

        self.assertIsInstance(beam.finish_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(beam.finish_reason.matched, 50256)

    @patch.object(SchedulerBeamSearchProcessorMixin, "_tail_str")
    def test_check_beam_finished_with_stop_string(self, mock_tail_str):
        """Test beam finish check with stop string."""
        mock_tail_str.return_value = "This is STOP"

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = True
        req.sampling_params.stop_strs = ["STOP", "END"]
        req.sampling_params.stop_regex_strs = []
        req.tokenizer = Mock()

        beam = BeamSearchSequence(
            tokens=[1, 2, 3, 4],
            cum_logprob=-5.0,
            text="other information",
        )

        is_finished = self.scheduler._check_beam_finished(req, beam)

        self.assertTrue(is_finished)
        self.assertIsInstance(beam.finish_reason, FINISH_MATCHED_STR)
        self.assertEqual(beam.finish_reason.matched, "STOP")
        mock_tail_str.assert_called_once_with(req, beam.tokens)

    @patch.object(SchedulerBeamSearchProcessorMixin, "_tail_str")
    def test_check_beam_finished_with_regex(self, mock_tail_str):
        """Test beam finish check with regex pattern."""
        mock_tail_str.return_value = "Call 123-4567 now"

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = True
        req.sampling_params.stop_strs = ["STOP"]
        req.sampling_params.stop_regex_strs = [r"\d{3}-\d{4}"]
        req.stop_token_ids = set()
        req.tokenizer = Mock()

        beam = BeamSearchSequence(
            tokens=[1, 2, 3, 4, 5],
            cum_logprob=-5.0,
            text="Call 123-4567 now",
        )

        is_finished = self.scheduler._check_beam_finished(req, beam)

        self.assertTrue(is_finished)
        self.assertIsInstance(beam.finish_reason, FINISHED_MATCHED_REGEX)
        self.assertEqual(beam.finish_reason.matched, r"\d{3}-\d{4}")
        mock_tail_str.assert_called_once_with(req, beam.tokens)

    @patch.object(SchedulerBeamSearchProcessorMixin, "_tail_str")
    def test_check_beam_not_finished(self, mock_tail_str):
        """Test beam that should not be marked as finished."""
        mock_tail_str.return_value = "Continue"

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = ["STOP"]
        req.sampling_params.stop_regex_strs = []
        req.stop_token_ids = {50256}
        req.tokenizer = Mock()

        beam = BeamSearchSequence(
            tokens=[1, 2, 3, 4],
            cum_logprob=-5.0,
            text="other information",
        )

        is_finished = self.scheduler._check_beam_finished(req, beam)

        self.assertFalse(is_finished)
        self.assertIsNone(beam.finish_reason)
        mock_tail_str.assert_called_once_with(req, beam.tokens)

    @patch.object(SchedulerBeamSearchProcessorMixin, "_tail_str")
    def test_check_beam_not_finished_with_ignore_eos(self, mock_tail_str):
        """Test beam not finished when ignore_eos=True even with EOS token."""
        mock_tail_str.return_value = "Text without stop"

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = True  # Key: ignore EOS tokens
        req.sampling_params.stop_strs = ["STOP"]
        req.sampling_params.stop_regex_strs = []
        req.stop_token_ids = {50256}
        req.tokenizer = Mock()

        beam = BeamSearchSequence(
            tokens=[1, 2, 3, 50256],
            cum_logprob=-5.0,
            text="Text without stop",
        )

        is_finished = self.scheduler._check_beam_finished(req, beam)

        self.assertFalse(is_finished)
        self.assertIsNone(beam.finish_reason)
        mock_tail_str.assert_called_once_with(req, beam.tokens)

    @patch.object(SchedulerBeamSearchProcessorMixin, "_tail_str")
    def test_check_beam_finished_with_empty_tail_str(self, mock_tail_str):
        """Test beam finish check when tail_str is empty (returns False early)."""
        mock_tail_str.return_value = ""  # Key: empty tail_str

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = ["STOP"]
        req.sampling_params.stop_regex_strs = [r"\d{3}-\d{4}"]
        req.stop_token_ids = set()
        req.tokenizer = Mock()

        beam = BeamSearchSequence(
            tokens=[1, 2, 3, 4],
            cum_logprob=-5.0,
            text="some text",
        )

        is_finished = self.scheduler._check_beam_finished(req, beam)

        # Should return False early when tail_str is empty
        self.assertFalse(is_finished)
        self.assertIsNone(beam.finish_reason)
        mock_tail_str.assert_called_once_with(req, beam.tokens)


class TestTailStr(unittest.TestCase):
    """Test _tail_str method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    def test_tail_str_with_stop_strings(self):
        """Test tail string extraction with stop strings."""
        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.stop_strs = ["STOP"]
        req.sampling_params.stop_regex_strs = []
        req.sampling_params.stop_str_max_len = 2
        req.sampling_params.stop_regex_max_len = 0
        req.tokenizer = Mock()
        req.tokenizer.decode = Mock(return_value="tail text")

        tokens = [1, 2, 3, 4, 5]
        result = self.scheduler._tail_str(req, tokens)

        self.assertEqual(result, "tail text")
        self.assertTrue(req.tokenizer.decode.called)

    def test_tail_str_with_regex(self):
        """Test tail string extraction with regex patterns."""
        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.stop_strs = []
        req.sampling_params.stop_regex_strs = [r"\d{3}"]
        req.sampling_params.stop_str_max_len = 0
        req.sampling_params.stop_regex_max_len = 2
        req.tokenizer = Mock()
        req.tokenizer.decode = Mock(return_value="text 123")

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        result = self.scheduler._tail_str(req, tokens)

        self.assertEqual(result, "text 123")


class TestExtractBeamTopkData(unittest.TestCase):
    """Test _extract_beam_topk_data method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    def test_extract_beam_topk_data(self):
        """Test extracting top-k tokens and logprobs for beams."""
        device = torch.device("cpu")

        batch = Mock()
        batch.reqs = [Mock(), Mock()]
        batch.reqs[0].beam_candidates = 4
        batch.reqs[1].beam_candidates = 8

        result = Mock()
        result.logits_output = Mock()
        result.logits_output.logprobs = torch.randn(6, 10, device=device)

        top_tokens, top_logprobs = self.scheduler._extract_beam_topk_data(batch, result)

        # Should return tensors with shape [num_beams, max_k]
        self.assertEqual(top_tokens.shape, (6, 8))
        self.assertEqual(top_logprobs.shape, (6, 8))

        for i in range(6):
            for j in range(7):
                self.assertGreaterEqual(
                    top_logprobs[i, j].item(), top_logprobs[i, j + 1].item()
                )


class TestProcessBeamSearchExpansion(unittest.TestCase):
    """Test _process_beam_search_expansion method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    @patch.object(SchedulerBeamSearchProcessorMixin, "_expand_and_prune_beams")
    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_create_completed_beams_for_finished_request",
    )
    def test_process_beam_search_expansion_normal_path(
        self, mock_create_completed, mock_expand_prune
    ):
        """Test normal expansion path (not finished, continues beam search)."""
        device = torch.device("cpu")

        req = Mock()
        req.beam_width = 2
        req.beam_candidates = 4
        req.beam_list = Mock()
        req.beam_list.incomplete = [
            BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0),
            BeamSearchSequence(tokens=[3, 4], cum_logprob=-3.0),
        ]
        req.beam_list.cum_logprobs = torch.tensor([-2.0, -3.0], device=device)
        req.beam_list.batch_slot_start_idx = 0
        req.sampling_params = Mock()
        req.sampling_params.max_new_tokens = 10
        req.to_finish = None

        batch = Mock()
        batch.device = device

        beam_output_top_tokens = torch.tensor(
            [[100, 200, 300, 400], [500, 600, 700, 800]], device=device
        )
        beam_output_top_logprobs = torch.tensor(
            [[-1.0, -2.0, -3.0, -4.0], [-1.5, -2.5, -3.5, -4.5]], device=device
        )

        mock_expand_prune.return_value = [0, 1]

        result = self.scheduler._process_beam_search_expansion(
            req, batch, 2, 4, beam_output_top_tokens, beam_output_top_logprobs
        )

        mock_expand_prune.assert_called_once()
        mock_create_completed.assert_not_called()
        self.assertIsNotNone(result)
        self.assertEqual(result.cpu().tolist(), [0, 1])

    @patch.object(SchedulerBeamSearchProcessorMixin, "_expand_and_prune_beams")
    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_create_completed_beams_for_finished_request",
    )
    def test_process_beam_search_expansion_finished_by_length(
        self, mock_create_completed, mock_expand_prune
    ):
        """Test expansion when request finishes due to max_new_tokens."""
        device = torch.device("cpu")

        req = Mock()
        req.beam_width = 2
        req.beam_candidates = 4
        req.beam_list = Mock()
        req.beam_list.incomplete = [
            BeamSearchSequence(tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9], cum_logprob=-2.0),
            BeamSearchSequence(tokens=[3, 4, 5, 6, 7, 8, 9, 10, 11], cum_logprob=-3.0),
        ]
        req.beam_list.cum_logprobs = torch.tensor([-2.0, -3.0], device=device)
        req.sampling_params = Mock()
        req.sampling_params.max_new_tokens = 10  # Will finish after this step
        req.to_finish = None

        batch = Mock()
        batch.device = device

        beam_output_top_tokens = torch.tensor(
            [[100, 200, 300, 400], [500, 600, 700, 800]], device=device
        )
        beam_output_top_logprobs = torch.tensor(
            [[-1.0, -2.0, -3.0, -4.0], [-1.5, -2.5, -3.5, -4.5]], device=device
        )

        result = self.scheduler._process_beam_search_expansion(
            req, batch, 2, 4, beam_output_top_tokens, beam_output_top_logprobs
        )

        mock_create_completed.assert_called_once()
        mock_expand_prune.assert_not_called()
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertIsNone(result)

    @patch.object(SchedulerBeamSearchProcessorMixin, "_expand_and_prune_beams")
    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_create_completed_beams_for_finished_request",
    )
    def test_process_beam_search_expansion_with_to_finish(
        self, mock_create_completed, mock_expand_prune
    ):
        """Test expansion when request has to_finish set."""
        device = torch.device("cpu")

        req = Mock()
        req.beam_width = 2
        req.beam_candidates = 4
        req.beam_list = Mock()
        req.beam_list.incomplete = [
            BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0),
            BeamSearchSequence(tokens=[3, 4], cum_logprob=-3.0),
        ]
        req.beam_list.cum_logprobs = torch.tensor([-2.0, -3.0], device=device)
        req.sampling_params = Mock()
        req.sampling_params.max_new_tokens = 10
        req.to_finish = FINISH_MATCHED_TOKEN(matched=50256)  # Pre-set finish reason

        batch = Mock()
        batch.device = device

        beam_output_top_tokens = torch.tensor(
            [[100, 200, 300, 400], [500, 600, 700, 800]], device=device
        )
        beam_output_top_logprobs = torch.tensor(
            [[-1.0, -2.0, -3.0, -4.0], [-1.5, -2.5, -3.5, -4.5]], device=device
        )

        result = self.scheduler._process_beam_search_expansion(
            req, batch, 2, 4, beam_output_top_tokens, beam_output_top_logprobs
        )

        mock_create_completed.assert_called_once()
        call_args = mock_create_completed.call_args[0]
        self.assertIsInstance(call_args[6], FINISH_MATCHED_TOKEN)
        mock_expand_prune.assert_not_called()
        self.assertEqual(req.finished_reason, req.to_finish)
        self.assertIsNone(result)

    @patch.object(SchedulerBeamSearchProcessorMixin, "_expand_and_prune_beams")
    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_create_completed_beams_for_finished_request",
    )
    def test_process_beam_search_expansion_with_empty_keep_indices(
        self, mock_create_completed, mock_expand_prune
    ):
        """Test expansion when _expand_and_prune_beams returns None (keep_last_beam_indices is empty)."""
        device = torch.device("cpu")

        req = Mock()
        req.beam_width = 2
        req.beam_candidates = 4
        req.beam_list = Mock()
        req.beam_list.incomplete = [
            BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0),
            BeamSearchSequence(tokens=[3, 4], cum_logprob=-3.0),
        ]
        req.beam_list.cum_logprobs = torch.tensor([-2.0, -3.0], device=device)
        req.sampling_params = Mock()
        req.sampling_params.max_new_tokens = 10
        req.to_finish = None

        batch = Mock()
        batch.device = device

        beam_output_top_tokens = torch.tensor(
            [[100, 200, 300, 400], [500, 600, 700, 800]], device=device
        )
        beam_output_top_logprobs = torch.tensor(
            [[-1.0, -2.0, -3.0, -4.0], [-1.5, -2.5, -3.5, -4.5]], device=device
        )

        # Mock _expand_and_prune_beams to return None (insufficient candidates)
        mock_expand_prune.return_value = None

        result = self.scheduler._process_beam_search_expansion(
            req, batch, 2, 4, beam_output_top_tokens, beam_output_top_logprobs
        )

        mock_expand_prune.assert_called_once()
        mock_create_completed.assert_not_called()
        self.assertIsNone(result)


class TestExpandAndPruneBeams(unittest.TestCase):
    """Test _expand_and_prune_beams method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    def test_expand_and_prune_beams_no_stop_conditions(self):
        """Test beam expansion without stop conditions (fast path)."""
        device = torch.device("cpu")

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = True
        req.sampling_params.stop_strs = []
        req.stop_token_ids = set()
        req.beam_list = Mock()
        req.beam_list.incomplete = [
            BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0),
            BeamSearchSequence(tokens=[3, 4], cum_logprob=-3.0),
        ]
        req.beam_list.completed = []
        req.beam_list.last_tokens = torch.tensor([2, 4], device=device)
        req.beam_list.cum_logprobs = torch.tensor([-2.0, -3.0], device=device)

        beam_width = 2
        topk = 4
        topk_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
        topk_values = torch.tensor(
            [-2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0], device=device
        )
        all_tokens_flat = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800], device=device
        )

        result = self.scheduler._expand_and_prune_beams(
            req, beam_width, topk, topk_indices, topk_values, all_tokens_flat
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(result), beam_width)
        self.assertEqual(len(req.beam_list.incomplete), beam_width)
        # First beam: tokens=[1, 2] + token_id=100, cum_logprob=-2.5
        self.assertEqual(req.beam_list.incomplete[0].tokens, [1, 2, 100])
        self.assertAlmostEqual(req.beam_list.incomplete[0].cum_logprob, -2.5)
        self.assertIsNone(req.beam_list.incomplete[0].finish_reason)
        # Second beam: tokens=[1, 2] + token_id=200, cum_logprob=-3.0
        self.assertEqual(req.beam_list.incomplete[1].tokens, [1, 2, 200])
        self.assertAlmostEqual(req.beam_list.incomplete[1].cum_logprob, -3.0)
        self.assertIsNone(req.beam_list.incomplete[1].finish_reason)

        self.assertEqual(len(req.beam_list.completed), 0)

        self.assertTrue(
            torch.equal(
                req.beam_list.last_tokens, torch.tensor([100, 200], device=device)
            )
        )

        self.assertTrue(
            torch.allclose(
                req.beam_list.cum_logprobs,
                torch.tensor([-2.5, -3.0], dtype=torch.float32, device=device),
            )
        )

    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_calculate_beam_score_with_eos_check",
        return_value=0.5,
    )
    def test_expand_and_prune_beams_with_eos(self, mock_calc_score):
        """Test beam expansion with EOS tokens (vectorized path)."""
        device = torch.device("cpu")

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = []
        req.stop_token_ids = {50256}
        req.beam_list = Mock()
        req.beam_list.incomplete = [
            BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0),
            BeamSearchSequence(tokens=[3, 4], cum_logprob=-3.0),
        ]
        req.beam_list.completed = []
        req.beam_list.last_tokens = torch.tensor([2, 4], device=device)
        req.beam_list.cum_logprobs = torch.tensor([-2.0, -3.0], device=device)

        beam_width = 2
        topk = 4
        # topk_indices: [0,1,4,5] map to beams [0,0,1,1]
        topk_indices = torch.tensor([0, 1, 4, 5], device=device)
        topk_values = torch.tensor([-2.5, -3.0, -3.5, -4.0], device=device)
        all_tokens_flat = torch.tensor(
            [50256, 200, 300, 400, 500, 600, 700, 800], device=device
        )

        result = self.scheduler._expand_and_prune_beams(
            req, beam_width, topk, topk_indices, topk_values, all_tokens_flat
        )

        self.assertEqual(len(result), beam_width)
        self.assertEqual(result, [0, 1])

        self.assertEqual(len(req.beam_list.completed), 1)
        self.assertEqual(req.beam_list.completed[0].tokens, [1, 2, 50256])
        self.assertEqual(req.beam_list.completed[0].cum_logprob, -2.5)
        self.assertIsInstance(
            req.beam_list.completed[0].finish_reason, FINISH_MATCHED_TOKEN
        )
        self.assertEqual(req.beam_list.completed[0].beam_score, 0.5)

        self.assertEqual(len(req.beam_list.incomplete), beam_width)
        self.assertEqual(req.beam_list.incomplete[0].tokens, [1, 2, 200])
        self.assertEqual(req.beam_list.incomplete[0].cum_logprob, -3.0)
        self.assertEqual(req.beam_list.incomplete[1].tokens, [3, 4, 500])
        self.assertEqual(req.beam_list.incomplete[1].cum_logprob, -3.5)

        mock_calc_score.assert_called_once()

    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_calculate_beam_score_with_eos_check",
        return_value=0.8,
    )
    @patch.object(SchedulerBeamSearchProcessorMixin, "_check_beam_finished")
    def test_expand_and_prune_beams_with_stop_strs(
        self, mock_check_finished, mock_calc_score
    ):
        """Test beam expansion with stop_strs (sequential check path)."""
        device = torch.device("cpu")

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = ["STOP"]  # Trigger sequential check
        req.stop_token_ids = set()
        req.beam_list = Mock()
        req.beam_list.incomplete = [
            BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0),
            BeamSearchSequence(tokens=[3, 4], cum_logprob=-3.0),
        ]
        req.beam_list.completed = []
        req.beam_list.last_tokens = torch.tensor([2, 4], device=device)
        req.beam_list.cum_logprobs = torch.tensor([-2.0, -3.0], device=device)

        beam_width = 2
        topk = 4  # 4 candidates per beam
        # topk_indices: [0,1,2,3,4,5,6,7] map to beams [0,0,0,0,1,1,1,1]
        # Total 8 candidates from 2 beams
        topk_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
        topk_values = torch.tensor(
            [-2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0], device=device
        )
        all_tokens_flat = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800], device=device
        )

        # Mock _check_beam_finished: first 6 candidates complete, last 2 incomplete
        # This ensures all 8 candidates are checked
        def check_finished_side_effect(req_arg, beam):
            # First 6 beams complete (tokens 100-500)
            if beam.tokens[-1] in [100, 200, 300, 400, 500, 600]:
                beam.finish_reason = FINISH_MATCHED_STR(matched="STOP")
                return True
            # Last 2 beams incomplete (tokens 700, 800)
            return False

        mock_check_finished.side_effect = check_finished_side_effect

        result = self.scheduler._expand_and_prune_beams(
            req, beam_width, topk, topk_indices, topk_values, all_tokens_flat
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(result), beam_width)

        self.assertEqual(mock_check_finished.call_count, 8)

        self.assertEqual(len(req.beam_list.completed), 6)
        self.assertEqual(req.beam_list.completed[0].tokens, [1, 2, 100])
        self.assertAlmostEqual(req.beam_list.completed[0].cum_logprob, -2.5)
        self.assertIsInstance(
            req.beam_list.completed[0].finish_reason, FINISH_MATCHED_STR
        )
        self.assertEqual(req.beam_list.completed[0].beam_score, 0.8)

        self.assertEqual(len(req.beam_list.incomplete), beam_width)
        self.assertEqual(req.beam_list.incomplete[0].tokens, [3, 4, 700])
        self.assertAlmostEqual(req.beam_list.incomplete[0].cum_logprob, -5.5)
        self.assertIsNone(req.beam_list.incomplete[0].finish_reason)
        self.assertEqual(req.beam_list.incomplete[1].tokens, [3, 4, 800])
        self.assertAlmostEqual(req.beam_list.incomplete[1].cum_logprob, -6.0)
        self.assertIsNone(req.beam_list.incomplete[1].finish_reason)

        self.assertTrue(
            torch.equal(
                req.beam_list.last_tokens, torch.tensor([700, 800], device=device)
            )
        )
        self.assertTrue(
            torch.allclose(
                req.beam_list.cum_logprobs,
                torch.tensor([-5.5, -6.0], dtype=torch.float32, device=device),
            )
        )

        self.assertEqual(mock_calc_score.call_count, 6)

    @patch.object(
        SchedulerBeamSearchProcessorMixin,
        "_calculate_beam_score_with_eos_check",
        return_value=-1.0,
    )
    @patch.object(SchedulerBeamSearchProcessorMixin, "_check_beam_finished")
    def test_expand_and_prune_beams_insufficient_candidates(
        self, mock_check_finished, mock_calc_score
    ):
        """Test beam expansion when insufficient candidates remain."""
        device = torch.device("cpu")

        req = Mock()
        req.sampling_params = Mock()
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = []
        req.stop_token_ids = {50256}
        req.beam_list = Mock()
        req.beam_list.incomplete = [
            BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0),
        ]
        req.beam_list.completed = []
        req.beam_list.last_tokens = torch.tensor([2], device=device)
        req.beam_list.cum_logprobs = torch.tensor([-2.0], device=device)

        beam_width = 3
        topk = 6
        topk_indices = torch.tensor([0, 1, 2, 3, 4, 5], device=device)
        topk_values = torch.tensor([-2.5, -3.0, -3.5, -4.0, -4.5, -5.0], device=device)
        all_tokens_flat = torch.tensor(
            [50256, 50256, 50256, 50256, 50256, 50256], device=device
        )

        result = self.scheduler._expand_and_prune_beams(
            req, beam_width, topk, topk_indices, topk_values, all_tokens_flat
        )

        self.assertIsNone(result)
        self.assertEqual(len(req.beam_list.completed), 6)
        self.assertEqual(len(req.beam_list.incomplete), 0)
        self.assertEqual(mock_calc_score.call_count, 6)
        for beam in req.beam_list.completed:
            self.assertEqual(beam.beam_score, -1.0)


class TestCreateCompletedBeamsForFinishedRequest(unittest.TestCase):
    """Test _create_completed_beams_for_finished_request method."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = create_mock_scheduler()

    def test_create_completed_beams_for_finished_request(self):
        """Test creating completed beams for a finished request."""
        device = torch.device("cpu")

        req = Mock()
        req.beam_list = Mock()
        req.beam_list.incomplete = [
            BeamSearchSequence(tokens=[1, 2], cum_logprob=-2.0),
            BeamSearchSequence(tokens=[3, 4], cum_logprob=-3.0),
        ]
        # Start with one existing completed beam
        existing_beam = BeamSearchSequence(
            tokens=[5, 6, 7],
            cum_logprob=-1.5,
            finish_reason=FINISH_MATCHED_TOKEN(matched=50256),
            beam_score=-0.5,
        )
        req.beam_list.completed = [existing_beam]
        req.beam_list.cum_logprobs = torch.tensor([-2.0, -3.0], device=device)

        beam_width = 2
        topk = 4
        topk_indices = torch.tensor([0, 1, 2, 3], device=device)
        topk_values = torch.tensor([-2.5, -3.0, -3.5, -4.0], device=device)
        all_tokens_flat = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800], device=device
        )
        will_finish_reason = FINISH_LENGTH(length=10)

        self.scheduler._create_completed_beams_for_finished_request(
            req,
            beam_width,
            topk,
            topk_indices,
            topk_values,
            all_tokens_flat,
            will_finish_reason,
        )

        self.assertEqual(len(req.beam_list.completed), 3)
        self.assertEqual(len(req.beam_list.incomplete), 0)

        self.assertEqual(req.beam_list.completed[0], existing_beam)

        for beam in req.beam_list.completed[1:]:
            self.assertEqual(beam.finish_reason, will_finish_reason)


class TestHandleBeamKVCache(unittest.TestCase):
    """Test _handle_beam_kv_cache method."""

    def setUp(self):
        """Set up test fixtures."""
        token_to_kv_pool_allocator = Mock()
        token_to_kv_pool_allocator.free = Mock()

        self.scheduler = create_mock_scheduler()

    @patch.object(SchedulerBeamSearchProcessorMixin, "_batch_collect_range_kv_indices")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_copy_kvcache_for_beams")
    def test_handle_beam_kv_cache_beam_pruning(
        self, mock_copy_kvcache, mock_collect_kv
    ):
        """Test KV cache handling when beams are pruned (some beams not kept).

        Scenario: Last round had 2 beams at req_to_token slots [0, 1].
        After beam search expansion and pruning, we still have 2 beams at slots [0, 1],
        but both copy from the last round's beam 0 (last_batch_slot_indices = [0, 0]).
        This means beam 1 was pruned and its KV cache should be freed.
        """
        device = torch.device("cpu")

        req = Mock()
        req.beam_width = 2
        req.beam_list = Mock()
        req.beam_list.batch_slot_start_idx = 0
        req.beam_list.prompt_lens = torch.tensor([5, 5], device=device)

        batch = Mock()
        batch.seq_lens = torch.tensor([10, 10], device=device)
        batch.req_pool_indices = torch.tensor([0, 1], device=device)
        batch.device = device

        # This means last round's beam 1 was pruned
        last_batch_slot_indices = torch.tensor([0, 0], device=device)

        req_to_token_pool = Mock()
        req_to_token_pool.req_to_token = torch.arange(20, device=device).reshape(2, 10)
        self.scheduler.req_to_token_pool = req_to_token_pool

        mock_collect_kv.return_value = torch.tensor(
            [100, 101, 102, 103, 104, 200, 201, 202, 203, 204], device=device
        )

        mock_copy_kvcache.return_value = torch.tensor(
            [100, 101, 102, 103, 104], device=device
        )

        self.scheduler._handle_beam_kv_cache(batch, [req], [last_batch_slot_indices])

        mock_collect_kv.assert_called_once()

        mock_copy_kvcache.assert_called_once()
        copy_args = mock_copy_kvcache.call_args[0]
        self.assertTrue(torch.equal(copy_args[0], torch.tensor([0, 0], device=device)))
        self.assertTrue(torch.equal(copy_args[1], torch.tensor([0, 1], device=device)))

        self.scheduler.token_to_kv_pool_allocator.free.assert_called_once()
        freed_indices = self.scheduler.token_to_kv_pool_allocator.free.call_args[0][0]
        expected_freed = torch.tensor([200, 201, 202, 203, 204], device=device)
        self.assertTrue(torch.equal(torch.sort(freed_indices)[0], expected_freed))

    @patch.object(SchedulerBeamSearchProcessorMixin, "_batch_collect_range_kv_indices")
    @patch.object(SchedulerBeamSearchProcessorMixin, "_copy_kvcache_for_beams")
    def test_handle_beam_kv_cache_multiple_requests(
        self, mock_copy_kvcache, mock_collect_kv
    ):
        """Test KV cache handling for multiple beam search requests."""
        device = torch.device("cpu")

        req1 = Mock()
        req1.beam_width = 2
        req1.beam_list = Mock()
        req1.beam_list.batch_slot_start_idx = 0
        req1.beam_list.prompt_lens = torch.tensor([5, 5], device=device)

        req2 = Mock()
        req2.beam_width = 3
        req2.beam_list = Mock()
        req2.beam_list.batch_slot_start_idx = 2
        req2.beam_list.prompt_lens = torch.tensor([6, 6, 6], device=device)

        batch = Mock()
        batch.seq_lens = torch.tensor([7, 7, 8, 8, 8], device=device)
        batch.req_pool_indices = torch.tensor([0, 1, 2, 3, 4], device=device)
        batch.device = device

        last_batch_slot_indices_1 = torch.tensor([0, 0], device=device)
        last_batch_slot_indices_2 = torch.tensor([2, 3, 4], device=device)

        req_to_token_pool = Mock()
        req_to_token_pool.req_to_token = torch.arange(60, device=device).reshape(5, 12)
        self.scheduler.req_to_token_pool = req_to_token_pool

        mock_collect_kv.return_value = torch.tensor(
            [100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205],
            device=device,
        )

        mock_copy_kvcache.return_value = torch.tensor(
            [100, 101, 104, 105, 200, 201, 202, 203, 204, 205],
            device=device,
        )

        self.scheduler._handle_beam_kv_cache(
            batch, [req1, req2], [last_batch_slot_indices_1, last_batch_slot_indices_2]
        )

        self.assertEqual(mock_collect_kv.call_count, 1)
        call_args = mock_collect_kv.call_args
        self.assertTrue(
            torch.equal(call_args[0][0], torch.tensor([0, 1, 2, 3, 4], device=device))
        )
        self.assertTrue(
            torch.equal(call_args[0][1], torch.tensor([7, 7, 8, 8, 8], device=device))
        )
        self.assertEqual(call_args[0][2], device)
        expected_prompt_lens = torch.tensor([5, 5, 6, 6, 6], device=device)
        self.assertTrue(torch.equal(call_args[0][3], expected_prompt_lens))

        mock_copy_kvcache.assert_called_once()
        copy_call_args = mock_copy_kvcache.call_args
        self.assertTrue(
            torch.equal(
                copy_call_args[0][0], torch.tensor([0, 0, 2, 3, 4], device=device)
            )
        )
        self.assertTrue(
            torch.equal(
                copy_call_args[0][1], torch.tensor([0, 1, 2, 3, 4], device=device)
            )
        )
        self.assertTrue(
            torch.equal(
                copy_call_args[0][2], torch.tensor([5, 5, 6, 6, 6], device=device)
            )
        )
        self.assertTrue(
            torch.equal(
                copy_call_args[0][3], torch.tensor([7, 7, 8, 8, 8], device=device)
            )
        )
        self.assertEqual(copy_call_args[0][4], device)

        self.scheduler.token_to_kv_pool_allocator.free.assert_called_once()
        freed_indices = self.scheduler.token_to_kv_pool_allocator.free.call_args[0][0]
        self.assertTrue(
            torch.equal(freed_indices, torch.tensor([102, 103], device=device))
        )


class TestBatchCollectRangeKVIndices(unittest.TestCase):
    """Test batch KV cache index collection."""

    def test_batch_collect_range_kv_indices_basic(self):
        """Test basic KV cache index collection."""
        device = torch.device("cpu")

        scheduler = create_mock_scheduler()
        scheduler.req_to_token_pool.req_to_token = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100, 101, 102, 103, 104],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 200, 201, 202, 203, 204],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 300, 301, 302, 303, 304],
            ],
            dtype=torch.int64,
            device=device,
        )

        pool_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
        seq_lens = torch.tensor([10, 10, 12], dtype=torch.int64, device=device)
        prefix_lens = torch.tensor([5, 8, 6], dtype=torch.int64, device=device)

        result = SchedulerBeamSearchProcessorMixin._batch_collect_range_kv_indices(
            scheduler, pool_indices, seq_lens, device, prefix_lens
        )

        expected = [15, 16, 17, 18, 19, 28, 29, 36, 37, 38, 39, 300, 301]
        result_cpu = result.cpu().tolist()
        self.assertEqual(result_cpu, expected)

    def test_batch_collect_range_kv_indices_no_prefix(self):
        """Test KV cache index collection without prefix."""
        device = torch.device("cpu")

        scheduler = create_mock_scheduler()
        scheduler.req_to_token_pool.req_to_token = torch.tensor(
            [
                [10, 11, 12, 13, 14],
                [20, 21, 22, 23, 24],
            ],
            dtype=torch.int64,
            device=device,
        )

        pool_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
        seq_lens = torch.tensor([3, 4], dtype=torch.int64, device=device)

        result = SchedulerBeamSearchProcessorMixin._batch_collect_range_kv_indices(
            scheduler, pool_indices, seq_lens, device, prefix_lens=None
        )

        expected = [10, 11, 12, 20, 21, 22, 23]
        self.assertEqual(result.tolist(), expected)


class TestCacheFinishedBeamSearch(unittest.TestCase):
    """Test caching finished beam search requests."""

    @patch("sglang.srt.managers.scheduler_beam_search_processor_mixin.release_kv_cache")
    @patch.object(
        SchedulerBeamSearchProcessorMixin, "_collect_beam_req_decode_kv_indices"
    )
    def test_cache_finished_beam_search(self, mock_collect_kv, mock_release_kv_cache):
        """Test releasing KV cache for finished beam search."""
        device = torch.device("cpu")

        batch = Mock()
        batch.device = device
        batch.req_pool_indices = torch.tensor([0, 1, 2], device=device)
        batch.seq_lens = torch.tensor([7, 7, 7], device=device)

        req = Mock()
        req.beam_width = 3
        req.beam_list = Mock()
        req.beam_list.batch_slot_start_idx = 0
        req.beam_list.prompt_lens = torch.tensor([5, 5, 5], device=device)
        req.finished = Mock(return_value=True)
        batch.reqs = [req]

        expected_kv = [105, 106, 205, 206, 305, 306]
        expected_pool = [0, 1, 2]
        mock_collect_kv.return_value = (
            torch.tensor(expected_kv, device=device),
            torch.tensor(expected_pool, device=device),
        )

        scheduler = create_mock_scheduler()

        SchedulerBeamSearchProcessorMixin._cache_finished_beam_search(scheduler, batch)

        mock_collect_kv.assert_called_once()

        kv_indices = (
            scheduler.token_to_kv_pool_allocator.free.call_args[0][0].cpu().tolist()
        )
        self.assertEqual(kv_indices, expected_kv)

        pool_indices = scheduler.req_to_token_pool.free.call_args[0][0]
        self.assertEqual(pool_indices, [0, 1, 2])

        mock_release_kv_cache.assert_called_once()


class TestCopyKVCacheForBeams(unittest.TestCase):
    """Test KV cache copying for beams with different scenarios."""

    def test_copy_kvcache_for_beams_single_group(self):
        """Test KV cache copying when all beams have same dimensions.

        Note: _copy_kvcache_group is not mocked because it's a simple function
        that directly tests the actual implementation.
        """
        device = torch.device("cpu")
        scheduler = create_mock_scheduler()

        scheduler.req_to_token_pool.req_to_token = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
                [30, 31, 32, 33, 34, 35],
                [40, 41, 42, 43, 44, 45],
            ],
            dtype=torch.int64,
            device=device,
        )

        src_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
        dst_pool_indices = torch.tensor([2, 3], dtype=torch.int64, device=device)
        prompt_lens = torch.tensor([2, 2], dtype=torch.int64, device=device)
        seq_lens_batch = torch.tensor([4, 4], dtype=torch.int64, device=device)

        result = SchedulerBeamSearchProcessorMixin._copy_kvcache_for_beams(
            scheduler,
            src_pool_indices,
            dst_pool_indices,
            prompt_lens,
            seq_lens_batch,
            device,
        )

        expected = [12, 13, 22, 23]
        self.assertEqual(result.cpu().tolist(), expected)

    def test_copy_kvcache_for_beams_multiple_groups(self):
        """Test KV cache copying when beams have different dimensions.

        Note: _copy_kvcache_group is not mocked because it's a simple function
        that directly tests the actual implementation.
        """
        device = torch.device("cpu")
        scheduler = create_mock_scheduler()

        scheduler.req_to_token_pool.req_to_token = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17],
                [20, 21, 22, 23, 24, 25, 26, 27],
                [30, 31, 32, 33, 34, 35, 36, 37],
                [40, 41, 42, 43, 44, 45, 46, 47],
            ],
            dtype=torch.int64,
            device=device,
        )

        src_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
        dst_pool_indices = torch.tensor([2, 3], dtype=torch.int64, device=device)
        prompt_lens = torch.tensor([2, 3], dtype=torch.int64, device=device)
        seq_lens_batch = torch.tensor([4, 6], dtype=torch.int64, device=device)

        result = SchedulerBeamSearchProcessorMixin._copy_kvcache_for_beams(
            scheduler,
            src_pool_indices,
            dst_pool_indices,
            prompt_lens,
            seq_lens_batch,
            device,
        )

        expected = [12, 13, 23, 24, 25]
        self.assertEqual(result.cpu().tolist(), expected)


class TestCopyKVCacheGroup(unittest.TestCase):
    """Test KV cache copying for beam search (_copy_kvcache_group method)."""

    def test_copy_kvcache_group_basic(self):
        """Test basic KV cache copying for a single group."""
        device = torch.device("cpu")

        # Create mock scheduler
        scheduler = create_mock_scheduler()
        scheduler.req_to_token_pool.req_to_token = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            ],
            dtype=torch.int64,
            device=device,
        )

        src_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
        dst_indices = torch.tensor([1, 2], dtype=torch.int64, device=device)
        prefix_len = 5
        seq_len = 8

        result = SchedulerBeamSearchProcessorMixin._copy_kvcache_group(
            scheduler, src_indices, dst_indices, prefix_len, seq_len
        )

        copied_data = scheduler.req_to_token_pool.req_to_token[1, prefix_len:seq_len]
        expected = torch.tensor([15, 16, 17], dtype=torch.int64, device=device)
        self.assertTrue(torch.equal(copied_data, expected))

        copied_data = scheduler.req_to_token_pool.req_to_token[2, prefix_len:seq_len]
        expected = torch.tensor([25, 26, 27], dtype=torch.int64, device=device)
        self.assertTrue(torch.equal(copied_data, expected))

        self.assertEqual(result.cpu().tolist(), [15, 16, 17, 25, 26, 27])

    def test_copy_kvcache_group_with_deduplication(self):
        """Test KV cache copying with duplicate source indices."""
        device = torch.device("cpu")

        scheduler = create_mock_scheduler()
        scheduler.req_to_token_pool.req_to_token = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
                [30, 31, 32, 33, 34, 35],
            ],
            dtype=torch.int64,
            device=device,
        )

        src_indices = torch.tensor([0, 0], dtype=torch.int64, device=device)
        dst_indices = torch.tensor([1, 2], dtype=torch.int64, device=device)
        prefix_len = 2
        seq_len = 4

        result = SchedulerBeamSearchProcessorMixin._copy_kvcache_group(
            scheduler, src_indices, dst_indices, prefix_len, seq_len
        )
        self.assertEqual(result.cpu().tolist(), [12, 13])


class TestCollectBeamReqDecodeKVIndices(unittest.TestCase):
    """Test collecting decode KV indices for beam requests."""

    def test_collect_beam_req_decode_kv_indices(self):
        """Test collecting decode portion KV indices."""
        device = torch.device("cpu")

        batch = Mock()
        batch.device = device
        batch.req_pool_indices = torch.tensor([0, 1, 2, 3, 4], device=device)
        batch.seq_lens = torch.tensor([7, 7, 8, 8, 8], device=device)

        req1 = Mock()
        req1.beam_width = 2
        req1.beam_list = Mock()
        req1.beam_list.batch_slot_start_idx = 0
        req1.beam_list.prompt_lens = torch.tensor([5, 5], device=device)

        req2 = Mock()
        req2.beam_width = 3
        req2.beam_list = Mock()
        req2.beam_list.batch_slot_start_idx = 2
        req2.beam_list.prompt_lens = torch.tensor([6, 6, 6], device=device)

        finished_reqs = [req1, req2]

        scheduler = create_mock_scheduler()
        scheduler.req_to_token_pool.req_to_token = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [10, 11, 12, 13, 14, 15, 26, 27, 28, 29],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                [30, 31, 32, 33, 34, 35, 46, 47, 48, 49],
                [30, 31, 32, 33, 34, 35, 56, 57, 58, 59],
            ],
            dtype=torch.int64,
            device=device,
        )

        kv_indices, pool_indices = (
            SchedulerBeamSearchProcessorMixin._collect_beam_req_decode_kv_indices(
                scheduler, batch, finished_reqs
            )
        )

        self.assertEqual(
            kv_indices.cpu().tolist(), [15, 16, 26, 36, 37, 46, 47, 56, 57]
        )
        self.assertEqual(len(pool_indices), 5)


class TestCalculateBeamScore(unittest.TestCase):
    """Test beam search score calculation methods."""

    def test_calculate_beam_score_basic(self):
        """Test basic beam score calculation without length penalty."""
        cum_logprob = -10.0
        seq_len = 5
        length_penalty = 1.0

        score = SchedulerBeamSearchProcessorMixin._calculate_beam_score(
            cum_logprob, seq_len, length_penalty
        )

        # Score should be cum_logprob / seq_len^1.0 = -10.0 / 5 = -2.0
        self.assertAlmostEqual(score, -2.0, places=5)

    def test_calculate_beam_score_with_length_penalty(self):
        """Test beam score calculation with different length penalties."""
        cum_logprob = -10.0
        seq_len = 4

        # Test with length_penalty > 1.0 (favor longer sequences)
        score = SchedulerBeamSearchProcessorMixin._calculate_beam_score(
            cum_logprob, seq_len, length_penalty=2.0
        )
        expected = -10.0 / (4**2.0)  # -10.0 / 16 = -0.625
        self.assertAlmostEqual(score, expected, places=5)

        # Test with length_penalty < 1.0 (favor shorter sequences)
        score = SchedulerBeamSearchProcessorMixin._calculate_beam_score(
            cum_logprob, seq_len, length_penalty=0.5
        )
        expected = -10.0 / (4**0.5)  # -10.0 / 2 = -5.0
        self.assertAlmostEqual(score, expected, places=5)

    def test_calculate_beam_score_with_eos_check(self):
        """Test beam score calculation with EOS token handling."""
        # Create a beam with EOS token at the end
        beam = BeamSearchSequence(
            tokens=[1, 2, 3, 50256],  # 50256 is EOS token
            cum_logprob=-8.0,
        )
        stop_token_ids = {50256}

        score = SchedulerBeamSearchProcessorMixin._calculate_beam_score_with_eos_check(
            beam, stop_token_ids, length_penalty=1.0
        )

        # Should use seq_len=3 (excluding EOS), so score = -8.0 / 3
        expected = -8.0 / 3.0
        self.assertAlmostEqual(score, expected, places=5)

    def test_calculate_beam_score_without_eos(self):
        """Test beam score calculation when last token is not EOS."""
        beam = BeamSearchSequence(
            tokens=[1, 2, 3, 4],
            cum_logprob=-12.0,
        )
        stop_token_ids = {50256}

        score = SchedulerBeamSearchProcessorMixin._calculate_beam_score_with_eos_check(
            beam, stop_token_ids, length_penalty=1.0
        )

        # Should use seq_len=4 (no EOS to exclude), so score = -12.0 / 4
        expected = -12.0 / 4.0
        self.assertAlmostEqual(score, expected, places=5)


if __name__ == "__main__":
    unittest.main()
