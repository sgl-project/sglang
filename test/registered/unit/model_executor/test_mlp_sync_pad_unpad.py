"""Unit tests for the DP-attention MLP-sync pad/unpad round-trip.

``prepare_mlp_sync_batch`` pads per-request tensors (positions / seq_lens /
req_pool_indices) by appending dummy rows after the real ones so all DP ranks
agree on tensor shapes. ``post_forward_mlp_sync_batch`` must slice them back so
post-forward consumers — seeded sampling (which asserts positions rows ==
sampling rows), ngram token-table updates — never see the padding.

Pure dataclass logic — CPU only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _mock_model_runner(seq_len_fill_value: int = 1) -> MagicMock:
    runner = MagicMock()
    runner.attn_backend.get_cuda_graph_seq_len_fill_value.return_value = (
        seq_len_fill_value
    )
    return runner


def _logits_output(num_rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        next_token_logits=torch.randn(num_rows, 16), hidden_states=None
    )


class TestMlpSyncPadUnpad(CustomTestCase):
    def test_decode_post_forward_unpads_per_request_tensors(self):
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=3,
            input_ids=torch.tensor([11, 12, 13]),
            req_pool_indices=torch.tensor([5, 6, 7]),
            seq_lens=torch.tensor([7, 8, 9]),
            out_cache_loc=torch.tensor([0, 1, 2]),
            seq_lens_sum=24,
            positions=torch.tensor([6, 7, 8]),
            seq_lens_cpu=torch.tensor([7, 8, 9]),
            lora_ids=[None, None, None],
        )
        # Mirror the decode arm of prepare_mlp_sync_batch: record the original
        # batch size, adopt the synced (padded) one, then pad the inputs.
        padded = 5
        fb._original_batch_size = fb.batch_size
        fb.batch_size = padded
        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=padded, bs=padded)

        # Padding appends dummy rows after the real ones.
        self.assertEqual(fb.positions.shape[0], padded)
        self.assertEqual(fb.seq_lens.shape[0], padded)
        self.assertEqual(fb.req_pool_indices.shape[0], padded)
        torch.testing.assert_close(fb.positions[:3], torch.tensor([6, 7, 8]))

        logits_output = _logits_output(padded)
        fb.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(fb.batch_size, 3)
        torch.testing.assert_close(fb.positions, torch.tensor([6, 7, 8]))
        torch.testing.assert_close(fb.seq_lens, torch.tensor([7, 8, 9]))
        torch.testing.assert_close(fb.req_pool_indices, torch.tensor([5, 6, 7]))
        torch.testing.assert_close(fb.seq_lens_cpu, torch.tensor([7, 8, 9]))
        self.assertEqual(logits_output.next_token_logits.shape[0], 3)
        # Seeded sampling asserts positions rows == sampled (real) rows.
        self.assertEqual(fb.positions.shape[0], fb.batch_size)

    def test_extend_post_forward_unpads_positions(self):
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=torch.arange(7),
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([3, 4]),
            out_cache_loc=torch.arange(7),
            seq_lens_sum=7,
            positions=torch.tensor([0, 1, 2, 0, 1, 2, 3]),
            seq_lens_cpu=torch.tensor([3, 4]),
            lora_ids=[None, None],
        )
        # Extend keeps batch_size; only token-level tensors get padded.
        fb._original_batch_size = fb.batch_size
        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=10, bs=2)

        self.assertEqual(fb.positions.shape[0], 10)

        logits_output = _logits_output(10)
        fb.post_forward_mlp_sync_batch(logits_output)

        torch.testing.assert_close(fb.positions, torch.tensor([0, 1, 2, 0, 1, 2, 3]))
        torch.testing.assert_close(fb.seq_lens, torch.tensor([3, 4]))
        # sample() derives prefill sampling positions from seq_lens - 1, so the
        # row count must match the real request count.
        self.assertEqual((fb.seq_lens - 1).shape[0], fb.batch_size)

    def test_draft_extend_dummy_request_pads_cpu_and_gpu_lens(self):
        spec_info = MagicMock()
        spec_info.num_tokens_per_req = 4
        spec_info.is_draft_input.return_value = False
        fb = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=1,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            extend_seq_lens=torch.empty(0, dtype=torch.int32),
            extend_prefix_lens=torch.empty(0, dtype=torch.int64),
            extend_seq_lens_cpu=[],
            extend_prefix_lens_cpu=[],
            extend_logprob_start_lens_cpu=[],
            spec_info=spec_info,
        )

        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=4, bs=1)

        torch.testing.assert_close(
            fb.extend_seq_lens, torch.tensor([4], dtype=torch.int32)
        )
        torch.testing.assert_close(fb.extend_prefix_lens, torch.tensor([0]))
        self.assertEqual(fb.extend_seq_lens_cpu, [4])
        self.assertEqual(fb.extend_prefix_lens_cpu, [0])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0])

    def test_empty_draft_keeps_normalized_dummy_logits(self):
        spec_info = MagicMock()
        spec_info.is_draft_input.return_value = True
        fb = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=1,
            input_ids=torch.tensor([0]),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([1]),
            seq_lens_sum=1,
            out_cache_loc=torch.tensor([0]),
            positions=torch.tensor([0]),
            seq_lens_cpu=torch.tensor([1]),
            spec_info=spec_info,
        )
        # Overlap may retain a scheduler dummy position while the previous
        # batch's draft state is empty/stale.
        fb._original_batch_size = 1
        fb._original_num_tokens = 1
        fb.hidden_states_backup = torch.empty((0, 8))
        logits_output = SimpleNamespace(
            next_token_logits=torch.randn(1, 16), hidden_states=torch.randn(1, 8)
        )

        fb.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(logits_output.next_token_logits.shape[0], 1)
        self.assertEqual(logits_output.hidden_states.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
