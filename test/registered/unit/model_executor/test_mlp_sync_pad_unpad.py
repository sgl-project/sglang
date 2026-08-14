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
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
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
    def test_init_mlp_sync_metadata_scales_speculative_request_width(self):
        spec_info = SimpleNamespace(
            num_tokens_per_req=4,
            num_tokens_for_logprob_per_req=2,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            input_ids=torch.arange(8),
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([5, 6]),
            out_cache_loc=torch.arange(8),
            seq_lens_sum=11,
            positions=torch.arange(8),
            spec_info=spec_info,
        )
        batch = SimpleNamespace(
            global_num_tokens=[2, 0, 3],
            global_num_tokens_for_logprob=[2, 0, 3],
            can_run_dp_cuda_graph=True,
        )

        fb.init_mlp_sync_metadata(batch, torch.device("cpu"))

        self.assertEqual(fb.original_global_num_tokens_cpu, [2, 0, 3])
        self.assertEqual(fb.global_num_tokens_cpu, [8, 0, 12])
        self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [4, 0, 6])
        torch.testing.assert_close(fb.global_num_tokens_gpu, torch.tensor([8, 0, 12]))
        torch.testing.assert_close(
            fb.global_num_tokens_for_logprob_gpu, torch.tensor([4, 0, 6])
        )
        self.assertTrue(fb.can_run_dp_cuda_graph)

    def test_draft_input_without_hidden_states_can_be_padded(self):
        spec_info = SimpleNamespace(
            is_draft_input=lambda: True,
            hidden_states=None,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=1,
            input_ids=torch.tensor([11]),
            req_pool_indices=torch.tensor([5]),
            seq_lens=torch.tensor([7]),
            out_cache_loc=torch.tensor([0]),
            seq_lens_sum=7,
            positions=torch.tensor([6]),
            seq_lens_cpu=torch.tensor([7]),
            lora_ids=[None],
            spec_info=spec_info,
        )

        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=2, bs=1)

        self.assertIsNone(spec_info.hidden_states)

    def test_dp_cuda_graph_batch_size_uses_raw_request_counts(self):
        fb = SimpleNamespace(original_global_num_tokens_cpu=[3, 11, 7])
        self.assertEqual(DecodeCudaGraphRunner._max_dp_batch_size(fb), 11)

        fb.original_global_num_tokens_cpu = None
        with self.assertRaisesRegex(RuntimeError, "raw per-rank request counts"):
            DecodeCudaGraphRunner._max_dp_batch_size(fb)

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


if __name__ == "__main__":
    unittest.main()
