"""CPU tests for speculative mRoPE position construction."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _forward_batch(batch_size: int) -> ForwardBatch:
    return ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=batch_size,
        input_ids=torch.empty(0, dtype=torch.int64),
        req_pool_indices=torch.arange(batch_size, dtype=torch.int32),
        seq_lens=torch.ones(batch_size, dtype=torch.int32),
        out_cache_loc=torch.empty(0, dtype=torch.int64),
        seq_lens_sum=batch_size,
    )


class TestSpecMropePositions(CustomTestCase):
    def test_text_only_ragged_positions_stay_flat(self):
        batch_size = 7
        positions = torch.arange(32, dtype=torch.int32)
        forward_batch = _forward_batch(batch_size)
        batch = SimpleNamespace(multimodal_inputs=[None] * batch_size)
        model_runner = SimpleNamespace(device=torch.device("cpu"))

        forward_batch.compute_spec_mrope_positions(
            model_runner, batch, seq_positions=positions
        )

        expected = positions.to(dtype=torch.int64).unsqueeze(0).repeat(3, 1)
        torch.testing.assert_close(forward_batch.mrope_positions, expected)
        self.assertEqual(forward_batch.mrope_positions.shape, (3, 32))
        self.assertEqual(forward_batch.mrope_positions.dtype, torch.int64)

    def test_text_only_rectangular_positions_are_unchanged(self):
        batch_size = 2
        positions = torch.tensor([[4, 5, 6], [10, 11, 12]], dtype=torch.int64)
        forward_batch = _forward_batch(batch_size)
        batch = SimpleNamespace(multimodal_inputs=[None] * batch_size)
        model_runner = SimpleNamespace(device=torch.device("cpu"))

        forward_batch.compute_spec_mrope_positions(
            model_runner, batch, seq_positions=positions
        )

        expected = positions.flatten().unsqueeze(0).repeat(3, 1)
        torch.testing.assert_close(forward_batch.mrope_positions, expected)

    def test_multimodal_rectangular_delta_behavior_is_unchanged(self):
        batch_size = 2
        positions = torch.tensor([[4, 5, 6], [10, 11, 12]], dtype=torch.int64)
        forward_batch = _forward_batch(batch_size)
        batch = SimpleNamespace(
            multimodal_inputs=[
                SimpleNamespace(mrope_position_delta=torch.tensor([[2]])),
                SimpleNamespace(mrope_position_delta=torch.tensor([[7]])),
            ]
        )
        model_runner = SimpleNamespace(device=torch.device("cpu"))

        forward_batch.compute_spec_mrope_positions(
            model_runner, batch, seq_positions=positions
        )

        expected_1d = torch.tensor([6, 7, 8, 17, 18, 19], dtype=torch.int64)
        expected = expected_1d.unsqueeze(0).repeat(3, 1)
        torch.testing.assert_close(forward_batch.mrope_positions, expected)


if __name__ == "__main__":
    unittest.main()
