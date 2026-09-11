"""Device-portable sampling contracts for linear UNO."""

import unittest
from unittest.mock import patch

import torch
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.hardware_backend.npu.graph_runner.uno_npu_graph_runner import (
    UnoNPUGraphRunner,
)
from sglang.srt.speculative.uno_utils import _dense_rejection_torch, _top_k
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnoDeviceSampling(CustomTestCase):
    def test_uno_npu_graph_runner_keeps_npu_capture_and_replay(self):
        self.assertTrue(issubclass(UnoNPUGraphRunner, NPUGraphRunner))
        self.assertIs(UnoNPUGraphRunner.execute, NPUGraphRunner.execute)

    def test_torch_topk_fallback_matches_reference(self):
        logits = torch.tensor([[0.1, 0.9, -0.2, 0.4]], dtype=torch.float32)
        expected_values, expected_indices = torch.topk(logits, k=3, dim=-1)

        with patch("sglang.srt.speculative.uno_utils._flashinfer_top_k", None):
            values, indices = _top_k(logits, 3)

        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(indices, expected_indices)

    def test_dense_rejection_fallback_accepts_prefix_and_samples_residual(self):
        candidates = torch.tensor([[7, 1, 2]], dtype=torch.int64)
        target_probs = torch.tensor(
            [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
        )
        draft_probs = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

        accepted, bonus = _dense_rejection_torch(
            candidates=candidates,
            target_probs=target_probs,
            draft_probs=draft_probs,
            accept_uniforms=torch.tensor([[0.5, 0.5, 0.5]]),
            final_uniforms=torch.tensor([0.0]),
        )

        self.assertEqual(accepted.tolist(), [1])
        self.assertEqual(bonus.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
