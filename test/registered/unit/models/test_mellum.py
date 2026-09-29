"""Unit tests for the Mellum model implementation."""

import unittest

import torch

from sglang.srt.models.mellum import MellumForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestMellumForCausalLM(CustomTestCase):
    def test_prepare_positions(self):
        model = MellumForCausalLM.__new__(MellumForCausalLM)
        positions = torch.tensor([[0, 1]], dtype=torch.int64).t()

        model.use_fused_qk_norm_rope = False
        self.assertIs(
            model._prepare_positions(positions, torch.device("cpu")), positions
        )

        model.use_fused_qk_norm_rope = True
        fused_positions = model._prepare_positions(positions, torch.device("cpu"))

        self.assertEqual(fused_positions.dtype, torch.int32)
        self.assertTrue(fused_positions.is_contiguous())
        self.assertEqual(fused_positions.shape, (2,))


if __name__ == "__main__":
    unittest.main()
