"""Regression contracts for DeepSeek/GLM ModelSlim shared experts on NPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import _get_shared_expert_fp8_block_size
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestModelSlimSharedExpertsNPU(CustomTestCase):
    def test_npu_mxfp8_does_not_require_generic_quant_config(self):
        # ModelSlimLinearMethod intentionally has no ``quant_config`` member.
        modelslim_method = SimpleNamespace()
        with patch.object(deepseek_v2, "_is_npu", True):
            block_size = _get_shared_expert_fp8_block_size(
                modelslim_method, modelslim_method
            )

        self.assertIsNone(block_size)

    def test_non_npu_preserves_matching_block_size_validation(self):
        gate_up = SimpleNamespace(
            quant_config=SimpleNamespace(weight_block_size=[128, 128])
        )
        down = SimpleNamespace(
            quant_config=SimpleNamespace(weight_block_size=[128, 128])
        )
        with patch.object(deepseek_v2, "_is_npu", False):
            block_size = _get_shared_expert_fp8_block_size(gate_up, down)

        self.assertEqual(block_size, [128, 128])


if __name__ == "__main__":
    unittest.main()
