"""Regression test for the DeepSeek AWQ dequantize helper on CUDA."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.models.deepseek_common import utils as deepseek_utils
from sglang.test.test_utils import CustomTestCase


class TestAwqDequantizeFunc(CustomTestCase):
    @patch.object(deepseek_utils, "_is_npu", False)
    @patch.object(deepseek_utils, "_is_hip", False)
    @patch.object(deepseek_utils, "_is_cuda", True)
    def test_cuda_branch_resolves(self):
        """The CUDA branch returns a dequantize callable; DeepSeek AWQ
        checkpoints with a quantized kv_b_proj call it while loading weights."""
        awq_dequantize = deepseek_utils.awq_dequantize_func()

        self.assertTrue(callable(awq_dequantize))


if __name__ == "__main__":
    unittest.main()
