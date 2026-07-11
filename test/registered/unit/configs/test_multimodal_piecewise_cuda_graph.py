"""Regression tests for multimodal piecewise CUDA graph opt-ins."""

import unittest

from sglang.srt.configs.model_config import (
    is_multimodal_piecewise_cuda_graph_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMultimodalPiecewiseCudaGraph(CustomTestCase):
    def test_kimi_k25_lm_prefill_is_opted_in(self):
        self.assertTrue(
            is_multimodal_piecewise_cuda_graph_supported(
                ["KimiK25ForConditionalGeneration"]
            )
        )

    def test_unknown_multimodal_arch_is_not_opted_in(self):
        self.assertFalse(
            is_multimodal_piecewise_cuda_graph_supported(
                ["UnknownVisionForConditionalGeneration"]
            )
        )


if __name__ == "__main__":
    unittest.main()
