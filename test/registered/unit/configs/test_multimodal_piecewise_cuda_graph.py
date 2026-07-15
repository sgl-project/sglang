"""Regression tests for multimodal piecewise CUDA graph opt-ins."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import (
    is_multimodal_mla_large_prefill_cuda_graph_supported,
    is_multimodal_piecewise_cuda_graph_supported,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
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

    def test_only_kimi_has_the_larger_mla_prefill_bucket_opt_in(self):
        self.assertTrue(
            is_multimodal_mla_large_prefill_cuda_graph_supported(
                ["KimiK25ForConditionalGeneration"]
            )
        )
        self.assertFalse(
            is_multimodal_mla_large_prefill_cuda_graph_supported(
                ["MiniMaxM3SparseForConditionalGeneration"]
            )
        )

    def test_supported_multimodal_model_upgrades_default_to_tc_piecewise(self):
        args = ServerArgs(model_path="dummy")
        args.model_config = SimpleNamespace(
            is_multimodal_piecewise_cuda_graph_supported=True
        )
        args.cuda_graph_config = CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.BREAKABLE)
        )
        args._cuda_graph_config_locked = set()

        with patch.object(
            ServerArgs, "_disable_tc_piecewise_cudagraph_if_incompatible"
        ) as disable_if_incompatible:
            args._apply_cuda_graph_compatibility()

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.TC_PIECEWISE)
        disable_if_incompatible.assert_called_once()


if __name__ == "__main__":
    unittest.main()
