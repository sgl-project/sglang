"""Regression tests for multimodal piecewise CUDA graph opt-ins."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import (
    is_multimodal_piecewise_cuda_graph_supported,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMultimodalPiecewiseCudaGraph(CustomTestCase):
    def _make_prefill_runner(self, backend):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.prefill_backend_name = backend
        runner.has_mha_companion_layers = backend == Backend.BREAKABLE
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.max_num_tokens = 16
        return runner

    def _make_multimodal_forward_batch(self):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=[object()],
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=[1, 2, 3, 4],
            extend_prefix_lens_cpu=[0],
        )

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

    def test_multimodal_inputs_keep_tc_piecewise_prefill_enabled(self):
        runner = self._make_prefill_runner(Backend.TC_PIECEWISE)

        self.assertTrue(runner.can_run_graph(self._make_multimodal_forward_batch()))

    def test_multimodal_inputs_keep_breakable_prefill_enabled(self):
        runner = self._make_prefill_runner(Backend.BREAKABLE)

        self.assertTrue(runner.can_run_graph(self._make_multimodal_forward_batch()))

    def test_breakable_prefill_rejects_nonzero_prefix(self):
        runner = self._make_prefill_runner(Backend.BREAKABLE)
        forward_batch = self._make_multimodal_forward_batch()
        forward_batch.extend_prefix_lens_cpu = [1]

        self.assertFalse(runner.can_run_graph(forward_batch))


if __name__ == "__main__":
    unittest.main()
