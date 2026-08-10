"""Regression tests for multimodal piecewise CUDA graph opt-ins."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.embedding_model_spec import resolve_embedding_model_spec
from sglang.srt.configs.model_config import (
    is_multimodal_piecewise_cuda_graph_supported,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
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
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = backend
        runner.has_mha_companion_layers = backend == Backend.BREAKABLE
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
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

        with (
            patch.object(
                ServerArgs, "_disable_tc_piecewise_cudagraph_if_incompatible"
            ) as disable_if_incompatible,
            patch.object(
                args, "_resolved_attention_backends", return_value=("fa3", "fa3")
            ),
        ):
            args._apply_cuda_graph_compatibility()

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.TC_PIECEWISE)
        disable_if_incompatible.assert_called_once()

    def test_trtllm_mla_stays_on_breakable(self):
        args = ServerArgs(model_path="dummy")
        # trtllm_mla skips the tc_piecewise upgrade and keeps breakable, which
        # now serves MLA by falling back to the flashinfer MLA impl for extend.
        args.model_config = SimpleNamespace(
            is_multimodal_piecewise_cuda_graph_supported=True,
            is_multimodal=False,
            is_multimodal_breakable_cuda_graph_supported=False,
            hf_config=SimpleNamespace(architectures=["DeepseekV2ForCausalLM"]),
        )
        args.cuda_graph_config = CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.BREAKABLE)
        )
        args._cuda_graph_config_locked = set()

        with (
            patch.object(
                args,
                "_resolved_attention_backends",
                return_value=("trtllm_mla", "trtllm_mla"),
            ),
            patch.object(args, "use_mla_backend", return_value=True),
        ):
            args._apply_cuda_graph_compatibility()

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)

    def test_explicit_tc_piecewise_overrides_trtllm_mla_default(self):
        args = ServerArgs(model_path="dummy")
        args.cuda_graph_config = CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.TC_PIECEWISE)
        )
        args._cuda_graph_config_locked = {(Phase.PREFILL, "backend")}

        with patch.object(
            args,
            "_resolved_attention_backends",
            return_value=("trtllm_mla", "trtllm_mla"),
        ):
            args._apply_cuda_graph_compatibility()

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.TC_PIECEWISE)

    def test_multimodal_inputs_keep_tc_piecewise_prefill_enabled(self):
        runner = self._make_prefill_runner(Backend.TC_PIECEWISE)

        self.assertTrue(runner.can_run_graph(self._make_multimodal_forward_batch()))

    def test_multimodal_inputs_keep_breakable_prefill_enabled(self):
        runner = self._make_prefill_runner(Backend.BREAKABLE)

        self.assertTrue(runner.can_run_graph(self._make_multimodal_forward_batch()))

    def test_breakable_prefill_takes_nonzero_prefix_on_cuda_only(self):
        runner = self._make_prefill_runner(Backend.BREAKABLE)
        forward_batch = self._make_multimodal_forward_batch()
        forward_batch.extend_prefix_lens_cpu = [1]

        target = "sglang.srt.model_executor.runner.prefill_cuda_graph_runner.is_cuda"
        with patch(target, return_value=True):
            self.assertTrue(runner.can_run_graph(forward_batch))
        with patch(target, return_value=False):
            self.assertFalse(runner.can_run_graph(forward_batch))

    def test_embedding_gemma_forces_breakable_prefill(self):
        args = ServerArgs(model_path="dummy")
        args.model_config = SimpleNamespace(
            is_embedding_gemma=True,
            is_multimodal=False,
            context_len=2048,
            hf_config=SimpleNamespace(architectures=["Gemma3TextModel"]),
        )
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL),
            prefill=PhaseConfig(backend=Backend.TC_PIECEWISE),
        )
        args.disable_radix_cache = False
        args.chunked_prefill_size = 2048

        with (
            patch.object(args, "get_model_config", return_value=args.model_config),
            patch("sglang.srt.server_args.is_cuda", return_value=True),
        ):
            args._handle_model_capability_adjustments()

        self.assertTrue(args.disable_radix_cache)
        self.assertEqual(args.chunked_prefill_size, -1)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)

    def test_encoder_embedding_model_enables_embedding_mode_without_flag(self):
        args = ServerArgs(model_path="dummy")
        args.is_embedding = False
        args.model_config = SimpleNamespace(
            embedding_model_spec=resolve_embedding_model_spec(
                ["BertModel"],
                is_embedding_requested=False,
                is_embedding_gemma=False,
            ),
            is_multimodal=False,
            hf_config=SimpleNamespace(architectures=["BertModel"]),
        )

        with patch.object(args, "get_model_config", return_value=args.model_config):
            args._handle_model_capability_adjustments()

        self.assertTrue(args.is_embedding)


if __name__ == "__main__":
    unittest.main()
