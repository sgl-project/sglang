"""Regression tests for multimodal piecewise CUDA graph opt-ins."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.cuda_graph_hook import (
    apply_cuda_graph_compatibility,
    apply_inkling_prefill_cuda_graph_default,
)
from sglang.srt.arg_groups.model_hook import handle_model_capability_adjustments
from sglang.srt.arg_groups.model_overrides.inkling import _inkling_overrides
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.configs.embedding_model_spec import resolve_embedding_model_spec
from sglang.srt.configs.model_config import (
    AttentionArch,
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
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestMultimodalPiecewiseCudaGraph(CustomTestCase):
    def _make_prefill_runner(self, backend):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner._qwen_bcg_hc_sidechannel = False
        runner._qwen_bcg_mtp_draft = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = backend
        runner.has_mha_companion_layers = backend == Backend.BREAKABLE
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_context_size = None
        runner.max_num_tokens = 16
        return runner

    def _make_multimodal_forward_batch(self):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=[object()],
            contains_mm_inputs=lambda: True,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=[1, 2, 3, 4],
            extend_prefix_lens_cpu=[0],
        )

    def test_trtllm_mla_stays_on_breakable(self):
        args = ServerArgs(model_path="dummy")
        # trtllm_mla keeps breakable, which
        # now serves MLA by falling back to the flashinfer MLA impl for extend.
        args._model_config = SimpleNamespace(
            is_multimodal_piecewise_cuda_graph_supported=True,
            is_multimodal=False,
            is_multimodal_breakable_cuda_graph_supported=False,
            attention_arch=AttentionArch.MLA,
            hf_config=SimpleNamespace(architectures=["DeepseekV2ForCausalLM"]),
        )
        args.cuda_graph_config = CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.BREAKABLE)
        )
        args._cuda_graph_config_locked = set()

        args.attention_backend = "trtllm_mla"

        apply_cuda_graph_compatibility(args)

        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )

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

    def test_inkling_full_prefill_graph_follows_the_attention_backend(self):
        # (platform, explicit backend flags) -> (attention backend, prefill graph)
        cases = [
            (dict(is_sm90=True, is_sm100=False), {}, "fa4", Backend.FULL),
            (dict(is_sm90=False, is_sm100=True), {}, "fa4", Backend.FULL),
            (dict(is_sm90=False, is_sm100=False), {}, "triton", None),
            (
                dict(is_sm90=True, is_sm100=False),
                dict(attention_backend="triton"),
                None,
                None,
            ),
            (
                dict(is_sm90=True, is_sm100=False),
                dict(attention_backend="fa4", prefill_attention_backend="triton"),
                None,
                None,
            ),
        ]
        for platform, flags, attention_backend, prefill_backend in cases:
            with self.subTest(platform=platform, flags=flags):
                args = ServerArgs(model_path="dummy", **flags)
                args._model_config = SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=["InklingForConditionalGeneration"]
                    ),
                )
                with override_platform(is_cuda=True, **platform):
                    overrides = _inkling_overrides(args, None)
                    apply_inkling_prefill_cuda_graph_default(args)

                self.assertEqual(overrides.get("attention_backend"), attention_backend)
                self.assertEqual(
                    resolution_result(args, "cuda_graph_backend_prefill"),
                    prefill_backend,
                )

    def test_embedding_gemma_forces_breakable_prefill(self):
        args = ServerArgs(model_path="dummy")
        args._model_config = SimpleNamespace(
            is_embedding_gemma=True,
            joint_head_config=None,
            is_multimodal=False,
            context_len=2048,
            hf_config=SimpleNamespace(architectures=["Gemma3TextModel"]),
        )
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL),
            prefill=PhaseConfig(backend=Backend.FULL),
        )
        args.disable_radix_cache = False
        args.chunked_prefill_size = 2048

        with (
            override_platform(is_cuda=True),
        ):
            handle_model_capability_adjustments(args)

        self.assertTrue(resolution_result(args, "disable_radix_cache"))
        self.assertEqual(resolution_result(args, "chunked_prefill_size"), -1)
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").decode.backend,
            Backend.DISABLED,
        )
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )

    def test_encoder_embedding_model_enables_embedding_mode_without_flag(self):
        args = ServerArgs(model_path="dummy")
        args.is_embedding = False
        args._model_config = SimpleNamespace(
            embedding_model_spec=resolve_embedding_model_spec(
                ["BertModel"],
                is_embedding_requested=False,
                is_embedding_gemma=False,
            ),
            joint_head_config=None,
            is_multimodal=False,
            hf_config=SimpleNamespace(architectures=["BertModel"]),
        )

        if True:  # the record already carries the seeded configuration
            handle_model_capability_adjustments(args)

        self.assertTrue(resolution_result(args, "is_embedding"))


if __name__ == "__main__":
    unittest.main()
