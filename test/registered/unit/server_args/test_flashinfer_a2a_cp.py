import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferA2APrefillCP(CustomTestCase):
    def _args(
        self,
        *,
        enable_dp_attention: bool,
        dp_size: int,
        enable_prefill_cp: bool,
        attn_cp_size: int,
    ) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        args.tp_size = 4
        args.dp_size = dp_size
        args.enable_dp_attention = enable_dp_attention
        args.enable_prefill_cp = enable_prefill_cp
        args.attn_cp_size = attn_cp_size
        args.moe_a2a_backend = "flashinfer"
        args.moe_runner_backend = "flashinfer_trtllm_routed"
        return args

    def test_parallelism_support_truth_table(self):
        for case in (
            dict(
                enable_dp_attention=True,
                dp_size=4,
                enable_prefill_cp=False,
                attn_cp_size=1,
                expected=True,
            ),
            dict(
                enable_dp_attention=False,
                dp_size=1,
                enable_prefill_cp=True,
                attn_cp_size=4,
                expected=True,
            ),
            dict(
                enable_dp_attention=False,
                dp_size=1,
                enable_prefill_cp=True,
                attn_cp_size=2,
                expected=False,
            ),
            dict(
                enable_dp_attention=False,
                dp_size=1,
                enable_prefill_cp=False,
                attn_cp_size=1,
                expected=False,
            ),
        ):
            with self.subTest(case=case):
                args = self._args(**{k: v for k, v in case.items() if k != "expected"})
                self.assertEqual(
                    args._supports_flashinfer_a2a_parallelism(), case["expected"]
                )

    def test_handle_a2a_moe_admits_full_prefill_cp_with_routed_runner(self):
        args = self._args(
            enable_dp_attention=False,
            dp_size=1,
            enable_prefill_cp=True,
            attn_cp_size=4,
        )
        args.get_model_config = lambda: SimpleNamespace(nvfp4_moe_meta=None)

        args._handle_a2a_moe()

        view = resolved_view(args)
        self.assertEqual(view.moe_a2a_backend, "flashinfer")
        self.assertEqual(view.moe_runner_backend, "flashinfer_trtllm_routed")

    def test_handle_a2a_moe_rejects_partial_prefill_cp_with_diagnostic(self):
        args = self._args(
            enable_dp_attention=False,
            dp_size=1,
            enable_prefill_cp=True,
            attn_cp_size=2,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "full prefill context parallelism with attn_cp_size == tp_size",
        ):
            args._handle_a2a_moe()

    def _breakable_args(
        self,
        *,
        enable_prefill_cp: bool,
        attn_cp_size: int,
        cp_strategy: str,
        attention_backend: str,
        moe_a2a_backend: str,
        moe_runner_backend: str,
        architecture: str = "GptOssForCausalLM",
    ) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        args.tp_size = 4
        args.dp_size = 1
        args.enable_dp_attention = False
        args.enable_prefill_cp = enable_prefill_cp
        args.attn_cp_size = attn_cp_size
        args.cp_strategy = cp_strategy
        args.attention_backend = attention_backend
        args.prefill_attention_backend = None
        args.decode_attention_backend = None
        args.moe_a2a_backend = moe_a2a_backend
        args.moe_runner_backend = moe_runner_backend
        args.dcp_size = 1
        args.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture]),
            is_multimodal=False,
            is_multimodal_breakable_cuda_graph_supported=False,
        )
        args.cuda_graph_config = CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.BREAKABLE)
        )
        return args

    def test_breakable_cuda_graph_support_truth_table(self):
        cases = (
            dict(
                name="validated",
                enable_prefill_cp=True,
                attn_cp_size=4,
                cp_strategy="zigzag",
                attention_backend="trtllm_mha",
                moe_a2a_backend="flashinfer",
                moe_runner_backend="flashinfer_trtllm_routed",
                cp_allowed=True,
                a2a_allowed=True,
                graph_allowed=True,
            ),
            dict(
                name="partial_cp",
                enable_prefill_cp=True,
                attn_cp_size=2,
                cp_strategy="zigzag",
                attention_backend="trtllm_mha",
                moe_a2a_backend="flashinfer",
                moe_runner_backend="flashinfer_trtllm_routed",
                cp_allowed=False,
                a2a_allowed=False,
                graph_allowed=False,
            ),
            dict(
                name="interleave",
                enable_prefill_cp=True,
                attn_cp_size=4,
                cp_strategy="interleave",
                attention_backend="trtllm_mha",
                moe_a2a_backend="flashinfer",
                moe_runner_backend="flashinfer_trtllm_routed",
                cp_allowed=False,
                a2a_allowed=False,
                graph_allowed=False,
            ),
            dict(
                name="wrong_attention",
                enable_prefill_cp=True,
                attn_cp_size=4,
                cp_strategy="zigzag",
                attention_backend="flashinfer",
                moe_a2a_backend="flashinfer",
                moe_runner_backend="flashinfer_trtllm_routed",
                cp_allowed=False,
                a2a_allowed=False,
                graph_allowed=False,
            ),
            dict(
                name="other_a2a",
                enable_prefill_cp=True,
                attn_cp_size=4,
                cp_strategy="zigzag",
                attention_backend="trtllm_mha",
                moe_a2a_backend="deepep",
                moe_runner_backend="deep_gemm",
                cp_allowed=True,
                a2a_allowed=False,
                graph_allowed=False,
            ),
            dict(
                name="wrong_runner",
                enable_prefill_cp=True,
                attn_cp_size=4,
                cp_strategy="zigzag",
                attention_backend="trtllm_mha",
                moe_a2a_backend="flashinfer",
                moe_runner_backend="triton",
                cp_allowed=True,
                a2a_allowed=False,
                graph_allowed=False,
            ),
            dict(
                name="no_cp_no_a2a",
                enable_prefill_cp=False,
                attn_cp_size=1,
                cp_strategy="zigzag",
                attention_backend="trtllm_mha",
                moe_a2a_backend="none",
                moe_runner_backend="triton",
                cp_allowed=False,
                a2a_allowed=True,
                graph_allowed=True,
            ),
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                kwargs = {
                    key: value
                    for key, value in case.items()
                    if key
                    not in {
                        "name",
                        "cp_allowed",
                        "a2a_allowed",
                        "graph_allowed",
                    }
                }
                args = self._breakable_args(**kwargs)
                self.assertEqual(
                    args._supports_breakable_prefill_cp(), case["cp_allowed"]
                )
                self.assertEqual(
                    args._supports_breakable_moe_a2a(), case["a2a_allowed"]
                )
                with patch.object(ServerArgs, "use_mla_backend", return_value=False):
                    args._disable_breakable_cudagraph_if_incompatible()
                self.assertEqual(
                    args.cuda_graph_config.prefill.backend,
                    (Backend.BREAKABLE if case["graph_allowed"] else Backend.DISABLED),
                )

    def test_breakable_cuda_graph_rejects_other_model_architectures(self):
        args = self._breakable_args(
            enable_prefill_cp=True,
            attn_cp_size=4,
            cp_strategy="zigzag",
            attention_backend="trtllm_mha",
            moe_a2a_backend="flashinfer",
            moe_runner_backend="flashinfer_trtllm_routed",
            architecture="LlamaForCausalLM",
        )

        self.assertFalse(args._supports_breakable_prefill_cp())
        self.assertFalse(args._supports_breakable_moe_a2a())
        with patch.object(ServerArgs, "use_mla_backend", return_value=False):
            args._disable_breakable_cudagraph_if_incompatible()
        self.assertEqual(
            args.cuda_graph_config.prefill.backend,
            Backend.DISABLED,
        )


if __name__ == "__main__":
    unittest.main()
