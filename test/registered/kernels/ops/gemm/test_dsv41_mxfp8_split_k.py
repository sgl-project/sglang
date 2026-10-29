"""B300 dense split-K: numerical accuracy, fallbacks and graph replay."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="2-gpu-gb300")


@unittest.skipUnless(get_device_sm() == 103, "B300 small-batch MXFP8 policy")
class TestDsv41Mxfp8SplitK(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(published_topology())
        torch.manual_seed(42)

    def _operands(self, m=6, n=1792, k=5120):
        from flashinfer import mxfp8_quantize

        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True, alignment=32)
        return x, w, wq, ws

    def _reference(self, x, w):
        from flashinfer import SfLayout, mxfp8_quantize

        def dequantize(t):
            q, s = mxfp8_quantize(
                t,
                is_sf_swizzled_layout=False,
                alignment=32,
                sf_swizzle_layout=SfLayout.layout_linear,
            )
            scale = torch.exp2(s.view(t.shape[0], -1).float() - 127)
            return q.float() * scale.repeat_interleave(32, dim=1)

        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return dequantize(x) @ dequantize(w).t()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    def _linear(self, x, wq, ws, **kwargs):
        return fp8_utils.flashinfer_mxfp8_blockscaled_linear(
            x, wq, ws, backend="cute-dsl", pin_tactic=True, **kwargs
        )

    @torch.inference_mode()
    def test_numerics_and_prequantized_input(self):
        """Check the split reduction against an independent dequantized GEMM."""
        from flashinfer import mxfp8_quantize

        for n in (1792, 1152):
            for m in (1, 2, 4, 6, 8):
                with self.subTest(m=m, n=n):
                    x, w, wq, ws = self._operands(m=m, n=n)
                    reference = self._reference(x, w)
                    with patch.object(
                        fp8_utils,
                        "flashinfer_mm_mxfp8",
                        wraps=fp8_utils.flashinfer_mm_mxfp8,
                    ) as gemm:
                        actual = self._linear(x, wq, ws, use_small_batch_split_k=True)
                    self.assertTrue(gemm.call_args.kwargs["use_split_k"])
                    self.assertTrue(torch.isfinite(actual).all())
                    error = torch.linalg.vector_norm(actual.float() - reference)
                    self.assertLess(
                        (error / torch.linalg.vector_norm(reference)).item(), 0.005
                    )
                    q, sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
                    prequantized = self._linear(
                        q, wq, ws, input_scale=sf, use_small_batch_split_k=True
                    )
                    torch.testing.assert_close(prequantized, actual, rtol=0, atol=0)

    @torch.inference_mode()
    def test_default_and_unsupported_cases_preserve_output(self):
        """The policy cannot change other M/N/K shapes, dtypes or hardware."""
        for m, n, k, dtype in (
            (9, 1792, 5120, torch.bfloat16),
            (1, 128, 5120, torch.bfloat16),
            (1, 1792, 2560, torch.bfloat16),
            (1, 1792, 5120, torch.float16),
        ):
            with self.subTest(m=m, n=n, k=k, dtype=dtype):
                x, _, wq, ws = self._operands(m=m, n=n, k=k)
                x = x.to(dtype)
                expected = self._linear(x, wq, ws)
                with patch.object(
                    fp8_utils,
                    "flashinfer_mm_mxfp8",
                    wraps=fp8_utils.flashinfer_mm_mxfp8,
                ) as gemm:
                    actual = self._linear(x, wq, ws, use_small_batch_split_k=True)
                self.assertFalse(gemm.call_args.kwargs.get("use_split_k", False))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        x, _, wq, ws = self._operands()
        expected = self._linear(x, wq, ws)
        with patch.object(fp8_utils, "get_device_sm", return_value=100):
            actual = self._linear(x, wq, ws, use_small_batch_split_k=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_deterministic_and_batch_invariant_bypass(self):
        """Both consistency controls must retain the original reduction."""
        from sglang.srt.batch_invariant_ops import (
            disable_batch_invariant_mode,
            enable_batch_invariant_mode,
        )

        x, _, wq, ws = self._operands()
        expected = self._linear(x, wq, ws)
        with (
            get_exec().deterministic.override(enable_deterministic_inference=True),
            patch.object(
                fp8_utils, "flashinfer_mm_mxfp8", wraps=fp8_utils.flashinfer_mm_mxfp8
            ) as gemm,
        ):
            actual = self._linear(x, wq, ws, use_small_batch_split_k=True)
        self.assertFalse(gemm.call_args.kwargs.get("use_split_k", False))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        enable_batch_invariant_mode()
        try:
            with patch.object(
                fp8_utils, "flashinfer_mm_mxfp8", wraps=fp8_utils.flashinfer_mm_mxfp8
            ) as gemm:
                actual = self._linear(x, wq, ws, use_small_batch_split_k=True)
            self.assertFalse(gemm.call_args.kwargs.get("use_split_k", False))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            disable_batch_invariant_mode()

    @torch.inference_mode()
    def test_truncated_scales_retain_validation(self):
        """A short swizzled buffer must fail validation before any GEMM launch."""
        from flashinfer import mxfp8_quantize

        x, _, wq, ws = self._operands()
        q, sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        for input_sf, weight_sf in ((sf[:-1], ws), (sf, ws[:-1])):
            for enabled in (False, True):
                with self.subTest(enabled=enabled, input_scale_size=input_sf.numel()):
                    with self.assertRaisesRegex(ValueError, "shape mismatch"):
                        self._linear(
                            q,
                            wq,
                            weight_sf,
                            input_scale=input_sf,
                            use_small_batch_split_k=enabled,
                        )

    def test_model_default_and_optout(self):
        """The unset default enables only target projections; zero opts out."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
        from sglang.test.layer_ut_utils import make_tp1_column_parallel_linear

        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 32],
            scale_fmt="ue8m0",
        )
        with patch.object(
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            decoder = torch.nn.Module()
            decoder.self_attn = torch.nn.Module()
            decoder.self_attn.wqkv_a = make_tp1_column_parallel_linear(
                config, 1792, 5120
            )
            decoder.self_attn.wq_b = make_tp1_column_parallel_linear(config, 1792, 5120)
            decoder.mlp = torch.nn.Module()
            decoder.mlp.shared_experts = torch.nn.Module()
            decoder.mlp.shared_experts.gate_up_proj = make_tp1_column_parallel_linear(
                config, 1152, 5120
            )
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([decoder])
        target = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"), model=model
        )
        projections = (
            decoder.self_attn.wqkv_a,
            decoder.self_attn.wq_b,
            decoder.mlp.shared_experts.gate_up_proj,
        )
        before = [p.quant_method.w8a8_mxfp8_linear for p in projections]
        with envs.SGLANG_OPT_DSV41_SMALL_BATCH_SPLIT_K.override(False):
            DeepseekV4ForCausalLM._configure_small_batch_split_k(target)
        for projection, original in zip(projections, before):
            self.assertIs(projection.quant_method.w8a8_mxfp8_linear, original)
        with envs.SGLANG_OPT_DSV41_SMALL_BATCH_SPLIT_K.override(False):
            # Exercise the unset default while restoring the caller's env on exit.
            envs.SGLANG_OPT_DSV41_SMALL_BATCH_SPLIT_K.clear()
            DeepseekV4ForCausalLM._configure_small_batch_split_k(target)
        for projection in (projections[0], projections[2]):
            self.assertTrue(
                projection.quant_method.w8a8_mxfp8_linear.keywords[
                    "use_small_batch_split_k"
                ]
            )
        self.assertIs(projections[1].quant_method.w8a8_mxfp8_linear, before[1])

    @torch.inference_mode()
    def test_graph_replay_bias_and_leading_dimensions(self):
        """Replay must consume changed inputs and preserve bias/output shaping."""
        x, _, wq, ws = self._operands()
        x = x.view(2, 3, 5120)
        bias = torch.randn(1792, device="cuda", dtype=torch.bfloat16)
        call = partial(
            self._linear, wq=wq, ws=ws, bias=bias, use_small_batch_split_k=True
        )
        call(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = call(x)
        for amplitude in (0, 0.1, 3):
            x.normal_().mul_(amplitude)
            graph.replay()
            expected = call(x)
            self.assertEqual(actual.shape, (2, 3, 1792))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        compiled = torch.compile(call, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
