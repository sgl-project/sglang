"""FP8 MoE runner fallback and activation dispatch contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.fp8 as fp8
from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestFp8MoeRunnerFallback(CustomTestCase):
    def _runner_backend_for(self, global_backend):
        method = fp8.Fp8MoEMethod.__new__(fp8.Fp8MoEMethod)
        with patch.object(fp8, "get_moe_runner_backend", return_value=global_backend):
            method.create_moe_runner(layer=None, moe_runner_config=MoeRunnerConfig())
        return method.runner.runner_backend

    def test_flashinfer_cutlass_falls_back_to_triton(self):
        self.assertTrue(
            self._runner_backend_for(MoeRunnerBackend.FLASHINFER_CUTLASS).is_triton()
        )

    def test_flashinfer_cutedsl_falls_back_to_triton(self):
        self.assertTrue(
            self._runner_backend_for(MoeRunnerBackend.FLASHINFER_CUTEDSL).is_triton()
        )

    def test_triton_is_kept(self):
        self.assertTrue(self._runner_backend_for(MoeRunnerBackend.TRITON).is_triton())


class TestDeepGemmActivationDispatch(CustomTestCase):
    def test_swiglu_limit_preserves_existing_pre_activation_clamp(self):
        from sglang.srt.layers.moe.moe_runner import deep_gemm

        for set_after_init in (False, True):
            with self.subTest(set_after_init=set_after_init):
                config = MoeRunnerConfig(
                    swiglu_limit=1.0,
                    gemm1_clamp_limit=None if set_after_init else 1.0,
                )
                with patch.object(
                    deep_gemm,
                    "get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_megamoe=lambda: False),
                ):
                    deep_gemm.DeepGemmRunnerCore(config)
                config.gemm1_clamp_limit = 1.0
                actual = self._run_activation(
                    swiglu_limit=config.swiglu_limit,
                    gemm1_clamp_limit=config.gemm1_clamp_limit,
                )
                torch.testing.assert_close(
                    actual, F.silu(torch.ones_like(actual)), rtol=0.08, atol=0.01
                )

    def test_zero_alpha_does_not_enable_post_silu_clamp(self):
        for limit in (None, 0.0, 1.0):
            with self.subTest(limit=limit):
                actual = self._run_activation(gemm1_alpha=0.0, gemm1_clamp_limit=limit)
                torch.testing.assert_close(
                    actual,
                    F.silu(torch.full_like(actual, 3.0)) * 3.0,
                    rtol=0.08,
                    atol=0.01,
                )

    def test_limit_only_uses_post_activation_clamp(self):
        for limit in (0.0, 1.0):
            with self.subTest(limit=limit):
                actual = self._run_activation(gemm1_clamp_limit=limit)
                torch.testing.assert_close(actual, torch.full_like(actual, limit**2))

    def test_contiguous_limit_only_clamp_matches_triton_runner(self):
        """Limit-only models (e.g. Step3p5, GigaChat3.5) must see the same
        post-SiLU clamp from DeepGEMM as from the Triton runner."""
        from sglang.srt.layers.moe.moe_runner import deep_gemm
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _swiglu_silu_clamp_mul,
        )

        triton_activation = _swiglu_silu_clamp_mul.__wrapped__
        x = torch.linspace(-12.0, 12.0, 4 * 512).reshape(4, 512)
        x = torch.cat((x, x.flip(-1)), dim=-1)
        for limit in (0.5, 7.0):
            with self.subTest(limit=limit):
                out = torch.empty(4, 512)
                deep_gemm._silu_and_mul_clamp(x, out, limit)
                torch.testing.assert_close(out, triton_activation(x, limit))

    def _run_activation(self, **kwargs):
        from sglang.kernels.ops.moe import ep_moe_kernels
        from sglang.srt.layers.moe.moe_runner import deep_gemm

        inputs = torch.full((1, 4, 1024), 3.0)
        counts = torch.tensor([4], dtype=torch.int32)

        def activation(input, *args, **options):
            # Only the accelerator launch is replaced; wrapper dispatch is real.
            output, scales = args[3], args[7]
            gate, up = input.chunk(2, dim=-1)
            limit = options["GEMM1_CLAMP_LIMIT"]
            value = F.silu(gate)
            if options["HAS_GEMM1_CLAMP_LIMIT"]:
                value = value.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            output.copy_(value * up)
            scales.fill_(1)

        def pre_activation(input, output, scales, group, counts, **options):
            gate, up = input.chunk(2, dim=-1)
            limit = options["swiglu_limit"]
            output.copy_(F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit))
            scales.fill_(1)

        with (
            patch.object(deep_gemm.deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False),
            patch.object(ep_moe_kernels, "_silu_and_mul_post_quant_kernel") as kernel,
            patch.object(deep_gemm, "silu_and_mul_masked_post_quant", pre_activation),
        ):
            kernel.__getitem__.return_value = activation
            output, _ = deep_gemm._varlen_deep_gemm_silu_mul_quant(
                inputs, counts, group_size=128, topk=1, **kwargs
            )
        return output.float()


if __name__ == "__main__":
    unittest.main()
