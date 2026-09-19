import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import mxfp4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPpuMxfp4W4A16(unittest.TestCase):
    @staticmethod
    def _exec():
        return SimpleNamespace(
            moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
        )

    def test_ppu_env_selects_repacked_w4a16(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(mxfp4, "get_device_sm", return_value=89),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=True,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MOE, "get", return_value=False
            ),
            patch.object(mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        ):
            method = mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")
            self.assertTrue(method.use_deepgemm_mxfp4_w4a16)
            self.assertFalse(method.use_marlin)
            self.assertTrue(method.is_deepgemm_moe_runner_backend_enabled())

    def test_ppu_w4a16_rejects_non_none_a2a(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(
                mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.DEEPEP
            ),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "moe-a2a-backend none"):
                mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")

    def test_ppu_w4a16_requires_explicit_deepgemm_backend(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "moe-runner-backend deep_gemm"):
                mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")

    def test_native_mxfp4_w4a4_is_not_gated_by_w4a16_jit_check(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(mxfp4, "get_device_sm", return_value=89),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=False,
            ),
            patch.object(mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False),
        ):
            method = mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")
            self.assertFalse(method.use_deepgemm_mxfp4_w4a16)
            self.assertTrue(method.is_deepgemm_moe_runner_backend_enabled())

    def test_ppu_converts_marlin_e8m0_scales_to_bf16(self):
        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deepgemm_mxfp4_w4a16 = True
        method.moe_runner_config = SimpleNamespace(gemm1_alpha=None)
        raw_scale = torch.tensor([127, 128], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )

        def prepare_marlin_scales(layer):
            layer.w13_weight_scale = torch.nn.Parameter(
                raw_scale.reshape(1, 1, 2), requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                raw_scale.reshape(1, 1, 2), requires_grad=False
            )

        layer = SimpleNamespace()
        with (
            patch(
                "sglang.srt.layers.quantization.marlin_utils."
                "check_moe_marlin_supports_layer",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.marlin_utils_fp4."
                "prepare_moe_mxfp4_layer_for_marlin",
                side_effect=prepare_marlin_scales,
            ),
            patch.object(mxfp4, "_is_ppu", True),
        ):
            method.process_weights_after_loading(layer)

        expected = torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.bfloat16)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.bfloat16)
        self.assertEqual(layer._mxfp4_backend, "marlin")
        self.assertTrue(torch.equal(layer.w13_weight_scale, expected))
        self.assertTrue(torch.equal(layer.w2_weight_scale, expected))

    def test_ppu_w4a16_uses_bf16_scales_in_runner(self):
        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_deepgemm_mxfp4_w4a16 = True
        method.runner = Mock()
        method.runner.run.return_value = sentinel.combine_input
        layer = SimpleNamespace(
            w13_weight=sentinel.w13_weight,
            w2_weight=sentinel.w2_weight,
            w13_weight_scale=torch.ones(1, dtype=torch.bfloat16),
            w2_weight_scale=torch.ones(1, dtype=torch.bfloat16),
        )

        with patch.object(mxfp4, "_is_ppu", True):
            self.assertIs(
                method.apply(layer, sentinel.dispatch_output), sentinel.combine_input
            )
        quant_info = method.runner.run.call_args.args[1]
        self.assertTrue(quant_info.use_mxfp4_w4a16)
        self.assertFalse(quant_info.use_int4_w4a16)
        self.assertEqual(quant_info.w13_scale.dtype, torch.bfloat16)
        self.assertEqual(quant_info.w2_scale.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
