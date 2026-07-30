import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.npu import device_operator as device_operator_module
from sglang.srt.hardware_backend.npu.device_capabilities import (
    NPUDeviceFamily,
    NPUFeature,
    get_npu_device_family,
    supports_npu_feature,
)
from sglang.srt.hardware_backend.npu.device_operator import NPUDeviceOperator
from sglang.srt.layers import layernorm as layernorm_module
from sglang.srt.layers.layernorm import Gemma3RMSNorm, GemmaRMSNorm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNPUDeviceCapabilities(unittest.TestCase):
    def setUp(self):
        get_npu_device_family.cache_clear()

    def tearDown(self):
        get_npu_device_family.cache_clear()

    def _torch_npu_for_soc(self, soc_version):
        get_soc_version = MagicMock(return_value=soc_version)
        torch_npu = SimpleNamespace(
            npu=SimpleNamespace(get_soc_version=get_soc_version)
        )
        return torch_npu, get_soc_version

    @staticmethod
    def _supports(*supported_features):
        return lambda feature: feature in supported_features

    @staticmethod
    def _triton_module(kernel):
        module = ModuleType("sgl_kernel_npu.norm.add_rmsnorm_bias")
        module.add_gemma_rms_norm = kernel
        return {
            "sgl_kernel_npu": ModuleType("sgl_kernel_npu"),
            "sgl_kernel_npu.norm": ModuleType("sgl_kernel_npu.norm"),
            "sgl_kernel_npu.norm.add_rmsnorm_bias": module,
        }

    def test_gemma_rms_norm_feature_matrix(self):
        cases = (
            (
                220,
                NPUDeviceFamily.ASCEND_910B,
                {NPUFeature.NATIVE_GEMMA_RMS_NORM},
            ),
            (
                225,
                NPUDeviceFamily.ASCEND_910B,
                {NPUFeature.NATIVE_GEMMA_RMS_NORM},
            ),
            (
                250,
                NPUDeviceFamily.ASCEND_910C,
                {NPUFeature.NATIVE_GEMMA_RMS_NORM},
            ),
            (
                255,
                NPUDeviceFamily.ASCEND_910C,
                {NPUFeature.NATIVE_GEMMA_RMS_NORM},
            ),
            (
                260,
                NPUDeviceFamily.ASCEND_950,
                {NPUFeature.TRITON_GEMMA_RMS_NORM},
            ),
            (999, NPUDeviceFamily.UNKNOWN, set()),
        )

        for soc_version, expected_family, expected_features in cases:
            with self.subTest(soc_version=soc_version):
                get_npu_device_family.cache_clear()
                torch_npu, _ = self._torch_npu_for_soc(soc_version)
                with patch.dict(sys.modules, {"torch_npu": torch_npu}):
                    self.assertEqual(get_npu_device_family(), expected_family)
                    for feature in NPUFeature:
                        self.assertEqual(
                            supports_npu_feature(feature),
                            feature in expected_features,
                        )

    def test_device_detection_is_cached(self):
        torch_npu, get_soc_version = self._torch_npu_for_soc(220)
        with patch.dict(sys.modules, {"torch_npu": torch_npu}):
            self.assertTrue(supports_npu_feature(NPUFeature.NATIVE_GEMMA_RMS_NORM))
            self.assertTrue(supports_npu_feature(NPUFeature.NATIVE_GEMMA_RMS_NORM))
        get_soc_version.assert_called_once_with()

    def test_detection_failure_uses_safe_fallback(self):
        get_soc_version = MagicMock(side_effect=RuntimeError("not initialized"))
        torch_npu = SimpleNamespace(
            npu=SimpleNamespace(get_soc_version=get_soc_version)
        )
        with patch.dict(sys.modules, {"torch_npu": torch_npu}):
            self.assertEqual(get_npu_device_family(), NPUDeviceFamily.UNKNOWN)
            for feature in NPUFeature:
                self.assertFalse(supports_npu_feature(feature))

    def test_gemma_rms_norm_uses_native_provider(self):
        input = torch.randn(2, 4)
        weight = torch.randn(4)
        output = torch.randn_like(input)
        native_kernel = MagicMock(return_value=(output, None))
        torch_npu = SimpleNamespace(npu_gemma_rms_norm=native_kernel)

        with (
            patch.dict(sys.modules, {"torch_npu": torch_npu}),
            patch.object(
                device_operator_module,
                "supports_npu_feature",
                side_effect=self._supports(NPUFeature.NATIVE_GEMMA_RMS_NORM),
            ),
        ):
            result = NPUDeviceOperator.gemma_rms_norm(input, weight, 1e-6)

        self.assertIs(result, output)
        native_kernel.assert_called_once_with(input, weight, 1e-6)

    def test_gemma_rms_norm_uses_triton_provider_and_restores_shape(self):
        input = torch.randn(2, 4, 3).transpose(1, 2)
        weight = torch.randn(4)
        triton_kernel = MagicMock(side_effect=lambda x, *_: (x.clone(), None))

        with (
            patch.dict(sys.modules, self._triton_module(triton_kernel)),
            patch.object(
                device_operator_module,
                "supports_npu_feature",
                side_effect=self._supports(NPUFeature.TRITON_GEMMA_RMS_NORM),
            ),
        ):
            result = NPUDeviceOperator.gemma_rms_norm(input, weight, 1e-6)

        self.assertEqual(result.shape, input.shape)
        triton_input, triton_weight, triton_residual, eps = triton_kernel.call_args.args
        self.assertEqual(triton_input.shape, (6, 4))
        self.assertTrue(triton_input.is_contiguous())
        self.assertTrue(triton_weight.is_contiguous())
        self.assertIsNone(triton_residual)
        self.assertEqual(eps, 1e-6)

    def test_gemma_rms_norm_uses_decomposed_fallback(self):
        input = torch.randn(2, 4)
        weight = torch.randn(4)
        output = torch.randn_like(input)
        fallback_kernel = MagicMock(return_value=(output, None))
        torch_npu = SimpleNamespace(npu_rms_norm=fallback_kernel)

        with (
            patch.dict(sys.modules, {"torch_npu": torch_npu}),
            patch.object(
                device_operator_module,
                "supports_npu_feature",
                side_effect=self._supports(),
            ),
        ):
            result = NPUDeviceOperator.gemma_rms_norm(input, weight, 1e-6)

        self.assertIs(result, output)
        args = fallback_kernel.call_args.args
        self.assertIs(args[0], input)
        torch.testing.assert_close(args[1], 1.0 + weight)
        self.assertEqual(args[2], 1e-6)

    def test_add_gemma_rms_norm_uses_native_provider(self):
        input = torch.randn(2, 4)
        residual = torch.randn_like(input)
        weight = torch.randn(4)
        output = torch.randn_like(input)
        residual_sum = torch.randn_like(input)
        native_kernel = MagicMock(return_value=(output, None, residual_sum))
        torch_npu = SimpleNamespace(npu_add_rms_norm=native_kernel)

        with (
            patch.dict(sys.modules, {"torch_npu": torch_npu}),
            patch.object(
                device_operator_module,
                "supports_npu_feature",
                side_effect=self._supports(NPUFeature.NATIVE_GEMMA_RMS_NORM),
            ),
        ):
            result = NPUDeviceOperator.add_gemma_rms_norm(input, weight, residual, 1e-6)

        self.assertIs(result[0], output)
        self.assertIs(result[1], residual_sum)
        args = native_kernel.call_args.args
        self.assertIs(args[0], residual)
        self.assertIs(args[1], input)
        torch.testing.assert_close(args[2], 1.0 + weight)
        self.assertEqual(args[3], 1e-6)

    def test_add_gemma_rms_norm_uses_triton_provider(self):
        input = torch.randn(2, 3, 4)
        residual = torch.randn_like(input)
        weight = torch.randn(4)
        triton_kernel = MagicMock(
            side_effect=lambda x, _, r, __: (x.clone(), r.clone())
        )

        with (
            patch.dict(sys.modules, self._triton_module(triton_kernel)),
            patch.object(
                device_operator_module,
                "supports_npu_feature",
                side_effect=self._supports(NPUFeature.TRITON_GEMMA_RMS_NORM),
            ),
        ):
            output, residual_sum = NPUDeviceOperator.add_gemma_rms_norm(
                input, weight, residual, 1e-6
            )

        self.assertEqual(output.shape, input.shape)
        self.assertEqual(residual_sum.shape, residual.shape)
        triton_input, _, triton_residual, _ = triton_kernel.call_args.args
        self.assertEqual(triton_input.shape, (6, 4))
        self.assertEqual(triton_residual.shape, (6, 4))

    def test_add_gemma_rms_norm_uses_decomposed_fallback(self):
        input = torch.randn(2, 4)
        residual = torch.randn_like(input)
        weight = torch.randn(4)
        output = torch.randn_like(input)
        fallback_kernel = MagicMock(return_value=(output, None))
        torch_npu = SimpleNamespace(npu_rms_norm=fallback_kernel)

        with (
            patch.dict(sys.modules, {"torch_npu": torch_npu}),
            patch.object(
                device_operator_module,
                "supports_npu_feature",
                side_effect=self._supports(),
            ),
        ):
            norm_output, residual_sum = NPUDeviceOperator.add_gemma_rms_norm(
                input, weight, residual, 1e-6
            )

        self.assertIs(norm_output, output)
        torch.testing.assert_close(residual_sum, input + residual)
        args = fallback_kernel.call_args.args
        torch.testing.assert_close(args[0], input + residual)
        torch.testing.assert_close(args[1], 1.0 + weight)
        self.assertEqual(args[2], 1e-6)

    def test_empty_triton_inputs_do_not_launch_kernel(self):
        input = torch.empty(0, 4)
        residual = torch.empty_like(input)
        weight = torch.randn(4)
        triton_kernel = MagicMock()

        with (
            patch.dict(sys.modules, self._triton_module(triton_kernel)),
            patch.object(
                device_operator_module,
                "supports_npu_feature",
                side_effect=self._supports(NPUFeature.TRITON_GEMMA_RMS_NORM),
            ),
        ):
            output = NPUDeviceOperator.gemma_rms_norm(input, weight, 1e-6)
            fused_output, residual_sum = NPUDeviceOperator.add_gemma_rms_norm(
                input, weight, residual, 1e-6
            )

        self.assertEqual(output.shape, input.shape)
        self.assertEqual(fused_output.shape, input.shape)
        self.assertEqual(residual_sum.shape, input.shape)
        triton_kernel.assert_not_called()

    def test_gemma_layers_delegate_normal_and_residual_paths(self):
        input = torch.randn(2, 4)
        residual = torch.randn_like(input)
        norm_output = torch.randn_like(input)
        residual_sum = torch.randn_like(input)

        for layer in (GemmaRMSNorm(4), Gemma3RMSNorm(4)):
            eps = (
                layer.variance_epsilon if isinstance(layer, GemmaRMSNorm) else layer.eps
            )
            with self.subTest(layer=type(layer).__name__):
                with (
                    patch.object(
                        layernorm_module.envs.SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM,
                        "get",
                        return_value=False,
                    ),
                    patch.object(
                        layernorm_module.NPUDeviceOperator,
                        "gemma_rms_norm",
                        return_value=norm_output,
                    ) as gemma_rms_norm,
                    patch.object(
                        layernorm_module.NPUDeviceOperator,
                        "add_gemma_rms_norm",
                        return_value=(norm_output, residual_sum),
                    ) as add_gemma_rms_norm,
                ):
                    self.assertIs(layer.forward_npu(input), norm_output)
                    fused_result = layer.forward_npu(input, residual)
                    self.assertIs(fused_result[0], norm_output)
                    self.assertIs(fused_result[1], residual_sum)

                gemma_rms_norm.assert_called_once_with(input, layer.weight, eps)
                add_gemma_rms_norm.assert_called_once_with(
                    input, layer.weight, residual, eps
                )


if __name__ == "__main__":
    unittest.main()
