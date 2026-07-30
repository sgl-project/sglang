import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.npu.device_capabilities import (
    NPUDeviceFamily,
    NPUFeature,
    get_npu_device_family,
    supports_npu_feature,
)
from sglang.srt.layers import layernorm as layernorm_module
from sglang.srt.layers.layernorm import GemmaRMSNorm
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

    def test_native_gemma_rms_norm_feature_matrix(self):
        cases = (
            (220, NPUDeviceFamily.A2, True),
            (225, NPUDeviceFamily.A2, True),
            (250, NPUDeviceFamily.A3, True),
            (255, NPUDeviceFamily.A3, True),
            (260, NPUDeviceFamily.A5, False),
            (999, NPUDeviceFamily.UNKNOWN, False),
        )

        for soc_version, expected_family, expected_support in cases:
            with self.subTest(soc_version=soc_version):
                get_npu_device_family.cache_clear()
                torch_npu, _ = self._torch_npu_for_soc(soc_version)
                with patch.dict(sys.modules, {"torch_npu": torch_npu}):
                    self.assertEqual(get_npu_device_family(), expected_family)
                    self.assertEqual(
                        supports_npu_feature(NPUFeature.NATIVE_GEMMA_RMS_NORM),
                        expected_support,
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
            self.assertFalse(supports_npu_feature(NPUFeature.NATIVE_GEMMA_RMS_NORM))

    def test_gemma_rms_norm_uses_native_kernel_when_supported(self):
        layer = GemmaRMSNorm(4)
        x = torch.randn(2, 4)
        native_kernel = MagicMock(return_value=(x, None))
        fallback_kernel = MagicMock()
        torch_npu = SimpleNamespace(
            npu_gemma_rms_norm=native_kernel,
            npu_rms_norm=fallback_kernel,
        )

        with (
            patch.object(layernorm_module, "torch_npu", torch_npu, create=True),
            patch.object(layernorm_module, "supports_npu_feature", return_value=True),
        ):
            result = layer.forward_npu(x)

        self.assertIs(result, x)
        native_kernel.assert_called_once_with(x, layer.weight, layer.variance_epsilon)
        fallback_kernel.assert_not_called()

    def test_gemma_rms_norm_uses_equivalent_fallback_when_unsupported(self):
        layer = GemmaRMSNorm(4)
        x = torch.randn(2, 4)
        fallback_kernel = MagicMock(return_value=(x, None))
        native_kernel = MagicMock()
        torch_npu = SimpleNamespace(
            npu_gemma_rms_norm=native_kernel,
            npu_rms_norm=fallback_kernel,
        )

        with (
            patch.object(layernorm_module, "torch_npu", torch_npu, create=True),
            patch.object(layernorm_module, "supports_npu_feature", return_value=False),
        ):
            result = layer.forward_npu(x)

        self.assertIs(result, x)
        fallback_kernel.assert_called_once()
        args = fallback_kernel.call_args.args
        self.assertIs(args[0], x)
        torch.testing.assert_close(args[1], 1.0 + layer.weight)
        self.assertEqual(args[2], layer.variance_epsilon)
        native_kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
