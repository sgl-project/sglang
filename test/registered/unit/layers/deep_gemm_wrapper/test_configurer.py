"""CPU regression tests for optional DeepGEMM availability detection."""

import builtins
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from sglang.srt.layers.deep_gemm_wrapper import configurer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepGemmAvailability(CustomTestCase):
    def test_missing_cuda_toolkit_disables_deep_gemm(self):
        """A wheel without a discoverable CUDA toolkit must not break model imports."""
        original_import = builtins.__import__

        def import_without_cuda_toolkit(name, *args, **kwargs):
            if name == "deep_gemm":
                raise AssertionError("Cannot find CUDA home")
            return original_import(name, *args, **kwargs)

        for sm in (90, 100, 120, 121):
            with (
                self.subTest(sm=sm),
                patch.multiple(configurer, _is_cuda=True, _is_musa=False),
                patch.object(configurer, "get_device_sm", return_value=sm),
                configurer.envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(True),
                patch("builtins.__import__", side_effect=import_without_cuda_toolkit),
            ):
                self.assertFalse(configurer._compute_enable_deep_gemm())

    def test_available_deep_gemm_remains_enabled(self):
        """The missing-toolkit fallback must not disable a working installation."""
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_einsum = lambda: None
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous = lambda: None
        deep_gemm.transform_sf_into_required_layout = lambda: None

        for sm in (90, 100, 120, 121):
            with (
                self.subTest(sm=sm),
                patch.multiple(configurer, _is_cuda=True, _is_musa=False),
                patch.object(configurer, "get_device_sm", return_value=sm),
                configurer.envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(True),
                patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            ):
                self.assertTrue(configurer._compute_enable_deep_gemm())


if __name__ == "__main__":
    unittest.main()
