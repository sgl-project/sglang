import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.layers.deep_gemm_wrapper import configurer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepGemmConfigurer(unittest.TestCase):
    def test_jit_disable_short_circuits_sm120_probe(self):
        with (
            patch.object(configurer, "_is_cuda", True),
            patch.object(configurer, "_is_musa", False),
            patch.object(
                configurer.envs.SGLANG_ENABLE_JIT_DEEPGEMM,
                "get",
                return_value=False,
            ),
            patch.object(configurer, "_sm120_deep_gemm_apis_available") as probe,
        ):
            self.assertFalse(configurer._compute_enable_deep_gemm())
        probe.assert_not_called()

    def test_sm120_requires_arch_marker_and_wo_a_apis(self):
        deep_gemm = types.ModuleType("deep_gemm")
        deep_gemm.fp8_einsum = lambda *args, **kwargs: None
        deep_gemm.transform_sf_into_required_layout = lambda *args, **kwargs: None

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            self.assertFalse(configurer._sm120_deep_gemm_apis_available())

        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            self.assertTrue(configurer._sm120_deep_gemm_apis_available())

    def test_sm120_old_deep_gemm_is_disabled(self):
        with (
            patch.object(configurer, "_is_cuda", True),
            patch.object(configurer, "_is_musa", False),
            patch.object(
                configurer.envs.SGLANG_ENABLE_JIT_DEEPGEMM,
                "get",
                return_value=True,
            ),
            patch.object(configurer, "get_device_sm", return_value=120),
            patch.object(
                configurer,
                "_sm120_deep_gemm_apis_available",
                return_value=False,
            ),
        ):
            self.assertFalse(configurer._compute_enable_deep_gemm())


if __name__ == "__main__":
    unittest.main()
