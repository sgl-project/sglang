import unittest
from unittest.mock import patch

from sglang.kernels.ops.layernorm import mhc
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMhcDeepGemmGate(CustomTestCase):
    def test_falls_back_when_deep_gemm_is_unavailable(self):
        with (
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
                False,
            ),
        ):
            self.assertFalse(mhc._use_deep_gemm_hc_prenorm())

    def test_preserves_deep_gemm_path_when_available(self):
        with (
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
                True,
            ),
        ):
            self.assertTrue(mhc._use_deep_gemm_hc_prenorm())

    def test_explicit_disable_takes_precedence(self):
        with (
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False),
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
                True,
            ),
        ):
            self.assertFalse(mhc._use_deep_gemm_hc_prenorm())


class TestMhcTilelangBackendGate(CustomTestCase):
    def test_cuda_keeps_env_selected_tilelang_pre_path(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
            patch.object(mhc, "is_hip", return_value=False),
        ):
            self.assertTrue(mhc._use_tilelang_mhc_pre())

    def test_hip_always_uses_torch_pre_path(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
            patch.object(mhc, "is_hip", return_value=True),
        ):
            self.assertFalse(mhc._use_tilelang_mhc_pre())

    def test_explicit_tilelang_pre_disable_takes_precedence(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False),
            patch.object(mhc, "is_hip", return_value=False),
        ):
            self.assertFalse(mhc._use_tilelang_mhc_pre())

    def test_cuda_keeps_env_selected_tilelang_post_path(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(True),
            patch.object(mhc, "is_hip", return_value=False),
        ):
            self.assertTrue(mhc._use_tilelang_mhc_post())

    def test_hip_always_uses_torch_post_path(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(True),
            patch.object(mhc, "is_hip", return_value=True),
        ):
            self.assertFalse(mhc._use_tilelang_mhc_post())

    def test_explicit_tilelang_post_disable_takes_precedence(self):
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(False),
            patch.object(mhc, "is_hip", return_value=False),
        ):
            self.assertFalse(mhc._use_tilelang_mhc_post())


if __name__ == "__main__":
    unittest.main()
