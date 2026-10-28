import unittest
from unittest.mock import patch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_fp8,
    is_unified_kv_triton,
    maybe_enable_gfx1250_v4_unified_kv,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx1250_supported
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGfx1250V4UnifiedKvGate(CustomTestCase):
    def setUp(self):
        env_gate._gfx1250_v4_unified = False

    def tearDown(self):
        env_gate._gfx1250_v4_unified = False

    def test_v41_does_not_enable(self):
        maybe_enable_gfx1250_v4_unified_kv("deepseek_v41")
        self.assertFalse(env_gate._gfx1250_v4_unified)

    def test_explicit_tilelang_does_not_enable(self):
        with patch.object(
            envs.SGLANG_HACK_FLASHMLA_BACKEND, "get", return_value="tilelang"
        ):
            maybe_enable_gfx1250_v4_unified_kv("deepseek_v4")
        self.assertFalse(env_gate._gfx1250_v4_unified)

    def test_v4_auto_enables_only_on_gfx1250(self):
        with patch.object(
            envs.SGLANG_HACK_FLASHMLA_BACKEND, "get", return_value="auto"
        ):
            maybe_enable_gfx1250_v4_unified_kv("deepseek_v4")
        if is_gfx1250_supported():
            self.assertTrue(env_gate._gfx1250_v4_unified)
            self.assertTrue(is_unified_kv_triton())
        else:
            self.assertFalse(env_gate._gfx1250_v4_unified)


class TestGfx1250V4UnifiedKvFp8Gate(CustomTestCase):
    """is_unified_kv_fp8 allows gfx1250 alongside gfx95 (both carry the OCP
    e4m3fn fp8_dtype; aiter ships gfx1250 kernels for this layout)."""

    def setUp(self):
        is_unified_kv_fp8.cache_clear()

    def tearDown(self):
        is_unified_kv_fp8.cache_clear()

    def test_fp8_allowed_on_gfx1250(self):
        with patch.object(
            env_gate, "is_unified_kv_triton", return_value=True
        ), patch.object(
            envs.SGLANG_DSV4_UNIFIED_KV_FP8, "get", return_value=True
        ), patch.object(
            env_gate, "is_gfx95_supported", return_value=False
        ), patch.object(
            env_gate, "is_gfx1250_supported", return_value=True
        ):
            self.assertTrue(is_unified_kv_fp8())

    def test_fp8_falls_back_off_gfx95_and_gfx1250(self):
        with patch.object(
            env_gate, "is_unified_kv_triton", return_value=True
        ), patch.object(
            envs.SGLANG_DSV4_UNIFIED_KV_FP8, "get", return_value=True
        ), patch.object(
            env_gate, "is_gfx95_supported", return_value=False
        ), patch.object(
            env_gate, "is_gfx1250_supported", return_value=False
        ):
            self.assertFalse(is_unified_kv_fp8())


if __name__ == "__main__":
    unittest.main()
