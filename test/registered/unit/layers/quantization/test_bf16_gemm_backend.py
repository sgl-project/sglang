"""Unit tests for architecture-aware BF16 GEMM backend selection."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.quantization import unquant
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBf16GemmBackend(CustomTestCase):
    def setUp(self):
        self._backend = unquant._BF16_GEMM_BACKEND
        self._gemm = unquant._cutedsl_bf16_gemm
        self._use_gemm = unquant._use_cutedsl_bf16_gemm

    def tearDown(self):
        unquant._BF16_GEMM_BACKEND = self._backend
        unquant._cutedsl_bf16_gemm = self._gemm
        unquant._use_cutedsl_bf16_gemm = self._use_gemm

    def _initialize(self, backend: str) -> None:
        cutedsl_module = ModuleType("sglang.kernels.ops.gemm.cutedsl_bf16_gemm")
        cutedsl_module.cutedsl_bf16_gemm = object()
        cutedsl_module.use_cutedsl_bf16_gemm = object()
        with (
            patch("sglang.srt.utils.is_sm100_supported", return_value=True),
            patch("sglang.srt.utils.get_device_sm", return_value=103),
            patch.dict(
                sys.modules,
                {
                    "sglang.kernels.ops.gemm.cutedsl_bf16_gemm": cutedsl_module,
                },
            ),
        ):
            unquant.initialize_bf16_gemm_config(
                SimpleNamespace(bf16_gemm_backend=backend)
            )

    def test_sm103_auto_falls_back_to_torch(self):
        """SM103 auto selection must avoid the 2-CTA CuTeDSL kernel that raises Xid 13."""
        self._initialize("auto")
        self.assertEqual(unquant.get_bf16_gemm_backend(), unquant.Bf16GemmBackend.TORCH)

    def test_sm103_rejects_explicit_cutedsl(self):
        """Explicit CuTeDSL must fail before an unsupported SM103 kernel can launch."""
        with self.assertRaisesRegex(ValueError, "requires an SM100 GPU"):
            self._initialize("cutedsl")


if __name__ == "__main__":
    unittest.main()
