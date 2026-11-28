"""Kimi-K3 registers its CuTe DSL TGV GEMM tactics only on CUDA.

The CuTe DSL GEMM module imports cuda.bindings and cutlass at module level, so
importing it on any other platform fails model construction. Pure Python: the
platform is patched and the GEMM module is replaced in ``sys.modules``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.models import kimi_k3
from sglang.test.test_utils import CustomTestCase

_CUTEDSL_GEMM_MODULE = "sglang.kernels.ops.gemm.cutedsl_bf16_gemm"


class TestKimiK3TgvRegistration(CustomTestCase):
    def test_non_cuda_platforms_never_import_the_cutedsl_gemm(self):
        platform = SimpleNamespace(is_cuda=False, device_sm=0)
        with (
            mock.patch.object(kimi_k3, "get_platform", return_value=platform),
            # A None entry makes any import of the module raise.
            mock.patch.dict(sys.modules, {_CUTEDSL_GEMM_MODULE: None}),
        ):
            self.assertIsNone(kimi_k3._register_k3_tgv_m_ranges())

    def test_cuda_registers_the_measured_ranges(self):
        for device_sm, expected in (
            (100, [kimi_k3._K3_TGV_M_RANGES]),
            (103, [kimi_k3._K3_TGV_M_RANGES, kimi_k3._K3_GB300_TGV_M_RANGES]),
        ):
            cutedsl_gemm = SimpleNamespace(register_tgv_m_ranges=mock.Mock())
            platform = SimpleNamespace(is_cuda=True, device_sm=device_sm)
            with (
                self.subTest(device_sm=device_sm),
                mock.patch.object(kimi_k3, "get_platform", return_value=platform),
                mock.patch.dict(sys.modules, {_CUTEDSL_GEMM_MODULE: cutedsl_gemm}),
            ):
                kimi_k3._register_k3_tgv_m_ranges()
                registered = [
                    call.args[0]
                    for call in cutedsl_gemm.register_tgv_m_ranges.call_args_list
                ]
                self.assertEqual(registered, expected)


if __name__ == "__main__":
    unittest.main()
