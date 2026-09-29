"""CPU regression tests for Thor's SM110 automatic backend selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import (
    ResolvedView,
    _moe_runner_backend_quant_constraints,
)
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.runtime_context import get_platform, override_platform
from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSm110BackendRouting(CustomTestCase):
    def test_sm110_probe_requires_major_11_and_cuda_12_8(self):
        try:
            for device_sm, cuda_version, expected in (
                (100, "13.0", False),
                (110, "12.7", False),
                (110, "12.8", True),
                (110, "13.0", True),
                (120, "13.0", False),
            ):
                with self.subTest(device_sm=device_sm, cuda=cuda_version):
                    with (
                        patch.object(common, "is_cuda", return_value=True),
                        patch.object(common, "get_device_sm", return_value=device_sm),
                        patch.object(common.torch.version, "cuda", cuda_version),
                    ):
                        common.is_sm110_supported.cache_clear()
                        self.assertEqual(get_platform().is_sm110, expected)
        finally:
            common.is_sm110_supported.cache_clear()

    def test_fp8_auto_routes_sm110_to_triton_only(self):
        for is_sm110, is_sm120, requested, expected in (
            (True, False, "auto", Fp8GemmRunnerBackend.TRITON),
            (False, True, "auto", Fp8GemmRunnerBackend.CUTLASS),
            (False, False, "auto", Fp8GemmRunnerBackend.AUTO),
            (True, False, "deep_gemm", Fp8GemmRunnerBackend.DEEP_GEMM),
        ):
            with self.subTest(sm110=is_sm110, sm120=is_sm120, requested=requested):
                exec_config = SimpleNamespace(
                    kernel=SimpleNamespace(fp8_gemm_runner_backend=requested)
                )
                with (
                    override_platform(is_sm110=is_sm110, is_sm120=is_sm120),
                    patch.object(fp8_utils, "get_exec", return_value=exec_config),
                    patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", None),
                ):
                    fp8_utils.initialize_fp8_gemm_config()
                    self.assertEqual(fp8_utils.get_fp8_gemm_runner_backend(), expected)

    def test_modelopt_fp4_auto_routes_sm110_to_flashinfer_cutlass_only(self):
        def view(quantization="modelopt_fp4", backend="auto"):
            return ResolvedView(
                SimpleNamespace(
                    quantization=quantization,
                    moe_runner_backend=backend,
                    moe_a2a_backend="none",
                )
            )

        for is_sm110, is_sm120, quantization, backend, expected in (
            (True, False, "modelopt_fp4", "auto", "flashinfer_cutlass"),
            (False, True, "modelopt_fp4", "auto", "flashinfer_cutlass"),
            (False, False, "modelopt_fp4", "auto", None),
            (True, False, "modelopt_fp4", "triton", None),
            (True, False, "modelopt_fp8", "auto", None),
        ):
            with self.subTest(
                sm110=is_sm110,
                sm120=is_sm120,
                quantization=quantization,
                backend=backend,
            ):
                with override_platform(is_sm110=is_sm110, is_sm120=is_sm120):
                    result = _moe_runner_backend_quant_constraints(
                        view(quantization, backend)
                    )
                    self.assertEqual(
                        result,
                        {"moe_runner_backend": expected} if expected else {},
                    )


if __name__ == "__main__":
    unittest.main()
