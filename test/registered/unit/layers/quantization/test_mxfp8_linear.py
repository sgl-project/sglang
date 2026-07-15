"""CPU contracts for the shared MXFP8 dense-linear path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMxfp8LinearMethod(CustomTestCase):
    def test_modelopt_mixed_auto_selects_cutlass(self):
        server_args = SimpleNamespace(
            fp8_gemm_runner_backend="auto", quantization="modelopt_mixed"
        )
        with (
            patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", None),
            patch.object(fp8_utils, "_is_sm100_supported", True),
            patch.object(fp8_utils, "is_flashinfer_available", return_value=True),
        ):
            fp8_utils.initialize_fp8_gemm_config(server_args)
            self.assertEqual(
                fp8_utils.get_fp8_gemm_runner_backend(),
                Fp8GemmRunnerBackend.FLASHINFER_CUTLASS,
            )

    def test_cutlass_scale_swizzle_matches_reference_layout(self):
        n, k = 128, 160
        scales = torch.randint(0, 256, (n, k // 32), dtype=torch.uint8)

        actual = Fp8LinearMethod._swizzle_mxfp8_cutlass_scale(scales, n, k).view(
            2, 32, 4, 4
        )
        padded = torch.nn.functional.pad(scales, (0, 3))
        for tile in range(2):
            expected = torch.stack(padded[:, tile * 4 : (tile + 1) * 4].chunk(4), dim=1)
            torch.testing.assert_close(actual[tile], expected)

    def test_cutlass_processing_preserves_canonical_weight(self):
        method = self._make_method()
        weight = torch.nn.Parameter(
            torch.empty((17, 160), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        layer = torch.nn.Module()
        layer.register_parameter("weight", weight)
        layer.register_parameter(
            "weight_scale_inv",
            torch.nn.Parameter(
                torch.empty((17, 5), dtype=torch.uint8), requires_grad=False
            ),
        )

        with patch(
            "sglang.srt.layers.quantization.fp8.get_fp8_gemm_runner_backend",
            return_value=Fp8GemmRunnerBackend.FLASHINFER_CUTLASS,
        ):
            method._process_mxfp8_linear_weight_scale(layer)

        self.assertIs(layer.weight, weight)
        self.assertEqual(layer.weight_mxfp8_cutlass.shape, (128, 160))
        self.assertEqual(layer.mxfp8_orig_n, 17)

    def test_cutlass_apply_slices_before_bias(self):
        method = self._make_method()
        method.w8a8_mxfp8_linear = lambda **_: torch.arange(
            3 * 128, dtype=torch.float32
        ).reshape(3, 128)
        layer = SimpleNamespace(
            weight_mxfp8_cutlass=torch.empty((128, 160)),
            weight_scale_inv_swizzled=torch.empty(0),
            mxfp8_orig_n=17,
        )
        bias = torch.arange(17, dtype=torch.float32)

        with patch(
            "sglang.srt.layers.quantization.fp8.get_fp8_gemm_runner_backend",
            return_value=Fp8GemmRunnerBackend.FLASHINFER_CUTLASS,
        ):
            output = method.apply(layer, torch.zeros((3, 160)), bias=bias)

        expected = (
            torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128)[:, :17] + bias
        )
        self.assertEqual(output.shape, (3, 17))
        self.assertTrue(output.is_contiguous())
        torch.testing.assert_close(output, expected)

    @staticmethod
    def _make_method() -> Fp8LinearMethod:
        method = Fp8LinearMethod.__new__(Fp8LinearMethod)
        method.quant_config = Fp8Config.__new__(Fp8Config)
        method.weight_scale_name = "weight_scale_inv"
        method.use_mxfp8 = True
        method.use_marlin = False
        return method


if __name__ == "__main__":
    unittest.main()
