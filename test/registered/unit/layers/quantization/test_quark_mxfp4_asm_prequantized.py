"""CPU-only tests for prequantized MXFP4 inputs to the Quark AITER ASM GEMM."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    QuarkW4A4MXFP4,
    _AsmShuffledMxfp4Input,
    _validate_asm_prequantized_input,
)
from sglang.test.test_utils import CustomTestCase

FP4X2 = quark_w4a4_mxfp4._aiter_fp4x2_dtype
E8M0 = quark_w4a4_mxfp4._aiter_e8m0_dtype


class TestQuarkMxfp4AsmPrequantizedInput(CustomTestCase):
    def setUp(self):
        self.layer = SimpleNamespace(weight=torch.empty((128, 128), dtype=torch.uint8))
        self.x_q = torch.empty((4, 128), dtype=FP4X2)
        self.x_scales = torch.empty((256, 8), dtype=E8M0)

    def test_tuple_contract(self):
        actual = _validate_asm_prequantized_input(self.layer, (self.x_q, self.x_scales))
        self.assertIs(actual[0], self.x_q)
        self.assertIs(actual[1], self.x_scales)

    def test_rejects_wrong_layout(self):
        with self.assertRaisesRegex(ValueError, "padded, shuffled"):
            _validate_asm_prequantized_input(self.layer, (self.x_q, self.x_scales[:4]))
        with self.assertRaisesRegex(ValueError, "native FP4x2"):
            _validate_asm_prequantized_input(
                self.layer, (self.x_q.view(torch.int8), self.x_scales)
            )

    def test_asm_gemm_consumes_tuple_without_requantizing(self):
        scheme = QuarkW4A4MXFP4({}, {}, is_checkpoint_mxfp4_serialized=True)
        layer = SimpleNamespace(
            use_aiter_asm_fp4_gemm=True,
            weight=self.layer.weight,
            weight_scale=torch.empty((128, 8), dtype=torch.uint8),
        )
        expected = torch.empty((4, 128), dtype=scheme.out_dtype)
        with (
            patch.object(
                quark_w4a4_mxfp4, "gemm_a4w4", return_value=expected, create=True
            ) as gemm,
            patch.object(quark_w4a4_mxfp4, "per_1x32_f4_quant", create=True) as quant,
        ):
            actual = scheme.apply_weights(
                layer, _AsmShuffledMxfp4Input((self.x_q, self.x_scales))
            )
        self.assertIs(actual, expected)
        quant.assert_not_called()
        # Already shuffled: the scales reach gemm_a4w4 untouched.
        self.assertIs(gemm.call_args.args[0], self.x_q)
        self.assertIs(gemm.call_args.args[2], self.x_scales)


if __name__ == "__main__":
    unittest.main()
