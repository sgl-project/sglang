"""CPU-only tests for prequantized MXFP4 inputs to the Quark AITER ASM GEMM."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    QuarkW4A4MXFP4,
    _AsmShuffledMxfp4Input,
    _validate_asm_prequantized_input,
)
from sglang.test.test_utils import CustomTestCase


class TestQuarkMxfp4AsmPrequantizedInput(CustomTestCase):
    def test_prequantized_asm_tuple_contract(self):
        layer = SimpleNamespace(weight=torch.empty((128, 128), dtype=torch.uint8))
        x_q = torch.empty((4, 128), dtype=quark_w4a4_mxfp4._aiter_fp4x2_dtype)
        x_scales = torch.empty((256, 8), dtype=quark_w4a4_mxfp4._aiter_e8m0_dtype)

        actual = _validate_asm_prequantized_input(layer, (x_q, x_scales))

        self.assertIs(actual[0], x_q)
        self.assertIs(actual[1], x_scales)

    def test_prequantized_asm_tuple_rejects_wrong_layout(self):
        layer = SimpleNamespace(weight=torch.empty((128, 128), dtype=torch.uint8))
        x_q = torch.empty((4, 128), dtype=quark_w4a4_mxfp4._aiter_fp4x2_dtype)
        x_scales = torch.empty((256, 8), dtype=quark_w4a4_mxfp4._aiter_e8m0_dtype)

        with self.assertRaisesRegex(ValueError, "padded, shuffled"):
            _validate_asm_prequantized_input(layer, (x_q, x_scales[:4]))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            _validate_asm_prequantized_input(
                layer,
                (
                    torch.empty((4, 256), dtype=quark_w4a4_mxfp4._aiter_fp4x2_dtype)[
                        :, ::2
                    ],
                    x_scales,
                ),
            )
        with self.assertRaisesRegex(ValueError, "native FP4x2"):
            _validate_asm_prequantized_input(
                layer,
                (torch.empty(x_q.shape, dtype=torch.int8, device=x_q.device), x_scales),
            )
        with self.assertRaisesRegex(ValueError, "divisible by 256"):
            _validate_asm_prequantized_input(
                SimpleNamespace(weight=torch.empty((128, 64), dtype=torch.uint8)),
                (
                    torch.empty((4, 64), dtype=quark_w4a4_mxfp4._aiter_fp4x2_dtype),
                    x_scales,
                ),
            )

    def test_scheme_prepares_consumer_specific_layout(self):
        scheme = QuarkW4A4MXFP4({}, {}, is_checkpoint_mxfp4_serialized=True)
        layer = SimpleNamespace(
            dequantized_bf16=False,
            input_size_per_partition=256,
            use_aiter_asm_fp4_gemm=False,
        )
        x = torch.empty((4, 64), dtype=torch.bfloat16)
        z = torch.empty_like(x)
        weight = torch.empty(64, dtype=torch.bfloat16)
        expected = (
            torch.empty((1, 128), dtype=torch.uint8),
            torch.empty((1, 8), dtype=torch.uint8),
        )
        quant = Mock(return_value=expected)
        can_use = Mock(return_value=True)

        with (
            patch.object(quark_w4a4_mxfp4, "_is_hip", True),
            patch.object(
                quark_w4a4_mxfp4,
                "_get_fused_rmsnorm_gated_ops",
                return_value=(can_use, quant),
            ),
        ):
            actual = scheme.prepare_fused_rmsnorm_gated_input(
                layer,
                x,
                z,
                weight,
                1e-6,
                num_heads=4,
                activation="silu",
            )
            self.assertIs(actual, expected)
            self.assertEqual(quant.call_args.kwargs["round_mode"], 2)
            self.assertFalse(quant.call_args.kwargs["shuffle_scales"])
            self.assertFalse(quant.call_args.kwargs["use_native_dtypes"])
            self.assertFalse(can_use.call_args.kwargs["shuffle_scales"])

            layer.use_aiter_asm_fp4_gemm = True
            quant.reset_mock()
            can_use.reset_mock()
            actual = scheme.prepare_fused_rmsnorm_gated_input(
                layer,
                x,
                z,
                weight,
                1e-6,
                num_heads=4,
                activation="silu",
            )
            self.assertIsInstance(actual, _AsmShuffledMxfp4Input)
            self.assertIs(actual[0], expected[0])
            self.assertIs(actual[1], expected[1])
            self.assertEqual(quant.call_args.kwargs["round_mode"], 1)
            self.assertTrue(quant.call_args.kwargs["shuffle_scales"])
            self.assertTrue(quant.call_args.kwargs["use_native_dtypes"])
            self.assertTrue(can_use.call_args.kwargs["shuffle_scales"])

    def test_scheme_falls_back_for_incompatible_consumer(self):
        scheme = QuarkW4A4MXFP4({}, {}, is_checkpoint_mxfp4_serialized=True)
        layer = SimpleNamespace(
            dequantized_bf16=False,
            input_size_per_partition=256,
            use_aiter_asm_fp4_gemm=True,
        )
        x = torch.empty((4, 64), dtype=torch.bfloat16)
        z = torch.empty_like(x)
        weight = torch.empty(64, dtype=torch.bfloat16)
        quant = Mock()

        with (
            patch.object(quark_w4a4_mxfp4, "_is_hip", True),
            patch.object(
                quark_w4a4_mxfp4,
                "_get_fused_rmsnorm_gated_ops",
                return_value=(Mock(return_value=True), quant),
            ),
            patch.object(quark_w4a4_mxfp4, "_aiter_asm_mxfp4_round_mode", 0),
        ):
            self.assertIsNone(
                scheme.prepare_fused_rmsnorm_gated_input(
                    layer,
                    x,
                    z,
                    weight,
                    1e-6,
                    num_heads=4,
                    activation="silu",
                )
            )
            quant.assert_not_called()

            layer.input_size_per_partition = 512
            self.assertIsNone(
                scheme.prepare_fused_rmsnorm_gated_input(
                    layer,
                    x,
                    z,
                    weight,
                    1e-6,
                    num_heads=4,
                    activation="silu",
                )
            )

            layer.dequantized_bf16 = True
            self.assertIsNone(
                scheme.prepare_fused_rmsnorm_gated_input(
                    layer,
                    x,
                    z,
                    weight,
                    1e-6,
                    num_heads=4,
                    activation="silu",
                )
            )

    def test_asm_gemm_consumes_prequantized_tuple_without_requantizing(self):
        scheme = QuarkW4A4MXFP4({}, {}, is_checkpoint_mxfp4_serialized=True)
        layer = SimpleNamespace(
            use_aiter_asm_fp4_gemm=True,
            weight=torch.empty((128, 128), dtype=torch.uint8),
            weight_scale=torch.empty((128, 8), dtype=torch.uint8),
        )
        x_q = torch.empty((4, 128), dtype=quark_w4a4_mxfp4._aiter_fp4x2_dtype)
        x_scales = torch.empty((256, 8), dtype=quark_w4a4_mxfp4._aiter_e8m0_dtype)
        expected = torch.empty((4, 128), dtype=scheme.out_dtype)

        with (
            patch.object(
                quark_w4a4_mxfp4,
                "gemm_a4w4",
                return_value=expected,
                create=True,
            ) as gemm,
            patch.object(
                quark_w4a4_mxfp4,
                "per_1x32_f4_quant",
                create=True,
            ) as quant,
        ):
            actual = scheme.apply_weights(
                layer, _AsmShuffledMxfp4Input((x_q, x_scales))
            )

        self.assertIs(actual, expected)
        quant.assert_not_called()
        gemm.assert_called_once()
        # Already shuffled: the scales reach gemm_a4w4 untouched.
        self.assertIs(gemm.call_args.args[0], x_q)
        self.assertIs(gemm.call_args.args[2], x_scales)

    def test_asm_gemm_shuffles_row_major_tuple_scales(self):
        scheme = QuarkW4A4MXFP4({}, {}, is_checkpoint_mxfp4_serialized=True)
        layer = SimpleNamespace(
            use_aiter_asm_fp4_gemm=True,
            weight=torch.empty((128, 128), dtype=torch.uint8),
            weight_scale=torch.empty((128, 8), dtype=torch.uint8),
        )
        x_q = torch.zeros((4, 128), dtype=torch.uint8)
        x_scales = torch.full((4, 8), 0x7F, dtype=torch.uint8)
        expected = torch.empty((4, 128), dtype=scheme.out_dtype)

        with patch.object(
            quark_w4a4_mxfp4, "gemm_a4w4", return_value=expected, create=True
        ) as gemm:
            actual = scheme.apply_weights(layer, (x_q, x_scales))

        self.assertIs(actual, expected)
        self.assertEqual(tuple(gemm.call_args.args[2].shape), (256, 8))

    def test_asm_gemm_keeps_plain_tensor_quantization_path(self):
        scheme = QuarkW4A4MXFP4({}, {}, is_checkpoint_mxfp4_serialized=True)
        layer = SimpleNamespace(
            use_aiter_asm_fp4_gemm=True,
            weight=torch.empty((128, 128), dtype=torch.uint8),
            weight_scale=torch.empty((128, 8), dtype=torch.uint8),
        )
        x = torch.empty((4, 256), dtype=torch.bfloat16)
        x_q = torch.empty((4, 128), dtype=quark_w4a4_mxfp4._aiter_fp4x2_dtype)
        x_scales = torch.empty((256, 8), dtype=quark_w4a4_mxfp4._aiter_e8m0_dtype)
        expected = torch.empty((4, 128), dtype=scheme.out_dtype)

        with (
            patch.object(
                quark_w4a4_mxfp4,
                "per_1x32_f4_quant",
                return_value=(x_q, x_scales),
                create=True,
            ) as quant,
            patch.object(
                quark_w4a4_mxfp4,
                "gemm_a4w4",
                return_value=expected,
                create=True,
            ) as gemm,
        ):
            actual = scheme.apply_weights(layer, x)

        self.assertIs(actual, expected)
        quant.assert_called_once_with(x)
        gemm.assert_called_once()

    def test_triton_gemm_keeps_existing_prequantized_tuple_path(self):
        scheme = QuarkW4A4MXFP4({}, {}, is_checkpoint_mxfp4_serialized=True)
        layer = SimpleNamespace(
            use_aiter_asm_fp4_gemm=False,
            weight=torch.empty((128, 64), dtype=torch.uint8),
            weight_scale=torch.empty((128, 4), dtype=torch.uint8),
        )
        x_q = torch.empty((4, 64), dtype=torch.uint8)
        x_scales = torch.empty((4, 4), dtype=torch.uint8)

        with (
            patch.object(quark_w4a4_mxfp4, "gemm_afp4wfp4", create=True) as gemm,
            patch.object(quark_w4a4_mxfp4, "dynamic_mxfp4_quant", create=True) as quant,
        ):
            actual = scheme.apply_weights(layer, (x_q, x_scales))

        self.assertEqual(actual.shape, (4, 128))
        quant.assert_not_called()
        gemm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
