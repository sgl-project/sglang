"""CPU guards for UE8M0 weight quantization and DeepGEMM requantization."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import call, patch

import torch
from compressed_tensors.quantization import QuantizationStrategy

import sglang.srt.layers.quantization.fp8_utils as fp8_utils
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.quantization import fp8 as fp8_quant
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.test.test_utils import CustomTestCase

BLOCK_SIZE = [128, 128]


def _make_params(n: int = 64, k: int = 128):
    weight = torch.nn.Parameter(torch.zeros((n, k)), requires_grad=False)
    weight_scale = torch.nn.Parameter(torch.ones((1, 1)), requires_grad=False)
    weight_scale.format_ue8m0 = False
    return weight, weight_scale


class TestUE8M0WeightQuantization(CustomTestCase):
    def test_fp32_exports_match_bf16_quantization(self):
        """Upcasting BF16 weights for export must preserve their FP8 encoding."""
        generator = torch.Generator().manual_seed(0)
        for shape in ((256, 512), (2, 128, 512)):
            with self.subTest(shape=shape):
                weight = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
                exported = weight.float()
                expected_weight, expected_scale = fp8_utils.quant_weight_ue8m0(
                    weight, BLOCK_SIZE
                )
                actual_weight, actual_scale = fp8_utils.quant_weight_ue8m0(
                    exported, BLOCK_SIZE
                )
                self.assertTrue(
                    torch.equal(
                        actual_weight.view(torch.uint8),
                        expected_weight.view(torch.uint8),
                    )
                )
                torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
                self.assertEqual(exported.dtype, torch.float32)
                torch.testing.assert_close(exported, weight.float(), rtol=0, atol=0)

    def test_fp32_values_are_not_rounded_through_bf16(self):
        """FP32 values above an FP8 midpoint must not become BF16 ties first."""
        weight = torch.zeros((128, 128), dtype=torch.float32)
        weight[0, 0] = 448.0
        # With scale 1, the E4M3 midpoint is 1.0625, exactly representable in BF16.
        weight[0, 1] = 1.063

        quantized, scale = fp8_utils.quant_weight_ue8m0(weight, BLOCK_SIZE)
        rounded, _ = fp8_utils.quant_weight_ue8m0(weight.bfloat16(), BLOCK_SIZE)

        self.assertEqual(scale.item(), 1.0)
        self.assertEqual(quantized[0, 1].float().item(), 1.125)
        self.assertEqual(rounded[0, 1].float().item(), 1.0)


class TestDeepGemmUE8M0Requant(CustomTestCase):
    def _enabled_deepgemm_ue8m0(self):
        return patch.multiple(
            deep_gemm_wrapper,
            ENABLE_JIT_DEEPGEMM=True,
            DEEPGEMM_SCALE_UE8M0=True,
        )

    def test_helper_requants_supported_deepgemm_bf16_once(self):
        weight, weight_scale = _make_params()

        with (
            self._enabled_deepgemm_ue8m0(),
            patch.object(fp8_utils, "requant_weight_ue8m0_inplace") as requant,
        ):
            fired = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                BLOCK_SIZE,
                use_deepgemm_runner=True,
                output_dtype=torch.bfloat16,
                weight_shape=weight.shape,
            )
            fired_again = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                BLOCK_SIZE,
                use_deepgemm_runner=True,
                output_dtype=torch.bfloat16,
                weight_shape=weight.shape,
            )

        self.assertTrue(fired)
        self.assertFalse(fired_again)
        self.assertTrue(weight_scale.format_ue8m0)
        requant.assert_called_once_with(weight, weight_scale, BLOCK_SIZE)

    def test_helper_skips_non_bf16_output(self):
        weight, weight_scale = _make_params()

        with (
            self._enabled_deepgemm_ue8m0(),
            patch.object(fp8_utils, "requant_weight_ue8m0_inplace") as requant,
        ):
            fired = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                BLOCK_SIZE,
                use_deepgemm_runner=True,
                output_dtype=torch.float16,
                weight_shape=weight.shape,
            )

        self.assertFalse(fired)
        self.assertFalse(weight_scale.format_ue8m0)
        requant.assert_not_called()

    def test_helper_skips_shape_deepgemm_will_not_run(self):
        weight, weight_scale = _make_params(n=96, k=128)

        with (
            self._enabled_deepgemm_ue8m0(),
            patch.object(fp8_utils, "requant_weight_ue8m0_inplace") as requant,
        ):
            fired = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                BLOCK_SIZE,
                use_deepgemm_runner=True,
                output_dtype=torch.bfloat16,
                weight_shape=weight.shape,
            )

        self.assertFalse(fired)
        self.assertFalse(weight_scale.format_ue8m0)
        requant.assert_not_called()

    def test_helper_skips_non_deepgemm_runner(self):
        weight, weight_scale = _make_params()

        with (
            self._enabled_deepgemm_ue8m0(),
            patch.object(fp8_utils, "requant_weight_ue8m0_inplace") as requant,
        ):
            fired = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                BLOCK_SIZE,
                use_deepgemm_runner=False,
                output_dtype=torch.bfloat16,
                weight_shape=weight.shape,
            )

        self.assertFalse(fired)
        self.assertFalse(weight_scale.format_ue8m0)
        requant.assert_not_called()

    def test_helper_skips_unsupported_block_size(self):
        weight, weight_scale = _make_params()
        unsupported_block_size = [128, 256]

        with (
            self._enabled_deepgemm_ue8m0(),
            patch.object(fp8_utils, "requant_weight_ue8m0_inplace") as requant,
        ):
            fired = fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
                weight,
                weight_scale,
                unsupported_block_size,
                use_deepgemm_runner=True,
                output_dtype=torch.bfloat16,
                weight_shape=weight.shape,
            )

        self.assertFalse(fired)
        self.assertFalse(weight_scale.format_ue8m0)
        requant.assert_not_called()

    def test_compressed_tensors_block_processing_preserves_ue8m0_marker(self):
        scheme = CompressedTensorsW8A8Fp8.__new__(CompressedTensorsW8A8Fp8)
        scheme.strategy = QuantizationStrategy.BLOCK
        scheme.is_static_input_scheme = False
        scheme.weight_block_size = BLOCK_SIZE
        scheme.w8a8_block_fp8_linear = (
            fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        )

        layer = torch.nn.Module()
        layer.weight, layer.weight_scale = _make_params()
        layer.orig_dtype = torch.bfloat16

        with (
            self._enabled_deepgemm_ue8m0(),
            patch.object(fp8_utils, "requant_weight_ue8m0_inplace") as requant,
        ):
            scheme.process_weights_after_loading(layer)
            scheme.process_weights_after_loading(layer)

        self.assertTrue(layer.weight_scale.format_ue8m0)
        requant.assert_called_once()

    def test_fp8_moe_requants_standard_layer_for_deepgemm(self):
        method = fp8_quant.Fp8MoEMethod.__new__(fp8_quant.Fp8MoEMethod)
        method.convert_mxfp8_to_block = False
        method.use_mxfp8 = False
        method.is_fp4_expert = False
        method.dequant_fp4_to_fp8 = False
        method.quant_config = unittest.mock.Mock(weight_block_size=BLOCK_SIZE)

        layer = torch.nn.Module()
        layer.w13_weight, layer.w13_weight_scale_inv = _make_params()
        layer.w2_weight, layer.w2_weight_scale_inv = _make_params()

        def _mark_ue8m0(weight, weight_scale, *args, **kwargs):
            weight_scale.format_ue8m0 = True
            return True

        with (
            patch.multiple(
                fp8_quant,
                _is_cpu=False,
                _is_fp8_fnuz=False,
                _use_aiter=False,
            ),
            patch.object(
                method, "is_deepgemm_moe_runner_backend_enabled", return_value=True
            ),
            patch.object(
                fp8_quant,
                "requant_block_scale_ue8m0_for_deepgemm",
                side_effect=_mark_ue8m0,
            ) as requant,
        ):
            method.process_weights_after_loading_block_quant(layer)

        self.assertEqual(
            requant.call_args_list,
            [
                call(
                    layer.w13_weight,
                    layer.w13_weight_scale_inv,
                    BLOCK_SIZE,
                    use_deepgemm_runner=True,
                    output_dtype=torch.bfloat16,
                    weight_shape=layer.w13_weight.shape[-2:],
                ),
                call(
                    layer.w2_weight,
                    layer.w2_weight_scale_inv,
                    BLOCK_SIZE,
                    use_deepgemm_runner=True,
                    output_dtype=torch.bfloat16,
                    weight_shape=layer.w2_weight.shape[-2:],
                ),
            ],
        )
        self.assertTrue(layer.w13_weight_scale_inv.format_ue8m0)
        self.assertTrue(layer.w2_weight_scale_inv.format_ue8m0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
