import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

# Load the quantization package first so `base_config`, `moe_methods`, and
# `linear_method_npu` initialize in dependency order. Importing
# `linear_method_npu` directly from a cold process triggers a circular import:
# linear_method_npu -> base_config -> quantization/__init__ ->
# gguf/unquant/gptq_moe -> moe_methods -> linear_method_npu (partially
# initialized, `_get_float8_e8m0fnu_dtype` not yet defined). Initializing the
# package first mirrors how the engine loads quantization at model-config time.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    npu_w8a8_mxfp8_linear,
)
from sglang.srt.hardware_backend.npu.quantization.w8a8_mxfp8 import (
    process_npu_arch35_mxfp8_linear_weights,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config


class TestNPUW8A8BlockFP8Linear(unittest.TestCase):
    def test_fp8_config_preserves_ue8m0_scale_format(self):
        quant_config = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "scale_fmt": "ue8m0",
            }
        )

        self.assertEqual(quant_config.scale_fmt, "ue8m0")

    def test_layout_only_ue8m0_conversion_preserves_fp8_weight(self):
        original_weight = torch.randint(1, 255, (128, 64), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(original_weight.clone(), requires_grad=False),
            weight_scale_inv=torch.nn.Parameter(
                torch.tensor([[2**-12]], dtype=torch.float32), requires_grad=False
            ),
        )

        process_npu_arch35_mxfp8_linear_weights(layer, [128, 128], scale_fmt="ue8m0")

        self.assertEqual(layer.weight.shape, (64, 128))
        torch.testing.assert_close(
            layer.weight.data.T.contiguous().view(torch.uint8),
            original_weight.view(torch.uint8),
        )
        self.assertEqual(layer.weight_scale_inv.shape, (1, 128, 2))
        self.assertTrue(
            torch.equal(
                layer.weight_scale_inv.data,
                torch.full((1, 128, 2), 0x73, dtype=torch.uint8),
            )
        )
        self.assertTrue(layer.weight_scale_inv.format_ue8m0)

    def test_rejects_non_ue8m0_scale_format(self):
        with self.assertRaisesRegex(ValueError, "scale_fmt='ue8m0'"):
            process_npu_arch35_mxfp8_linear_weights(
                SimpleNamespace(), [128, 128], scale_fmt="float32"
            )

    def test_rejects_non_fp8_weight(self):
        with self.assertRaisesRegex(ValueError, "expects float8_e4m3fn weights"):
            npu_w8a8_mxfp8_linear(
                torch.empty(1, 128, dtype=torch.bfloat16),
                torch.empty(128, 64, dtype=torch.bfloat16),
                [128, 128],
                torch.empty(1),
            )

    def test_quantizes_flattened_input_and_restores_batch_shape(self):
        input_tensor = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)
        bias = torch.randn(64, dtype=torch.float32)
        quantized = torch.empty(6, 128, dtype=torch.float8_e4m3fn)
        input_scale = torch.empty(6, 2, 2, dtype=torch.uint8)
        matmul_output = torch.randn(6, 64, dtype=torch.bfloat16)

        npu_ops = MagicMock()
        npu_ops.npu_dynamic_mx_quant.return_value = (quantized, input_scale)
        npu_ops.npu_quant_matmul.return_value = matmul_output
        with patch.object(torch.ops, "npu", npu_ops, create=True):
            output = npu_w8a8_mxfp8_linear(
                input_tensor,
                weight,
                [64, 128],
                weight_scale,
                bias=bias,
            )

        self.assertEqual(output.shape, (2, 3, 64))
        quant_call = npu_ops.npu_dynamic_mx_quant.call_args
        self.assertEqual(quant_call.args[0].shape, (6, 128))
        self.assertEqual(quant_call.kwargs["dst_type"], torch.float8_e4m3fn)

        matmul_call = npu_ops.npu_quant_matmul.call_args
        self.assertIs(matmul_call.args[0], quantized)
        self.assertIs(matmul_call.args[1], weight)
        self.assertIs(matmul_call.kwargs["scale"], weight_scale)
        self.assertIs(matmul_call.kwargs["pertoken_scale"], input_scale)
        self.assertIs(matmul_call.kwargs["bias"], bias)
        self.assertEqual(matmul_call.kwargs["group_sizes"], (1, 1, 32))

    def test_rejects_noncontiguous_input(self):
        input_tensor = torch.randn(2, 3, 128, dtype=torch.bfloat16).transpose(0, 1)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)

        with self.assertRaisesRegex(RuntimeError, "view size is not compatible"):
            npu_w8a8_mxfp8_linear(input_tensor, weight, [64, 128], weight_scale)

    def test_preserves_supported_input_dtype(self):
        input_tensor = torch.randn(2, 128, dtype=torch.float16)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)
        npu_ops = MagicMock()
        npu_ops.npu_dynamic_mx_quant.return_value = (
            torch.empty(2, 128, dtype=torch.float8_e4m3fn),
            torch.empty(2, 2, 2, dtype=torch.uint8),
        )
        npu_ops.npu_quant_matmul.return_value = torch.empty(2, 64)

        with patch.object(torch.ops, "npu", npu_ops, create=True):
            npu_w8a8_mxfp8_linear(input_tensor, weight, [128, 128], weight_scale)

        self.assertEqual(
            npu_ops.npu_quant_matmul.call_args.kwargs["output_dtype"],
            torch.float16,
        )

    def test_converts_bias_to_float32(self):
        input_tensor = torch.randn(2, 128, dtype=torch.bfloat16)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)
        bias = torch.randn(64, dtype=torch.bfloat16)
        npu_ops = MagicMock()
        npu_ops.npu_dynamic_mx_quant.return_value = (
            torch.empty(2, 128, dtype=torch.float8_e4m3fn),
            torch.empty(2, 2, 2, dtype=torch.uint8),
        )
        npu_ops.npu_quant_matmul.return_value = torch.empty(2, 64)

        with patch.object(torch.ops, "npu", npu_ops, create=True):
            npu_w8a8_mxfp8_linear(
                input_tensor,
                weight,
                [128, 128],
                weight_scale,
                bias=bias,
            )

        quant_bias = npu_ops.npu_quant_matmul.call_args.kwargs["bias"]
        self.assertEqual(quant_bias.dtype, torch.float32)
        torch.testing.assert_close(quant_bias, bias.float())


if __name__ == "__main__":
    unittest.main()
