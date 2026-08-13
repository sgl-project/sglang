import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    npu_w8a8_block_fp8_linear,
)


class TestNPUW8A8BlockFP8Linear(unittest.TestCase):
    def test_rejects_unsupported_block_size(self):
        with self.assertRaisesRegex(ValueError, r"block_size \[128, 128\]"):
            npu_w8a8_block_fp8_linear(
                torch.empty(1, 128, dtype=torch.bfloat16),
                torch.empty(128, 64, dtype=torch.float8_e4m3fn),
                [64, 128],
                torch.empty(1),
            )

    def test_rejects_non_fp8_weight(self):
        with self.assertRaisesRegex(ValueError, "expects float8_e4m3fn weights"):
            npu_w8a8_block_fp8_linear(
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
            output = npu_w8a8_block_fp8_linear(
                input_tensor,
                weight,
                [128, 128],
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
            npu_w8a8_block_fp8_linear(input_tensor, weight, [128, 128], weight_scale)

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
            npu_w8a8_block_fp8_linear(
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
