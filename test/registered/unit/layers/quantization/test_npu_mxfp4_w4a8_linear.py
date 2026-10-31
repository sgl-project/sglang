"""CPU regression tests for Ascend MXFP4 W4A8 Linear helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Initialize the quantization package before the NPU module to avoid the
# base_config -> quantization -> moe_methods -> linear_method_npu import cycle.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization import linear_method_npu
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP4W4A8LinearMethod,
    NPUMXFP4W4A8OfflineLinearMethod,
    _prepare_mxfp4_w4a8_bias,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNPUMXFP4W4A8Linear(unittest.TestCase):
    def test_preserves_no_bias_path(self):
        self.assertIsNone(_prepare_mxfp4_w4a8_bias(None))

    def test_formats_vector_bias_for_a8w4_matmul(self):
        bias = torch.arange(8, dtype=torch.float32)

        quant_bias = _prepare_mxfp4_w4a8_bias(bias)

        self.assertEqual(quant_bias.dtype, torch.bfloat16)
        self.assertEqual(quant_bias.shape, (1, 8))
        self.assertTrue(torch.equal(quant_bias, bias.to(torch.bfloat16).unsqueeze(0)))

    def test_preserves_matrix_shape(self):
        bias = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)

        quant_bias = _prepare_mxfp4_w4a8_bias(bias)

        self.assertEqual(quant_bias.dtype, torch.bfloat16)
        self.assertEqual(quant_bias.shape, (1, 8))
        self.assertIs(quant_bias, bias)


class TestNPUMXFP4W4A8LinearExecution(unittest.TestCase):
    """Check execution with identical prepared tensors, not weight quantization."""

    def test_online_offline_execution_contract(self):
        # CPU tensors stand in for post-load NZ weights and paired UE8M0 scales.
        # Their transpose views must be passed to matmul without repacking.
        weight = torch.arange(32 * 8, dtype=torch.uint8).reshape(8, 32).T
        weight_scale = (
            torch.arange(16, dtype=torch.uint8).reshape(8, 1, 2).transpose(0, 1)
        )
        layer = SimpleNamespace(weight=weight, weight_scale=weight_scale)
        methods = (NPUMXFP4W4A8LinearMethod(), NPUMXFP4W4A8OfflineLinearMethod())
        e8m0_dtype = object()
        fp4_dtype = 296  # torch_npu's packed-FP4 enum; no NPU plugin needed here.

        for shape in ((6, 64), (2, 3, 64), (2, 1, 3, 64)):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                x = torch.arange(384, dtype=dtype).reshape(shape)
                for bias in (
                    None,
                    torch.arange(8, dtype=torch.float32),
                    torch.arange(8, dtype=torch.float16),
                    torch.arange(8, dtype=torch.bfloat16).reshape(1, 8),
                ):
                    with self.subTest(shape=shape, dtype=dtype, bias=bias):
                        outputs = []
                        output_dtype = (
                            dtype
                            if dtype in (torch.float16, torch.bfloat16)
                            else torch.bfloat16
                        )
                        quantized_x = torch.empty(6, 64, dtype=torch.float8_e4m3fn)
                        dynamic_scale = torch.empty(6, 1, 2, dtype=torch.uint8)
                        matmul_output = torch.arange(48, dtype=output_dtype).reshape(
                            6, 8
                        )
                        for method in methods:
                            npu_ops = MagicMock()
                            npu_ops.npu_dynamic_mx_quant.return_value = (
                                quantized_x,
                                dynamic_scale,
                            )
                            npu_ops.npu_quant_matmul.return_value = matmul_output
                            with (
                                patch.object(torch.ops, "npu", npu_ops, create=True),
                                patch.object(
                                    linear_method_npu,
                                    "_get_float8_e8m0fnu_dtype",
                                    return_value=e8m0_dtype,
                                ),
                                patch.object(
                                    linear_method_npu,
                                    "_get_float4_e2m1fn_x2_dtype",
                                    return_value=fp4_dtype,
                                ),
                            ):
                                output = method.apply(layer, x, bias)

                            npu_ops.npu_dynamic_mx_quant.assert_called_once()
                            quant_call = npu_ops.npu_dynamic_mx_quant.call_args
                            self.assertEqual(
                                quant_call.kwargs, {"dst_type": torch.float8_e4m3fn}
                            )
                            self.assertEqual(quant_call.args[0].dtype, output_dtype)
                            torch.testing.assert_close(
                                quant_call.args[0], x.to(output_dtype).reshape(6, 64)
                            )
                            npu_ops.npu_quant_matmul.assert_called_once()
                            matmul_call = npu_ops.npu_quant_matmul.call_args
                            self.assertEqual(len(matmul_call.args), 3)
                            self.assertIs(matmul_call.args[0], quantized_x)
                            self.assertIs(matmul_call.args[1], weight)
                            self.assertIs(matmul_call.args[2], weight_scale)
                            kwargs = matmul_call.kwargs.copy()
                            self.assertIs(kwargs.pop("pertoken_scale"), dynamic_scale)
                            quant_bias = kwargs.pop("bias")
                            self.assertEqual(
                                kwargs,
                                {
                                    "scale_dtype": e8m0_dtype,
                                    "pertoken_scale_dtype": e8m0_dtype,
                                    "output_dtype": output_dtype,
                                    "x2_dtype": fp4_dtype,
                                    "group_sizes": [0, 0, 32],
                                },
                            )
                            if bias is None:
                                self.assertIsNone(quant_bias)
                            else:
                                self.assertEqual(quant_bias.dtype, torch.bfloat16)
                                self.assertEqual(quant_bias.shape, (1, 8))
                                torch.testing.assert_close(
                                    quant_bias, bias.to(torch.bfloat16).reshape(1, 8)
                                )
                                if bias.dtype == torch.bfloat16 and bias.dim() == 2:
                                    self.assertIs(quant_bias, bias)
                            self.assertEqual(output.shape, (*shape[:-1], 8))
                            self.assertEqual(output.dtype, output_dtype)
                            torch.testing.assert_close(
                                output, matmul_output.reshape(*shape[:-1], 8)
                            )
                            self.assertIs(layer.weight, weight)
                            self.assertIs(layer.weight_scale, weight_scale)
                            outputs.append(output)
                        torch.testing.assert_close(
                            outputs[0], outputs[1], rtol=0, atol=0
                        )

    def test_both_entries_delegate_to_shared_execution(self):
        layer = SimpleNamespace(weight=object(), weight_scale=object())
        x, bias, expected = object(), object(), object()
        for method in (NPUMXFP4W4A8LinearMethod(), NPUMXFP4W4A8OfflineLinearMethod()):
            with (
                self.subTest(method=type(method).__name__),
                patch.object(
                    linear_method_npu, "npu_w4a8_mxfp4_linear", return_value=expected
                ) as execute,
            ):
                self.assertIs(method.apply(layer, x, bias), expected)
                execute.assert_called_once_with(
                    x, layer.weight, layer.weight_scale, bias
                )

    def test_noncontiguous_input(self):
        x = torch.arange(384, dtype=torch.bfloat16).reshape(2, 3, 64).transpose(0, 1)
        layer = SimpleNamespace(
            weight=torch.empty(32, 8, dtype=torch.uint8),
            weight_scale=torch.empty(1, 8, 2, dtype=torch.uint8),
        )
        self.assertFalse(x.is_contiguous())
        for method in (NPUMXFP4W4A8LinearMethod(), NPUMXFP4W4A8OfflineLinearMethod()):
            with self.subTest(method=type(method).__name__):
                npu_ops = MagicMock()
                npu_ops.npu_dynamic_mx_quant.return_value = (
                    torch.empty(6, 64, dtype=torch.float8_e4m3fn),
                    torch.empty(6, 1, 2, dtype=torch.uint8),
                )
                npu_ops.npu_quant_matmul.return_value = torch.empty(6, 8, dtype=x.dtype)
                with (
                    patch.object(torch.ops, "npu", npu_ops, create=True),
                    patch.object(
                        linear_method_npu,
                        "_get_float4_e2m1fn_x2_dtype",
                        return_value=296,
                    ),
                ):
                    output = method.apply(layer, x)
                torch.testing.assert_close(
                    npu_ops.npu_dynamic_mx_quant.call_args.args[0], x.reshape(6, 64)
                )
                self.assertEqual(output.shape, (3, 2, 8))


if __name__ == "__main__":
    unittest.main()
