"""CPU/mock contracts for the Diffusion ModelSlim MXFP8 adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

# Initialize the quantization package before the shared NPU module.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp8_scheme import (
    ModelSlimMXFP8Scheme,
)
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP8LinearMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDiffusionModelSlimMXFP8(unittest.TestCase):
    def test_checkpoint_layout_and_partial_scale_blocks(self):
        for k in (32, 64, 96, 4304):
            with self.subTest(k=k):
                layer = torch.nn.Module()
                scheme = ModelSlimMXFP8Scheme()
                loader = lambda *args: None
                scheme.create_weights(
                    layer, k, [8], k, 8, torch.bfloat16, weight_loader=loader
                )
                blocks = (k + 31) // 32
                self.assertEqual(layer.weight.shape, (8, k))
                self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(layer.weight_scale.shape, (8, blocks))
                self.assertEqual(layer.weight_scale.dtype, torch.uint8)
                self.assertIs(layer.weight.weight_loader, loader)
                self.assertIs(layer.weight_scale.weight_loader, loader)
                weight_bytes = torch.arange(8 * k, dtype=torch.uint8).reshape(8, k)
                scales = torch.arange(8 * blocks, dtype=torch.uint8).reshape(8, blocks)
                layer.weight.data.copy_(weight_bytes.view(torch.float8_e4m3fn))
                layer.weight_scale.data.copy_(scales)
                npu_ops = MagicMock()
                with patch.object(torch.ops, "npu", npu_ops, create=True):
                    scheme.process_weights_after_loading(layer)
                self.assertEqual(npu_ops.mock_calls, [])
                torch.testing.assert_close(
                    layer.weight.view(torch.uint8), weight_bytes.T
                )
                paired = F.pad(scales, (0, blocks % 2), value=0).reshape(8, -1, 2)
                torch.testing.assert_close(
                    layer.weight_scale_inv, paired.transpose(0, 1)
                )
                self.assertNotIn("weight_scale", dict(layer.named_parameters()))
                self.assertIsInstance(scheme.kernel, NPUMXFP8LinearMethod)

    def test_same_prepared_tensors_reach_shared_execution(self):
        weight = torch.empty(8, 96, dtype=torch.float8_e4m3fn).T
        weight_scale = torch.empty(8, 2, 2, dtype=torch.uint8).transpose(0, 1)
        layer = SimpleNamespace(
            weight=weight, weight_scale_inv=weight_scale, bias_fp32=None
        )
        scheme = ModelSlimMXFP8Scheme()
        for shape in ((6, 96), (2, 3, 96), (2, 1, 3, 96)):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                x = torch.arange(576, dtype=dtype).reshape(shape)
                for bias in (
                    None,
                    torch.arange(8, dtype=torch.float32),
                    torch.arange(8, dtype=torch.float16),
                ):
                    with self.subTest(shape=shape, dtype=dtype, bias=bias):
                        out_dtype = dtype if dtype != torch.float32 else torch.bfloat16
                        qx = torch.empty(6, 96, dtype=torch.float8_e4m3fn)
                        x_scale = torch.empty(6, 2, 2, dtype=torch.uint8)
                        expected = torch.arange(48, dtype=out_dtype).reshape(6, 8)
                        outputs = []
                        # The online Diffusion config selects this same SRT method.
                        for execute in (
                            NPUMXFP8LinearMethod().apply,
                            scheme.apply_weights,
                        ):
                            ops = MagicMock()
                            ops.npu_dynamic_mx_quant.return_value = (qx, x_scale)
                            ops.npu_quant_matmul.return_value = expected
                            with patch.object(torch.ops, "npu", ops, create=True):
                                output = execute(layer, x, bias)
                            ops.npu_dynamic_mx_quant.assert_called_once()
                            call = ops.npu_dynamic_mx_quant.call_args
                            torch.testing.assert_close(
                                call.args[0], x.to(out_dtype).reshape(6, 96)
                            )
                            self.assertEqual(
                                call.kwargs, {"dst_type": torch.float8_e4m3fn}
                            )
                            ops.npu_quant_matmul.assert_called_once()
                            call = ops.npu_quant_matmul.call_args
                            self.assertIs(call.args[0], qx)
                            self.assertIs(call.args[1], weight)
                            self.assertIs(call.args[2], weight_scale)
                            kwargs = call.kwargs.copy()
                            self.assertIs(kwargs.pop("pertoken_scale"), x_scale)
                            quant_bias = kwargs.pop("bias")
                            if bias is None:
                                self.assertIsNone(quant_bias)
                            else:
                                torch.testing.assert_close(quant_bias, bias.float())
                            e8m0 = getattr(torch, "float8_e8m0fnu", None)
                            self.assertEqual(
                                kwargs,
                                {
                                    "scale_dtype": e8m0,
                                    "pertoken_scale_dtype": e8m0,
                                    "output_dtype": out_dtype,
                                    "group_sizes": [1, 1, 32],
                                },
                            )
                            self.assertEqual(output.dtype, out_dtype)
                            self.assertEqual(output.shape, (*shape[:-1], 8))
                            outputs.append(output)
                        torch.testing.assert_close(
                            outputs[0], outputs[1], rtol=0, atol=0
                        )

    def test_noncontiguous_input_and_updated_bias(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.empty(8, 64, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(8, 2, dtype=torch.uint8), requires_grad=False
        )
        layer.bias = torch.nn.Parameter(
            torch.arange(8, dtype=torch.float16), requires_grad=False
        )
        scheme = ModelSlimMXFP8Scheme()
        scheme.process_weights_after_loading(layer)
        self.assertIsNone(layer.bias_fp32)
        layer.bias.data.add_(1)
        x = torch.arange(384, dtype=torch.bfloat16).reshape(2, 3, 64).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        ops = MagicMock()
        ops.npu_dynamic_mx_quant.return_value = (
            torch.empty(6, 64, dtype=torch.float8_e4m3fn),
            torch.empty(6, 1, 2, dtype=torch.uint8),
        )
        ops.npu_quant_matmul.return_value = torch.zeros(6, 8, dtype=x.dtype)
        with patch.object(torch.ops, "npu", ops, create=True):
            output = scheme.apply_weights(layer, x, layer.bias)
        torch.testing.assert_close(
            ops.npu_dynamic_mx_quant.call_args.args[0], x.reshape(6, 64)
        )
        torch.testing.assert_close(
            ops.npu_quant_matmul.call_args.kwargs["bias"], layer.bias.float()
        )
        self.assertEqual(output.shape, (3, 2, 8))


if __name__ == "__main__":
    unittest.main()
