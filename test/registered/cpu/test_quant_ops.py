import unittest
from types import SimpleNamespace

import sgl_kernel  # noqa: F401
import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from sglang.srt.utils import cpu_has_amx_support
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


class TestCPUQuantOps(CustomTestCase):
    def test_per_token_quant_int8_python_dispatch_cpu(self):
        """Public CPU dispatch keeps the CUDA dtype and scale-shape contract."""
        x = torch.linspace(-1.0, 1.0, steps=2 * 3 * 33).reshape(2, 3, 33)
        x = x.to(torch.bfloat16)

        x_q, scales, x_sum = per_token_quant_int8(x, cal_sum=True)

        self.assertEqual(x_q.dtype, torch.int8)
        self.assertEqual(x_q.shape, x.shape)
        self.assertEqual(scales.shape, x.shape[:-1] + (1,))
        torch.testing.assert_close(
            x_q.float() * scales.float(), x.float(), atol=0.02, rtol=0.02
        )
        torch.testing.assert_close(x_sum, x.sum(dim=-1))

    @unittest.skipUnless(
        cpu_has_amx_support(),
        "Compressed-tensors W8A8 INT8 CPU path requires the Intel AMX backend.",
    )
    def test_compressed_tensors_w8a8_int8_cpu(self):
        x = torch.randn(3, 64, dtype=torch.bfloat16)
        weight = torch.randint(-16, 16, (64, 64), dtype=torch.int8)
        weight_scale = torch.rand(64, dtype=torch.float32) / 16
        bias = torch.randn(64, dtype=torch.float32) / 10
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale=torch.nn.Parameter(weight_scale, requires_grad=False),
        )
        scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, False, True)
        scheme.process_weights_after_loading(layer)
        self.assertTrue(layer.use_intel_amx_backend)

        out = scheme.apply_weights(layer, x, bias)
        ref = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            x, layer.weight, layer.weight_scale, bias, x.dtype, True
        )

        torch.testing.assert_close(out, ref)

    def test_compressed_tensors_w8a8_int8_cpu_rejects_unpacked_weight(self):
        """Unsupported CPU dimensions fail before using the packed AMX contract."""
        x = torch.randn(3, 33, dtype=torch.bfloat16)
        weight = torch.randint(-16, 16, (16, 33), dtype=torch.int8)
        weight_scale = torch.rand(16, dtype=torch.float32) / 16
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale=torch.nn.Parameter(weight_scale, requires_grad=False),
        )
        scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, False, True)
        scheme.process_weights_after_loading(layer)

        self.assertFalse(layer.use_intel_amx_backend)
        with self.assertRaisesRegex(NotImplementedError, "AMX-packed weights"):
            scheme.apply_weights(layer, x, None)


if __name__ == "__main__":
    unittest.main()
