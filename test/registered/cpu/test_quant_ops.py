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


def _reference_w8a8_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    dequant_weight = weight.float() * weight_scale.float().view(-1, 1)
    out = x.float().matmul(dequant_weight.t())
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def _requantize_tensor_scale_int8(
    weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    max_w_scale = weight_scale.max()
    int8_info = torch.iinfo(torch.int8)
    requantized_weight = weight.clone()
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        weight_dq = weight[start:end, :].float() * weight_scale[idx]
        requantized_weight[start:end, :] = torch.clamp(
            torch.round(weight_dq / max_w_scale),
            int8_info.min,
            int8_info.max,
        ).to(torch.int8)
        start = end
    return requantized_weight, max_w_scale.expand(weight.size(0)).clone()


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
        x = torch.linspace(-0.8, 0.9, steps=3 * 64, dtype=torch.float32).reshape(3, 64)
        x = x.to(torch.bfloat16)
        weight = (torch.arange(64 * 64).reshape(64, 64) % 33 - 16).to(torch.int8)
        weight_scale = torch.linspace(0.005, 0.02, steps=64, dtype=torch.float32)
        bias = torch.linspace(-0.1, 0.1, steps=64, dtype=torch.float32)
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale=torch.nn.Parameter(weight_scale, requires_grad=False),
        )
        scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, False, True)
        scheme.process_weights_after_loading(layer)
        self.assertTrue(layer.use_intel_amx_backend)

        out = scheme.apply_weights(layer, x, bias)
        ref = _reference_w8a8_int8_linear(x, weight, weight_scale, bias)

        torch.testing.assert_close(out, ref, atol=0.12, rtol=0.05)

    @unittest.skipUnless(
        cpu_has_amx_support(),
        "Compressed-tensors W8A8 INT8 CPU path requires the Intel AMX backend.",
    )
    def test_compressed_tensors_w8a8_int8_cpu_tensor_scales(self):
        x = torch.linspace(-0.7, 0.8, steps=3 * 64, dtype=torch.float32).reshape(3, 64)
        x = x.to(torch.bfloat16)
        weight = (torch.arange(64 * 64).reshape(64, 64) % 41 - 20).to(torch.int8)
        weight_scale = torch.tensor([0.01, 0.04], dtype=torch.float32)
        bias = torch.linspace(-0.05, 0.05, steps=64, dtype=torch.float32)
        logical_widths = [32, 32]
        ref_weight, ref_weight_scale = _requantize_tensor_scale_int8(
            weight, weight_scale, logical_widths
        )
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale=torch.nn.Parameter(weight_scale, requires_grad=False),
            logical_widths=logical_widths,
        )
        scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.TENSOR, False, True)
        scheme.process_weights_after_loading(layer)
        self.assertTrue(layer.use_intel_amx_backend)
        self.assertEqual(layer.weight_scale.numel(), weight.size(0))

        out = scheme.apply_weights(layer, x, bias)
        ref = _reference_w8a8_int8_linear(x, ref_weight, ref_weight_scale, bias)

        torch.testing.assert_close(out, ref, atol=0.16, rtol=0.06)

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
