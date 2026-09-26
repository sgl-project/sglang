import unittest
from types import SimpleNamespace

import sgl_kernel  # noqa: F401
import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


class TestCPUQuantOps(CustomTestCase):
    def test_per_token_quant_int8_python_dispatch_cpu(self):
        x = torch.randn(4, 33, dtype=torch.bfloat16)

        x_q, scales, x_sum = per_token_quant_int8(x, cal_sum=True)
        ref_q, ref_scales = torch.ops.sgl_kernel.per_token_quant_int8_cpu(
            x.contiguous()
        )

        torch.testing.assert_close(x_q, ref_q)
        torch.testing.assert_close(scales, ref_scales)
        torch.testing.assert_close(x_sum, x.sum(dim=-1))

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

        out = scheme.apply_weights(layer, x, bias)
        ref = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            x, layer.weight, layer.weight_scale, bias, x.dtype, True
        )

        torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    unittest.main()
