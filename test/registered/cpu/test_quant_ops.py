import unittest

import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    static_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


def _dequant_fp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float()


def _ue8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    return (scale.to(torch.int32) << 23).view(torch.float32)


class TestCPUQuantOps(CustomTestCase):
    def test_per_token_group_quant_fp8_cpu(self):
        x = torch.linspace(-2.0, 2.0, steps=128, dtype=torch.float32).reshape(2, 64)

        q, scale = torch.ops.sgl_kernel.per_token_group_quant_fp8_cpu(x, 32, 1e-10)
        ref_q, ref_scale = per_token_group_quant_fp8(x, 32)

        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (2, 2))
        torch.testing.assert_close(scale, ref_scale)
        torch.testing.assert_close(q.float(), ref_q.float())
        torch.testing.assert_close(
            _dequant_fp8(q, scale.repeat_interleave(32, dim=-1)),
            x,
            atol=0.08,
            rtol=0.08,
        )

    def test_scaled_fp8_quant_cpu_dynamic_and_static(self):
        x = torch.randn(3, 64, dtype=torch.bfloat16) / 4

        q, scale = scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(scale.shape, (3, 1))
        torch.testing.assert_close(
            _dequant_fp8(q, scale), x.float(), atol=0.02, rtol=0.02
        )

        static_scale = torch.tensor([0.01], dtype=torch.float32)
        static_q, returned_scale = static_quant_fp8(x, static_scale)
        self.assertIs(returned_scale, static_scale)
        torch.testing.assert_close(
            static_q.float(),
            torch.clamp(x.float() / static_scale, -448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .float(),
        )

    def test_scaled_fp8_quant_cpu_padding(self):
        x = torch.randn(2, 32, dtype=torch.float32) / 3

        q, scale = torch.ops.sgl_kernel.scaled_fp8_quant_cpu(x, None, 4, True)

        self.assertEqual(q.shape, (4, 32))
        self.assertEqual(scale.shape, (4, 1))
        torch.testing.assert_close(
            _dequant_fp8(q[:2], scale[:2]), x, atol=0.02, rtol=0.02
        )
        torch.testing.assert_close(q[2:].float(), torch.zeros(2, 32))
        torch.testing.assert_close(scale[2:], torch.zeros(2, 1))

    def test_mxfp8_group_quantize_cpu(self):
        x = torch.randn(3, 64, dtype=torch.float32) / 5

        q, scale_u8 = mxfp8_group_quantize(x)

        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale_u8.dtype, torch.uint8)
        self.assertEqual(scale_u8.shape, (3, 2))
        dequant = q.float() * _ue8m0_to_float(scale_u8).repeat_interleave(32, dim=-1)
        torch.testing.assert_close(dequant, x, atol=0.04, rtol=0.04)


if __name__ == "__main__":
    unittest.main()
