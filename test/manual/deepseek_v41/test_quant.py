"""Pure-torch fp8 / fp4 quantization against the reference TileLang kernels."""

import unittest

import torch
from ref_loader import RefTestCase, assert_equal, randomize_, report, requires_ref

from sglang.srt.layers.attention.dsv4.torch_quant import (
    fake_quant_fp4,
    fake_quant_fp8,
    naive_linear,
    quant_fp8_act,
)

E8M0 = torch.float8_e8m0fnu


def sample_activations(rows: int = 64, cols: int = 256) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(1)
    magnitude = torch.exp2(
        torch.randint(-12, 12, (rows, 1), generator=gen, device="cpu").float()
    )
    x = torch.randn(rows, cols, generator=gen, device="cpu") * magnitude
    # Exact power-of-two block maxima probe the ceil(log2) boundary.
    x[0, :32] = 0
    x[0, 0] = 448.0
    x[1, 0] = 6.0 * 2**-3
    x[2, 0] = 224.0
    x[3, :] = 0
    return x.to(device="cuda", dtype=torch.bfloat16)


@requires_ref
class TestQuant(RefTestCase):
    def test_act_quant_fp8(self):
        x = sample_activations()
        ref_values, ref_scale = self.kernel.act_quant(x, 32, "ue8m0", E8M0)
        values, scale = quant_fp8_act(x)
        self.assertEqual(scale.tolist(), ref_scale.float().tolist())
        assert_equal(values.view(torch.uint8), ref_values.view(torch.uint8))

    def test_fake_quant_fp8(self):
        x = sample_activations()
        expected = self.kernel.act_quant(x.clone(), 32, "ue8m0", E8M0, inplace=True)
        assert_equal(fake_quant_fp8(x), expected)

    def test_fake_quant_fp4(self):
        x = sample_activations()
        expected = self.kernel.fp4_act_quant(x.clone(), 32, inplace=True)
        assert_equal(fake_quant_fp4(x), expected)

    def _check_linear(self, weight_dtype, name):
        model = self.model
        linear = model.Linear(256, 192, dtype=weight_dtype)
        randomize_(linear)
        x = sample_activations(rows=48, cols=256)
        expected = model.linear(x, linear.weight)
        actual = naive_linear(x, linear.weight, linear.scale)
        report(name, actual, expected)
        torch.testing.assert_close(actual, expected, rtol=2**-6, atol=0.0)

    def test_fp8_linear(self):
        self._check_linear(torch.float8_e4m3fn, "fp8 linear")

    def test_fp4_linear(self):
        self._check_linear(torch.float4_e2m1fn_x2, "fp4 linear")


if __name__ == "__main__":
    unittest.main()
