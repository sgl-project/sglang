"""DeepSeek V4 dense block-fp8 weights dequantized to bf16 at load.

Block-fp8 dense weights are materialised as bf16 once at load so oneDNN serves
the GEMM directly. This checks the dequantization against an independent
reference, including the ragged tail blocks that the repeat_interleave has to
truncate.

Usage:
python3 -m unittest test_dsv4_dense_fp8_dequant.TestDsv4DenseFp8Dequant
"""

import unittest

import torch

from sglang.srt.layers.quantization.fp8 import _xpu_dequant_block_fp8
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")

FP8 = torch.float8_e4m3fn


def reference_dequant(weight, scale, block_size):
    """Per element, independent of the implementation's expand and slice."""
    block_n, block_k = block_size
    n, k = weight.shape
    rows = torch.arange(n, device=weight.device) // block_n
    cols = torch.arange(k, device=weight.device) // block_k
    s = scale.to(torch.float32)[rows][:, cols]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def make_case(n, k, block_size, device, seed=0):
    block_n, block_k = block_size
    g = torch.Generator(device="cpu").manual_seed(seed)
    weight = (torch.randn(n, k, generator=g) * 0.25).to(FP8).to(device)
    scale = (
        torch.rand(
            (n + block_n - 1) // block_n,
            (k + block_k - 1) // block_k,
            generator=g,
        )
        * 0.5
        + 0.5
    ).to(device)
    return weight, scale


class TestDsv4DenseFp8Dequant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.xpu.is_available():
            raise unittest.SkipTest("XPU required")
        cls.device = torch.device("xpu")

    def _check(self, n, k, block_size):
        weight, scale = make_case(n, k, block_size, self.device)
        got = _xpu_dequant_block_fp8(weight, scale, block_size)
        want = reference_dequant(weight, scale, block_size)
        self.assertEqual(got.shape, (n, k))
        self.assertEqual(got.dtype, torch.bfloat16)
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)

    def test_aligned_shape(self):
        self._check(256, 512, (128, 128))

    def test_ragged_rows_and_columns(self):
        """n and k not multiples of the block: the expanded scale is truncated."""
        self._check(200, 300, (128, 128))

    def test_single_block(self):
        self._check(64, 64, (128, 128))

    def test_non_square_block(self):
        self._check(256, 512, (64, 128))

    def test_scale_is_actually_applied(self):
        """Negative control: a different scale must give a different result."""
        weight, scale = make_case(256, 512, (128, 128), self.device)
        base = _xpu_dequant_block_fp8(weight, scale, (128, 128))
        other = _xpu_dequant_block_fp8(weight, scale * 2.0, (128, 128))
        self.assertGreater((base.float() - other.float()).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
