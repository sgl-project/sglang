"""RMSNorm's CUDA path hands FP32 inputs to the native implementation."""

import unittest

import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRMSNormFp32(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.norm = RMSNorm(64, eps=1e-6)
        self.norm.weight.data.normal_(1.0, 0.1)
        self.x = torch.randn(5, 64)

    def test_without_residual(self):
        expected = self.norm.forward_native(self.x.clone())
        torch.testing.assert_close(
            self.norm.forward_cuda(self.x.clone()), expected, rtol=0, atol=0
        )

    def test_with_residual(self):
        residual = torch.randn(5, 64)
        expected = self.norm.forward_native(self.x.clone(), residual.clone())
        actual = self.norm.forward_cuda(self.x.clone(), residual.clone())
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
