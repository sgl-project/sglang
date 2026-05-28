"""Tests for Gemma4RMSNorm forward_xpu dispatch."""

import unittest

import torch


class TestGemma4RMSNormXPU(unittest.TestCase):
    def setUp(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        torch.manual_seed(42)
        from sglang.srt.layers.layernorm import Gemma4RMSNorm

        self.norm = Gemma4RMSNorm(128, eps=1e-6, scale_shift=1.0).to("xpu")

    def test_2d_input(self):
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="xpu")
        out = self.norm.forward_xpu(x)
        self.assertEqual(out.shape, (4, 128))
        ref = self.norm.forward_native(x)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)

    def test_3d_input(self):
        x = torch.randn(4, 8, 128, dtype=torch.bfloat16, device="xpu")
        out = self.norm.forward_xpu(x)
        self.assertEqual(out.shape, (4, 8, 128))
        ref = self.norm.forward_native(x)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)

    def test_scale_shift_zero(self):
        from sglang.srt.layers.layernorm import Gemma4RMSNorm

        norm0 = Gemma4RMSNorm(128, eps=1e-6, scale_shift=0.0).to("xpu")
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="xpu")
        out = norm0.forward_xpu(x)
        ref = norm0.forward_native(x)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)

    def test_empty_tensor(self):
        x = torch.empty(0, 128, dtype=torch.bfloat16, device="xpu")
        out = self.norm.forward_xpu(x)
        self.assertEqual(out.shape, (0, 128))


if __name__ == "__main__":
    unittest.main()
