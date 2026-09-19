"""SANA-Video eager convolution fusions preserve native rounding and fallback."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.multimodal_gen.runtime.models.dits.sana as sana
from sglang.kernels.ops.diffusion import BitExactFusionGate
from sglang.multimodal_gen.runtime.models.dits.sana_video import GLUMBTempConv
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestSanaVideoEagerConv(CustomTestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        for name in ("_SANA_CONV_SILU", "_SANA_CONV_GLU"):
            original = getattr(sana, name)
            self.addCleanup(setattr, sana, name, original)
            setattr(sana, name, BitExactFusionGate("test"))
        torch.manual_seed(42)

    @staticmethod
    def reference(module, x):
        b, t, h, w, c = x.shape
        z = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        z = F.silu(module.conv_inverted(z))
        z = module.conv_depth(z)
        value, gate = z.chunk(2, dim=1)
        z = module.conv_point(value * F.silu(gate))
        z = z.reshape(b, t, c, h * w).permute(0, 2, 1, 3)
        z = z + module.conv_temp(z)
        return z.permute(0, 2, 3, 1).reshape(b, t, h, w, c)

    @torch.inference_mode()
    def test_native_video_shape_and_convolution_bias_rounding(self):
        module = GLUMBTempConv(2240, 3.0).cuda().bfloat16().eval()
        x = torch.randn(1, 21, 30, 52, 2240, device="cuda", dtype=torch.bfloat16)
        before = x.clone()
        for _ in range(2):
            actual = module(x)
            self.assertTrue(torch.equal(actual, self.reference(module, x)))
            self.assertTrue(sana._SANA_CONV_SILU.verified)
            self.assertTrue(sana._SANA_CONV_GLU.verified)
        self.assertTrue(torch.equal(x, before))

    @torch.inference_mode()
    def test_replay_uses_changed_inputs(self):
        module = GLUMBTempConv(32, 3.0).cuda().bfloat16().eval()
        x = torch.randn(2, 3, 7, 11, 32, device="cuda", dtype=torch.bfloat16)
        module(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = module(x)
        x.add_(0.5)
        graph.replay()
        self.assertTrue(torch.equal(actual, self.reference(module, x)))

    @torch.inference_mode()
    def test_image_default_stream_policy_is_preserved(self):
        conv = torch.nn.Conv2d(32, 64, 1).cuda().bfloat16()
        x = torch.randn(2, 32, 7, 11, device="cuda", dtype=torch.bfloat16)
        with patch.object(sana, "fused_bias_silu", side_effect=AssertionError):
            actual = sana.sana_conv_bias_silu(conv, x)
        self.assertTrue(torch.equal(actual, F.silu(conv(x))))
        self.assertFalse(sana._SANA_CONV_SILU.verified)

    @torch.inference_mode()
    def test_mismatch_permanently_falls_back(self):
        conv = torch.nn.Conv2d(32, 64, 1).cuda().bfloat16()
        x = torch.randn(2, 32, 7, 11, device="cuda", dtype=torch.bfloat16)
        with patch.object(
            sana, "fused_bias_silu", side_effect=lambda z, b: torch.zeros_like(z)
        ) as fused:
            for _ in range(2):
                actual = sana.sana_conv_bias_silu(conv, x, allow_eager=True)
                self.assertTrue(torch.equal(actual, F.silu(conv(x))))
        self.assertEqual(fused.call_count, 1)
        self.assertTrue(sana._SANA_CONV_SILU.disabled)

    def test_grad_enabled_default_stream_keeps_torch(self):
        conv = torch.nn.Conv2d(4, 8, 1).cuda().bfloat16()
        x = torch.randn(
            1, 4, 3, 5, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        with patch.object(sana, "fused_bias_silu", side_effect=AssertionError):
            actual = sana.sana_conv_bias_silu(conv, x, allow_eager=True)
        actual.float().sum().backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
