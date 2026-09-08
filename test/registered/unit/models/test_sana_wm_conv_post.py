"""SANA-WM conv post-processing parity, rounding, and graph replay."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.multimodal_gen.runtime.models.dits.sana_wm_components as wm
from sglang.kernels.ops.diffusion import BitExactFusionGate
from sglang.kernels.ops.diffusion.activation.sana_conv_post_triton import (
    can_use_fused_bias_glu,
    can_use_fused_bias_silu,
    fused_bias_glu,
    fused_bias_silu,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-large")


class TestSanaWMConvPost(CustomTestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.original_gate = wm._SANA_WM_CONV_POST
        wm._SANA_WM_CONV_POST = BitExactFusionGate("test", per_signature=True)
        self.addCleanup(setattr, wm, "_SANA_WM_CONV_POST", self.original_gate)
        torch.manual_seed(42)

    @torch.inference_mode()
    def test_post_kernels_both_layouts_and_bias_modes(self):
        for shape in [(3, 34, 17, 23), (2, 13440, 22, 40), (2, 34, 1, 1)]:
            for layout in [torch.contiguous_format, torch.channels_last]:
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(
                    memory_format=layout
                )
                bias = torch.randn(shape[1], device="cuda", dtype=x.dtype)
                biased = x + bias[None, :, None, None]
                self.assertTrue(torch.equal(fused_bias_silu(x, bias), F.silu(biased)))
                for b, z in [(bias, biased), (None, x)]:
                    a, g = z.chunk(2, dim=1)
                    actual = fused_bias_glu(x, b)
                    self.assertTrue(torch.equal(actual, a * F.silu(g)))
                    self.assertTrue(actual.is_contiguous(memory_format=layout))
        # Use a real spatial slice; width-one tensors remain contiguous.
        sliced = torch.empty(2, 34, 7, 10, device="cuda", dtype=x.dtype)[:, :, :, ::2]
        self.assertFalse(can_use_fused_bias_silu(sliced, bias))
        self.assertFalse(can_use_fused_bias_glu(sliced, None))
        self.assertFalse(can_use_fused_bias_glu(x.float(), None))

    @staticmethod
    def reference(module, x):
        z = module.depth_conv(F.silu(module.inverted_conv.conv(x)))
        a, g = z.chunk(2, dim=1)
        return a * F.silu(g)

    @torch.inference_mode()
    def test_native_convolution_and_depthwise_bias_rounding(self):
        for channels, hidden, shape in [
            (32, 96, (3, 32, 17, 23)),
            (2240, 6720, (14, 2240, 22, 40)),
        ]:
            module = wm.GLUMBConvTemp(channels, hidden).cuda().bfloat16()
            for seed in (0, 1):
                torch.manual_seed(seed)
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                self.assertTrue(
                    torch.equal(module._spatial_glu(x), self.reference(module, x))
                )
                self.assertTrue(wm._SANA_WM_CONV_POST.verified)
                self.assertFalse(wm._SANA_WM_CONV_POST.disabled)
            # Guard the numerical distinction which forbids bias extraction.
            z = module.inverted_conv(x)
            conv = module.depth_conv.conv
            split = F.conv2d(z, conv.weight, None, padding=1, groups=conv.groups)
            split = split + conv.bias[None, :, None, None]
            self.assertFalse(torch.equal(conv(z), split))

    @torch.inference_mode()
    def test_graph_replay_updates_inputs_and_streaming_tail(self):
        module = wm.GLUMBConvTemp(32, 96).cuda().bfloat16()
        # Exercise a nonzero temporal convolution instead of its zero init.
        module.t_conv.weight.normal_(std=0.01)
        x = torch.randn(2, 3 * 7 * 11, 32, device="cuda", dtype=torch.bfloat16)
        tail = torch.randn(2, 32, 1, 77, device="cuda", dtype=x.dtype)
        module(x, (3, 7, 11), ffn_tail=tail, save_ffn_tail=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, actual_tail = module(
                x, (3, 7, 11), ffn_tail=tail, save_ffn_tail=True
            )
        x.add_(0.25)
        tail.mul_(0.5)
        graph.replay()
        with patch.object(wm._SANA_WM_CONV_POST, "disabled", True):
            expected, expected_tail = module(
                x, (3, 7, 11), ffn_tail=tail, save_ffn_tail=True
            )
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_tail, expected_tail))

    @torch.inference_mode()
    def test_unverified_capture_and_mismatch_use_reference(self):
        module = wm.GLUMBConvTemp(32, 96).cuda().bfloat16()
        x = torch.randn(3, 32, 17, 23, device="cuda", dtype=torch.bfloat16)
        expected = self.reference(module, x)
        with patch.object(wm, "fused_bias_silu") as fused:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = module._spatial_glu(x)
            graph.replay()
            fused.assert_not_called()
        self.assertTrue(torch.equal(actual, expected))
        with patch.object(
            wm, "fused_bias_glu", return_value=torch.zeros_like(expected)
        ):
            actual = module._spatial_glu(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(wm._SANA_WM_CONV_POST.disabled)
        with patch.object(wm, "fused_bias_silu") as fused:
            actual = module._spatial_glu(x)
            fused.assert_not_called()
        self.assertTrue(torch.equal(actual, expected))

    def test_grad_enabled_uses_differentiable_reference(self):
        module = wm.GLUMBConvTemp(32, 96).cuda().bfloat16()
        x = torch.randn(
            2, 32, 7, 11, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        with patch.object(wm, "fused_bias_silu") as fused:
            actual = module._spatial_glu(x)
            actual.float().sum().backward()
            fused.assert_not_called()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
