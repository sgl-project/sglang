# SPDX-License-Identifier: Apache-2.0
"""Lossless Wan normalization post-ops, dtype boundaries, and CUDA replay."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import sglang.multimodal_gen.runtime.models.vaes.wan_vae_cuda_opt as wan
from sglang.kernels.ops.diffusion import wan_norm_silu_post
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import VaeFastPathGate
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def reference(x, gamma, bias, scale):
    return F.silu(F.normalize(x, dim=1) * scale * gamma + bias)


class TestWanNormSiLUPost(CustomTestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("NVIDIA CUDA required")
        torch.manual_seed(1729)

    def assertBitsEqual(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertTrue(
            torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        )

    @staticmethod
    def module(channels=64, *, bias=False, dtype=torch.bfloat16):
        gamma = torch.nn.Parameter(
            torch.randn(channels, 1, 1, 1, device="cuda", dtype=dtype)
        )
        offset = torch.nn.Parameter(torch.randn_like(gamma)) if bias else 0.0
        norm = SimpleNamespace(gamma=gamma, bias=offset, scale=channels**0.5)
        return wan.FusedWanRMSNormSiLU(norm, VaeFastPathGate())

    @torch.inference_mode()
    def test_autocast_promotion_layout_and_affine(self):
        for dtype in (torch.bfloat16, torch.float32):
            for affine in (torch.bfloat16, torch.float32):
                for layout in (torch.contiguous_format, torch.channels_last_3d):
                    for bias in (False, True):
                        with self.subTest(
                            dtype=dtype, affine=affine, layout=layout, bias=bias
                        ):
                            module = self.module(bias=bias, dtype=affine)
                            x = torch.randn(
                                2, 64, 3, 8, 12, device="cuda", dtype=dtype
                            ).to(memory_format=layout)
                            original = x.clone()
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                expected = reference(
                                    x, module.gamma, module.bias, module.scale
                                )
                                actual = module(x)
                                self.assertBitsEqual(actual, expected)
                                self.assertTrue(module._post_gate.verified)
                                self.assertFalse(module._post_gate.disabled)
                                x.add_(0.125)
                                self.assertBitsEqual(
                                    module(x),
                                    reference(
                                        x, module.gamma, module.bias, module.scale
                                    ),
                                )
                            self.assertTrue(torch.equal(original + 0.125, x))

    @torch.inference_mode()
    def test_all_finite_bf16_and_fp32_silu_boundaries(self):
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        values = bits.view(torch.bfloat16)
        finite = values[torch.isfinite(values)].reshape(1, 64, 1, 1, 1020)
        fp32 = (
            torch.tensor(
                [
                    -110.0,
                    -104.0,
                    -100.0,
                    -90.0,
                    -88.0,
                    -80.0,
                    -20.0,
                    -1.0,
                    -0.0,
                    0.0,
                    1e-40,
                    -1e-40,
                    1.0,
                    20.0,
                    80.0,
                    100.0,
                ],
                device="cuda",
            )
            .repeat(64)
            .reshape(1, 64, 1, 1, 16)
        )
        for domain in (finite, fp32):
            for layout in (torch.contiguous_format, torch.channels_last_3d):
                x = domain.to(memory_format=layout)
                denominator = torch.ones((1, 1, *x.shape[2:]), device="cuda")
                gamma = torch.ones((64, 1, 1, 1), device="cuda", dtype=torch.bfloat16)
                expected = F.silu((x / denominator) * 1.0 * gamma + 0.0)
                self.assertBitsEqual(
                    wan_norm_silu_post(x, denominator, gamma, scale=1.0), expected
                )

    @torch.inference_mode()
    def test_graph_replay_recomputes_norm_and_affine(self):
        for layout in (torch.contiguous_format, torch.channels_last_3d):
            module = self.module(bias=True)
            x = torch.randn(2, 64, 3, 8, 12, device="cuda", dtype=torch.bfloat16).to(
                memory_format=layout
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                module(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = module(x)
                x.mul_(0.5)
                module.gamma.add_(0.25)
                module.bias.sub_(0.125)
                graph.replay()
                self.assertBitsEqual(
                    actual, reference(x, module.gamma, module.bias, module.scale)
                )


if __name__ == "__main__":
    unittest.main()
