# SPDX-License-Identifier: Apache-2.0
"""Packed-row writer contract: reference parity, masked scatter, and RoPE."""

import unittest

import torch

from sglang.srt.layers.attention.dsa.nvfp4_k_cache import (
    dequantize_nvfp4_k_cache_paged,
    dequantize_nvfp4_k_cache_paged_reference,
    quantize_nvfp4_k_cache_into,
    quantize_nvfp4_k_cache_into_reference,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-small")


class TestGlmDsaNvfp4Cache(unittest.TestCase):
    def test_reference_parity(self):
        if not torch.cuda.is_available():
            self.skipTest("Requires CUDA")
        for count in (1, 24, 64, 65, 256, 2048):
            with self.subTest(count=count):
                torch.manual_seed(count)
                latent = torch.randn(count, 512, dtype=torch.bfloat16, device="cuda")
                rope = torch.randn(count, 64, dtype=torch.bfloat16, device="cuda")
                latent[0, :16] = 0
                latent[0, 16:20] = torch.tensor(
                    [torch.nan, torch.inf, -torch.inf, -0.0], device="cuda"
                )
                loc = torch.arange(count, dtype=torch.int32, device="cuda") + 2
                if count > 1:
                    loc[1] = -1
                scale = torch.tensor([0.006], device="cuda")
                actual = torch.full(
                    (count + 4, 1, 416), 0x5A, dtype=torch.uint8, device="cuda"
                )
                expected = actual.clone()
                quantize_nvfp4_k_cache_into_reference(
                    latent, rope, expected, loc, scale
                )
                quantize_nvfp4_k_cache_into(latent, rope, actual, loc, scale)
                self.assertTrue(torch.equal(actual, expected))
                indices = torch.cat(
                    (loc, torch.tensor([count + 10], device="cuda", dtype=torch.int32))
                )
                decoded = dequantize_nvfp4_k_cache_paged(
                    actual, indices, scale, dtype=torch.float32
                )
                reference = dequantize_nvfp4_k_cache_paged_reference(
                    expected, indices, scale, dtype=torch.float32
                )
                self.assertTrue(torch.equal(decoded, reference))
                self.assertTrue(
                    torch.equal(decoded[:1, 0, 512:].to(torch.bfloat16), rope[:1])
                )
                before = actual.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    quantize_nvfp4_k_cache_into(latent, rope, actual, loc, scale)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual, before))


if __name__ == "__main__":
    unittest.main()
