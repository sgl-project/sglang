# SPDX-License-Identifier: Apache-2.0
"""Numerical A/B for #38573 without sm90 Cutlass.

Checkpoint scales use group_size=64; pre-fix cutlass path used chunk_size=128.
Oracle: W[n,k] *= scale[n, k // chunk_size].
"""

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _dequant_group(
    w: torch.Tensor, scale: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    _, k = w.shape
    idx = (torch.arange(k, device=w.device) // chunk_size).clamp(max=scale.shape[1] - 1)
    return w.to(torch.float32) * scale[:, idx]


class TestW4AFP8ChunkSizeNumericalAB(CustomTestCase):
    def test_g64_scales_with_chunk_128_diverge(self):
        torch.manual_seed(0)
        n, k, group_size, wrong_chunk = 8, 256, 64, 128
        w = torch.randint(-8, 8, (n, k), dtype=torch.int8).float()
        n_groups = k // group_size
        scale = (
            torch.arange(1, n_groups + 1, dtype=torch.float32)
            .view(1, -1)
            .expand(n, -1)
            .contiguous()
        )
        scale = scale * torch.linspace(0.01, 0.08, n).view(-1, 1)

        ref = _dequant_group(w, scale, chunk_size=group_size)
        buggy = _dequant_group(w, scale, chunk_size=wrong_chunk)
        self.assertGreater((ref - buggy).abs().max().item(), 1e-3)

        fixed = _dequant_group(w, scale, chunk_size=group_size)
        self.assertTrue(torch.allclose(fixed, ref))

    def test_g128_chunk_128_identical(self):
        torch.manual_seed(1)
        n, k = 4, 256
        w = torch.randn(n, k)
        scale = torch.randn(n, k // 128)
        a = _dequant_group(w, scale, 128)
        b = _dequant_group(w, scale, 128)
        self.assertTrue(torch.allclose(a, b))
