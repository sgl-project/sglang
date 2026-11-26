"""SGLANG_K3_TARGET_DENSE_FP8: StaticFP8Weight on K3-shaped dense linears
against the bf16 F.linear it replaces, including a prefill chunk past the
per-launch token cap."""

import unittest

import torch

from sglang.srt.models.kimi_k3_amd.dense_fp8 import StaticFP8Weight
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

# (N, K): KDA fused q/k/v/g at TP8, merged MoE front at TP8
SHAPES = [(4 * 12 * 128, 7168), (896 + 3584 + 512, 7168)]


class TestK3DenseFP8(CustomTestCase):
    def test_matches_bf16_linear(self):
        torch.manual_seed(0)
        for n, k in SHAPES:
            w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
            fp8 = StaticFP8Weight(w)
            for m in (1, 8, 64, 1024, StaticFP8Weight._MAX_TOKENS + 17):
                with self.subTest(n=n, m=m):
                    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
                    ref = torch.nn.functional.linear(x, w).float()
                    out = fp8(x).float()
                    self.assertEqual(out.shape, (m, n))
                    rel = (out - ref).norm() / ref.norm()
                    self.assertLess(rel.item(), 0.06)

    def test_empty_batch(self):
        w = torch.randn(256, 7168, device="cuda").to(torch.bfloat16)
        out = StaticFP8Weight(w)(
            torch.empty(0, 7168, device="cuda", dtype=torch.bfloat16)
        )
        self.assertEqual(out.shape, (0, 256))


if __name__ == "__main__":
    unittest.main()
