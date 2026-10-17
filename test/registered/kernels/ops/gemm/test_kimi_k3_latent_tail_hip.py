"""The gfx950 Kimi-K3 latent-MoE tail kernels against the unfused norm, GEMM and add chain."""

import unittest

import torch

from sglang.kernels.ops.gemm.kimi_k3_latent_tail_hip import (
    PARTIAL_SPLITS,
    k3_latent_tail,
    k3_latent_tail_partials,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the fused tail serves Kimi-K3's gfx950 TP8 decode path.
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3: 3584-wide latent, 7168 hidden; the column tail owns 1/8 of the up_proj rows.
L, H, EPS = 3584, 7168, 1e-5


def _beyond_ulp(got, want):
    want = want.float()
    return ((got.float() - want).abs() > want.abs() * 2**-7 + 1e-3).sum().item()


@unittest.skipUnless(is_gfx95_supported(), "requires gfx950")
class TestKimiK3LatentTailHip(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        g = torch.Generator(device="cuda").manual_seed(0)
        cls.up = (torch.randn(H, L, generator=g, device="cuda") * 0.02).bfloat16()
        cls.norm_w = (torch.rand(L, generator=g, device="cuda") + 0.5).bfloat16()
        cls.g = g

    def _inputs(self, rows):
        g = self.g
        latent = (torch.randn(rows, L, generator=g, device="cuda") * 3).bfloat16()
        shared = torch.randn(rows, H, generator=g, device="cuda").bfloat16()
        prefix = (torch.randn(rows, H, generator=g, device="cuda") * 4).bfloat16()
        rstd = torch.rsqrt(latent.float().pow(2).mean(-1, keepdim=True) + EPS)
        scaled = (latent.float() * self.norm_w.float()).bfloat16().float()
        proj = (rstd * (scaled @ self.up.float().t())).bfloat16()
        return latent, shared, prefix, proj

    @torch.inference_mode()
    def test_tail_matches_norm_projection_and_add(self):
        """One launch equals bf16(rmsnorm(latent) @ up.T) + shared [+ prefix] with
        an fp32 add, within one bf16 rounding (but where the GEMM's summation order
        flips the projection's rounding ahead of a cancelling add)."""
        for rows in (1, 4, 32):
            latent, shared, prefix, proj = self._inputs(rows)
            for with_prefix in (False, True):
                want = proj.float() + shared.float()
                if with_prefix:
                    want = want + prefix.float()
                got = k3_latent_tail(
                    latent,
                    self.norm_w,
                    EPS,
                    self.up,
                    shared,
                    prefix if with_prefix else None,
                )
                with self.subTest(rows=rows, prefix=with_prefix):
                    self.assertLessEqual(
                        _beyond_ulp(got, want.bfloat16()), got.numel() // 1000
                    )

    @torch.inference_mode()
    def test_partials_sum_to_the_tail_projection(self):
        """The column tail's split-K partials of 1/8 of the up_proj rows, finished
        as their consumer does (sum over chunks / rms), give the same projection."""
        rows_n = H // 8
        up_slice = self.up[3 * rows_n : 4 * rows_n]
        for rows in (1, 4, 64):
            latent, _, _, proj = self._inputs(rows)
            part, sq = k3_latent_tail_partials(latent, self.norm_w, up_slice)
            self.assertEqual(tuple(part.shape), (PARTIAL_SPLITS, rows, rows_n))
            inv_rms = torch.rsqrt(sq.sum(0) / L + EPS)
            got = (part.sum(0) * inv_rms[:, None]).bfloat16()
            with self.subTest(rows=rows):
                torch.testing.assert_close(
                    sq.sum(0), latent.float().pow(2).sum(-1), rtol=1e-5, atol=1e-3
                )
                self.assertEqual(_beyond_ulp(got, proj[:, 3 * rows_n : 4 * rows_n]), 0)


if __name__ == "__main__":
    unittest.main()
