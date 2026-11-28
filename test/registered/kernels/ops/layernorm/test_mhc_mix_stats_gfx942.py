"""DeepSeek-V4 mHC mix statistics on gfx942 take the fused Triton reduce + sinkhorn and match the fp64 forms."""

import unittest
from unittest import mock

import torch

from sglang.srt.utils import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.mhc import hc_split_sinkhorn_torch
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, suite="base-b-test-1-gpu-small-amd")

HC, H = 4, 5120
MIX = (2 + HC) * HC
ITERS, RMS_EPS, HC_EPS = 20, 1e-6, 1e-6


@unittest.skipUnless(is_gfx942_supported(), "the gfx942 mix-stats route")
class TestMixStatsGfx942(CustomTestCase):
    def test_fused_route_matches_fp64(self):
        from sglang.kernels.ops.layernorm import mhc as mhc_ops
        from sglang.srt.models import deepseek_v4_mhc

        g = torch.Generator(device="cpu").manual_seed(0)
        fn = (torch.randn(MIX, HC * H, generator=g) * 0.02).cuda()
        scale = torch.tensor([0.7, 1.3, 0.9]).cuda()
        base = (torch.randn(MIX, generator=g) * 0.3).cuda()
        cfg = deepseek_v4_mhc.HcConfig(
            mult=HC,
            sinkhorn_iters=ITERS,
            eps=HC_EPS,
            rms_eps=RMS_EPS,
            hidden=H,
            pre_from_prev=True,
            cp_prefill=False,
        )
        hc = deepseek_v4_mhc.HcSubLayer(cfg, fn, scale, base, norm=None)
        for m in (1, 6, 132, 4096):
            with self.subTest(m=m):
                x = torch.randn(m, HC, H, generator=g).to(torch.bfloat16).cuda()
                with mock.patch.object(
                    mhc_ops,
                    "hc_mix_stats_sinkhorn",
                    wraps=mhc_ops.hc_mix_stats_sinkhorn,
                ) as fused:
                    got = deepseek_v4_mhc._mix_stats_impl(hc, x)
                self.assertEqual(fused.call_count, 1)

                xf = x.flatten(1).double()
                rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + RMS_EPS)
                mixes = ((xf @ fn.double().T) * rsqrt).float().unsqueeze(1)
                want = hc_split_sinkhorn_torch(mixes, scale, base, HC, ITERS, HC_EPS)
                for name, a, b in zip(("pre", "post", "comb"), got, want):
                    err = (a.float() - b.squeeze(1).float()).abs().max().item()
                    self.assertLess(err, 1e-4, f"{name} max abs error {err:.2e}")


if __name__ == "__main__":
    unittest.main()
