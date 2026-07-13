"""Unit test for the portable Triton hc_split_sinkhorn fallback (mHC / DeepSeek-V4).

Validates the vendor-neutral Triton kernel against a pure-torch reference of the
TileLang kernel math. This is the fallback used on backends without TileLang
(e.g. ROCm/AMD). No server, no model loading.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.layers import mhc
from sglang.test.test_utils import CustomTestCase


def _sinkhorn_ref(mixes, hc_scale, hc_base, hc, iters, eps):
    """Pure-torch reference mirroring hc_split_sinkhorn_kernel_ in mhc.py.

    mixes: [n, (2 + hc) * hc]  ->  pre/post: [n, hc], comb: [n, hc, hc]
    """
    n = mixes.shape[0]
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = (mixes[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(n, hc, hc)

    row_max = comb.max(dim=2, keepdim=True).values
    comb = torch.exp(comb - row_max)
    row_sum = comb.sum(dim=2, keepdim=True)
    comb = comb / row_sum + eps
    col_sum = comb.sum(dim=1, keepdim=True)
    comb = comb / (col_sum + eps)
    for _ in range(iters - 1):
        row_sum = comb.sum(dim=2, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=1, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb


class TestMhcSinkhornTriton(CustomTestCase):
    EPS = 1e-6

    def _check(self, n, hc, iters, seed=0):
        torch.manual_seed(seed)
        dev = "cuda"
        mix_hc = (2 + hc) * hc
        mixes = torch.randn(n, mix_hc, device=dev, dtype=torch.float32)
        scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
        base = torch.randn(mix_hc, device=dev, dtype=torch.float32)

        ref_pre, ref_post, ref_comb = _sinkhorn_ref(
            mixes, scale, base, hc, iters, self.EPS
        )

        # Exercise the fallback via the 3D public API shape [b, s, mix_hc].
        pre, post, comb = mhc._hc_split_sinkhorn_triton(
            mixes.view(1, n, mix_hc), scale, base, hc, iters, self.EPS
        )
        pre = pre.reshape(n, hc)
        post = post.reshape(n, hc)
        comb = comb.reshape(n, hc, hc)

        torch.testing.assert_close(pre, ref_pre, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(post, ref_post, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(comb, ref_comb, atol=1e-5, rtol=1e-4)

    def test_default_dsv4_shape(self):
        # DeepSeek-V4: hc_mult=4, 20 Sinkhorn iterations.
        for n in (1, 7, 128, 4096):
            with self.subTest(n=n):
                self._check(n=n, hc=4, iters=20)

    def test_varied_iters(self):
        for iters in (1, 2, 5, 10):
            with self.subTest(iters=iters):
                self._check(n=512, hc=4, iters=iters)

    def test_varied_hc(self):
        for hc in (2, 4, 8):
            with self.subTest(hc=hc):
                self._check(n=256, hc=hc, iters=20)

    def test_dispatch_selects_triton_without_tilelang(self):
        # When TileLang is unavailable (or non-CUDA), the public dispatcher must
        # route to the Triton fallback and stay numerically correct.
        if mhc._HAS_TILELANG and mhc._IS_CUDA:
            self.skipTest("TileLang path active; fallback dispatch not exercised here")
        n, hc, iters = 1024, 4, 20
        mix_hc = (2 + hc) * hc
        torch.manual_seed(1)
        mixes = torch.randn(1, n, mix_hc, device="cuda", dtype=torch.float32)
        scale = torch.rand(3, device="cuda", dtype=torch.float32) + 0.5
        base = torch.randn(mix_hc, device="cuda", dtype=torch.float32)

        pre, post, comb = mhc.hc_split_sinkhorn(mixes, scale, base, hc, iters, self.EPS)
        r_pre, r_post, r_comb = _sinkhorn_ref(
            mixes.view(n, mix_hc), scale, base, hc, iters, self.EPS
        )
        torch.testing.assert_close(pre.reshape(n, hc), r_pre, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(post.reshape(n, hc), r_post, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            comb.reshape(n, hc, hc), r_comb, atol=1e-5, rtol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
