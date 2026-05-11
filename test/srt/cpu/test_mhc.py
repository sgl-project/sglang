import itertools
import unittest

import torch
import torch.nn.functional as F
from utils import precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


def _ref_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    hc = hc_mult

    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )

    comb = mixes[:, 2 * hc :].view(-1, hc, hc) * hc_scale[2] + hc_base[2 * hc :].view(
        hc, hc
    )
    comb = (comb - comb.amax(dim=-1, keepdim=True)).exp()

    row_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / row_sum + eps
    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + eps)

    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + eps)

    return pre, post, comb


def _ref_hc_pre(
    x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_norm_eps, hc_eps
):
    dtype = x.dtype
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x_flat, hc_fn.float()) * rsqrt
    pre, post, comb = _ref_hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
    )
    y = (pre.unsqueeze(-1) * x.float()).sum(dim=1)
    return y.to(dtype), post, comb


def _ref_hc_post(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)


def _ref_hc_head(x, hc_fn, hc_scale, hc_base, hc_eps, norm_eps):
    dtype = x.dtype
    shape = x.size()
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape).float(), dim=1)
    return y.to(dtype)


def _make_inputs(T, hc, d, dtype=torch.bfloat16, seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    mix_hc = (2 + hc) * hc

    x = torch.randn(T, hc, d, dtype=dtype, generator=gen)
    hc_fn = torch.randn(mix_hc, hc * d, dtype=torch.float32, generator=gen) * 0.02
    hc_scale = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32)
    hc_base = torch.zeros(mix_hc, dtype=torch.float32)
    return x, hc_fn, hc_scale, hc_base


def _make_post_inputs(T, hc, d, dtype=torch.bfloat16, seed=7):
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn(T, d, dtype=dtype, generator=gen)
    residual = torch.randn(T, hc, d, dtype=dtype, generator=gen)
    post = torch.rand(T, hc, dtype=torch.float32, generator=gen)
    comb = torch.rand(T, hc, hc, dtype=torch.float32, generator=gen)
    comb = comb / comb.sum(dim=-1, keepdim=True)
    comb = comb / comb.sum(dim=-2, keepdim=True)
    return x, residual, post, comb


class TestMhcAccuracy(CustomTestCase):
    T = [1, 64]
    HC = [2, 4]
    D = [128, 141]
    DTYPE = [torch.bfloat16, torch.float16]
    SINKHORN_ITERS = 20
    RMS_EPS = 1e-5
    HC_EPS = 1e-6
    NORM_EPS = 1e-5

    def _pre_accuracy_test(self, T, hc, d, dtype):
        x, hc_fn, hc_scale, hc_base = _make_inputs(
            T, hc, d, dtype=dtype, seed=2000 + T + d + hc
        )

        ref_y, ref_post, ref_comb = _ref_hc_pre(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc,
            self.SINKHORN_ITERS,
            self.RMS_EPS,
            self.HC_EPS,
        )
        out_y, out_post, out_comb = torch.ops.sgl_kernel.hc_pre_fused_cpu(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc,
            self.SINKHORN_ITERS,
            self.RMS_EPS,
            self.HC_EPS,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_y, out_y, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_post, out_post, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_comb, out_comb, atol=atol, rtol=rtol)

    def _post_accuracy_test(self, T, hc, d, dtype):
        x, residual, post, comb = _make_post_inputs(
            T, hc, d, dtype=dtype, seed=3000 + T + d + hc
        )

        ref_out = _ref_hc_post(x, residual, post, comb)
        out = torch.ops.sgl_kernel.hc_post_fused_cpu(x, residual, post, comb)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def _head_accuracy_test(self, T, hc, d, dtype):
        gen = torch.Generator()
        gen.manual_seed(4000 + T + d + hc)
        x = torch.randn(T, hc, d, dtype=dtype, generator=gen)
        hc_fn = torch.randn(hc, hc * d, dtype=torch.float32, generator=gen) * 0.02
        hc_scale = torch.tensor(1.0, dtype=torch.float32)
        hc_base = torch.zeros(hc, dtype=torch.float32)

        ref_out = _ref_hc_head(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)
        out = torch.ops.sgl_kernel.hc_head_fused_cpu(
            x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def test_mhc_accuracy(self):
        for T, hc, d, dtype in itertools.product(self.T, self.HC, self.D, self.DTYPE):
            with self.subTest(op="hc_pre", T=T, hc=hc, d=d, dtype=dtype):
                self._pre_accuracy_test(T, hc, d, dtype)
            with self.subTest(op="hc_post", T=T, hc=hc, d=d, dtype=dtype):
                self._post_accuracy_test(T, hc, d, dtype)
            with self.subTest(op="hc_head", T=T, hc=hc, d=d, dtype=dtype):
                self._head_accuracy_test(T, hc, d, dtype)


if __name__ == "__main__":
    unittest.main()
