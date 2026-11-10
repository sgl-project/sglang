"""CPU golden for DSV4.1 mHC Sinkhorn (hc_mult=4, 20 iters, FP32).

GPU kernel vs this golden lives in test/registered/kernel/jit/test_dsv41_sm70_hc.py
and is not registered for GPU CI.
"""

import unittest

import torch

from sglang.kernels.ops.elementwise.sm70_dsv41_hc_mix import (
    DSV41_HC_DIM,
    DSV41_HC_MULT,
    DSV41_HIDDEN,
    DSV41_MIX_HC,
    dsv41_mhc_shapes_match,
)
from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sequential_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Left-to-right 4-wide reduction, matching `_hc_split_sinkhorn_torch`."""
    b, s, _ = mixes.size()
    hc = hc_mult
    flat = mixes.reshape(-1, (2 + hc) * hc).float()
    scale = hc_scale.float()
    base = hc_base.float()

    pre = torch.sigmoid(flat[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(flat[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (flat[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(-1, hc, hc)

    row_max = comb[..., 0]
    for k in range(1, hc):
        row_max = torch.maximum(row_max, comb[..., k])
    comb = torch.exp(comb - row_max.unsqueeze(-1))
    row_sum = comb[..., 0]
    for k in range(1, hc):
        row_sum = row_sum + comb[..., k]
    comb = comb / row_sum.unsqueeze(-1) + eps
    col_sum = comb[:, 0, :]
    for j in range(1, hc):
        col_sum = col_sum + comb[:, j, :]
    comb = comb / (col_sum.unsqueeze(1) + eps)

    for _ in range(sinkhorn_iters - 1):
        row_sum = comb[..., 0]
        for k in range(1, hc):
            row_sum = row_sum + comb[..., k]
        comb = comb / (row_sum.unsqueeze(-1) + eps)
        col_sum = comb[:, 0, :]
        for j in range(1, hc):
            col_sum = col_sum + comb[:, j, :]
        comb = comb / (col_sum.unsqueeze(1) + eps)

    return (
        pre.reshape(b, s, hc).to(mixes.dtype),
        post.reshape(b, s, hc).to(mixes.dtype),
        comb.reshape(b, s, hc, hc).to(mixes.dtype),
    )


class TestDsv41MhcSinkhorn(CustomTestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_dsv41_shapes_are_not_qwen_hc(self):
        self.assertEqual(DSV41_HIDDEN, 5120)
        self.assertEqual(DSV41_HC_MULT, 4)
        self.assertEqual(DSV41_HC_DIM, 20480)
        self.assertEqual(DSV41_MIX_HC, 24)
        self.assertNotEqual(DSV41_HIDDEN, 2560)
        self.assertNotEqual(DSV41_HC_DIM, 10240)

    def test_shapes_match_rejects_qwen_geometry(self):
        x = torch.zeros(1, 4, 2560, dtype=torch.float16)
        hc_fn = torch.zeros(24, 10240, dtype=torch.float32)
        self.assertFalse(dsv41_mhc_shapes_match(x, hc_fn, hc_mult=4))
        x_ok = torch.zeros(1, 4, 5120, dtype=torch.float16)
        fn_ok = torch.zeros(24, 20480, dtype=torch.float32)
        self.assertTrue(dsv41_mhc_shapes_match(x_ok, fn_ok, hc_mult=4))
        self.assertFalse(dsv41_mhc_shapes_match(x_ok, fn_ok, hc_mult=8))

    def test_torch_sinkhorn_matches_sequential_reduction(self):
        torch.manual_seed(0)
        mixes = torch.randn(2, 1, 24, dtype=torch.float32)
        hc_scale = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32)
        hc_base = torch.randn(24, dtype=torch.float32) * 0.1
        pre_t, post_t, comb_t = _hc_split_sinkhorn_torch(
            mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
        )
        pre_s, post_s, comb_s = _sequential_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
        )
        torch.testing.assert_close(pre_t, pre_s, rtol=0, atol=0)
        torch.testing.assert_close(post_t, post_s, rtol=0, atol=0)
        torch.testing.assert_close(comb_t, comb_s, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
