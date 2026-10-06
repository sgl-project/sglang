"""Numerical regression for the opt-in shifted mHC prefill boundary on GB300.

Run directly with Python; checks the production decomposition and a second chained boundary.
"""

import unittest

import torch

from sglang.kernels.ops.layernorm.mhc import hc_mix_stats_sinkhorn_bf16x3
from sglang.kernels.ops.layernorm.mhc_mega import mhc_mega_boundary
from sglang.kernels.ops.layernorm.mhc_post_combine_norm_prefill import (
    mhc_post_combine_norm_prefill,
)


@torch.inference_mode()
def check_shifted_mega_mhc_prefill(tokens):
    torch.manual_seed(42)
    hidden = 5120
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(tokens, 4, hidden, device="cuda", dtype=torch.bfloat16)
    pre = torch.rand(tokens, 4, device="cuda")
    post = torch.rand(tokens, 4, device="cuda")
    comb = torch.rand(tokens, 4, 4, device="cuda")
    comb /= comb.sum(-1, keepdim=True)
    fn = torch.randn(24, 4 * hidden, device="cuda") * 0.01
    scale = torch.randn(3, device="cuda") * 0.1
    base = torch.randn(24, device="cuda") * 0.1
    weight = torch.empty(hidden, device="cuda", dtype=torch.bfloat16).uniform_(0.9, 1.1)
    hi = fn.bfloat16()
    mid = (fn - hi.float()).bfloat16()
    lo = (fn - hi.float() - mid.float()).bfloat16()
    parts = (hi, mid, lo)
    for _ in range(2):
        expected_residual, expected_norm = mhc_post_combine_norm_prefill(
            x, residual, post, comb, pre, weight, 1e-6
        )
        expected_stats = hc_mix_stats_sinkhorn_bf16x3(
            expected_residual.flatten(1), parts, scale, base, 20, 1e-6, 1e-6
        )
        actual_residual, actual_norm, actual_stats = mhc_mega_boundary(
            x,
            residual,
            pre,
            post,
            comb,
            fn,
            scale,
            base,
            weight,
            1e-6,
            1e-6,
            1e-6,
            20,
        )
        # BF16 post rounding can be amplified by the carried collapse;
        # FP32 mixing statistics use tighter tolerances.
        for actual, expected in zip(
            (actual_residual, actual_norm), (expected_residual, expected_norm)
        ):
            torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=1e-2)
        for actual, expected in zip(actual_stats, expected_stats):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-3)
        residual, x = actual_residual, actual_norm
        pre, post, comb = actual_stats


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "DeepGEMM Mega mHC requires Blackwell",
)
class TestShiftedMegaMhcPrefill(unittest.TestCase):
    def test_4k(self):
        check_shifted_mega_mhc_prefill(4096)

    def test_8k(self):
        check_shifted_mega_mhc_prefill(8192)

    def test_16k(self):
        check_shifted_mega_mhc_prefill(16384)


if __name__ == "__main__":
    unittest.main()
