"""Unit tests for srt/utils/skinny_gemm_pad.py -- no server, no model loading."""

import unittest

import torch

from sglang.srt.utils.skinny_gemm_pad import (
    SM120_SKINNY_GEMM_MIN_ELEMS,
    SM120_SKINNY_GEMM_MIN_K,
    apply_with_padded_rows,
    skinny_gemm_pad_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# ((m, n, k), pad target) measured on an RTX 5090 for every Qwen3.5-family
# in_proj_ba shape (N = 2 * num_v_heads / TP, K = hidden_size).
MEASURED = [
    # 27B TP=2: cliff at 2 <= M <= 8, gone at M=9.
    ((8, 48, 5120), 9),
    ((2, 48, 5120), 9),
    ((9, 48, 5120), 0),
    ((1, 48, 5120), 0),
    # 27B TP=1: cliff only at 2 <= M <= 4; M=8 must stay untouched.
    ((4, 96, 5120), 5),
    ((5, 96, 5120), 0),
    ((8, 96, 5120), 0),
    # 122B TP=4 (K=4096); 397B TP=2 and TP=4 (K=6144).
    ((15, 32, 4096), 16),
    ((16, 32, 4096), 0),
    ((5, 64, 6144), 6),
    ((6, 64, 6144), 0),
    ((10, 32, 6144), 11),
    ((11, 32, 6144), 0),
    # 35B-A3B (K=2048): no cliff at any M.
    ((8, 64, 2048), 0),
    ((2, 32, 2048), 0),
]


class TestSkinnyGemmPadRows(CustomTestCase):
    def test_measured_shapes(self):
        for (m, n, k), want in MEASURED:
            self.assertEqual(skinny_gemm_pad_rows(m=m, n=n, k=k), want, (m, n, k))

    def test_pad_target_is_the_first_row_past_the_threshold(self):
        for n, k in [(48, 5120), (96, 5120), (32, 4096), (64, 6144), (32, 6144)]:
            pad_to = skinny_gemm_pad_rows(m=2, n=n, k=k)
            self.assertGreaterEqual(pad_to * n * k, SM120_SKINNY_GEMM_MIN_ELEMS)
            self.assertLess((pad_to - 1) * n * k, SM120_SKINNY_GEMM_MIN_ELEMS)
            for m in range(2, pad_to):
                self.assertEqual(skinny_gemm_pad_rows(m=m, n=n, k=k), pad_to)
            self.assertEqual(skinny_gemm_pad_rows(m=pad_to, n=n, k=k), 0)

    def test_gemv_and_short_k_are_never_padded(self):
        self.assertEqual(skinny_gemm_pad_rows(m=1, n=48, k=5120), 0)
        self.assertEqual(
            skinny_gemm_pad_rows(m=8, n=64, k=SM120_SKINNY_GEMM_MIN_K - 1), 0
        )
        self.assertEqual(skinny_gemm_pad_rows(m=8, n=64, k=2048, min_k=2048), 16)
        self.assertEqual(skinny_gemm_pad_rows(m=8, n=48, k=5120, min_elems=1 << 20), 0)


class TestApplyWithPaddedRows(CustomTestCase):
    def test_padded_rows_are_dropped_and_others_pass_through(self):
        torch.manual_seed(0)
        w = torch.randn(48, 64)
        calls = []

        def linear(x):
            calls.append(x.shape[0])
            return x @ w.T

        for m in (1, 2, 8, 9, 16):
            x = torch.randn(m, 64)
            out = apply_with_padded_rows(linear, x, pad_to=9)
            torch.testing.assert_close(out, x @ w.T)
        # M=1 (GEMV) and M >= pad_to run unpadded; 2 <= M < pad_to run at pad_to.
        self.assertEqual(calls, [1, 9, 9, 9, 16])


if __name__ == "__main__":
    unittest.main()
