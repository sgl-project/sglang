"""The batch-invariant matmul kernels must be bit-reproducible.

Deterministic inference swaps ``mm`` for the Triton kernels in
``batch_invariant_ops``.  The segmented split-K kernel picks its tiles, warps
and split from a launch table, so the bits must not depend on that choice, nor
on the batch shape: otherwise greedy requests stop being repeat- and
batch-invariant.  (The persistent and segmented kernels accumulate in a
different order and are deliberately not compared against each other.)
"""

import unittest
from unittest import mock

import torch

import sglang.srt.batch_invariant_ops.batch_invariant_ops as bio
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

# (M, N, K): decode-shaped GEMMs plus a K that is not a multiple of the 64-wide
# K block.
SHAPES = [
    (1, 6144, 4096),
    (7, 4096, 1100),
    (16, 2048, 2048),
    (64, 1536, 2048),
]

# (BLOCK_M, BLOCK_N, num_warps, num_stages, split) handed to the same kernel.
LAUNCHES = [
    (32, 128, 4, 3, False),
    (32, 128, 4, 3, True),
    (64, 128, 4, 3, False),
    (128, 128, 8, 3, False),
]


class TestBatchInvariantMatmul(CustomTestCase):
    def _inputs(self, m: int, n: int, k: int):
        generator = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(
            (m, k), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        b = torch.randn(
            (k, n), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        return a, b

    def _assert_same_bits(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        self.assertEqual(out.shape, ref.shape)
        torch.testing.assert_close(
            out.view(torch.int16), ref.view(torch.int16), rtol=0, atol=0
        )

    def _segmented(self, a, b, launch):
        with mock.patch.object(bio, "_segmented_matmul_launch", lambda **_: launch):
            return bio._matmul_segmented_triton(a, b, torch.bfloat16)

    def test_launch_does_not_change_bits(self):
        for m, n, k in SHAPES:
            a, b = self._inputs(m, n, k)
            ref = self._segmented(a, b, LAUNCHES[0])
            for launch in LAUNCHES[1:]:
                with self.subTest(M=m, N=n, K=k, launch=launch):
                    self._assert_same_bits(self._segmented(a, b, launch), ref)

    def test_row_is_batch_invariant(self):
        n, k = 2048, 2048
        a, b = self._inputs(64, n, k)
        ref = self._segmented(a, b, LAUNCHES[0])
        for m in (1, 4, 16, 64):
            rows = [0] if m == 1 else [0, 1, m - 1]
            out = self._segmented(a[:m], b, LAUNCHES[0])
            for row in rows:
                with self.subTest(M=m, row=row):
                    self._assert_same_bits(out[row], ref[row])

    def test_repeatable(self):
        a, b = self._inputs(16, 2048, 2048)
        self._assert_same_bits(
            self._segmented(a, b, LAUNCHES[0]), self._segmented(a, b, LAUNCHES[0])
        )


if __name__ == "__main__":
    unittest.main()
