"""The gfx950 segmented batch-invariant matmul: bitwise checks on the GPU."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")

# (N, K) of bf16 linear layers: dense and MoE projections, a router (N = 32), and
# K = 2880, which is not a multiple of the segment size.
SHAPES = [(4096, 4096), (24576, 4096), (4096, 12288), (5120, 2880), (32, 2880)]
FULL_M = 4096
ROWS = [1, 7, 32, 33, 64, 200, 256, 257, 1024, 2048]


def _inputs(N, K, M=FULL_M, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)
    return a, (w / K**0.5).t()  # b as the linear layers pass it: (K, N), transposed


def _mm(a, b, bias=None):
    return batch_invariant_ops._matmul_segmented_triton(
        a=a, b=b, bias=bias, out_dtype=torch.bfloat16
    )


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 only")
class TestSegmentedMatmul(CustomTestCase):
    def test_rows_do_not_depend_on_the_batch(self):
        for N, K in SHAPES:
            a, b = _inputs(N, K)
            full = _mm(a, b)
            for m in ROWS:
                with self.subTest(N=N, K=K, M=m):
                    self.assertTrue(torch.equal(_mm(a[:m], b), full[:m]))

    def test_rows_do_not_depend_on_the_batch_with_bias(self):
        N, K = 5120, 2880
        a, b = _inputs(N, K)
        bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        full = _mm(a, b, bias)
        for m in ROWS:
            with self.subTest(M=m):
                self.assertTrue(torch.equal(_mm(a[:m], b, bias), full[:m]))

    def test_same_output_from_run_to_run(self):
        for N, K in SHAPES:
            for m in (64, 1024, FULL_M):
                a, b = _inputs(N, K, M=m)
                first = _mm(a, b)
                with self.subTest(N=N, K=K, M=m):
                    for _ in range(3):
                        self.assertTrue(torch.equal(_mm(a, b), first))

    def test_split_and_tiles_do_not_change_the_bits(self):
        """Segments in separate programs or in one program, and every tile in the
        launch table, give the same bits as the default launch."""
        launches = [
            (32, 64, 4, 2, True),
            (64, 64, 4, 2, True),
            (128, 128, 8, 3, True),
            (32, 64, 4, 1, False),
            (128, 128, 8, 1, False),
            (128, 256, 8, 1, False),
            (256, 128, 8, 1, False),
        ]
        for N, K in SHAPES:
            a, b = _inputs(N, K, M=512)
            ref = _mm(a, b)
            for launch in launches:
                with (
                    self.subTest(N=N, K=K, launch=launch),
                    patch.object(
                        batch_invariant_ops,
                        "_segmented_matmul_launch",
                        lambda M, N, launch=launch: launch,
                    ),
                ):
                    self.assertTrue(torch.equal(_mm(a, b), ref))

    def test_close_to_fp32(self):
        for N, K in SHAPES:
            a, b = _inputs(N, K, M=256)
            ref = a.float() @ b.float()
            err = (_mm(a, b).float() - ref).norm() / ref.norm()
            with self.subTest(N=N, K=K):
                self.assertLess(err.item(), 1e-2)


if __name__ == "__main__":
    unittest.main()
