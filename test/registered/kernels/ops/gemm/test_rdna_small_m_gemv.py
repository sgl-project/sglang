"""Tests the RDNA (gfx11) split-K bf16 GEMV used for single-row decode projections.

The shape gate is exercised on every GPU (the arch check is patched out, so the
accept/reject rules are covered on hosts without a gfx11 device); the kernel is
compared against an fp32 reference and rocBLAS on the shapes that occur in decode.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.gemm import rdna_small_m_gemv as splitk
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

# Shapes from a 0.6B decode step: (n, k) of qkv / o / gate_up / down.
DECODE_SHAPES = [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072)]


class TestGate(CustomTestCase):
    def setUp(self):
        # The gate's last condition is the gfx11 arch check; pin it so the shape and
        # dtype rules are testable on hosts without a gfx11 device.
        patcher = mock.patch.object(splitk, "_is_rdna", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_accepts_decode_shapes(self):
        for n, k in DECODE_SHAPES:
            with self.subTest(n=n, k=k):
                self.assertTrue(splitk.use_rdna_splitk_gemv(1, n, k, torch.bfloat16))

    def test_rejects_non_single_row(self):
        # The tl.sum reduction is single-row by construction; multi-row must not
        # silently route here.
        self.assertFalse(splitk.use_rdna_splitk_gemv(2, 4096, 1024, torch.bfloat16))
        self.assertFalse(splitk.use_rdna_splitk_gemv(64, 4096, 1024, torch.bfloat16))

    def test_rejects_wide_head(self):
        # Wide layers already saturate the bus; split-K loses there.
        self.assertFalse(splitk.use_rdna_splitk_gemv(1, 151936, 1024, torch.bfloat16))

    def test_accepts_large_split_k(self):
        # 5120x17408 is a 27B down_proj: split_k=34, measured 1.37x over rocBLAS.
        self.assertTrue(splitk.use_rdna_splitk_gemv(1, 5120, 17408, torch.bfloat16))

    def test_rejects_unsupported_dtype_and_shape(self):
        self.assertFalse(splitk.use_rdna_splitk_gemv(1, 4096, 1024, torch.float16))
        # k must be a multiple of BLOCK_K, n a multiple of BLOCK_N.
        self.assertFalse(splitk.use_rdna_splitk_gemv(1, 4096, 1000, torch.bfloat16))
        self.assertFalse(splitk.use_rdna_splitk_gemv(1, 4100, 1024, torch.bfloat16))
        # split_k == 1 has no occupancy problem to fix.
        self.assertFalse(splitk.use_rdna_splitk_gemv(1, 4096, 512, torch.bfloat16))

    def test_rejects_other_arches(self):
        with mock.patch.object(splitk, "_is_rdna", return_value=False):
            self.assertFalse(splitk.use_rdna_splitk_gemv(1, 4096, 1024, torch.bfloat16))

    def test_env_var_disables_the_path(self):
        with mock.patch.object(splitk, "_is_disabled", return_value=True):
            self.assertFalse(splitk.use_rdna_splitk_gemv(1, 4096, 1024, torch.bfloat16))


@unittest.skipUnless(is_hip(), "the split-K GEMV path is only wired up on ROCm")
class TestKernel(CustomTestCase):
    def _run_case(self, n, k, seed=0):
        torch.manual_seed(seed)
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
        out = splitk.rdna_splitk_gemv(x, w)
        ref = x.float() @ w.float().t()
        blas = (x @ w.t()).float()
        self.assertEqual(out.shape, (1, n))
        self.assertEqual(out.dtype, torch.bfloat16)
        err = (out.float() - ref).abs().max().item()
        err_blas = (blas - ref).abs().max().item()
        # One bf16 ulp at the output scale: the error both paths can reach from the
        # final store alone, and it grows with sqrt(K) for the accumulated part.
        step = max(ref.abs().max().item() * 2**-9, 1e-6)
        self.assertLess(err, 10 * step, (n, k, err, step))
        # fp32 accumulation is at least as accurate as rocBLAS bf16 output.
        self.assertLessEqual(err, err_blas * 2 + 1e-3, (n, k, err, err_blas))
        self.assertFalse(torch.isnan(out).any().item(), (n, k))

    def test_decode_shapes(self):
        for n, k in DECODE_SHAPES:
            with self.subTest(n=n, k=k):
                self._run_case(n, k)

    def test_large_model_shapes(self):
        # Qwen3.5-4B / Qwen3.8-27B dense shapes, including a split_k=34 down_proj.
        for n, k in [(10240, 2560), (18432, 2560), (2560, 9216), (5120, 17408)]:
            with self.subTest(n=n, k=k):
                self._run_case(n, k)

    def test_batch_invariant(self):
        """Same output regardless of what else is in flight."""
        torch.manual_seed(1)
        x = torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(4096, 1024, dtype=torch.bfloat16, device="cuda") * 0.05
        a = splitk.rdna_splitk_gemv(x, w)
        splitk.rdna_splitk_gemv(
            torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda"),
            torch.randn(1024, 2048, dtype=torch.bfloat16, device="cuda") * 0.05,
        )
        b = splitk.rdna_splitk_gemv(x, w)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
