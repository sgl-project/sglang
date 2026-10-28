"""Tests the RDNA (gfx11) split-K bf16 GEMV used for single-row decode projections.

The gate is exercised on CPU; the kernel is compared against an fp32 reference and
rocBLAS on the shapes that actually occur in decode.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-large")

# Shapes from a 0.6B decode step: (n, k) of qkv / o / gate_up / down.
DECODE_SHAPES = [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072)]


def _is_rdna() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx11")


class TestGate(unittest.TestCase):
    def test_accepts_decode_shapes(self):
        from sglang.kernels.ops.gemm.rdna_small_m_gemv import use_rdna_splitk_gemv

        for n, k in DECODE_SHAPES:
            with self.subTest(n=n, k=k):
                self.assertTrue(use_rdna_splitk_gemv(1, n, k, torch.bfloat16))

    def test_rejects_non_single_row(self):
        from sglang.kernels.ops.gemm.rdna_small_m_gemv import use_rdna_splitk_gemv

        # The tl.sum reduction is single-row by construction; multi-row must not
        # silently route here.
        self.assertFalse(use_rdna_splitk_gemv(2, 4096, 1024, torch.bfloat16))
        self.assertFalse(use_rdna_splitk_gemv(64, 4096, 1024, torch.bfloat16))

    def test_rejects_wide_head(self):
        from sglang.kernels.ops.gemm.rdna_small_m_gemv import use_rdna_splitk_gemv

        # Wide layers already saturate the bus; split-K loses there.
        self.assertFalse(use_rdna_splitk_gemv(1, 151936, 1024, torch.bfloat16))

    def test_accepts_large_split_k(self):
        from sglang.kernels.ops.gemm.rdna_small_m_gemv import use_rdna_splitk_gemv

        # 5120x17408 is a 27B down_proj: split_k=34, measured 1.37x over rocBLAS.
        self.assertTrue(use_rdna_splitk_gemv(1, 5120, 17408, torch.bfloat16))

    def test_rejects_unsupported_dtype_and_shape(self):
        from sglang.kernels.ops.gemm.rdna_small_m_gemv import use_rdna_splitk_gemv

        self.assertFalse(use_rdna_splitk_gemv(1, 4096, 1024, torch.float16))
        # k must be a multiple of BLOCK_K, n a multiple of BLOCK_N.
        self.assertFalse(use_rdna_splitk_gemv(1, 4096, 1000, torch.bfloat16))
        self.assertFalse(use_rdna_splitk_gemv(1, 4100, 1024, torch.bfloat16))


@unittest.skipIf(not _is_rdna(), "split-K GEMV is gated to gfx11")
class TestKernel(unittest.TestCase):
    def _run_case(self, n, k, seed=0):
        from sglang.kernels.ops.gemm.rdna_small_m_gemv import rdna_splitk_gemv

        torch.manual_seed(seed)
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
        out = rdna_splitk_gemv(x, w)
        ref = x.float() @ w.float().t()
        blas = (x @ w.t()).float()
        self.assertEqual(out.shape, (1, n))
        self.assertEqual(out.dtype, torch.bfloat16)
        # fp32 accumulation is at least as accurate as rocBLAS bf16 output.
        self.assertLess((out.float() - ref).abs().max().item(), 0.05)
        self.assertLessEqual(
            (out.float() - ref).abs().max().item(),
            (blas - ref).abs().max().item() * 2 + 1e-3,
        )

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
        from sglang.kernels.ops.gemm.rdna_small_m_gemv import rdna_splitk_gemv

        torch.manual_seed(1)
        x = torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(4096, 1024, dtype=torch.bfloat16, device="cuda") * 0.05
        a = rdna_splitk_gemv(x, w)
        rdna_splitk_gemv(torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda"),
                         torch.randn(1024, 2048, dtype=torch.bfloat16,
                                     device="cuda") * 0.05)
        b = rdna_splitk_gemv(x, w)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
