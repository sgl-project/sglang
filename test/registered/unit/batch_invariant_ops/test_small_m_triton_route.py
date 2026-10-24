"""SGLANG_BATCH_INVARIANT_OPS_SMALL_M_TRITON: on SM120/121, bf16 GEMMs with M <= the cap run
the persistent Triton kernel with a small-M tile instead of DeepGEMM. The route is only valid
if its output is DeepGEMM's bit for bit, so every check here is exact (rtol=0, atol=0)."""

import unittest
from unittest import mock

import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

# (N, K) of Qwen3.8-27B's bf16 linears: narrow N, a 96-wide N, the longest K, a wide N.
SHAPES = [(96, 5120), (5120, 6144), (5120, 17408), (14336, 5120)]
CAP = 16


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if get_device_sm() not in (120, 121):
        return "the route exists only on SM120/121"
    if not batch_invariant_ops.ENABLE_JIT_DEEPGEMM:
        return "DeepGEMM is unavailable, so there is no reference"
    return None


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class TestSmallMTritonRoute(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_cap = batch_invariant_ops._SMALL_M_TRITON_MAX
        cls._saved_cmp = batch_invariant_ops._ENABLE_MM_COMPARISON_TEST
        batch_invariant_ops._SMALL_M_TRITON_MAX = CAP
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = False
        cls.gen = torch.Generator(device="cuda").manual_seed(0)

    @classmethod
    def tearDownClass(cls):
        batch_invariant_ops._SMALL_M_TRITON_MAX = cls._saved_cap
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = cls._saved_cmp

    def _operands(self, M, N, K):
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=self.gen)
        # matmul_persistent's DeepGEMM path takes b as the transpose of a row-major [N, K]
        b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", generator=self.gen).T
        return a, b

    def _deepgemm(self, a, b, bias=None):
        out = batch_invariant_ops._matmul_persistent_deepgemm(
            a, b, out_dtype=torch.bfloat16, bias=bias
        )
        # Fail closed: an all-zero or non-finite reference would make equality vacuous.
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertTrue((out != 0).any().item())
        return out

    def test_route_equals_deepgemm(self):
        for N, K in SHAPES:
            a, b = self._operands(CAP, N, K)
            for M in range(1, CAP + 1):
                with self.subTest(M=M, N=N, K=K):
                    out = batch_invariant_ops._matmul_small_m_triton(
                        a[:M], b, torch.bfloat16
                    )
                    torch.testing.assert_close(
                        out, self._deepgemm(a[:M], b), rtol=0, atol=0
                    )

    def test_route_is_batch_invariant(self):
        for N, K in SHAPES:
            a, b = self._operands(CAP, N, K)
            one = batch_invariant_ops._matmul_small_m_triton(a[:1], b, torch.bfloat16)
            for M in range(2, CAP + 1):
                with self.subTest(M=M, N=N, K=K):
                    out = batch_invariant_ops._matmul_small_m_triton(
                        a[:M], b, torch.bfloat16
                    )
                    torch.testing.assert_close(out[:1], one, rtol=0, atol=0)

    def test_dispatch(self):
        N, K = SHAPES[1]
        a, b = self._operands(CAP + 1, N, K)
        bias = torch.randn(N, dtype=torch.bfloat16, device="cuda", generator=self.gen)
        route = batch_invariant_ops._matmul_small_m_triton
        with mock.patch.object(
            batch_invariant_ops, "_matmul_small_m_triton", wraps=route
        ) as spy:
            out = batch_invariant_ops.matmul_persistent(a[:CAP], b, bias=bias)
            self.assertEqual(spy.call_count, 1)
            torch.testing.assert_close(
                out, self._deepgemm(a[:CAP], b, bias=bias), rtol=0, atol=0
            )
            # Above the cap, and for a non-bf16 output, DeepGEMM runs as before.
            batch_invariant_ops.matmul_persistent(a, b)
            batch_invariant_ops.matmul_persistent(a[:1], b, out_dtype=torch.float32)
            self.assertEqual(spy.call_count, 1)


if __name__ == "__main__":
    unittest.main()
