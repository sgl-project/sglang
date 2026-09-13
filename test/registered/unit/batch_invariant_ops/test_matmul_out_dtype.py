"""Batch-invariant matmul with an explicit output dtype.

Covers the fp32 store the DeepSeek router GEMM needs under deterministic
inference: the reduction must stay put while only the store widens.
"""

import unittest

import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
    _matmul_persistent_triton,
    matmul_persistent,
    set_batch_invariant_mode,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# 1-gpu-large (H100) rather than 1-gpu-small (5090): DeepGEMM is disabled on
# SM120, so only the large runner exercises the fp32 DeepGEMM store.
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
# MI300 (gfx942) has a 64KB shared memory limit but the kernel needs 66KB.
register_amd_ci(est_time=20, suite="nightly-amd-1-gpu-mi35x", nightly=True)

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)

# Just to get the logging out of the way
with set_batch_invariant_mode(True):
    pass


class TestMatmulOutDtype(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Cross-checks DeepGEMM against Triton wherever DeepGEMM is available.
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = True

    @classmethod
    def tearDownClass(cls):
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = False

    def _router_gemm_operands(self, M, K=1024, N=256):
        """Operands shaped like the router GEMM: rhs is a transposed [N, K] weight."""
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.bfloat16)
        w = torch.randn(N, K, dtype=torch.bfloat16)
        return a, w.t()

    def test_fp32_out_is_the_same_reduction(self):
        """Widening the store must not perturb the reduction, for one fixed kernel."""
        a, b = self._router_gemm_operands(M=64)
        out_bf16 = _matmul_persistent_triton(a=a, b=b)
        out_fp32 = _matmul_persistent_triton(a=a, b=b, out_dtype=torch.float32)

        self.assertEqual(out_fp32.dtype, torch.float32)
        self.assertTrue(torch.equal(out_fp32.bfloat16(), out_bf16))

    def test_fp32_out_keeps_accumulator_precision(self):
        """out_dtype=fp32 must store the fp32 accumulator, not a bf16 round trip."""
        a, b = self._router_gemm_operands(M=64)
        with set_batch_invariant_mode(True):
            out_bf16 = matmul_persistent(a=a, b=b)
            out_fp32 = matmul_persistent(a=a, b=b, out_dtype=torch.float32)

        self.assertEqual(out_bf16.dtype, torch.bfloat16)
        self.assertEqual(out_fp32.dtype, torch.float32)
        # It has to carry values bf16 cannot represent.
        self.assertFalse(torch.equal(out_fp32, out_bf16.float()))
        ref = a.double() @ b.double()
        self.assertLess(
            (out_fp32.double() - ref).abs().max().item(),
            (out_bf16.double() - ref).abs().max().item(),
        )

    def test_mm_out_dtype_reaches_the_batch_invariant_kernel(self):
        """torch.mm(out_dtype=fp32) must ask the kernel, not cast a bf16 result."""
        a, b = self._router_gemm_operands(M=64)
        with set_batch_invariant_mode(True):
            got = torch.mm(a, b, out_dtype=torch.float32)
            expected = matmul_persistent(a=a, b=b, out_dtype=torch.float32)
            rounded = matmul_persistent(a=a, b=b).float()

        self.assertEqual(got.dtype, torch.float32)
        self.assertTrue(torch.equal(got, expected))
        self.assertFalse(torch.equal(got, rounded))

    def test_fp32_out_is_batch_invariant(self):
        """The wider store must not reintroduce a token-count dependence."""
        a, b = self._router_gemm_operands(M=256)
        with set_batch_invariant_mode(True):
            full = torch.mm(a, b, out_dtype=torch.float32)
            for M in (1, 2, 4, 5, 16, 17, 64):
                with self.subTest(M=M):
                    self.assertTrue(
                        torch.equal(
                            torch.mm(a[:M], b, out_dtype=torch.float32), full[:M]
                        )
                    )


if __name__ == "__main__":
    unittest.main()
