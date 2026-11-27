"""The Triton BF16 x BF16 -> FP32 GEMM (SGLANG_OPT_BF16_FP32_GEMM_ALGO=triton).

Checks accuracy against an FP64 reference and that the bits do not depend on where
the operands live, on DeepSeek-V4.1-Flash TP4 shapes. cuBLAS FP32-output bits change
for operands that are not 16-byte aligned; see
benchmark/kernels/gemm/repro_cublas_bf16_fp32_sm80.py.
Runs on any CUDA GPU.
"""

import unittest

import torch

from sglang.kernels.ops.gemm.bf16_fp32_triton import linear_bf16_fp32_triton
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

# (M, N, K): LM head / draft head, indexer and compressor projections, router.
SHAPES = (
    (1, 32320, 5120),
    (5, 32320, 5120),
    (5, 32320, 256),
    (128, 512, 5120),
    (710, 512, 5120),
    (256, 384, 5120),
    (9, 384, 5120),
    (710, 128, 512),
    (4, 32, 5120),
    (3, 1, 5376),
    (65, 384, 5120),
)
OFFSETS = (1, 2, 4, 8, 64, 1024)


def _operands(m, n, k):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda").bfloat16()
    w = (torch.randn(n, k, device="cuda") / k**0.5).bfloat16()
    return x, w


def _shifted(t, offset):
    buf = torch.empty(t.numel() + offset, dtype=t.dtype, device=t.device)
    view = buf[offset:].view(t.shape)
    view.copy_(t)
    return view


class TestLinearBf16Fp32Triton(CustomTestCase):
    def test_accurate(self):
        for m, n, k in SHAPES:
            with self.subTest(m=m, n=n, k=k):
                x, w = _operands(m, n, k)
                exact = x.double() @ w.double().T
                got = linear_bf16_fp32_triton(x, w)
                self.assertEqual(got.dtype, torch.float32)
                self.assertLessEqual(
                    (got.double() - exact).norm().item(), 1e-5 * exact.norm().item()
                )
                # FP32 accumulation, rounded once to BF16, should nearly always
                # land on the correctly rounded product.
                wrong = (got.bfloat16() != exact.bfloat16()).sum().item()
                self.assertLessEqual(wrong, max(2, 0.004 * exact.numel()))

    def test_bits_do_not_depend_on_addresses(self):
        for m, n, k in SHAPES:
            with self.subTest(m=m, n=n, k=k):
                x, w = _operands(m, n, k)
                first = linear_bf16_fp32_triton(x, w).clone()
                for offset in OFFSETS:
                    self.assertTrue(
                        torch.equal(
                            linear_bf16_fp32_triton(_shifted(x, offset), w), first
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            linear_bf16_fp32_triton(x, _shifted(w, offset)), first
                        )
                    )

    def test_out_and_leading_dims(self):
        x, w = _operands(6, 384, 5120)
        out = torch.empty(2, 3, 384, device="cuda", dtype=torch.float32)
        got = linear_bf16_fp32_triton(x.view(2, 3, 5120), w, out)
        self.assertEqual(got.data_ptr(), out.data_ptr())
        self.assertTrue(torch.equal(got.view(6, 384), linear_bf16_fp32_triton(x, w)))
        self.assertEqual(linear_bf16_fp32_triton(x[:0], w).shape, (0, 384))


if __name__ == "__main__":
    unittest.main()
