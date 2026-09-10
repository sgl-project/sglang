"""Numerical and input ownership checks for decode projections."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.wo_a_bf16_gemv import wo_a_bf16_gemv
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-large")


class TestDecodeProjections(CustomTestCase):
    def test_grouped_gemv(self):
        for seed in (0, 17, 20260908):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                x = torch.randn(1, 2, 4096, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
                original_x, original_w = x.clone(), w.clone()
                result = wo_a_bf16_gemv(x, w)
                reference = torch.einsum("tgd,grd->tgr", x, w)
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertTrue(result.is_contiguous())
                torch.testing.assert_close(result, reference, atol=1e-3, rtol=8e-3)
                # Sample both groups against CPU FP64 accumulation independently
                # of the cuBLAS algorithm selected by the serving reference.
                exact = torch.einsum(
                    "tgd,grd->tgr", x.cpu().double(), w[:, ::31].cpu().double()
                ).bfloat16()
                torch.testing.assert_close(
                    result[:, :, ::31].cpu(), exact, atol=1e-3, rtol=8e-3
                )
                self.assertTrue(torch.equal(x, original_x))
                self.assertTrue(torch.equal(w, original_w))


if __name__ == "__main__":
    unittest.main()
