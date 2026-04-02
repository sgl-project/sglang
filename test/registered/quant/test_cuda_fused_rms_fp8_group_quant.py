import unittest

import torch

from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    sglang_fused_rms_fp8_group_quant,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, suite="stage-b-test-1-gpu-large")


class TestCudaFusedRMSFP8GroupQuant(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("CUDA FP8 is not available")

    @staticmethod
    def _rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        rms = torch.rsqrt(
            (x.to(torch.float32) * x.to(torch.float32)).mean(dim=-1, keepdim=True) + eps
        )
        return (x.to(torch.float32) * rms) * w.to(torch.float32)

    def test_fused_rms_fp8_group_quant_no_residual(self):
        torch.manual_seed(0)
        m, n = 64, 4096
        eps = 1e-6
        group_size = 128

        x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, device="cuda", dtype=torch.bfloat16)

        (q, s), _, _, res = sglang_fused_rms_fp8_group_quant(
            x,
            w,
            eps,
            group_size=group_size,
            dtype_quant=torch.float8_e4m3fn,
            res1=None,
            column_major_scales=True,
            scale_tma_aligned=False,
        )

        y_ref = self._rmsnorm_ref(x, w, eps)
        q_ref, s_ref = per_token_group_quant_fp8(
            y_ref,
            group_size=group_size,
            column_major_scales=True,
            scale_tma_aligned=False,
        )

        self.assertIsNone(res)
        torch.testing.assert_close(s, s_ref, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(
            q.to(torch.float16), q_ref.to(torch.float16), atol=0.2, rtol=0.2
        )

    def test_fused_rms_fp8_group_quant_with_residual(self):
        torch.manual_seed(1)
        m, n = 32, 7168
        eps = 1e-6
        group_size = 128

        x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        res = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        res_ref = x + res

        (q, s), _, _, res_out = sglang_fused_rms_fp8_group_quant(
            x,
            w,
            eps,
            group_size=group_size,
            dtype_quant=torch.float8_e4m3fn,
            res1=res,
            column_major_scales=True,
            scale_tma_aligned=False,
        )

        y_ref = self._rmsnorm_ref(res_ref, w, eps)
        q_ref, s_ref = per_token_group_quant_fp8(
            y_ref,
            group_size=group_size,
            column_major_scales=True,
            scale_tma_aligned=False,
        )

        torch.testing.assert_close(res_out, res_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(s, s_ref, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(
            q.to(torch.float16), q_ref.to(torch.float16), atol=0.2, rtol=0.2
        )


if __name__ == "__main__":
    unittest.main()
