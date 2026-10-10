"""
Tests the weight-only int8 GEMV (Triton) against an fp32 product on the
de-quantized weights, on the shapes and row counts a speculative draft runs.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

# max |out - ref| / max |ref|. fp16 products with fp32 accumulation measured
# 2.5e-3 to 2.9e-3 on the shapes below.
_TOLERANCE = 5e-3


@unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA device")
class TestInt8WeightOnlyGemv(CustomTestCase):
    def _run_case(self, m, n, k, dtype=torch.bfloat16, seed=0):
        from sglang.kernels.ops.gemm.int8_weight_only_gemv import (
            int8_weight_only_gemv,
            int8_weight_only_gemv_supported,
        )
        from sglang.srt.speculative.draft_int8 import (
            quantize_int8_rowwise,
        )

        torch.manual_seed(seed)
        x = torch.randn(m, k, device="cuda").to(dtype)
        q, scale = quantize_int8_rowwise(torch.randn(n, k, device="cuda") * 0.02)
        self.assertTrue(int8_weight_only_gemv_supported(x, q), (m, n, k))
        out = int8_weight_only_gemv(x=x, w=q, scale=scale)
        ref = x.double() @ (q.double() * scale.double()[:, None]).t()
        err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(tuple(out.shape), (m, n))
        self.assertLess(err, _TOLERANCE, (m, n, k, err))
        self.assertFalse(torch.isnan(out).any().item(), (m, n, k))

    def test_draft_shapes(self):
        # The large linears of a 5120-wide DSpark draft (Qwen3.8-27B):
        # mlp_gate_up, mlp_down, attn_qkv, attn_o.
        for n, k in [(34816, 5120), (5120, 17408), (6144, 5120), (5120, 4096)]:
            self._run_case(m=8, n=n, k=k)

    def test_row_counts(self):
        for m in [1, 7, 8, 15, 16]:
            self._run_case(m=m, n=6144, k=5120, seed=m)

    def test_fp16_activations(self):
        self._run_case(m=8, n=6144, k=5120, dtype=torch.float16)

    def test_predicate(self):
        from sglang.kernels.ops.gemm.int8_weight_only_gemv import (
            int8_weight_only_gemv_supported,
        )

        q = torch.zeros(6144, 5120, dtype=torch.int8, device="cuda")
        x = torch.zeros(8, 5120, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(int8_weight_only_gemv_supported(x, q))
        # more rows than the verify window, fp32 activations, a CPU tensor, a
        # row count or a width the tile does not divide: all keep the dense path.
        self.assertFalse(int8_weight_only_gemv_supported(x.repeat(3, 1), q))
        self.assertFalse(int8_weight_only_gemv_supported(x.float(), q))
        self.assertFalse(int8_weight_only_gemv_supported(x.cpu(), q.cpu()))
        self.assertFalse(int8_weight_only_gemv_supported(x, q[:6100]))
        self.assertFalse(int8_weight_only_gemv_supported(x[:, :5000], q[:, :5000]))


if __name__ == "__main__":
    unittest.main()
