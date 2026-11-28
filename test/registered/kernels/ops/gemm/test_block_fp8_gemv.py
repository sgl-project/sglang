"""
Tests the single-token W8A16 GEMV for 128x128 block-FP8 weights against an
fp32 dequantized reference.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _reference(x, w, s):
    n, k = w.shape
    scale = s[: (n + 127) // 128, : k // 128].float()
    scale = scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return x.float() @ (w.float() * scale).t()


@unittest.skipIf(not _is_sm90(), "block-FP8 GEMV is enabled on SM90 only")
class TestBlockFp8Gemv(unittest.TestCase):
    def _run_case(self, n, k, *, padded_scale=False, bias=False, seed=0):
        from sglang.kernels.ops.gemm.block_fp8_gemv import block_fp8_gemv

        torch.manual_seed(seed)
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        w = (torch.randn(n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        rows, cols = (n + 127) // 128, k // 128
        if padded_scale:
            # TMA-aligned layouts pad the scale tensor; only the logical
            # [ceil(N/128), K/128] corner may be read.
            s = torch.full((rows + 3, cols + 5), float("nan"), device="cuda")
            s[:rows, :cols] = torch.rand(rows, cols, device="cuda") * 0.02 + 1e-3
        else:
            s = torch.rand(rows, cols, device="cuda") * 0.02 + 1e-3
        b = torch.randn(n, dtype=torch.bfloat16, device="cuda") if bias else None
        out = block_fp8_gemv(x, w, s, b)
        ref = _reference(x, w, s)
        if b is not None:
            ref = ref + b.float()
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        self.assertEqual(tuple(out.shape), (1, n))
        self.assertLess(rel, 5e-3, (n, k, rel))

    def test_model_shapes(self):
        # Qwen3.6-35B-A3B dense projections and a DeepSeek-V3-style wide K.
        for n, k in [(12288, 2048), (9216, 2048), (2048, 4096), (2048, 512), (1536, 7168)]:
            self._run_case(n, k)

    def test_tail_rows_and_non_power_of_two_k(self):
        # N % BLOCK_N != 0 takes the masked tail; K = 2560 / 1536 force a
        # smaller K tile than the 8K-element default.
        for n, k in [(576, 7168), (5128, 2560), (3072, 1536)]:
            self._run_case(n, k)

    def test_strided_scale_and_bias(self):
        self._run_case(9216, 2048, padded_scale=True, bias=True)

    def test_predicate(self):
        from sglang.kernels.ops.gemm.block_fp8_gemv import use_block_fp8_gemv

        w = torch.empty(2048, 4096, device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.empty(16, 32, device="cuda")
        x1 = torch.empty(1, 4096, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(use_block_fp8_gemv(x1, w, s))
        # Batched input, fp32 activations, non-fp32 scales, and K not a
        # multiple of the 128 block must all fall back.
        self.assertFalse(use_block_fp8_gemv(x1.expand(2, -1), w, s))
        self.assertFalse(use_block_fp8_gemv(x1.float(), w, s))
        self.assertFalse(use_block_fp8_gemv(x1, w, s.to(torch.bfloat16)))
        w_odd = torch.empty(2048, 4000, device="cuda", dtype=torch.float8_e4m3fn)
        self.assertFalse(
            use_block_fp8_gemv(x1.new_empty(1, 4000), w_odd, s)
        )


if __name__ == "__main__":
    unittest.main()
