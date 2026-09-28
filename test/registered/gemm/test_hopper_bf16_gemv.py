"""
Tests the Hopper single-token bf16 GEMV JIT kernel against torch (cuBLAS +
fp32 reference) on the dispatch domains where the backend enables it.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-large")


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@unittest.skipIf(not _is_sm90(), "Hopper bf16 GEMV requires an SM90 GPU")
class TestHopperBf16Gemv(unittest.TestCase):
    def _run_case(self, n, k, seed=0):
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv

        torch.manual_seed(seed)
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
        out = hopper_bf16_gemv(x, w)
        ref = x.float() @ w.float().t()
        cub = (x @ w.t()).float()
        err = (out.float() - ref).abs().max().item()
        err_cub = (cub - ref).abs().max().item()
        # fp32 accumulation + single warp-tree reduction: at least as tight as
        # cuBLAS (which split-K reduces) against the fp32 reference.
        self.assertLessEqual(err, max(err_cub * 2.0, 1e-2), (n, k, err, err_cub))
        self.assertFalse(torch.isnan(out).any().item(), (n, k))

    def test_dispatch_domain_shapes(self):
        # Representative dense-decode shapes (Qwen3.6-27B): out_proj/attn_o,
        # attn_qkv, mlp_down, mlp_gate_up, in_proj_ba.
        for n, k in [
            (5120, 6144),
            (8192, 5120),
            (5120, 17408),
            (34816, 5120),
            (96, 5120),
        ]:
            self._run_case(n, k)

    def test_tail_rows(self):
        # N not divisible by rows_per_block exercises the guarded tail path.
        for n in [104, 5128, 8200]:
            self._run_case(n, 5120)

    def test_predicate(self):
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import use_hopper_bf16_gemv

        self.assertTrue(use_hopper_bf16_gemv(1, 5120, 6144))
        self.assertTrue(use_hopper_bf16_gemv(1, 34816, 5120))
        # batched decode, odd K, huge N (lm_head), and the near-optimal-cuBLAS
        # mid-N band must all fall back.
        self.assertFalse(use_hopper_bf16_gemv(2, 5120, 6144))
        self.assertFalse(use_hopper_bf16_gemv(1, 5120, 6000))
        self.assertFalse(use_hopper_bf16_gemv(1, 248320, 5120))
        self.assertFalse(use_hopper_bf16_gemv(1, 16384, 5120))


if __name__ == "__main__":
    unittest.main()
