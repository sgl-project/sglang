"""An untaken K3SortPlan shared GEMM runs the same kernel as a direct call."""

import unittest

import torch

from sglang.kernels.ops.moe import k3_moe_epi
from sglang.kernels.ops.moe.moe_sorting_small import K3SortPlan, k3_sort_plan
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

N, I = 3584, 1024


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None, "ROCm GPU required"
)
class TestK3SortPlanSharedGemm(CustomTestCase):
    def test_untaken_shared_gemm_matches_standalone(self):
        torch.manual_seed(0)
        w_down = torch.randn(N, I, device="cuda", dtype=torch.bfloat16) / I**0.5
        for m in (1, 8, 16, 24, 40, 64):
            with self.subTest(m=m):
                act = torch.randn(m, I, device="cuda", dtype=torch.bfloat16)
                topk_ids = torch.zeros(m, 16, device="cuda", dtype=torch.int32)
                want = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
                k3_moe_epi.shared_gemm_preact(act, w_down, want)
                got = torch.empty_like(want)
                with k3_sort_plan(K3SortPlan(shared_gemm=(topk_ids, act, w_down, got))):
                    pass
                torch.testing.assert_close(got, want, rtol=0, atol=0)
                ref = (act.float() @ w_down.float().t()).to(torch.bfloat16)
                torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
