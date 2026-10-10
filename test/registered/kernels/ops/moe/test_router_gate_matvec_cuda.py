"""
The Hopper router-gate matvec must match an fp32 reference for both gate
dtypes and keep the top-k routing of the triton matvec it replaces.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(is_sm90_supported(), "router gate matvec targets SM90")
class TestRouterGateMatvecCuda(CustomTestCase):
    def test_matches_fp32_reference(self):
        from sglang.kernels.ops.moe.router_gate_matvec_cuda import (
            router_gate_matvec_cuda,
            use_router_gate_matvec_cuda,
        )

        for w_dtype in (torch.bfloat16, torch.float32):
            for m in (1, 3, 6, 8):
                torch.manual_seed(m)
                x = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda")
                w = (torch.randn(512, 4096, device="cuda") * 0.02).to(w_dtype)
                self.assertTrue(use_router_gate_matvec_cuda(x, w))
                out = router_gate_matvec_cuda(x, w)
                ref = x.float() @ w.float().t()
                torch.testing.assert_close(out, ref, atol=2e-4, rtol=1e-5)

    def test_routing_matches_triton_matvec(self):
        """Summation-order differences small enough to pass assert_close can
        still flip near-tie expert choices; allow at most one flip."""
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
        from sglang.kernels.ops.moe.router import router_gate_matvec
        from sglang.kernels.ops.moe.router_gate_matvec_cuda import (
            router_gate_matvec_cuda,
        )

        topk_kwargs = dict(
            scoring_func="sigmoid",
            num_fused_shared_experts=1,
            renormalize=True,
            routed_scaling_factor=2.5,
            num_expert_group=8,
            topk_group=4,
        )
        flips = 0
        for seed in range(20):
            torch.manual_seed(100 + seed)
            x = torch.randn(6, 4096, dtype=torch.bfloat16, device="cuda")
            w = (torch.randn(512, 4096, device="cuda") * 0.02).to(torch.bfloat16)
            bias = torch.randn(512, device="cuda") * 0.1
            with envs.SGLANG_ENABLE_FP8_BLOCK_GEMV.override(False):
                ref = router_gate_matvec(x, w)
            out = router_gate_matvec_cuda(x, w)
            _, ids_ref = moe_fused_gate(ref, bias, 9, **topk_kwargs)
            _, ids = moe_fused_gate(out, bias, 9, **topk_kwargs)
            flips += (ids != ids_ref).any(dim=1).sum().item()
        self.assertLessEqual(flips, 1, flips)

    def test_predicate_rejects_unsupported_shapes(self):
        """The kernel asserts on K not a multiple of the block chunk and on
        M > 8, so the dispatch predicate must send those back to triton."""
        from sglang.kernels.ops.moe.router_gate_matvec_cuda import (
            use_router_gate_matvec_cuda,
        )

        x = torch.randn(6, 4096, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(use_router_gate_matvec_cuda(x, w))
        self.assertFalse(
            use_router_gate_matvec_cuda(
                x[:, :4000].contiguous(), w[:, :4000].contiguous()
            )
        )
        x9 = torch.randn(9, 4096, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(use_router_gate_matvec_cuda(x9, w))
        self.assertFalse(use_router_gate_matvec_cuda(x, w.to(torch.float16)))


if __name__ == "__main__":
    unittest.main()
