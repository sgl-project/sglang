"""Kimi-K3 routed MoE end to end through the AITER runner: fused SiTU + top-k,
the K3 sort plan (sort + MXFP8 quant + shared down GEMM in one launch, or the
two-launch mid-batch route/sort), and aiter fused_moe, against the same runner
with aiter's own sort and quant and a torch shared expert."""

import os
import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

# K3 / K3.1 TP8 routed experts and shared expert
E, TOPK, DIM, INTER, SHARED_INTER = 896, 16, 3584, 384, 1024
BETA, LINEAR_BETA = 4.0, 25.0
# K3's MI355 deploy: fp8 activations into the fp4 experts
os.environ.setdefault("AITER_SITUV2_A8W4", "1")


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(), "gfx950 (MI35x) only"
)
class TestK3MoeSortPlanE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
            AiterRunnerCore,
        )
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        torch.manual_seed(0)
        dev = torch.device("cuda", 0)

        def mxfp4(rows, cols, gate_up):
            """Random fp4 experts in the layout the K3 MXFP4 loader shuffles to."""
            q, s = dynamic_mxfp4_quant(
                (torch.randn(E * rows, cols, device=dev) * 0.02).to(torch.bfloat16)
            )
            q = shuffle_weight_a16w4(
                q.view(torch.uint8).view(E, rows, cols // 2), 16, gate_up
            )
            s = shuffle_scale_a16w4(s.view(torch.uint8), E, gate_up)
            q = q.view(torch.float4_e2m1fn_x2)
            q.is_shuffled = True
            return q, s

        w13, s13 = mxfp4(2 * INTER, DIM, True)
        w2, s2 = mxfp4(DIM, INTER, False)
        cls.quant_info = AiterMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            quant_type=AiterQuantType.PER_1X32,
            w13_scale=s13,
            w2_scale=s2,
        )
        cls.runner = AiterRunnerCore(
            MoeRunnerConfig(
                num_experts=E,
                num_local_experts=E,
                hidden_size=DIM,
                intermediate_size_per_partition=INTER,
                top_k=TOPK,
                activation="situ",
                is_gated=True,
                gemm1_alpha=BETA,
                gemm1_clamp_limit=LINEAR_BETA,
                gate_up_interleaved=False,
            )
        )
        cls.w_down = (
            torch.randn(DIM, SHARED_INTER, device=dev) / SHARED_INTER**0.5
        ).to(torch.bfloat16)
        cls.bias = torch.randn(E, device=dev) * 0.1
        cls.dev = dev

    def _inputs(self, m):
        g = torch.Generator(device=self.dev).manual_seed(m)
        x = torch.randn(m, DIM, generator=g, device=self.dev).to(torch.bfloat16)
        logits = torch.randn(m, E, generator=g, device=self.dev)
        gate_up = torch.randn(m, 2 * SHARED_INTER, generator=g, device=self.dev)
        return x, logits, gate_up.to(torch.bfloat16)

    def _ref_topk(self, logits):
        scores = logits.sigmoid()
        ids = (scores + self.bias).topk(TOPK, dim=1).indices
        w = scores.gather(1, ids)
        return ids.to(torch.int32), w / w.sum(1, keepdim=True)

    def _ref_shared(self, gate_up):
        from sglang.srt.layers.activation import SituAndMul

        act = SituAndMul(beta=BETA, linear_beta=LINEAR_BETA).forward_native(
            gate_up.float()
        )
        return act @ self.w_down.float().t()

    def _routed(self, x, ids, w):
        from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerInput

        runner_input = AiterRunnerInput(
            hidden_states=x,
            topk_ids=ids,
            topk_weights=w,
            quant_type=self.quant_info.quant_type,
        )
        return self.runner.run(runner_input, self.quant_info, {}).hidden_states

    def _assert_rel(self, out, ref, tol, msg):
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        self.assertLess(rel, tol, f"{msg}: rel err {rel:.3e}")

    def test_small_batch_fused_topk_sort_quant_shared(self):
        """M * top_k <= 256: one launch sorts, quantizes and runs the shared GEMM."""
        from sglang.kernels.ops.moe import k3_moe_epi
        from sglang.kernels.ops.moe.k3_sort_plan import K3SortPlan, k3_sort_plan

        for m in (1, 2, 4, 8, 12, 16):
            with self.subTest(m=m):
                x, logits, gate_up = self._inputs(m)
                ids = torch.empty(m, TOPK, dtype=torch.int32, device=self.dev)
                w = torch.empty(m, TOPK, dtype=torch.float32, device=self.dev)
                act = torch.empty(
                    m, SHARED_INTER, dtype=torch.bfloat16, device=self.dev
                )
                shared = torch.empty(m, DIM, dtype=torch.bfloat16, device=self.dev)
                k3_moe_epi.situ_topk(
                    gate_up, act, BETA, LINEAR_BETA, logits, self.bias, ids, w
                )
                plan = K3SortPlan(shared_gemm=(ids, act, self.w_down, shared))
                with k3_sort_plan(plan):
                    out = self._routed(x, ids, w)
                    self.assertIsNone(plan.shared_gemm, "the sort took the GEMM")

                ref_ids, ref_w = self._ref_topk(logits)
                got_ids, order = ids.sort(dim=1)
                want_ids, want_order = ref_ids.sort(dim=1)
                torch.testing.assert_close(got_ids, want_ids, rtol=0, atol=0)
                torch.testing.assert_close(
                    w.gather(1, order),
                    ref_w.gather(1, want_order),
                    rtol=1e-5,
                    atol=1e-6,
                )
                self._assert_rel(shared, self._ref_shared(gate_up), 2e-2, "shared")
                self._assert_rel(out, self._routed(x, ref_ids, ref_w), 1e-2, "routed")

    def test_mid_batch_route_sort(self):
        """Batches past the one-launch sort take the K3 route/sort pair."""
        from sglang.kernels.ops.moe import k3_route_sort_mid
        from sglang.kernels.ops.moe.k3_sort_plan import k3_sort_plan

        for m in (24, 64, 128, k3_route_sort_mid.max_tokens()):
            with self.subTest(m=m):
                x, logits, _ = self._inputs(m)
                ids, w = self._ref_topk(logits)
                with k3_sort_plan():
                    out = self._routed(x, ids, w)
                self._assert_rel(out, self._routed(x, ids, w), 1e-2, "routed")


if __name__ == "__main__":
    unittest.main()
