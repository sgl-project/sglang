"""DeepSeek-V4 MoE on ROCm: the router GEMM keeps fp32 scores and skipped reductions leave the mHC post pending."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip(), "requires HIP")
class TestRouterFp32(CustomTestCase):
    """The HIP branch of MoEGate.forward keeps the router GEMM output in fp32."""

    def setUp(self):
        from sglang.srt.models.deepseek_v2 import MoEGate
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.forward = MoEGate.forward

    def test_close_scores_and_mutable_graph(self):
        """BF16 output rounding must not collapse distinct expert scores."""
        # Exact BF16 operands produce 16 distinct scores near 1; rounding the
        # GEMM output to BF16 would collapse them before expert selection.
        weight = torch.zeros(384, 5120, device="cuda", dtype=torch.bfloat16)
        weight[:, 0] = 1
        weight[:16, 1] = torch.arange(16, device="cuda") / 4096
        gate = SimpleNamespace(
            weight=weight, is_deepseek_v4=True, tiny_router_gemm_max_tokens=0
        )
        for rows in (512,):
            with self.subTest(rows=rows):
                x = torch.zeros(rows, 5120, device="cuda", dtype=torch.bfloat16)
                x[:, :2] = 1
                self.forward(gate, x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self.forward(gate, x)
                for sign in (1, -1):
                    x[:, 1] = sign
                    graph.replay()
                    expected = (
                        1
                        + sign
                        * torch.arange(16, device="cuda", dtype=torch.float32)
                        / 4096
                    )
                    self.assertEqual(output.dtype, torch.float32)
                    torch.testing.assert_close(
                        output[:, :16], expected.expand(rows, -1), rtol=0, atol=0
                    )
                    self.assertEqual(torch.unique(output[0, :16]).numel(), 16)


@unittest.skipUnless(is_hip(), "requires ROCm")
class TestMoeSkippedReductionKeepsPost(CustomTestCase):
    """When the MoE output is reduce-scattered or the all-reduce is fused elsewhere, the
    layer must not reach the fused all-reduce + mHC post: the post stays pending and no
    collective is issued (a TP4 MoE with fake experts on one rank)."""

    def test_skip_predicates_bypass_the_fused_reduction(self):
        from sglang.srt.layers.moe.mhc_post_fusion import (
            MhcPostFusion,
            use_mhc_post_fusion,
        )
        from sglang.srt.runtime_context import get_forward, reset_context
        from sglang.test.dsv4_moe_stub import make_dsv4_moe_stub
        from sglang.test.test_utils import publish_build_topology

        # the skip predicates read the published MoE widths
        publish_build_topology(tp_size=4)
        self.addCleanup(reset_context)
        x = torch.ones(1, 5120, device="cuda", dtype=torch.bfloat16)
        for dual in (False, True):
            for flag in ("mlp_reduce_scatter", "fuse_mlp_allreduce"):
                with self.subTest(dual=dual, flag=flag):
                    moe = make_dsv4_moe_stub(0, dual=dual, shared_tp1=False)
                    state = MhcPostFusion(None, None, None, None)
                    with (
                        get_forward().scoped(
                            **{flag: True}, flashinfer_trtllm_bypass=False
                        ),
                        use_mhc_post_fusion(state),
                        # the skip predicate lives inside post_experts_all_reduce:
                        # mock the collectives it would issue
                        mock.patch(
                            "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce"
                        ) as reduction,
                        mock.patch(
                            "sglang.srt.distributed.communication_op.moe_tensor_model_parallel_all_reduce"
                        ) as moe_reduction,
                        mock.patch(
                            "sglang.srt.models.deepseek_v2.get_exec",
                            return_value=SimpleNamespace(
                                moe=SimpleNamespace(enable_eplb=False)
                            ),
                        ),
                    ):
                        actual = moe(x)
                    self.assertIsNone(state.output)
                    reduction.assert_not_called()
                    moe_reduction.assert_not_called()
                    torch.testing.assert_close(actual, x * 1.5, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
