"""DeepSeek-V4 MoE on ROCm: skipped reductions leave the mHC post pending."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")


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
