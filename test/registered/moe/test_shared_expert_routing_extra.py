"""Extended shared-slot routing and compiled model-context coverage."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    biased_grouped_topk_impl,
    select_experts,
)
from sglang.srt.runtime_context import get_context, get_model, get_parallel
from sglang.srt.utils import get_device, is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="extra-a", runner_config="1-gpu-small")


@unittest.skipUnless(is_cuda(), "covers the CUDA routers")
class TestSelectExpertsPerRankSharedSlots(CustomTestCase):
    """Shared slots must not alter routed selection or normalization."""

    NUM_TOKENS, TOP_K, EP_SIZE, EP_RANK = 32, 8, 8, 3
    # (num_experts, num_expert_group, topk_group, scoring_func, biased); each row
    # reaches a different router.
    ROUTERS = [
        (256, 8, 4, "sigmoid", True),  # DeepSeek-V3 grouped routing
        (384, 8, 4, "sigmoid", True),  # grouped, outside FlashInfer's shapes
        (64, 8, 3, "softmax", False),  # grouped, no correction bias
        (256, None, None, "sqrtsoftplus", True),  # DeepSeek-V4 routing
        (128, None, None, "softmax", False),
    ]

    def _route(
        self, *, num_fused_shared_experts, logits, bias, router, renormalize=True
    ):
        _, num_expert_group, topk_group, scoring_func, _ = router
        topk_config = TopKConfig(
            top_k=self.TOP_K + num_fused_shared_experts,
            renormalize=renormalize,
            use_grouped_topk=num_expert_group is not None,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            shared_experts=SharedExpertsConfig(
                num_fused_shared_experts=num_fused_shared_experts
            ),
            correction_bias=bias,
            scoring_func=scoring_func,
            routed_scaling_factor=2.5,
            allow_routed_experts_capture=False,
        )
        with (
            patch.object(
                topk_module,
                "has_per_rank_fused_shared_slots",
                side_effect=lambda num_fused: num_fused > 0,
            ),
            # Model one rank of TP=EP; routing itself issues no collectives.
            get_parallel().override(
                tp_size=self.EP_SIZE,
                tp_rank=self.EP_RANK,
                tp_group=None,
                attn_tp_size=self.EP_SIZE,
                attn_tp_rank=self.EP_RANK,
                attn_tp_group=None,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                moe_ep_size=self.EP_SIZE,
                moe_ep_rank=self.EP_RANK,
                moe_ep_group=None,
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_dp_size=1,
                moe_dp_rank=0,
            ),
        ):
            out = select_experts(
                torch.empty(logits.shape[0], 16, device=logits.device),
                logits,
                topk_config,
            )
        return out.topk_ids.long(), out.topk_weights

    def test_routes_like_no_fusion(self):
        k = self.TOP_K
        for router in self.ROUTERS:
            num_experts, _, _, _, biased = router
            with self.subTest(router=router):
                g = torch.Generator(device="cpu").manual_seed(num_experts)
                logits = torch.randn(self.NUM_TOKENS, num_experts, generator=g)
                bias = 0.1 * torch.randn(num_experts, generator=g) if biased else None
                logits = logits.to(get_device())
                bias = bias.to(get_device()) if biased else None
                ref_ids, ref_weights = self._route(
                    num_fused_shared_experts=0, logits=logits, bias=bias, router=router
                )
                ids, weights = self._route(
                    num_fused_shared_experts=1, logits=logits, bias=bias, router=router
                )

                self.assertEqual(ids.shape, (self.NUM_TOKENS, k + 1))
                torch.testing.assert_close(
                    weights[:, :k].sum(-1), torch.ones_like(weights[:, 0])
                )
                # Rank r's routed block starts at r * (num_local_routed + 1).
                num_local_routed = num_experts // self.EP_SIZE
                expected_ids = ref_ids + ref_ids // num_local_routed
                order, ref_order = ids[:, :k].argsort(-1), expected_ids.argsort(-1)
                self.assertTrue(
                    torch.equal(
                        ids[:, :k].gather(-1, order), expected_ids.gather(-1, ref_order)
                    )
                )
                torch.testing.assert_close(
                    weights[:, :k].gather(-1, order), ref_weights.gather(-1, ref_order)
                )
                home_shared_id = (
                    self.EP_RANK * (num_local_routed + 1) + num_local_routed
                )
                self.assertTrue(torch.all(ids[:, k] == home_shared_id))

    def test_unnormalized_global_shared_contribution_is_one(self):
        """Raw routed scores must not multiply the global shared contribution."""
        logits = torch.zeros(4, 256, device=get_device())
        rsf = 2.5
        with get_context().override_server_args(
            moe_a2a_backend="none", moe_runner_backend="triton"
        ):
            for folded in (False, True):
                with self.subTest(folded=folded):
                    config = TopKConfig(
                        top_k=9,
                        use_grouped_topk=True,
                        num_expert_group=8,
                        topk_group=4,
                        renormalize=False,
                        shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
                        correction_bias=torch.zeros(256, device=get_device()),
                        scoring_func="sigmoid",
                        routed_scaling_factor=rsf,
                        apply_routed_scaling_factor_on_output=folded,
                        allow_routed_experts_capture=False,
                    )
                    output = select_experts(logits, logits, config)
                    torch.testing.assert_close(
                        output.topk_weights[:, :8],
                        torch.full_like(
                            output.topk_weights[:, :8], 0.5 * (rsf if folded else 1.0)
                        ),
                    )
                    shared = output.topk_weights[:, -1] * (1.0 if folded else rsf)
                    torch.testing.assert_close(shared, torch.ones_like(shared))

    def test_unnormalized_weights_stay_raw(self):
        """Unnormalized routing must preserve raw selected sigmoid scores."""
        k, num_experts = self.TOP_K, 256
        router = (num_experts, 8, 4, "sigmoid", True)
        g = torch.Generator(device="cpu").manual_seed(0)
        logits = torch.randn(self.NUM_TOKENS, num_experts, generator=g).to(get_device())
        bias = (0.1 * torch.randn(num_experts, generator=g)).to(get_device())
        _, ref_ids = biased_grouped_topk_impl(
            logits,
            logits,
            bias,
            topk=k,
            renormalize=False,
            num_expert_group=8,
            topk_group=4,
        )
        num_local_routed = num_experts // self.EP_SIZE
        for num_fused_shared_experts in (0, 1):
            with self.subTest(num_fused_shared_experts=num_fused_shared_experts):
                ids, weights = self._route(
                    num_fused_shared_experts=num_fused_shared_experts,
                    logits=logits,
                    bias=bias,
                    router=router,
                    renormalize=False,
                )
                routed_ids = ids[:, :k]
                if num_fused_shared_experts:
                    routed_ids = routed_ids - routed_ids // (num_local_routed + 1)
                self.assertTrue(
                    torch.equal(
                        routed_ids.sort(-1).values, ref_ids.long().sort(-1).values
                    )
                )
                torch.testing.assert_close(
                    weights[:, :k], logits.sigmoid().gather(-1, routed_ids)
                )


class TestSharedExpertsCompiledContext(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disable_shared_experts_fusion=False
        )
        override.install()
        self.addCleanup(override.restore)

    def test_compile_reads_the_selected_model_config(self):
        # Bag leaves are plain attributes: fullgraph tracing must work and must
        # distinguish two runners even when the tensor shapes match.
        def compute(x):
            config = get_model().shared_experts
            return x + config.num_fused_shared_experts + config.num_shared_topk_slots

        compiled = torch.compile(compute, backend="eager", fullgraph=True)
        for config, expected in (
            (SharedExpertsConfig(num_fused_shared_experts=2), 7),
            (SharedExpertsConfig(num_fused_shared_experts=1, use_waterfill=True), 4),
            (SharedExpertsConfig(), 3),
        ):
            with get_model().override(shared_experts=config):
                self.assertEqual(compiled(torch.tensor([3])).item(), expected)


if __name__ == "__main__":
    unittest.main()
