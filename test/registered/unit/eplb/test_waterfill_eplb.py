"""Unit tests for Waterfill and EPLB updater compatibility."""

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.runtime_context import get_model
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK, TopKConfig, TopKOutputFormat
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase


class _FakeExpertParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_local_experts = 5
        self.weight = nn.Parameter(
            torch.arange(10, dtype=torch.float32).reshape(self.num_local_experts, 2)
        )
        self.correction_bias = nn.Parameter(torch.ones(self.num_local_experts))
        self.global_scale = nn.Parameter(torch.ones(self.num_local_experts))
        self.global_scale._sglang_require_global_experts = True


class TestWaterfillEPLB(CustomTestCase):
    def test_deepseek_moe_get_moe_weights_excludes_fused_shared_slot(self):
        experts = _FakeExpertParam()
        moe = SimpleNamespace(num_fused_shared_experts=1, experts=experts)
        shared_before = experts.weight.data[-1].clone()

        weights = DeepseekV2MoE.get_moe_weights(moe)

        self.assertEqual(len(weights), 1)
        self.assertEqual(
            weights[0].shape,
            (experts.num_local_experts - moe.num_fused_shared_experts, 2),
        )

        weights[0][-1].zero_()
        self.assertTrue(torch.equal(experts.weight.data[-2], torch.zeros(2)))
        self.assertTrue(torch.equal(experts.weight.data[-1], shared_before))

    def test_deepseek_moe_get_moe_weights_keeps_full_shape_without_fusion(self):
        experts = _FakeExpertParam()
        moe = SimpleNamespace(num_fused_shared_experts=0, experts=experts)
        weights = DeepseekV2MoE.get_moe_weights(moe)

        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0].shape, (experts.num_local_experts, 2))

    def test_topk_recorder_ids_exclude_per_rank_fused_shared_slots(self):
        topk_ids = torch.tensor([[0, 33, 263, 256]], dtype=torch.int32)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
            routed_scaling_factor=1.0,
        )
        dispatch_info = SimpleNamespace(
            num_physical_experts=264, ep_dispatch_algorithm="static"
        )

        def fake_eplb_postprocess(
            ids, expert_location_dispatch_info, num_token_non_padded
        ):
            return ids

        with (
            patch.object(topk_module, "_is_cuda", True),
            patch.object(topk_module, "_use_aiter", False),
            patch.object(
                topk_module, "has_per_rank_fused_shared_slots", return_value=True
            ),
            get_parallel().override(moe_ep_size=8, moe_ep_rank=7),
            patch.object(
                topk_module,
                "_biased_grouped_topk_postprocess",
                side_effect=fake_eplb_postprocess,
            ),
        ):
            processed_ids, _, recorder_ids = topk_module._post_process_topk_ids(
                topk_ids=topk_ids.clone(),
                topk_weights=topk_weights.clone(),
                topk_config=topk_config,
                router_logits=torch.empty((1, 256)),
                layer_id=0,
                expert_location_dispatch_info=dispatch_info,
            )

        self.assertTrue(torch.equal(processed_ids, torch.tensor([[0, 34, 270, 271]])))
        self.assertTrue(torch.equal(recorder_ids, torch.tensor([[0, 33, 263]])))

    def test_eplb_remaps_every_column_of_routed_only_per_rank_ids(self):
        """EPLB must remap all routed columns before shared-slot append and capture."""
        topk_ids = torch.tensor([[0, 33, 262]], dtype=torch.int32)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
            routed_scaling_factor=1.0,
        )
        dispatch_info = SimpleNamespace(
            num_physical_experts=264, ep_dispatch_algorithm="static"
        )

        def fake_eplb_postprocess(
            ids, expert_location_dispatch_info, num_token_non_padded
        ):
            return ids + 1

        with (
            patch.object(topk_module, "_is_cuda", True),
            patch.object(topk_module, "_use_aiter", False),
            patch.object(
                topk_module, "has_per_rank_fused_shared_slots", return_value=True
            ),
            get_parallel().override(moe_ep_size=8, moe_ep_rank=7),
            patch.object(
                topk_module,
                "_biased_grouped_topk_postprocess",
                side_effect=fake_eplb_postprocess,
            ),
        ):
            processed_ids, _, recorder_ids = topk_module._post_process_topk_ids(
                topk_ids=topk_ids.clone(),
                topk_weights=topk_weights.clone(),
                topk_config=topk_config,
                router_logits=torch.empty((1, 256)),
                layer_id=0,
                expert_location_dispatch_info=dispatch_info,
            )

        # Physical ids 1, 34, 263 in the EP8 per-rank layout; 271 is rank 7's shared slot.
        self.assertTrue(torch.equal(processed_ids, torch.tensor([[1, 35, 270, 271]])))
        self.assertTrue(torch.equal(recorder_ids, torch.tensor([[1, 34, 263]])))

    def test_topk_recorder_ids_exclude_global_fused_shared_slot(self):
        topk_ids = torch.tensor([[0, 33, 200, 256]], dtype=torch.int32)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
            routed_scaling_factor=1.0,
        )
        dispatch_info = SimpleNamespace(
            num_physical_experts=256, ep_dispatch_algorithm="static"
        )

        def fake_eplb_postprocess(
            ids, expert_location_dispatch_info, num_token_non_padded
        ):
            return ids + 1

        with (
            patch.object(topk_module, "_is_cuda", True),
            patch.object(topk_module, "_use_aiter", False),
            patch.object(
                topk_module, "has_per_rank_fused_shared_slots", return_value=False
            ),
            patch.object(
                topk_module,
                "_biased_grouped_topk_postprocess",
                side_effect=fake_eplb_postprocess,
            ),
        ):
            processed_ids, _, recorder_ids = topk_module._post_process_topk_ids(
                topk_ids=topk_ids.clone(),
                topk_weights=topk_weights.clone(),
                topk_config=topk_config,
                router_logits=torch.empty((1, 256)),
                layer_id=0,
                expert_location_dispatch_info=dispatch_info,
            )

        self.assertTrue(torch.equal(processed_ids, torch.tensor([[1, 34, 201, 256]])))
        self.assertTrue(torch.equal(recorder_ids, torch.tensor([[1, 34, 201]])))

    def test_the_shared_expert_column_has_exactly_one_writer(self):
        """Shared slots must be written exactly once, by routing or Waterfill."""
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

        for enable_waterfill in (False, True):
            with self.subTest(enable_waterfill=enable_waterfill):
                with get_model().override(
                    shared_experts=SharedExpertsConfig(
                        num_fused_shared_experts=1, use_waterfill=enable_waterfill
                    )
                ):
                    topk = TopK(top_k=8, layer_id=0)
                    hash_topk = HashTopK(
                        topk=8, num_experts=256, vocab_size=4, layer_id=0
                    )
                with get_model().override(shared_experts=SharedExpertsConfig()):
                    unfused_topk = TopK(top_k=8, layer_id=0)

                for layer, router_top_k, router_shared_columns in (
                    (
                        topk,
                        topk.topk_config.top_k,
                        topk.topk_config.num_shared_topk_slots,
                    ),
                    (hash_topk, hash_topk.topk, hash_topk.num_shared_topk_slots),
                ):
                    self.assertEqual(layer.enable_waterfill, enable_waterfill)
                    self.assertEqual(
                        router_shared_columns + int(layer.enable_waterfill), 1
                    )
                    self.assertEqual(router_top_k - router_shared_columns, 8)
                self.assertEqual(hash_topk.tid2eid.shape[1], 8)
                if enable_waterfill:
                    self.assertEqual(
                        topk.topk_config.output_format, TopKOutputFormat.STANDARD
                    )

                self.assertFalse(unfused_topk.enable_waterfill)
                self.assertEqual(unfused_topk.topk_config.top_k, 8)
                self.assertEqual(unfused_topk.topk_config.num_shared_topk_slots, 0)


if __name__ == "__main__":
    unittest.main()
