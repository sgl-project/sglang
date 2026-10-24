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

from sglang.srt.eplb.eplb_manager import get_routed_experts_weights_of_layer
from sglang.srt.eplb.expert_location_updater import _log_p2p_op_metrics
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK, TopKConfig, TopKOutputFormat
from sglang.srt.models import glm5_next as glm5_next_module
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.models.glm5_next import _collect_stage_routed_experts_weights
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import LazyValue
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


class _HashForbiddenParameter(nn.Parameter):
    def __hash__(self):
        raise AssertionError("EPLB weight discovery must not hash expert parameters")


class _FakeMoE(nn.Module):
    def __init__(self, *, num_fused_shared_experts: int):
        super().__init__()
        self.num_fused_shared_experts = num_fused_shared_experts
        self.experts = _FakeExpertParam()


class _FakeGlmMoE(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def get_moe_weights(self):
        return self.value


class TestWaterfillEPLB(CustomTestCase):
    def test_eplb_metrics_map_global_peers_to_stage_local_slots(self):
        op = SimpleNamespace(
            op=torch.distributed.isend,
            peer=3,
            tensor=torch.ones(2, dtype=torch.float32),
        )

        with self.assertLogs(
            "sglang.srt.eplb.expert_location_updater", level="INFO"
        ) as logs:
            _log_p2p_op_metrics(
                [(0, [op])],
                peer_ranks=(2, 3),
                rank_to_node=(5, 5),
                self_node_id=5,
            )

        self.assertIn("isend_nbytes_of_gpu=[0, 8]", logs.output[0])
        self.assertIn("isend_nbytes_of_node=[8]", logs.output[0])

    def test_glm5_weight_collection_skips_pp_missing_layers(self):
        expected = [torch.ones(1)]
        local_layer = SimpleNamespace(mlp=_FakeGlmMoE(expected))
        model = SimpleNamespace(
            start_layer=1,
            end_layer=2,
            layers=[nn.Module(), local_layer, nn.Module()],
        )

        with patch.object(glm5_next_module, "Glm5NextMoE", _FakeGlmMoE):
            actual = _collect_stage_routed_experts_weights(model)

        self.assertEqual(actual, {1: expected})

    def test_eplb_weight_map_resolver_materializes_private_lazy_value(self):
        model = nn.Module()
        expected = {3: [torch.ones(1)]}
        model._routed_experts_weights_of_layer = LazyValue(lambda: expected)

        self.assertIs(get_routed_experts_weights_of_layer(model), expected)

    def test_lazy_value_materialize_preserves_creator_attribute_error(self):
        def fail():
            raise AttributeError("original creator failure")

        with self.assertRaisesRegex(AttributeError, "original creator failure"):
            LazyValue(fail).materialize()

    def test_deepseek_moe_get_moe_weights_excludes_fused_shared_slot(self):
        moe = _FakeMoE(num_fused_shared_experts=1)
        experts = moe.experts
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
        moe = _FakeMoE(num_fused_shared_experts=0)
        experts = moe.experts
        weights = DeepseekV2MoE.get_moe_weights(moe)

        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0].shape, (experts.num_local_experts, 2))

    def test_deepseek_moe_get_moe_weights_ignores_module_back_reference(self):
        """EPLB discovery must avoid module traversal and parameter hashing."""

        moe = _FakeMoE(num_fused_shared_experts=0)
        experts = moe.experts
        experts.add_module("runner_layer", experts)
        experts.weight = _HashForbiddenParameter(experts.weight.detach())

        weights = DeepseekV2MoE.get_moe_weights(moe)

        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0].data_ptr(), experts.weight.data.data_ptr())

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
