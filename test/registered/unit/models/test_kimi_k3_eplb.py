import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

import sglang.srt.models.kimi_k3 as kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def make_module(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


class TestKimiK3Eplb(CustomTestCase):
    def test_expert_metadata(self):
        for groups in (None, 0, 1, 2):
            with self.subTest(groups=groups):
                config = SimpleNamespace(
                    num_hidden_layers=5, num_experts=4, num_expert_group=groups
                )
                for cls, argument in (
                    (kimi_k3.KimiK3LinearForCausalLM, config),
                    (
                        kimi_k3.KimiK3ForConditionalGeneration,
                        SimpleNamespace(text_config=config),
                    ),
                ):
                    metadata = cls.get_model_config_for_expert_location(argument)
                    self.assertEqual(metadata.num_layers, 5)
                    self.assertEqual(metadata.num_logical_experts, 4)
                    self.assertEqual(metadata.num_groups, groups or None)
                config.num_experts = None
                self.assertIsNone(
                    kimi_k3.KimiK3LinearForCausalLM.get_model_config_for_expert_location(
                        config
                    )
                )
                self.assertIsNone(
                    kimi_k3.KimiK3ForConditionalGeneration.get_model_config_for_expert_location(
                        SimpleNamespace(text_config=config)
                    )
                )

    def test_routed_paths_pass_dispatch_info(self):
        hidden = torch.ones(1, 3)
        info = object()
        topk_output = object()
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                moe = make_module(kimi_k3.KimiK3MoE)
                moe.layer_idx = 3
                moe._route_quant_fuse_eligible = False
                moe.topk = Mock(return_value=topk_output)
                moe.experts = Mock(return_value=hidden)
                with (
                    patch.object(
                        kimi_k3.ExpertLocationDispatchInfo,
                        "init_new",
                        return_value=info,
                    ) as init,
                    patch.object(
                        kimi_k3.zero_copy_context,
                        "set_moe_output",
                        return_value=nullcontext(),
                    ),
                    patch.object(kimi_k3.route_quant_handoff, "clear"),
                ):
                    if deferred:
                        moe._forward_routed_deferred(hidden, None, hidden)
                        expert_call = moe.experts.forward_deferred_finalize
                    else:
                        moe._forward_routed(hidden, None, hidden, hidden)
                        expert_call = moe.experts
                init.assert_called_once_with(layer_id=3)
                moe.topk.assert_called_once_with(
                    hidden, None, expert_location_dispatch_info=info
                )
                expert_call.assert_called_once_with(hidden, topk_output)

    def test_dispatch_disables_fused_router(self):
        for algorithm in ("static", "dynamic", "fake", "lp"):
            with self.subTest(algorithm=algorithm):
                moe = make_module(kimi_k3.KimiK3MoE)
                moe._eligible_for_fused_front = False
                with patch.object(
                    kimi_k3,
                    "get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(ep_dispatch_algorithm=algorithm)
                    ),
                ):
                    self.assertFalse(moe._routing_contract_ok)

    def test_weight_views_exclude_non_routed_parameters(self):
        moe = make_module(kimi_k3.KimiK3MoE)
        moe._use_mega_moe = False
        experts = nn.Module()
        experts.num_local_experts = 4
        for name, shape in (
            ("weight", (4, 3, 2)),
            ("scale", (4, 1)),
            ("correction_bias", (4,)),
            ("global_scale", (4,)),
            ("scalar", ()),
            ("other", (3, 2)),
        ):
            experts.register_parameter(
                name, nn.Parameter(torch.ones(shape), requires_grad=False)
            )
        experts.global_scale._sglang_require_global_experts = True
        moe.experts = experts
        weights = moe.get_moe_weights()
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0].data_ptr(), experts.weight.data_ptr())
        self.assertEqual(weights[1].data_ptr(), experts.scale.data_ptr())
        weights[0][0, 0, 0] = 7
        self.assertEqual(experts.weight[0, 0, 0].item(), 7)

    def test_pp_layer_ids_and_multimodal_forwarding(self):
        moe = make_module(kimi_k3.KimiK3MoE)
        weight = torch.ones(4, 2)
        moe.get_moe_weights = Mock(return_value=[weight])
        text = make_module(kimi_k3.KimiK3LinearForCausalLM)
        text.model = SimpleNamespace(
            start_layer=2,
            end_layer=4,
            layers={
                0: SimpleNamespace(mlp=moe),
                2: SimpleNamespace(mlp=nn.Identity()),
                3: SimpleNamespace(mlp=moe),
                4: SimpleNamespace(mlp=moe),
            },
        )
        wrapper = make_module(kimi_k3.KimiK3ForConditionalGeneration)
        wrapper.language_model = text
        result = wrapper.routed_experts_weights_of_layer
        self.assertEqual(list(result), [3])
        self.assertIs(result[3][0], weight)
        moe.get_moe_weights.assert_called_once_with()

    def test_megamoe_rebalance_rejected(self):
        moe = make_module(kimi_k3.KimiK3MoE)
        moe._use_mega_moe = True
        with self.assertRaisesRegex(NotImplementedError, "MegaMoE"):
            moe.get_moe_weights()


if __name__ == "__main__":
    unittest.main()
