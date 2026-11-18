"""Kimi text and multimodal models expose the same initial expert layout, and a
K3 MoE layer leaves no routing state behind for the next one."""

import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.layers.moe import route_quant_handoff
from sglang.srt.layers.moe.moe_runner import aiter_route_sort
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
    KimiK3MoE,
)
from sglang.srt.runtime_context import get_context, get_exec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKimiK3ExpertPlacement(CustomTestCase):
    def setUp(self):
        super().setUp()
        scope = get_context().override_server_args(
            enable_eplb=False, ep_num_redundant_experts=0
        )
        scope.install()
        self.addCleanup(scope.restore)

    def test_text_and_multimodal_metadata(self):
        config = KimiLinearConfig(
            num_hidden_layers=93, num_experts=896, num_expert_group=1
        )
        with get_exec().moe.override(enable_eplb=False, ep_num_redundant_experts=0):
            text = KimiK3LinearForCausalLM.get_model_config_for_expert_location(config)
            multimodal = (
                KimiK3ForConditionalGeneration.get_model_config_for_expert_location(
                    SimpleNamespace(text_config=config)
                )
            )
        self.assertEqual(text, multimodal)
        self.assertEqual(
            (text.num_layers, text.num_logical_experts, text.num_groups), (93, 896, 1)
        )

    def test_rejects_unimplemented_weight_relocation(self):
        config = KimiLinearConfig()
        for online, redundant in [(True, 0), (False, 8)]:
            with (
                self.subTest(online=online, redundant=redundant),
                get_exec().moe.override(
                    enable_eplb=online, ep_num_redundant_experts=redundant
                ),
            ):
                with self.assertRaisesRegex(ValueError, "without online EPLB"):
                    KimiK3MoE(config)


def _unfused_moe(
    *,
    route_sort_consumer: Optional[aiter_route_sort.AiterRouteSortConsumer],
    registrations: list,
) -> SimpleNamespace:
    def route_without_fused_moe(hidden_states, *, prefix_sum, valid_token_count):
        registrations.append({"M": hidden_states.shape[0]})
        return hidden_states

    return SimpleNamespace(
        _row_front_out=None,
        _eligible_for_fused_front=False,
        _forward_unfused=route_without_fused_moe,
        _route_sort_consumer=route_sort_consumer,
    )


class TestKimiK3MoERouteSortLifetime(CustomTestCase):
    def test_unconsumed_route_sort_is_dropped_on_the_unfused_path(self):
        """AITER matches a registered sort by topk tensor address, so one that no
        fused_moe consumed must not outlive the layer, whichever path routed it."""
        registrations = []
        moe = _unfused_moe(
            route_sort_consumer=aiter_route_sort.AiterRouteSortConsumer(),
            registrations=registrations,
        )
        with patch.object(
            aiter_route_sort, "_registrar", return_value=registrations.append
        ):
            KimiK3MoE.forward(moe, torch.zeros(2, 4))
        self.assertIsNone(registrations[-1])
        # The request no routing call consumed is dropped with it.
        self.assertEqual(route_quant_handoff.take_route_sort(), (None, None))

    def test_route_sort_request_needs_unrewritten_routing(self):
        """AITER matches the sort to the router's own topk tensors, so a request is
        dropped when select_experts rewrites them (EPLB remap, simulated routing,
        shared slots), and otherwise carries its padding row count."""
        non_padded = torch.tensor(3)
        consumer = aiter_route_sort.AiterRouteSortConsumer()
        try:
            route_quant_handoff.stage_route_sort(consumer)
            route_quant_handoff.bind_select_experts(non_padded, fused_routing_ok=False)
            self.assertIsNone(route_quant_handoff.take_route_sort()[0])
            route_quant_handoff.stage_route_sort(consumer)
            route_quant_handoff.bind_select_experts(non_padded, fused_routing_ok=True)
            self.assertEqual(
                route_quant_handoff.take_route_sort(), (consumer, non_padded)
            )
            # Consumed once; the runner still sees the consumer until clear().
            self.assertIsNone(route_quant_handoff.take_route_sort()[0])
            self.assertIs(route_quant_handoff.staged_route_sort(), consumer)
        finally:
            route_quant_handoff.clear()

    def test_layer_without_route_sort_never_probes_aiter(self):
        """CUDA, non-AITER runners and EP layers cannot consume a router-written
        sort, so they skip the cleanup and never import AITER."""
        registrations = []
        moe = _unfused_moe(route_sort_consumer=None, registrations=registrations)
        with patch.object(aiter_route_sort, "_registrar") as registrar:
            KimiK3MoE.forward(moe, torch.zeros(2, 4))
        registrar.assert_not_called()
        self.assertEqual(registrations, [{"M": 2}])


if __name__ == "__main__":
    unittest.main()
