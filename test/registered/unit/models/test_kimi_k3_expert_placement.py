"""Kimi text and multimodal models expose the same initial expert layout, and a
K3 MoE layer leaves no routing state behind for the next one."""

import unittest
from types import MethodType, SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe import route_quant_handoff, route_sort
from sglang.srt.layers.moe.moe_runner import aiter_route_sort
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
    KimiK3MoE,
)
from sglang.srt.models.kimi_k3_amd import moe as kimi_k3_amd_moe
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

    moe = SimpleNamespace(
        _row_front_out=None,
        _eligible_for_fused_front=False,
        _forward_unfused=route_without_fused_moe,
        _route_sort_consumer=route_sort_consumer,
    )
    moe._route_sort_scope = MethodType(KimiK3MoE._route_sort_scope, moe)
    return moe


class _MoeOnRunner(kimi_k3_amd_moe.KimiK3MoEHipMixin):
    """A MoE layer whose route-sort consumer comes from the mixin's own lookup."""

    def __init__(self, runner):
        self.experts = SimpleNamespace(runner=runner)


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
        self.assertIsNone(route_sort.staged())

    def test_route_sort_request_needs_unrewritten_routing(self):
        """AITER matches the sort to the router's own topk tensors, so a request is
        dropped when select_experts rewrites them (EPLB remap, simulated routing,
        shared slots), and otherwise carries its padding row count."""
        non_padded = torch.tensor(3)
        consumer = MagicMock()
        try:
            route_sort.stage(consumer)
            route_sort.bind_select_experts(non_padded, fused_routing_ok=False)
            self.assertIsNone(route_sort._take()[0])
            route_sort.stage(consumer)
            route_sort.bind_select_experts(non_padded, fused_routing_ok=True)
            self.assertEqual(route_sort._take(), (consumer, non_padded))
            # Consumed once; the runner still sees the consumer until clear().
            self.assertIsNone(route_sort._take()[0])
            self.assertIs(route_sort.staged(), consumer)
        finally:
            route_sort.clear(consumer)
        consumer.drop.assert_called_once_with()

    def test_route_sort_and_route_quant_stashes_clear_independently(self):
        """The routed-quant handoff is staged and cleared inside the layer's
        route-sort scope, so neither may clear the other's request."""
        consumer, x = MagicMock(), torch.zeros(2, 4)
        try:
            route_sort.stage(consumer)
            route_quant_handoff.stage(x)
            route_quant_handoff.clear()
            self.assertIs(route_sort.staged(), consumer)
            route_quant_handoff.stage(x)
            route_sort.clear(consumer)
            self.assertIs(route_quant_handoff._handoff.request_x, x)
        finally:
            route_quant_handoff.clear()
            route_sort.clear(consumer)

    def test_layer_off_rocm_neither_reads_the_switch_nor_asks_its_runner(self):
        """Off ROCm the radix-4 router writes no MoE sort: the layer's scope is a
        no-op and never reads SGLANG_K3_RADIX4_TOPK or asks its runner."""
        runner = MagicMock()
        moe = _MoeOnRunner(runner)
        with (
            patch.object(kimi_k3_amd_moe, "_is_hip", False),
            patch.object(envs.SGLANG_K3_RADIX4_TOPK, "get") as switch,
        ):
            with moe._route_sort_scope():
                self.assertIsNone(route_sort.staged())
        switch.assert_not_called()
        runner.route_sort_consumer.assert_not_called()

    def test_layer_on_rocm_stages_its_runners_consumer_around_the_forward(self):
        """On ROCm with SGLANG_K3_RADIX4_TOPK the layer asks its runner once and
        stages that consumer only while the scope is open."""
        runner = MagicMock()
        moe = _MoeOnRunner(runner)
        consumer = runner.route_sort_consumer.return_value
        with (
            patch.object(kimi_k3_amd_moe, "_is_hip", True),
            envs.SGLANG_K3_RADIX4_TOPK.override(True),
        ):
            for _ in range(2):
                with moe._route_sort_scope():
                    self.assertIs(route_sort.staged(), consumer)
                self.assertIsNone(route_sort.staged())
            # A layer without a runner has no consumer to stage.
            with _MoeOnRunner(None)._route_sort_scope():
                self.assertIsNone(route_sort.staged())
        runner.route_sort_consumer.assert_called_once_with()
        self.assertEqual(consumer.drop.call_count, 2)

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
