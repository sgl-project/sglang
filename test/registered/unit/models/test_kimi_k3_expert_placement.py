"""Kimi text and multimodal models expose the same initial expert layout, and a
K3 MoE layer leaves no routing state behind for the next one."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.layers.moe import aiter_presort
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


class TestKimiK3MoEPresortLifetime(CustomTestCase):
    def test_unconsumed_presort_is_cleared_on_the_unfused_path(self):
        """AITER matches a registered presort by topk tensor address, so one that no
        fused_moe consumed must not outlive the layer, whichever path routed it."""
        registrations = []

        def route_without_fused_moe(hidden_states, *, prefix_sum, valid_token_count):
            registrations.append({"M": hidden_states.shape[0]})
            return hidden_states

        moe = SimpleNamespace(
            _eligible_for_fused_front=False,
            _forward_unfused=route_without_fused_moe,
        )
        with patch.object(
            aiter_presort, "_registrar", return_value=registrations.append
        ):
            KimiK3MoE.forward(moe, torch.zeros(2, 4))
        self.assertIsNone(registrations[-1])


if __name__ == "__main__":
    unittest.main()
