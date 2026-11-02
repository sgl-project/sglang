"""Kimi text and multimodal models expose the same initial expert layout."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.kimi_linear import KimiLinearConfig
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


if __name__ == "__main__":
    unittest.main()
