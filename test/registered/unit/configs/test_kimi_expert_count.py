"""Kimi expert-count compatibility with shared routed-expert capture."""

import unittest

from sglang.srt.configs.kimi_k3 import KimiK3Config
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKimiExpertCount(unittest.TestCase):
    def test_k3_capture_reads_checkpoint_expert_count(self):
        config = KimiK3Config(
            text_config={"num_experts": 896, "num_experts_per_token": 16}
        )
        # This is the attribute read by RoutedExpertsCapturer.__init__.
        self.assertEqual(config.text_config.num_experts_per_tok, 16)

    def test_alias_tracks_updates_in_both_directions(self):
        config = KimiLinearConfig(num_experts_per_token=8)
        config.num_experts_per_token = 16
        self.assertEqual(config.num_experts_per_tok, 16)
        config.num_experts_per_tok = 4
        self.assertEqual(config.num_experts_per_token, 4)

    def test_k3_round_trip_preserves_checkpoint_field(self):
        config = KimiK3Config(text_config={"num_experts_per_token": 16})
        serialized = config.to_dict()
        self.assertEqual(serialized["text_config"]["num_experts_per_token"], 16)
        self.assertNotIn("num_experts_per_tok", serialized["text_config"])
        restored = KimiK3Config.from_dict(serialized)
        self.assertEqual(restored.text_config.num_experts_per_tok, 16)


if __name__ == "__main__":
    unittest.main()
