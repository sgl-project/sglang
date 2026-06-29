"""Unit tests for srt/configs/qwen3_5.py."""

import unittest

from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class TestQwen35Config(unittest.TestCase):
    def test_dense_text_config_clears_inherited_moe_sizes(self):
        cfg = Qwen3_5TextConfig()

        self.assertIsNone(cfg.moe_intermediate_size)
        self.assertIsNone(cfg.shared_expert_intermediate_size)

    def test_moe_text_config_keeps_moe_sizes(self):
        cfg = Qwen3_5MoeTextConfig()

        self.assertEqual(cfg.moe_intermediate_size, 512)
        self.assertEqual(cfg.shared_expert_intermediate_size, 512)


if __name__ == "__main__":
    unittest.main()
