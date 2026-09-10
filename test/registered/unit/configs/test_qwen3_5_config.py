"""Dense Qwen3_5TextConfig must not surface MoE-only sizes inherited from Qwen3NextConfig.

If it does, check_quantized_moe_compatibility treats block-quantized FP8 dense
checkpoints as MoE and rejects valid TP sharding at startup.
"""

import unittest

from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestQwen35ConfigMoEInheritance(CustomTestCase):
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
