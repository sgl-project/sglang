"""Regression tests for Cohere2MoeConfig import (sgl-project/sglang#28233)."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCohere2MoeConfig(CustomTestCase):
    def test_configs_package_imports(self):
        """Importing the configs package must not crash at module load."""
        import sglang.srt.configs  # noqa: F401

    def test_config_instantiates(self):
        from sglang.srt.configs.cohere2_moe import Cohere2MoeConfig

        cfg = Cohere2MoeConfig()
        self.assertEqual(cfg.model_type, "cohere2_moe")

    def test_post_init_defaults_num_key_value_heads(self):
        """``__post_init__`` must still run and default kv heads to attn heads."""
        from sglang.srt.configs.cohere2_moe import Cohere2MoeConfig

        cfg = Cohere2MoeConfig()
        self.assertEqual(cfg.num_key_value_heads, cfg.num_attention_heads)


if __name__ == "__main__":
    unittest.main()
