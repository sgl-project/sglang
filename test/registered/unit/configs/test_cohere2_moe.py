# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC

"""Unit tests for srt/configs/cohere2_moe.py"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

import unittest
from sglang.srt.configs.cohere2_moe import Cohere2MoeConfig
from sglang.test.test_utils import CustomTestCase


class TestCohere2MoeConfig(CustomTestCase):
    def test_init_config(self) -> None:
        config = Cohere2MoeConfig()
        self.assertEqual(config.model_type, "cohere2_moe")
        self.assertEqual(config.hidden_size, 8192)
        # Default value post-init check
        self.assertEqual(config.num_key_value_heads, config.num_attention_heads)


if __name__ == "__main__":
    unittest.main()
