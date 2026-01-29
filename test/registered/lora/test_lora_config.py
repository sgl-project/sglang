# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for LoRAConfig functionality."""

import unittest

from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-a-test-2")


class TestLoRAConfigFilterAddedTokens(unittest.TestCase):
    """Test cases for LoRAConfig.filter_added_tokens method."""

    def _create_config_with_added_tokens(self, added_tokens: dict) -> LoRAConfig:
        """Helper to create a LoRAConfig with specified added_tokens."""
        config_dict = {
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        return LoRAConfig.from_dict(config_dict, added_tokens_config=added_tokens)

    def test_filter_no_added_tokens(self):
        """Test filtering when there are no added tokens."""
        config = self._create_config_with_added_tokens(None)
        self.assertEqual(config.lora_added_tokens_size, 0)

        config.filter_added_tokens(base_vocab_size=32000)
        self.assertEqual(config.lora_added_tokens_size, 0)
        self.assertIsNone(config.added_tokens_config)

    def test_filter_empty_added_tokens(self):
        """Test filtering when added_tokens is empty dict."""
        config = self._create_config_with_added_tokens({})
        self.assertEqual(config.lora_added_tokens_size, 0)

        config.filter_added_tokens(base_vocab_size=32000)
        self.assertEqual(config.lora_added_tokens_size, 0)

    def test_filter_all_fake_tokens(self):
        """Test filtering when all tokens are fake (ID < base_vocab_size)."""
        # These tokens have IDs less than base_vocab_size (32000)
        added_tokens = {
            "<pad>": 0,
            "<eos>": 2,
            "<bos>": 1,
        }
        config = self._create_config_with_added_tokens(added_tokens)
        self.assertEqual(config.lora_added_tokens_size, 3)

        config.filter_added_tokens(base_vocab_size=32000)
        self.assertEqual(config.lora_added_tokens_size, 0)
        self.assertEqual(config.added_tokens_config, {})

    def test_filter_all_real_tokens(self):
        """Test filtering when all tokens are real (ID >= base_vocab_size)."""
        base_vocab_size = 32000
        # These tokens have IDs >= base_vocab_size
        added_tokens = {
            "<new_token_1>": 32000,
            "<new_token_2>": 32001,
            "<new_token_3>": 32002,
        }
        config = self._create_config_with_added_tokens(added_tokens)
        self.assertEqual(config.lora_added_tokens_size, 3)

        config.filter_added_tokens(base_vocab_size=base_vocab_size)
        self.assertEqual(config.lora_added_tokens_size, 3)
        self.assertEqual(config.added_tokens_config, added_tokens)

    def test_filter_mixed_tokens(self):
        """Test filtering with both fake and real tokens."""
        base_vocab_size = 32000
        added_tokens = {
            # Fake tokens (ID < base_vocab_size)
            "<pad>": 0,
            "<eos>": 2,
            # Real tokens (ID >= base_vocab_size)
            "<new_token_1>": 32000,
            "<new_token_2>": 32001,
        }
        config = self._create_config_with_added_tokens(added_tokens)
        self.assertEqual(config.lora_added_tokens_size, 4)

        config.filter_added_tokens(base_vocab_size=base_vocab_size)
        self.assertEqual(config.lora_added_tokens_size, 2)
        self.assertEqual(
            config.added_tokens_config,
            {"<new_token_1>": 32000, "<new_token_2>": 32001},
        )

    def test_filter_boundary_token(self):
        """Test token exactly at base_vocab_size boundary."""
        base_vocab_size = 32000
        added_tokens = {
            "<at_boundary>": 32000,  # Exactly at boundary, should be kept
            "<below_boundary>": 31999,  # Just below, should be filtered
        }
        config = self._create_config_with_added_tokens(added_tokens)

        config.filter_added_tokens(base_vocab_size=base_vocab_size)
        self.assertEqual(config.lora_added_tokens_size, 1)
        self.assertEqual(config.added_tokens_config, {"<at_boundary>": 32000})

    def test_filter_idempotent(self):
        """Test that calling filter_added_tokens multiple times is safe."""
        base_vocab_size = 32000
        added_tokens = {
            "<fake>": 100,
            "<real>": 32000,
        }
        config = self._create_config_with_added_tokens(added_tokens)

        # First call
        config.filter_added_tokens(base_vocab_size=base_vocab_size)
        self.assertEqual(config.lora_added_tokens_size, 1)

        # Second call should not change anything
        config.filter_added_tokens(base_vocab_size=base_vocab_size)
        self.assertEqual(config.lora_added_tokens_size, 1)
        self.assertEqual(config.added_tokens_config, {"<real>": 32000})


class TestLoRAConfigEffectiveTargetModules(unittest.TestCase):
    """Test cases for LoRAConfig.effective_target_modules attribute."""

    def test_from_dict_has_no_effective_modules(self):
        """Test that from_dict creates config without effective_target_modules."""
        config_dict = {
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        config = LoRAConfig.from_dict(config_dict)

        # When created from dict (no path), effective_target_modules should be None
        self.assertIsNone(config.effective_target_modules)
        # target_modules should still be set
        self.assertEqual(config.target_modules, ["q_proj", "k_proj", "v_proj"])


if __name__ == "__main__":
    unittest.main(warnings="ignore")
