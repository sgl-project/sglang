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

import unittest

from sglang.srt.lora.utils import get_normalized_target_modules


class TestLoRAUtils(unittest.TestCase):
    """Test utilities for LoRA functionality."""

    def test_get_normalized_target_modules_base_names(self):
        """Test normalization with base module names."""
        input_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "o_proj",
        ]
        expected = {"qkv_proj", "gate_up_proj", "down_proj", "o_proj"}
        result = get_normalized_target_modules(input_modules)
        self.assertEqual(result, expected)

    def test_get_normalized_target_modules_prefixed_names(self):
        """Test normalization with prefixed module names (e.g., feed_forward.gate_proj)."""
        input_modules = [
            "feed_forward.gate_proj",
            "feed_forward.up_proj",
            "feed_forward.down_proj",
            "attention.q_proj",
            "attention.k_proj",
            "attention.v_proj",
            "attention.o_proj",
        ]
        expected = {"gate_up_proj", "down_proj", "qkv_proj", "o_proj"}
        result = get_normalized_target_modules(input_modules)
        self.assertEqual(result, expected)

    def test_get_normalized_target_modules_mixed_format(self):
        """Test normalization with mixed base and prefixed module names."""
        input_modules = [
            "q_proj",
            "feed_forward.gate_proj",
            "attention.k_proj",
            "down_proj",
        ]
        expected = {"qkv_proj", "gate_up_proj", "down_proj"}
        result = get_normalized_target_modules(input_modules)
        self.assertEqual(result, expected)

    def test_get_normalized_target_modules_compatibility_scenario(self):
        """Test the specific compatibility scenario that was failing."""
        # Simulate server target modules (from --lora-target-modules all)
        server_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "qkv_proj",
            "gate_up_proj",
        ]
        server_normalized = get_normalized_target_modules(server_modules)

        # Simulate adapter target modules (from PEFT config)
        adapter_modules = [
            "feed_forward.gate_proj",
            "feed_forward.up_proj",
            "feed_forward.down_proj",
            "o_proj",
            "q_proj",
            "k_proj",
            "v_proj",
        ]
        adapter_normalized = get_normalized_target_modules(adapter_modules)

        # Both should normalize to the same set for compatibility
        self.assertEqual(server_normalized, adapter_normalized)

        # Verify adapter is subset of server (compatibility check)
        self.assertTrue(adapter_normalized.issubset(server_normalized))

    def test_get_normalized_target_modules_empty_input(self):
        """Test normalization with empty input."""
        result = get_normalized_target_modules([])
        self.assertEqual(result, set())

    def test_get_normalized_target_modules_unknown_modules(self):
        """Test normalization with unknown module names."""
        input_modules = ["unknown_proj", "custom.module_proj"]
        expected = {"unknown_proj", "module_proj"}
        result = get_normalized_target_modules(input_modules)
        self.assertEqual(result, expected)

    def test_get_normalized_target_modules_duplicate_handling(self):
        """Test that duplicate inputs result in unique outputs."""
        input_modules = [
            "q_proj",
            "k_proj",
            "q_proj",
            "feed_forward.gate_proj",
            "gate_proj",
        ]
        # Both q_proj and k_proj -> qkv_proj, both gate_proj variants -> gate_up_proj
        expected = {"qkv_proj", "gate_up_proj"}
        result = get_normalized_target_modules(input_modules)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
