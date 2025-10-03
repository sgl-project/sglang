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

"""
Unit tests for LoRA eviction policies.
Tests LRU, FIFO, and adapter pinning functionality.
"""

import multiprocessing as mp
import unittest

import torch

from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase


class TestLoRAEvictionPolicy(unittest.TestCase):
    """Unit tests for LoRA eviction policies."""

    def test_lru_eviction_policy(self):
        """Test LRU eviction policy unit functionality."""
        lru = get_eviction_policy("lru")
        adapters = ["adapter_1", "adapter_2", "adapter_3", "adapter_4"]

        # Mark adapters as used in order
        for adapter in adapters:
            lru.mark_used(adapter)

        # Reuse some to change LRU order
        # Order should now be: adapter_1 (oldest), adapter_3, adapter_2, adapter_4 (newest)
        lru.mark_used("adapter_2")
        lru.mark_used("adapter_4")

        # Test victim selection - should select adapter_1 (least recently used)
        candidates = set(adapters)
        victim = lru.select_victim(candidates)
        self.assertEqual(victim, "adapter_1")

        # Test with pinned adapter (simulate pre-filtering in mem_pool)
        pinned = {"adapter_1"}  # Pin the LRU adapter
        filtered_candidates = candidates - pinned  # Pre-filter like mem_pool does
        victim = lru.select_victim(filtered_candidates)
        self.assertEqual(victim, "adapter_3")  # Should select next LRU

        # Test removal
        lru.remove("adapter_1")
        victim = lru.select_victim(candidates)
        self.assertEqual(victim, "adapter_3")

    def test_fifo_eviction_policy(self):
        """Test FIFO eviction policy unit functionality."""
        fifo = get_eviction_policy("fifo")
        adapters = ["adapter_1", "adapter_2", "adapter_3", "adapter_4"]

        # Mark adapters as used in order
        for adapter in adapters:
            fifo.mark_used(adapter)

        # Reuse some (should not affect FIFO order)
        fifo.mark_used("adapter_3")
        fifo.mark_used("adapter_1")

        # Test victim selection - should select adapter_1 (first inserted)
        candidates = set(adapters)
        victim = fifo.select_victim(candidates)
        self.assertEqual(victim, "adapter_1")

        # Test with pinned adapter (simulate pre-filtering in mem_pool)
        pinned = {"adapter_1"}  # Pin the first adapter
        filtered_candidates = candidates - pinned  # Pre-filter like mem_pool does
        victim = fifo.select_victim(filtered_candidates)
        self.assertEqual(victim, "adapter_2")  # Should select next FIFO

        # Test removal
        fifo.remove("adapter_1")
        victim = fifo.select_victim(candidates)
        self.assertEqual(victim, "adapter_2")

    def test_eviction_policy_factory(self):
        """Test eviction policy factory function."""
        # Test valid policies
        lru = get_eviction_policy("lru")
        fifo = get_eviction_policy("fifo")

        self.assertIsNotNone(lru)
        self.assertIsNotNone(fifo)

        # Test invalid policy
        with self.assertRaises(ValueError):
            get_eviction_policy("invalid_policy")

    def test_lru_vs_fifo_behavior(self):
        """Test that LRU and FIFO behave differently."""
        lru = get_eviction_policy("lru")
        fifo = get_eviction_policy("fifo")

        adapters = ["adapter_1", "adapter_2", "adapter_3"]

        # Both policies: mark all adapters as used
        for adapter in adapters:
            lru.mark_used(adapter)
            fifo.mark_used(adapter)

        # LRU: reuse first adapter (should move it to end)
        lru.mark_used("adapter_1")
        # FIFO: reuse first adapter (should not change order)
        fifo.mark_used("adapter_1")

        candidates = set(adapters)

        # LRU should select adapter_2 (now least recently used)
        lru_victim = lru.select_victim(candidates)
        # FIFO should still select adapter_1 (first inserted)
        fifo_victim = fifo.select_victim(candidates)

        # They should select different victims
        self.assertNotEqual(lru_victim, fifo_victim)
        self.assertEqual(lru_victim, "adapter_2")  # LRU behavior
        self.assertEqual(fifo_victim, "adapter_1")  # FIFO behavior


class TestLoRAEvictionPolicyIntegration(CustomTestCase):
    """Integration tests for LoRA eviction policies with SRTRunner."""

    BASE_MODEL = "/workspace/models/llama3-1-8-b"
    ADAPTER_PATH = "/workspace/adapters/llama_3_1_8B_adapter"
    ADAPTER_NAMES = ["adapter_1", "adapter_2", "adapter_3"]
    PROMPT = "What is artificial intelligence?"

    def _get_runner_config(self, eviction_policy: str, max_loras: int = 2):
        """Get common SRTRunner configuration."""
        return {
            "model_path": self.BASE_MODEL,
            "torch_dtype": torch.float16,
            "model_type": "generation",
            "lora_paths": None,
            "max_loras_per_batch": max_loras,
            "lora_backend": "triton",
            "enable_lora": True,
            "max_lora_rank": 256,
            "lora_target_modules": ["all"],
            "lora_eviction_policy": eviction_policy,
        }

    def _run_eviction_test(self, eviction_policy: str, pinned: bool = False):
        """Run eviction test with specified policy and adapters."""
        with SRTRunner(**self._get_runner_config(eviction_policy)) as runner:
            # Load adapters (pin first one if pinned=True)
            for i, name in enumerate(self.ADAPTER_NAMES):
                runner.load_lora_adapter(
                    lora_name=name,
                    lora_path=self.ADAPTER_PATH,
                    pinned=(pinned and i == 0),
                )

            # Verify all adapters work after eviction
            for adapter_name in self.ADAPTER_NAMES:
                output = runner.forward(
                    [self.PROMPT], max_new_tokens=32, lora_paths=[adapter_name]
                )
                self.assertIsNotNone(output.output_strs[0])
                self.assertGreater(len(output.output_strs[0]), 0)

    def test_lru_eviction(self):
        """Test LRU eviction policy with actual inference."""
        self._run_eviction_test("lru", pinned=False)

    def test_lru_eviction_with_pinning(self):
        """Test LRU eviction policy with adapter pinning."""
        self._run_eviction_test("lru", pinned=True)

    def test_fifo_eviction(self):
        """Test FIFO eviction policy with actual inference."""
        self._run_eviction_test("fifo", pinned=False)

    def test_fifo_eviction_with_pinning(self):
        """Test FIFO eviction policy with adapter pinning."""
        self._run_eviction_test("fifo", pinned=True)

    def test_base_model_eviction_with_pinned(self):
        """Test that base model (None) can be evicted when slots are needed."""
        with SRTRunner(**self._get_runner_config("lru")) as runner:
            # Load: 1 pinned + 2 unpinned adapters (3 total, pool size 2)
            adapter_names = ["pinned", "normal", "extra"]
            for i, name in enumerate(adapter_names):
                runner.load_lora_adapter(
                    lora_name=name, lora_path=self.ADAPTER_PATH, pinned=(i == 0)
                )

            # Verify all adapters work (triggers eviction and reloading)
            for name in adapter_names:
                output = runner.forward(
                    [self.PROMPT], max_new_tokens=32, lora_paths=[name]
                )
                self.assertIsNotNone(output.output_strs[0])
                self.assertGreater(len(output.output_strs[0]), 0)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(verbosity=2)
