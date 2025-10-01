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
        print("Testing LRU eviction policy...")

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
        pinned = set()  # No pinned adapters

        victim = lru.select_victim(candidates)
        print(f"LRU victim: {victim} (expected: adapter_1)")
        self.assertEqual(victim, "adapter_1")

        # Test with pinned adapter (simulate pre-filtering in mem_pool)
        pinned = {"adapter_1"}  # Pin the LRU adapter
        filtered_candidates = candidates - pinned  # Pre-filter like mem_pool does
        victim = lru.select_victim(filtered_candidates)
        print(f"LRU victim with pinning: {victim} (expected: adapter_3)")
        self.assertEqual(victim, "adapter_3")  # Should select next LRU

        # Test removal
        lru.remove("adapter_1")
        victim = lru.select_victim(candidates)
        print(f"LRU victim after removal: {victim} (expected: adapter_3)")
        self.assertEqual(victim, "adapter_3")

        print("LRU eviction policy tests passed")

    def test_fifo_eviction_policy(self):
        """Test FIFO eviction policy unit functionality."""
        print("Testing FIFO eviction policy...")

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
        pinned = set()  # No pinned adapters

        victim = fifo.select_victim(candidates)
        print(f"FIFO victim: {victim} (expected: adapter_1)")
        self.assertEqual(victim, "adapter_1")

        # Test with pinned adapter (simulate pre-filtering in mem_pool)
        pinned = {"adapter_1"}  # Pin the first adapter
        filtered_candidates = candidates - pinned  # Pre-filter like mem_pool does
        victim = fifo.select_victim(filtered_candidates)
        print(f"FIFO victim with pinning: {victim} (expected: adapter_2)")
        self.assertEqual(victim, "adapter_2")  # Should select next FIFO

        # Test removal
        fifo.remove("adapter_1")
        victim = fifo.select_victim(candidates)
        print(f"FIFO victim after removal: {victim} (expected: adapter_2)")
        self.assertEqual(victim, "adapter_2")

        print("FIFO eviction policy tests passed")

    def test_eviction_policy_factory(self):
        """Test eviction policy factory function."""
        print("Testing eviction policy factory...")

        # Test valid policies
        lru = get_eviction_policy("lru")
        fifo = get_eviction_policy("fifo")

        self.assertIsNotNone(lru)
        self.assertIsNotNone(fifo)

        # Test invalid policy
        with self.assertRaises(ValueError):
            get_eviction_policy("invalid_policy")

        print("Eviction policy factory tests passed")

    def test_lru_vs_fifo_behavior(self):
        """Test that LRU and FIFO behave differently."""
        print("Testing LRU vs FIFO behavioral differences...")

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
        pinned = set()

        # LRU should select adapter_2 (now least recently used)
        lru_victim = lru.select_victim(candidates)
        # FIFO should still select adapter_1 (first inserted)
        fifo_victim = fifo.select_victim(candidates)

        print(f"LRU victim: {lru_victim}, FIFO victim: {fifo_victim}")

        # They should select different victims
        self.assertNotEqual(lru_victim, fifo_victim)
        self.assertEqual(lru_victim, "adapter_2")  # LRU behavior
        self.assertEqual(fifo_victim, "adapter_1")  # FIFO behavior

        print("LRU vs FIFO behavioral difference tests passed")


class TestLoRAEvictionPolicyIntegration(CustomTestCase):
    """Integration tests for LoRA eviction policies with SRTRunner."""

    BASE_MODEL = "/workspace/models/llama3-1-8-b"
    ADAPTER_PATH = "/workspace/adapters/llama_3_1_8B_adapter"
    ADAPTER_NAMES = ["adapter_1", "adapter_2", "adapter_3"]  # 3 adapters, pool size 2
    PROMPT = "What is artificial intelligence?"

    def _run_eviction_test(self, eviction_policy: str, pinned: bool = False):
        """Run eviction test with specified policy and adapters."""
        print(f"\n{'='*60}")
        print(f"Testing {eviction_policy.upper()} eviction policy (pinned={pinned})")
        print(f"{'='*60}")

        with SRTRunner(
            self.BASE_MODEL,
            torch_dtype=torch.float16,
            model_type="generation",
            lora_paths=None,  # Don't preload, dynamically load adapters
            max_loras_per_batch=2,  # Pool size 2, will load 3 adapters
            lora_backend="triton",
            enable_lora=True,
            max_lora_rank=256,
            lora_target_modules=["all"],
            lora_eviction_policy=eviction_policy,
        ) as runner:
            # Load all 3 adapters
            # For pinned test: adapter_1 is pinned, adapter_2 will be evicted when loading adapter_3
            # For normal test: eviction based on policy (LRU/FIFO)
            for i, name in enumerate(self.ADAPTER_NAMES):
                pin_flag = pinned and i == 0
                print(f"\n  Loading adapter: {name} (pinned={pin_flag})")
                runner.load_lora_adapter(
                    lora_name=name, lora_path=self.ADAPTER_PATH, pinned=pin_flag
                )
                print(f"    Loaded successfully")

            # Test all adapters to verify they work after eviction
            print(f"\n--- Testing inference with eviction ---")
            for adapter_name in self.ADAPTER_NAMES:
                print(f"\n  Attempting inference with adapter: {adapter_name}")

                try:
                    output = runner.forward(
                        [self.PROMPT],
                        max_new_tokens=32,
                        lora_paths=[adapter_name],
                    )

                    # Verify output is generated
                    self.assertIsNotNone(output.output_strs[0])
                    self.assertGreater(len(output.output_strs[0]), 0)
                    print(f"    ✓ Success! Output: {output.output_strs[0][:50]}...")
                except Exception as e:
                    print(
                        f"    ✗ Failed with error: {type(e).__name__}: {str(e)[:200]}"
                    )
                    raise

        print(f"\n✓ {eviction_policy.upper()} test passed (pinned={pinned})")

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


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(verbosity=2)
