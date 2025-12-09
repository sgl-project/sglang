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
Tests LRU and FIFO eviction behavior.
"""

import unittest

from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=200, suite="nightly-1-gpu", nightly=True)


class TestLoRAEvictionPolicy(unittest.TestCase):
    """Unit tests for LoRA eviction policies."""

    def _test_eviction_policy(
        self, policy_name, access_sequence, candidates, expected_victim
    ):
        """
        Helper to test eviction policy with given access pattern.

        Args:
            policy_name: Name of eviction policy ("lru" or "fifo")
            access_sequence: List of adapter IDs in access order
            candidates: Set of adapter IDs that can be evicted
            expected_victim: Expected adapter ID to be evicted
        """
        policy = get_eviction_policy(policy_name)

        # Simulate access pattern
        for adapter_id in access_sequence:
            policy.mark_used(adapter_id)

        # Select victim from candidates
        victim = policy.select_victim(candidates)
        self.assertEqual(
            victim,
            expected_victim,
            f"{policy_name.upper()}: Expected {expected_victim}, got {victim}",
        )

    def test_lru_basic(self):
        """Test LRU selects least recently used adapter."""
        self._test_eviction_policy(
            "lru",
            access_sequence=["lora1", "lora2", "lora3", "lora4"],
            candidates={"lora1", "lora2", "lora3", "lora4"},
            expected_victim="lora1",
        )

    def test_lru_with_reuse(self):
        """Test LRU updates order on reuse."""
        self._test_eviction_policy(
            "lru",
            access_sequence=["lora1", "lora2", "lora3", "lora4", "lora1"],
            candidates={"lora1", "lora2", "lora3", "lora4"},
            expected_victim="lora2",
        )

    def test_lru_multiple_reuse(self):
        """Test LRU with multiple reuses."""
        self._test_eviction_policy(
            "lru",
            access_sequence=["lora1", "lora2", "lora3", "lora1", "lora2"],
            candidates={"lora1", "lora2", "lora3"},
            expected_victim="lora3",
        )

    def test_lru_with_subset_candidates(self):
        """Test LRU with subset of candidates."""
        self._test_eviction_policy(
            "lru",
            access_sequence=["lora1", "lora2", "lora3", "lora4"],
            candidates={"lora2", "lora3", "lora4"},
            expected_victim="lora2",
        )

    def test_lru_base_model_priority(self):
        """Test LRU prioritizes base model for eviction."""
        self._test_eviction_policy(
            "lru",
            access_sequence=["lora1", "lora2", "lora3"],
            candidates={None, "lora1", "lora2", "lora3"},
            expected_victim=None,
        )

    def test_fifo_basic(self):
        """Test FIFO selects first inserted adapter."""
        self._test_eviction_policy(
            "fifo",
            access_sequence=["lora1", "lora2", "lora3", "lora4"],
            candidates={"lora1", "lora2", "lora3", "lora4"},
            expected_victim="lora1",
        )

    def test_fifo_ignores_reuse(self):
        """Test FIFO ignores reuse."""
        self._test_eviction_policy(
            "fifo",
            access_sequence=[
                "lora1",
                "lora2",
                "lora3",
                "lora4",
                "lora4",
                "lora3",
                "lora2",
                "lora1",
            ],
            candidates={"lora1", "lora2", "lora3", "lora4"},
            expected_victim="lora1",
        )

    def test_fifo_with_subset_candidates(self):
        """Test FIFO with subset of candidates."""
        self._test_eviction_policy(
            "fifo",
            access_sequence=["lora1", "lora2", "lora3", "lora4"],
            candidates={"lora2", "lora3", "lora4"},
            expected_victim="lora2",
        )

    def test_fifo_base_model_priority(self):
        """Test FIFO prioritizes base model for eviction."""
        self._test_eviction_policy(
            "fifo",
            access_sequence=["lora1", "lora2", "lora3"],
            candidates={None, "lora1", "lora2", "lora3"},
            expected_victim=None,
        )

    def test_policy_remove(self):
        """Test that remove() correctly updates internal state."""
        lru = get_eviction_policy("lru")
        lru.mark_used("lora1")
        lru.mark_used("lora2")
        lru.mark_used("lora3")

        # Remove lora1, so lora2 becomes LRU
        lru.remove("lora1")
        victim = lru.select_victim({"lora1", "lora2", "lora3"})
        self.assertEqual(victim, "lora2")

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
        access_sequence = ["lora1", "lora2", "lora3", "lora1"]
        candidates = {"lora1", "lora2", "lora3"}

        lru = get_eviction_policy("lru")
        for adapter_id in access_sequence:
            lru.mark_used(adapter_id)
        lru_victim = lru.select_victim(candidates)

        fifo = get_eviction_policy("fifo")
        for adapter_id in access_sequence:
            fifo.mark_used(adapter_id)
        fifo_victim = fifo.select_victim(candidates)

        self.assertNotEqual(lru_victim, fifo_victim)
        self.assertEqual(lru_victim, "lora2")
        self.assertEqual(fifo_victim, "lora1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
