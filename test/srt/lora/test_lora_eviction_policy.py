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

import unittest
from sglang.srt.lora.eviction_policy import get_eviction_policy


class TestLoRAEvictionPolicy(unittest.TestCase):
    """Unit tests for LoRA eviction policies."""

    def test_lru_eviction_policy(self):
        """Test LRU eviction policy unit functionality."""
        print("Testing LRU eviction policy...")
        
        lru = get_eviction_policy('lru')
        adapters = ['adapter_1', 'adapter_2', 'adapter_3', 'adapter_4']
        
        # Mark adapters as used in order
        for adapter in adapters:
            lru.mark_used(adapter)
        
        # Re-use some to change LRU order
        # Order should now be: adapter_1 (oldest), adapter_3, adapter_2, adapter_4 (newest)
        lru.mark_used('adapter_2')
        lru.mark_used('adapter_4')
        
        # Test victim selection - should select adapter_1 (least recently used)
        candidates = set(adapters)
        pinned = set()  # No pinned adapters
        
        victim = lru.select_victim(candidates, pinned, {})
        print(f"LRU victim: {victim} (expected: adapter_1)")
        self.assertEqual(victim, 'adapter_1')
        
        # Test with pinned adapter
        pinned = {'adapter_1'}  # Pin the LRU adapter
        victim = lru.select_victim(candidates, pinned, {})
        print(f"LRU victim with pinning: {victim} (expected: adapter_3)")
        self.assertEqual(victim, 'adapter_3')  # Should select next LRU
        
        # Test removal
        lru.remove('adapter_1')
        victim = lru.select_victim(candidates, set(), {})
        print(f"LRU victim after removal: {victim} (expected: adapter_3)")
        self.assertEqual(victim, 'adapter_3')
        
        print("LRU eviction policy tests passed")

    def test_fifo_eviction_policy(self):
        """Test FIFO eviction policy unit functionality."""
        print("Testing FIFO eviction policy...")
        
        fifo = get_eviction_policy('fifo')
        adapters = ['adapter_1', 'adapter_2', 'adapter_3', 'adapter_4']
        
        # Mark adapters as used in order
        for adapter in adapters:
            fifo.mark_used(adapter)
        
        # Re-use some (should not affect FIFO order)
        fifo.mark_used('adapter_3')
        fifo.mark_used('adapter_1')
        
        # Test victim selection - should select adapter_1 (first inserted)
        candidates = set(adapters)
        pinned = set()  # No pinned adapters
        
        victim = fifo.select_victim(candidates, pinned, {})
        print(f"FIFO victim: {victim} (expected: adapter_1)")
        self.assertEqual(victim, 'adapter_1')
        
        # Test with pinned adapter
        pinned = {'adapter_1'}  # Pin the first adapter
        victim = fifo.select_victim(candidates, pinned, {})
        print(f"FIFO victim with pinning: {victim} (expected: adapter_2)")
        self.assertEqual(victim, 'adapter_2')  # Should select next FIFO
        
        # Test removal
        fifo.remove('adapter_1')
        victim = fifo.select_victim(candidates, set(), {})
        print(f"FIFO victim after removal: {victim} (expected: adapter_2)")
        self.assertEqual(victim, 'adapter_2')
        
        print("FIFO eviction policy tests passed")

    def test_eviction_policy_factory(self):
        """Test eviction policy factory function."""
        print("Testing eviction policy factory...")
        
        # Test valid policies
        lru = get_eviction_policy('lru')
        fifo = get_eviction_policy('fifo')
        
        self.assertIsNotNone(lru)
        self.assertIsNotNone(fifo)
        
        # Test invalid policy
        with self.assertRaises(ValueError):
            get_eviction_policy('invalid_policy')
        
        print("Eviction policy factory tests passed")

    def test_lru_vs_fifo_behavior(self):
        """Test that LRU and FIFO behave differently."""
        print("Testing LRU vs FIFO behavioral differences...")
        
        lru = get_eviction_policy('lru')
        fifo = get_eviction_policy('fifo')
        
        adapters = ['adapter_1', 'adapter_2', 'adapter_3']
        
        # Both policies: mark all adapters as used
        for adapter in adapters:
            lru.mark_used(adapter)
            fifo.mark_used(adapter)
        
        # LRU: re-use first adapter (should move it to end)
        lru.mark_used('adapter_1')
        # FIFO: re-use first adapter (should not change order)
        fifo.mark_used('adapter_1')
        
        candidates = set(adapters)
        pinned = set()
        
        # LRU should select adapter_2 (now least recently used)
        lru_victim = lru.select_victim(candidates, pinned, {})
        # FIFO should still select adapter_1 (first inserted)
        fifo_victim = fifo.select_victim(candidates, pinned, {})
        
        print(f"LRU victim: {lru_victim}, FIFO victim: {fifo_victim}")
        
        # They should select different victims
        self.assertNotEqual(lru_victim, fifo_victim)
        self.assertEqual(lru_victim, 'adapter_2')  # LRU behavior
        self.assertEqual(fifo_victim, 'adapter_1')  # FIFO behavior
        
        print("LRU vs FIFO behavioral difference tests passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)