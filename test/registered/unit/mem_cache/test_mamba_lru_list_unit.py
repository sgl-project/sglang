import unittest
from array import array

import torch

from sglang.srt.mem_cache.mamba_radix_cache import LRUList, TreeNode
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMambaRadixCacheLRUList(CustomTestCase):
    """Test two-tier LRU list behavior for full KV cache vs Mamba states."""

    def test_lru_list_mamba_vs_full_isolation(self):
        full_lru = LRUList(mamba=False)
        mamba_lru = LRUList(mamba=True)

        node1 = TreeNode(id=1)
        node1.value = torch.tensor([0, 1])
        node1.mamba_value = torch.tensor([0])

        node2 = TreeNode(id=2)
        node2.value = torch.tensor([2, 3])
        node2.mamba_value = torch.tensor([1])

        full_lru.insert_mru(node1)
        full_lru.insert_mru(node2)

        mamba_lru.insert_mru(node1)
        mamba_lru.insert_mru(node2)

        # In both, node2 is MRU, node1 is LRU
        self.assertEqual(full_lru._get_lru().id, 1)
        self.assertEqual(mamba_lru._get_lru().id, 1)

        # Reset node1 in Mamba LRU only
        mamba_lru.reset_node_mru(node1)

        # Now in Mamba LRU node2 is LRU, but in Full LRU node1 remains LRU
        self.assertEqual(full_lru._get_lru().id, 1)
        self.assertEqual(mamba_lru._get_lru().id, 2)


class TestMambaRadixCacheTreeOperations(CustomTestCase):
    """Tests the CPU-level primitives of MambaRadixCache: TreeNode state flags and
    the two-tier (full vs mamba) LRU list ordering/eviction."""

    def test_match_post_processor_device_and_host(self):
        """Verify node state separation for GPU (L1) hits vs CPU host (L2) hits."""
        root = TreeNode(id=0)
        root.key = RadixKey(array("q"), None)
        root.value = []

        # Node A: Tokens [1, 2, 3] on GPU VRAM
        node_a = TreeNode(id=1)
        node_a.parent = root
        node_a.key = RadixKey(array("q", [1, 2, 3]))
        node_a.value = torch.tensor([10, 11, 12], dtype=torch.int64)
        node_a.mamba_value = torch.tensor([0], dtype=torch.int64)
        root.children[node_a.key] = node_a

        # Node B: Tokens [4, 5, 6] evicted from GPU to CPU Host RAM
        node_b = TreeNode(id=2)
        node_b.parent = node_a
        node_b.key = RadixKey(array("q", [4, 5, 6]))
        node_b.value = None  # Evicted from GPU
        node_b.mamba_value = None  # Evicted from GPU
        node_b.host_value = torch.tensor([104, 105, 106], dtype=torch.int64)
        node_b.mamba_host_value = torch.tensor([5], dtype=torch.int64)
        node_a.children[node_b.key] = node_b

        # Node A is on GPU
        self.assertFalse(node_a.evicted)
        self.assertFalse(node_a.mamba_evicted)

        # Node B is on CPU host (backuped) but evicted from GPU
        self.assertTrue(node_b.evicted)
        self.assertTrue(node_b.mamba_evicted)
        self.assertTrue(node_b.backuped)
        self.assertTrue(node_b.mamba_backuped)

    def test_evict_mamba_host_lru_order(self):
        """Verify host Mamba LRU eviction order and state transition."""
        mamba_lru = LRUList(mamba=True)

        node_a = TreeNode(id=101)
        node_a.mamba_value = torch.tensor([1], dtype=torch.int64)
        node_a.mamba_host_value = torch.tensor([10], dtype=torch.int64)

        node_b = TreeNode(id=102)
        node_b.mamba_value = torch.tensor([2], dtype=torch.int64)
        node_b.mamba_host_value = torch.tensor([20], dtype=torch.int64)

        node_c = TreeNode(id=103)
        node_c.mamba_value = torch.tensor([3], dtype=torch.int64)
        node_c.mamba_host_value = torch.tensor([30], dtype=torch.int64)

        # Insert into LRU list: node_a -> node_b -> node_c (node_c is MRU, node_a is LRU)
        mamba_lru.insert_mru(node_a)
        mamba_lru.insert_mru(node_b)
        mamba_lru.insert_mru(node_c)

        # Oldest LRU node is node_a
        lru_candidate = mamba_lru._get_lru()
        self.assertEqual(lru_candidate.id, 101)

        # Evict node_a
        mamba_lru._remove_node(lru_candidate)
        del mamba_lru.cache[lru_candidate.id]
        lru_candidate.mamba_value = None
        lru_candidate.mamba_host_value = None

        self.assertTrue(lru_candidate.mamba_evicted)
        self.assertFalse(lru_candidate.mamba_backuped)
        self.assertTrue(node_b.mamba_backuped)
        self.assertTrue(node_c.mamba_backuped)

        # Next LRU candidate is now node_b
        next_lru = mamba_lru._get_lru()
        self.assertEqual(next_lru.id, 102)


if __name__ == "__main__":
    unittest.main()
