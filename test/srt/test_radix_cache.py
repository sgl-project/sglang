import unittest

import torch


class MockReqToTokenPool:
    def __init__(self):
        self.released = []

    def free(self, req_pool_idx):
        self.released.append(req_pool_idx)


class MockKVAllocator:
    def __init__(self):
        self.freed = []
        self.device = torch.device("cpu")

    def free(self, tensor):
        self.freed.append(tensor)


class RadixCacheTest(unittest.TestCase):
    def setUp(self):
        from sglang.srt.mem_cache.radix_cache import RadixCache

        self.token_pool = MockReqToTokenPool()
        self.kv_pool = MockKVAllocator()
        self.cache = RadixCache(
            req_to_token_pool=self.token_pool,
            token_to_kv_pool_allocator=self.kv_pool,
            page_size=1,
            disable=False,
            enable_kv_cache_events=False,
        )

    def test_insert_and_match(self):
        key = [1, 2, 3, 4]
        value = torch.tensor(key, dtype=torch.int64)
        self.cache.insert(key, value)

        matched, last_node = self.cache.match_prefix(key)
        self.assertTrue(torch.equal(matched, value))
        self.assertEqual(last_node.key, key)

    def test_evict_removes_evicted_node(self):
        key = [10, 20, 30]
        value = torch.tensor(key, dtype=torch.int64)
        self.cache.insert(key, value.clone())

        # Ensure evictable size reflects insertion
        self.assertEqual(self.cache.evictable_size(), len(value))

        self.cache.evict(len(value))
        # After eviction, evictable size should drop
        self.assertEqual(self.cache.evictable_size(), 0)

        # All memory should be marked freed
        self.assertTrue(any(torch.equal(t, value) for t in self.kv_pool.freed))

    def test_lock_ref_prevents_eviction(self):
        key = [100, 101, 102]
        value = torch.tensor(key, dtype=torch.int64)
        self.cache.insert(key, value)

        # Get the inserted node
        _, node = self.cache.match_prefix(key)

        self.cache.inc_lock_ref(node)
        self.cache.evict(len(value))

        # Node should not be evicted
        self.assertIsNotNone(node.value)
        self.assertEqual(self.cache.evictable_size(), 0)

        self.cache.dec_lock_ref(node)
        self.cache.evict(len(value))

        # Now it should be evicted
        self.assertIsNone(node.value)
        self.assertEqual(self.cache.evictable_size(), 0)

    def test_evict_heap_promotion_of_parent(self):
        key1 = [1, 2]
        key2 = [1, 2, 3]

        val1 = torch.tensor(key1, dtype=torch.int64)
        val2 = torch.tensor(key2, dtype=torch.int64)

        self.cache.insert(key1, val1)
        self.cache.insert(key2, val2)

        _, child_node = self.cache.match_prefix(key2)
        parent_node = child_node.parent

        # Lock the child (which also protects the parent)
        self.cache.inc_lock_ref(child_node)

        expected_size = len(val1) + 1  # [1,2] + [3]
        self.assertEqual(self.cache.protected_size(), expected_size)
        self.assertEqual(self.cache.evictable_size(), 0)

        # Unlock once to revert lock_ref back to 0
        self.cache.dec_lock_ref(child_node)

        self.assertEqual(self.cache.protected_size(), 0)
        self.assertEqual(self.cache.evictable_size(), expected_size)

        # Evict all 3 tokens ([3] and [1, 2])
        self.cache.evict(expected_size)

        # Both nodes are now evicted
        self.assertIsNone(child_node.value)
        self.assertIsNone(parent_node.value)

        # No evictable bytes remain
        self.assertEqual(self.cache.evictable_size(), 0)


if __name__ == "__main__":
    unittest.main()
