import unittest

from sglang.srt.mem_cache.shared_kv_pool import SharedKVPoolControlPlane


class TestSharedKVPoolControlPlane(unittest.TestCase):
    def test_global_leases_are_exclusive_and_reusable(self):
        pool = SharedKVPoolControlPlane(num_pages=3)
        first = pool.reserve("rank-0", 2)
        self.assertIsNotNone(first)
        self.assertIsNone(pool.reserve("rank-1", 2))

        pool.release_write(first.lease_id)
        second = pool.reserve("rank-1", 3)
        self.assertEqual(second.page_ids, (0, 1, 2))

    def test_published_prefix_is_visible_across_replicas(self):
        pool = SharedKVPoolControlPlane(num_pages=4)
        write = pool.reserve("rank-0", 2)
        pool.publish_prefix((1, 2, 3), write.lease_id)

        read = pool.acquire_prefix((1, 2, 3, 4))
        self.assertIsNotNone(read)
        self.assertEqual(read.key, (1, 2, 3))
        self.assertEqual(read.page_ids, write.page_ids)
        self.assertFalse(pool.evict_prefix((1, 2, 3)))

        pool.release_read(read.lease_id)
        self.assertTrue(pool.evict_prefix((1, 2, 3)))
        self.assertEqual(pool.available_pages(), 4)

    def test_page_generation_rejects_stale_write_lease(self):
        pool = SharedKVPoolControlPlane(num_pages=1)
        write = pool.reserve("rank-0", 1)
        pool.release_write(write.lease_id)
        newer = pool.reserve("rank-1", 1)
        self.assertGreater(newer.generations[0], write.generations[0])
        with self.assertRaisesRegex(ValueError, "unknown or already released"):
            pool.publish_prefix((1,), write.lease_id)


if __name__ == "__main__":
    unittest.main()
