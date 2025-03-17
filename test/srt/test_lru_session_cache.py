import unittest

import torch

from sglang.srt.mem_cache.session_cache.lru_cache import (
    LRUSessionCache,
    LRUSessionCacheEntry,
    LRUSessionCacheStatus,
)


class TestLRUSessionCacheEntry(unittest.TestCase):
    def setUp(self):
        self.kv_indices = torch.tensor([1, 2, 3])
        self.entry = LRUSessionCacheEntry("session_1", self.kv_indices)

    def test_initialization(self):
        self.assertEqual(self.entry.sid, "session_1")
        self.assertTrue(torch.equal(self.entry.kv_indices, self.kv_indices))
        self.assertEqual(self.entry.status, LRUSessionCacheStatus.UNFINISHED)
        self.assertEqual(self.entry.lock_ref, 0)
        self.assertEqual(self.entry.length, 3)
        self.assertIsNone(self.entry.prev)
        self.assertIsNone(self.entry.next)

    def test_set_kv_indices(self):
        new_kv_indices = torch.tensor([1, 2, 3, 4, 5])
        self.entry.set_kv_indices(new_kv_indices)
        self.assertTrue(torch.equal(self.entry.kv_indices, new_kv_indices))
        self.assertEqual(self.entry.length, 5)

    def test_lock_ref(self):
        self.entry.inc_lock_ref()
        self.entry.inc_lock_ref()
        self.assertEqual(self.entry.get_lock_ref(), 2)
        self.entry.dec_lock_ref()
        self.assertEqual(self.entry.get_lock_ref(), 1)

    def test_status_methods(self):
        self.assertTrue(self.entry.is_unfinished())
        self.entry.set_status(LRUSessionCacheStatus.FINISHED)
        self.assertTrue(self.entry.is_finished())
        self.entry.set_status(LRUSessionCacheStatus.LOADING)
        self.assertTrue(self.entry.is_loading())
        self.entry.set_status(LRUSessionCacheStatus.LOADED)
        self.assertTrue(self.entry.is_loaded())
        self.entry.set_status(LRUSessionCacheStatus.WRITING)
        self.assertTrue(self.entry.is_writing())
        self.entry.set_status(LRUSessionCacheStatus.WRITTEN)
        self.assertTrue(self.entry.is_written())


class TestLRUSessionCache(unittest.TestCase):
    def setUp(self):
        self.cache = LRUSessionCache()
        self.entry1 = LRUSessionCacheEntry("session_1", torch.tensor([1, 2, 3]))
        self.entry2 = LRUSessionCacheEntry("session_2", torch.tensor([4, 5]))
        self.entry3 = LRUSessionCacheEntry("session_3", torch.tensor([6, 7]))

    def test_initialization(self):
        self.assertEqual(self.cache.head.sid, "head")
        self.assertEqual(self.cache.tail.sid, "tail")
        self.assertEqual(len(self.cache.cache), 0)

    def test_set_and_get(self):
        self.cache.set(self.entry1.sid, self.entry1.kv_indices)
        self.assertEqual(len(self.cache.cache), 1)
        retrieved_entry = self.cache.get("session_1")
        self.assertIsNotNone(retrieved_entry)
        self.assertEqual(retrieved_entry.sid, "session_1")
        self.assertTrue(
            torch.equal(retrieved_entry.kv_indices, torch.tensor([1, 2, 3]))
        )

    def test_delete(self):
        self.cache.set(self.entry1.sid, self.entry1.kv_indices)
        deleted_entry = self.cache.delete("session_1")
        self.assertIsNotNone(deleted_entry)
        self.assertEqual(deleted_entry.sid, "session_1")
        self.assertEqual(len(self.cache.cache), 0)
        self.assertIsNone(self.cache.get("session_1"))

    def test_evict(self):
        self.cache.set(self.entry1.sid, self.entry1.kv_indices)
        self.cache.set(self.entry2.sid, self.entry2.kv_indices)
        self.cache.evict()
        self.assertEqual(len(self.cache.cache), 1)
        self.assertIsNone(self.cache.get("session_1"))

    def test_evict_by_condition(self):
        self.cache.set(self.entry1.sid, self.entry1.kv_indices)
        self.cache.set(self.entry2.sid, self.entry2.kv_indices)
        self.cache.set(self.entry3.sid, self.entry3.kv_indices)

        entry = self.cache.get(self.entry1.sid)
        entry.lock_ref = 1
        evicted_entry = self.cache.evict_by_cond(lambda entry: entry.lock_ref <= 0)
        self.assertIsNotNone(evicted_entry)
        self.assertEqual(evicted_entry.sid, "session_2")
        self.assertEqual(len(self.cache.cache), 2)
        self.assertIsNone(self.cache.get("session_2"))

    def test_exist(self):
        self.cache.set(self.entry1.sid, self.entry1.kv_indices)
        self.assertTrue(self.cache.exist("session_1"))
        self.assertFalse(self.cache.exist("session_2"))

    def test_move_to_front(self):
        self.cache.set(self.entry1.sid, self.entry1.kv_indices)
        self.cache.set(self.entry2.sid, self.entry2.kv_indices)
        self.assertEqual(self.cache.head.next.sid, "session_2")
        self.cache.get("session_1")
        self.assertEqual(self.cache.head.next.sid, "session_1")

    def test_update_existing_entry(self):
        self.cache.set(self.entry1.sid, self.entry1.kv_indices)
        self.cache.set("session_1", torch.tensor([4, 5, 6]))
        retrieved_entry = self.cache.get("session_1")
        self.assertTrue(
            torch.equal(retrieved_entry.kv_indices, torch.tensor([4, 5, 6]))
        )


if __name__ == "__main__":
    unittest.main()
