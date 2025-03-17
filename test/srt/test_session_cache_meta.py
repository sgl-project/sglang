import unittest

from sglang.srt.mem_cache.session_cache.base_meta import (
    MemKVSessionCacheMetaManager,
    SessionCacheEntry,
    SessionCacheMeta,
)


class TestSessionCacheMeta(unittest.TestCase):
    def setUp(self):
        self.connector = "redis://ip:port"
        self.sid = "session_id"
        self.uri1 = "path1"
        self.uri2 = "path2"
        self.entry1 = SessionCacheEntry(self.connector, self.uri1, 0, 10)
        self.entry2 = SessionCacheEntry(self.connector, self.uri2, 10, 10)
        self.meta = SessionCacheMeta(self.sid, [self.entry1, self.entry2])

    def test_initialization(self):
        self.assertEqual(self.meta.sid, self.sid)
        self.assertEqual(len(self.meta.entries), 2)
        self.assertEqual(self.meta.entries[0].uri, self.uri1)
        self.assertEqual(self.meta.entries[1].uri, self.uri2)
        self.assertEqual(self.meta.entries[0].connector, self.connector)
        self.assertEqual(self.meta.entries[1].connector, self.connector)

    def test_get_next_index(self):
        self.assertEqual(self.meta._get_next_index(), 2)

    def test_get_new_uri(self):
        new_uri = self.meta._get_new_uri("prefix")
        self.assertEqual(new_uri, "prefix/2")

    def test_append(self):
        new_entry = SessionCacheEntry(self.connector, "path3", 20, 10)
        self.meta.append(new_entry)
        self.assertEqual(len(self.meta.entries), 3)
        self.assertEqual(self.meta.entries[2].uri, "path3")

    def test_get_length(self):
        length = self.meta.get_length()
        self.assertEqual(length, 20)

    def test_get_uris(self):
        uris = self.meta.get_uris()
        self.assertEqual(uris, ["path1", "path2"])

    def test_get_entries(self):
        entries = self.meta.get_entries()
        self.assertEqual(entries, [self.entry1, self.entry2])

    def test_get_next_entry_info(self):
        uri, offset, length = self.meta.get_next_entry_info("prefix", 30)
        self.assertEqual(uri, "prefix/2")
        self.assertEqual(offset, 20)
        self.assertEqual(length, 10)


class TestMemKVSessionCacheMetaManager(unittest.TestCase):
    def setUp(self):
        self.manager = MemKVSessionCacheMetaManager()
        self.connector = "redis://ip:port"
        self.sid = "session_id"
        self.uri1 = "path1"
        self.uri2 = "path2"
        self.entry1 = SessionCacheEntry(self.connector, self.uri1, 0, 10)
        self.entry2 = SessionCacheEntry(self.connector, self.uri2, 10, 10)
        self.meta = SessionCacheMeta(self.sid, [self.entry1, self.entry2])

    def test_initialization(self):
        self.assertEqual(len(self.manager.entries), 0)

    def test_reset(self):
        self.manager.save(self.sid, self.meta)
        self.manager.reset()
        self.assertEqual(len(self.manager.entries), 0)

    def test_save_and_load(self):
        self.manager.save(self.sid, self.meta)
        loaded_meta = self.manager.load(self.sid)
        self.assertEqual(loaded_meta.sid, self.meta.sid)
        self.assertEqual(len(loaded_meta.entries), 2)

    def test_exist(self):
        self.assertFalse(self.manager.exist(self.sid))
        self.manager.save(self.sid, self.meta)
        self.assertTrue(self.manager.exist(self.sid))

    def test_load_nonexistent_session(self):
        loaded_meta = self.manager.load("nonexistent_session")
        self.assertIsNone(loaded_meta)


if __name__ == "__main__":
    unittest.main()
