import unittest

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.local_store.local_page_index import LocalPageIndex
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

KV, SWA = PoolName.KV, PoolName.SWA


def _store(index, writes):
    reservation = index.reserve(writes)
    index.publish(reservation)


class TestLocalPageIndex(CustomTestCase):
    def test_write_is_visible_only_after_publish(self):
        index = LocalPageIndex(4)
        reservation = index.reserve([(KV, ["a"])])
        self.assertEqual(index.page_exists(["a"], KV), [False])
        index.publish(reservation)
        self.assertEqual(index.page_exists(["a"], KV), [True])
        self.assertEqual(index.page_exists(["a"], SWA), [False])

    def test_eviction_keeps_prefix_head(self):
        index = LocalPageIndex(3)
        _store(index, [(KV, ["head", "mid", "tail"])])
        _store(index, [(KV, ["other"])])
        self.assertEqual(
            index.page_exists(["head", "mid", "tail"], KV), [True, True, False]
        )

    def test_lookup_pin_blocks_eviction_until_released(self):
        index = LocalPageIndex(1)
        _store(index, [(KV, ["a"])])
        index.pin_lookup("rid", ["a"])
        self.assertIsNone(index.reserve([(KV, ["b"])]))
        index.release_lookup("rid")
        self.assertIsNotNone(index.reserve([(KV, ["b"])]))

    def test_failed_reserve_rolls_back(self):
        index = LocalPageIndex(2)
        _store(index, [(KV, ["a"])])
        index.pin_lookup("rid", ["a"])
        self.assertIsNone(index.reserve([(KV, ["b"]), (SWA, ["a", "c"])]))
        self.assertEqual(list(index.entries), ["a"])
        self.assertEqual(
            (index.entries["a"].pending, index.entries["a"].pins), (set(), 1)
        )

    def test_abort_keeps_another_write_pending(self):
        index = LocalPageIndex(2)
        index.reserve([(KV, ["a"])])
        index.abort(index.reserve([(SWA, ["a"])]))
        self.assertEqual(index.entries["a"].pending, {KV})


if __name__ == "__main__":
    unittest.main()
