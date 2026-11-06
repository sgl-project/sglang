"""CPU tests for same-key HF3FS metadata reservations."""

import threading
import unittest

from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import RankMetadata
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRankMetadataReservations(unittest.TestCase):
    def setUp(self):
        self.meta = RankMetadata(4)

    def reserve(self, *keys):
        return self.meta.reserve_and_allocate_page_indices([(key, "") for key in keys])

    def assert_partition(self):
        free = self.meta.free_pages
        mapped = list(self.meta.key_to_index.values())
        pending = list(self.meta.pending_key_to_index.values())
        self.assertEqual(len(free), len(set(free)))
        self.assertEqual(len(mapped), len(set(mapped)))
        self.assertEqual(len(pending), len(set(pending)))
        self.assertEqual(set(free) & set(mapped), set())
        self.assertEqual(set(free) & set(pending), set())
        self.assertEqual(set(mapped) & set(pending), set())
        self.assertEqual(len(free) + len(mapped) + len(pending), self.meta.num_pages)

    def confirm(self, *pairs):
        self.meta.confirm_write(list(pairs), [])

    def test_duplicate_key_in_same_batch_is_skipped(self):
        first, duplicate = self.reserve("same", "same")
        self.assertEqual(first[0], False)
        self.assertGreaterEqual(first[1], 0)
        self.assertEqual(duplicate, (False, -1))
        self.confirm(("same", first[1]))
        self.assertEqual(self.meta.get_page_indices(["same"]), [first[1]])
        self.assert_partition()

    def test_cross_request_duplicate_is_skipped_until_confirm(self):
        first = self.reserve("same")[0]
        second = self.reserve("same")[0]
        self.assertEqual(second, (False, -1))
        self.assertEqual(self.meta.pending_key_to_index, {"same": first[1]})
        self.confirm(("same", first[1]))
        self.assertEqual(self.reserve("same"), [(True, first[1])])
        self.assert_partition()

    def test_both_confirmation_orders_preserve_unique_ownership(self):
        for order in (("A", "B"), ("B", "A")):
            self.meta.clear_all()
            reservations = {
                "A": self.reserve("same")[0],
                "B": self.reserve("same")[0],
            }
            self.assertGreaterEqual(reservations["A"][1], 0)
            self.assertEqual(reservations["B"], (False, -1))
            for writer in order:
                self.confirm(("same", reservations[writer][1]))
                self.assert_partition()
            self.assertEqual(
                self.meta.get_page_indices(["same"]), [reservations["A"][1]]
            )

    def test_serial_same_and_different_keys_remain_supported(self):
        first = self.reserve("same")[0]
        self.confirm(("same", first[1]))
        repeat = self.reserve("same")[0]
        self.assertEqual(repeat, (True, first[1]))
        left = self.reserve("left")[0]
        right = self.reserve("right")[0]
        self.confirm(("left", left[1]), ("right", right[1]))
        self.assertEqual(
            self.meta.get_page_indices(["same", "left", "right"]),
            [first[1], left[1], right[1]],
        )
        self.assert_partition()

    def test_release_after_failed_write_allows_retry(self):
        first = self.reserve("same")[0][1]
        self.meta.confirm_write([], [first])
        retry = self.reserve("same")[0]
        self.assertEqual(retry[0], False)
        self.assertGreaterEqual(retry[1], 0)
        self.confirm(("same", retry[1]))
        self.assertEqual(self.meta.get_page_indices(["same"]), [retry[1]])
        self.assert_partition()

    def test_duplicate_confirm_and_release_are_idempotent(self):
        page = self.reserve("same")[0][1]
        self.confirm(("same", page))
        self.confirm(("same", page))
        self.meta.confirm_write([], [page, page])
        self.assertEqual(self.meta.get_page_indices(["same"]), [page])
        self.assert_partition()

    def test_clear_and_delete_keep_partition(self):
        pending = self.reserve("pending")[0][1]
        self.meta.delete_keys(["pending"])
        self.assertEqual(self.reserve("pending"), [(False, -1)])
        self.meta.clear_all()
        self.confirm(("pending", pending))
        self.assertEqual(self.meta.get_page_indices(["pending"]), [None])
        page = self.reserve("committed")[0][1]
        self.confirm(("committed", page))
        self.meta.delete_keys(["committed"])
        self.assert_partition()
        self.assertEqual(len(self.meta.free_pages), self.meta.num_pages)

    def test_lru_eviction_reuses_only_committed_page(self):
        self.meta = RankMetadata(2)
        a = self.reserve("a")[0][1]
        b = self.reserve("b")[0][1]
        self.confirm(("a", a), ("b", b))
        self.meta.get_page_indices(["b"])
        c = self.reserve("c")[0]
        self.assertEqual(c, (False, a))
        self.assert_partition()

    def test_full_pool_of_pending_writes_does_not_share_or_raise(self):
        first = self.reserve("a", "b", "c", "d")
        self.assertTrue(all(not existed and index >= 0 for existed, index in first))
        self.assertEqual(self.reserve("e"), [(False, -1)])
        self.assert_partition()

    def test_concurrent_reservations_have_one_writer(self):
        barrier = threading.Barrier(9)
        results = []
        lock = threading.Lock()

        def reserve_once():
            barrier.wait(timeout=5)
            result = self.reserve("same")[0]
            with lock:
                results.append(result)

        threads = [threading.Thread(target=reserve_once) for _ in range(8)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

        winners = [index for existed, index in results if index >= 0]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(set(winners)), 1)
        self.assertEqual(results.count((False, -1)), 7)
        self.assert_partition()


if __name__ == "__main__":
    unittest.main(verbosity=2)
