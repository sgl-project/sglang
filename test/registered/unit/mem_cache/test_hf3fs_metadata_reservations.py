"""CPU regressions for duplicate HF3FS writes and metadata ownership."""

import concurrent.futures
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsMockClient
from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
    RankMetadata,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRankMetadataReservations(unittest.TestCase):
    def setUp(self):
        self.meta = RankMetadata(4)

    def reserve(self, *keys):
        return self.meta.reserve_and_allocate_page_indices([(key, "") for key in keys])

    def assert_partition(self, in_flight=()):
        free = self.meta.free_pages
        mapped = list(self.meta.key_to_index.values())
        self.assertEqual(len(free), len(set(free)))
        self.assertEqual(len(mapped), len(set(mapped)))
        groups = [set(free), set(mapped), set(in_flight)]
        for i in range(len(groups)):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        self.assertEqual(set.union(*groups), set(range(self.meta.num_pages)))

    def test_both_confirmation_orders_keep_first_mapping(self):
        for order in ((0, 1), (1, 0)):
            with self.subTest(order=order):
                self.meta.clear_all()
                pages = [self.reserve("same")[0][1] for _ in range(2)]
                self.assertNotEqual(*pages)
                self.meta.confirm_write([("same", pages[order[0]])], [])
                published = self.meta.get_page_indices(["same"])
                self.meta.confirm_write([("same", pages[order[1]])], [])
                self.assertEqual(self.meta.get_page_indices(["same"]), published)
                self.assertEqual(published, [pages[order[0]]])
                self.assertIn(pages[order[1]], self.meta.free_pages)
                self.assert_partition()

    def test_duplicate_key_in_same_batch(self):
        pairs = self.reserve("same", "same", "different")
        self.assertTrue(all(not exists and page >= 0 for exists, page in pairs))
        self.meta.confirm_write(
            [(key, pair[1]) for key, pair in zip(("same", "same", "different"), pairs)],
            [],
        )
        self.assertEqual(self.meta.get_page_indices(["same"]), [pairs[0][1]])
        self.assert_partition()

    def test_abandoned_reservation_does_not_block_same_key(self):
        abandoned = self.reserve("same")[0][1]
        retry = self.reserve("same")[0]
        self.assertFalse(retry[0])
        self.assertGreaterEqual(retry[1], 0)
        self.assertNotEqual(retry[1], abandoned)
        self.meta.confirm_write([("same", retry[1])], [])
        self.assertEqual(self.meta.exists_keys(["same"]), [True])
        # Abandoned pages still leak, as on main; this patch adds no recovery.
        self.assert_partition([abandoned])

    def test_release_failed_duplicate_does_not_release_published_page(self):
        first, second = [self.reserve("same")[0][1] for _ in range(2)]
        self.meta.confirm_write([("same", first)], [second, second])
        self.assertEqual(self.meta.get_page_indices(["same"]), [first])
        self.assert_partition()

    def test_same_page_confirmation_is_idempotent(self):
        page = self.reserve("same")[0][1]
        for _ in range(2):
            self.meta.confirm_write([("same", page)], [])
        self.assertNotIn(page, self.meta.free_pages)
        self.assert_partition()

    def test_capacity_refusal_and_release_retry(self):
        pages = [pair[1] for pair in self.reserve("a", "b", "c", "d")]
        self.assertEqual(self.reserve("extra"), [(False, -1)])
        self.meta.confirm_write([], pages)
        retry = self.reserve("extra")[0][1]
        self.meta.confirm_write([("extra", retry)], [])
        self.assert_partition()

    def test_serial_existing_key_lru_delete_and_clear(self):
        self.meta = RankMetadata(2)
        a, b = [pair[1] for pair in self.reserve("a", "b")]
        self.meta.confirm_write([("a", a), ("b", b)], [])
        self.assertEqual(self.reserve("a"), [(True, a)])
        self.assertEqual(self.reserve("c"), [(False, b)])
        self.meta.confirm_write([("c", b)], [])
        self.assertEqual(self.meta.delete_keys(["a", "missing"]), 1)
        self.assert_partition()
        self.meta.clear_all()
        self.assertFalse(self.meta.key_to_index)
        self.assert_partition()

    def test_concurrent_writers_reclaim_duplicates(self):
        self.meta = RankMetadata(8)
        barrier = threading.Barrier(8)

        def write():
            exists, page = self.reserve("same")[0]
            self.assertFalse(exists)
            self.assertGreaterEqual(page, 0)
            barrier.wait(timeout=5)
            self.meta.confirm_write([("same", page)], [])
            return page

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            pages = list(executor.map(lambda _: write(), range(8)))
        self.assertEqual(len(set(pages)), 8)
        self.assertEqual(len(self.meta.key_to_index), 1)
        self.assert_partition()


class TestHF3FSClientReservations(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.metadata = Hf3fsLocalMetadataClient()
        self.metadata.initialize(0, 512)
        self.client = Hf3fsMockClient(str(Path(directory.name) / "kv"), 512 * 16, 16, 8)
        self.addCleanup(self.client.close)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.addCleanup(self.executor.shutdown)

    def store(self):
        store = HiCacheHF3FS.__new__(HiCacheHF3FS)
        store.rank = 0
        store.skip_backup = False
        store.metadata_client = self.metadata
        store.clients = [self.client]
        store.bytes_per_page = 16
        store.entries = 8
        store.executor = self.executor
        store.ac = SimpleNamespace(next=lambda: 0)
        store.enable_storage_metrics = False
        return store

    def write(self, store, keys, pooled):
        values = [torch.full((16,), 37, dtype=torch.uint8) for _ in keys]
        if not pooled:
            return store._batch_set(keys, values)
        store._pool_storage_ctx = {
            PoolName.KV: SimpleNamespace(
                namespace=PoolName.KV, bytes_per_page=16, clients=[self.client]
            )
        }
        store.registered_pools = {
            PoolName.KV: SimpleNamespace(
                page_size=1, get_data_page=lambda index, flat: values[index]
            )
        }
        store._pool_component_key = lambda key, name: key
        transfer = PoolTransfer(
            name=PoolName.KV, keys=keys, host_indices=torch.arange(len(keys))
        )
        return store._pool_batch_set(transfer)

    def test_duplicate_write_succeeds_and_later_batch_is_written(self):
        for pooled in (False, True):
            with self.subTest(pooled=pooled):
                self.metadata.clear(0)
                writer_a, writer_b = self.store(), self.store()
                original_write = self.client.batch_write
                a_reserved = threading.Event()
                resume_a = threading.Event()
                a_result = []

                def delayed_write(offsets, values):
                    if threading.current_thread().name.startswith("writer-a"):
                        a_reserved.set()
                        if not resume_a.wait(timeout=10):
                            raise TimeoutError("writer B did not finish")
                    return original_write(offsets, values)

                self.client.batch_write = delayed_write
                # Use separate executors so B can finish while A is paused.
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="writer-a"
                ) as a_executor:
                    writer_a.executor = a_executor
                    thread = threading.Thread(
                        target=lambda: a_result.extend(
                            self.write(writer_a, ["shared"], pooled)
                        )
                    )
                    thread.start()
                    try:
                        self.assertTrue(a_reserved.wait(timeout=5))
                        batch = ["prefix", "shared"] + [f"b-{i}" for i in range(126)]
                        self.assertTrue(all(self.write(writer_b, batch, pooled)))
                        published = self.metadata.get_page_indices(0, ["shared"])
                        self.assertTrue(
                            all(self.write(writer_b, ["later-suffix"], pooled))
                        )
                    finally:
                        resume_a.set()
                        thread.join(timeout=10)
                        self.client.batch_write = original_write
                self.assertFalse(thread.is_alive())
                self.assertEqual(a_result, [True])
                self.assertEqual(
                    self.metadata.get_page_indices(0, ["shared"]), published
                )
                self.assertEqual(self.metadata.exists(0, ["later-suffix"]), [True])
                actual = torch.zeros(16, dtype=torch.uint8)
                self.assertEqual(
                    self.client.batch_read([published[0] * 16], [actual]), [16]
                )
                self.assertTrue(
                    torch.equal(actual, torch.full((16,), 37, dtype=torch.uint8))
                )
                meta = self.metadata._get_metadata(0, PoolName.KV)
                self.assertEqual(len(meta.free_pages) + len(meta.key_to_index), 512)

    def test_write_failure_releases_page_and_allows_retry(self):
        for pooled in (False, True):
            with self.subTest(pooled=pooled):
                self.metadata.clear(0)
                original_write = self.client.batch_write
                self.client.batch_write = lambda offsets, values: [0] * len(offsets)
                try:
                    self.assertEqual(
                        self.write(self.store(), ["retry"], pooled), [False]
                    )
                finally:
                    self.client.batch_write = original_write
                meta = self.metadata._get_metadata(0, PoolName.KV)
                self.assertEqual(len(meta.free_pages), 512)
                self.assertEqual(self.write(self.store(), ["retry"], pooled), [True])

    def test_capacity_refusal_still_returns_false(self):
        for pooled in (False, True):
            with self.subTest(pooled=pooled):
                self.metadata.clear(0)
                self.metadata.reserve_and_allocate_page_indices(
                    0, [(f"in-flight-{i}", "") for i in range(512)]
                )
                self.assertEqual(self.write(self.store(), ["full"], pooled), [False])
                self.assertEqual(self.metadata.exists(0, ["full"]), [False])

    def test_abandoned_reservation_can_be_written_by_another_store(self):
        for pooled in (False, True):
            with self.subTest(pooled=pooled):
                self.metadata.clear(0)
                self.metadata.reserve_and_allocate_page_indices(0, [("abandoned", "")])
                self.assertEqual(
                    self.write(self.store(), ["abandoned"], pooled), [True]
                )
                self.assertEqual(self.metadata.exists(0, ["abandoned"]), [True])


if __name__ == "__main__":
    unittest.main(verbosity=2)
