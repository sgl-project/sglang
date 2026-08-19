"""Unit tests for the HF3FS HiCache storage backend -- no server, no model loading."""

import os
import shutil
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BYTES_PER_PAGE = 64
NUM_PAGES = 8


class MockHostKVCache:
    """Minimal HostKVCache stand-in with a page-first (zero-copy) layout."""

    def __init__(self, num_pages: int = NUM_PAGES, page_size: int = 1):
        self.page_size = page_size
        self.layout = "page_first"
        self.dtype = torch.uint8
        self.buffer = torch.arange(
            num_pages * BYTES_PER_PAGE, dtype=torch.int64
        ).remainder(256)
        self.buffer = self.buffer.to(torch.uint8).view(num_pages, BYTES_PER_PAGE)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        return self.buffer[int(index) // self.page_size]


class TestHiCacheHF3FSSkipBackup(CustomTestCase):
    """`_batch_set` must honour its `List[bool]` contract on every return path."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_store(self, rank: int, is_mla_model: bool = True) -> HiCacheHF3FS:
        store = HiCacheHF3FS(
            rank=rank,
            file_path=os.path.join(self.temp_dir, f"hicache.{rank}.bin"),
            file_size=NUM_PAGES * BYTES_PER_PAGE,
            numjobs=1,
            bytes_per_page=BYTES_PER_PAGE,
            entries=2,
            client_timeout=1,
            dtype=torch.uint8,
            metadata_client=Hf3fsLocalMetadataClient(),
            is_mla_model=is_mla_model,
            is_page_first_layout=True,
            use_mock_client=True,
        )
        self.stores.append(store)
        store.register_mem_pool_host(MockHostKVCache())
        return store

    def test_batch_set_returns_one_entry_per_key_on_skip_backup(self):
        store = self._make_store(rank=1)
        self.assertTrue(store.skip_backup)

        keys = ["page0", "page1", "page2"]
        results = store._batch_set(keys, values=None)

        self.assertIsInstance(results, list)
        self.assertEqual(results, [True] * len(keys))

    def test_batch_set_v1_result_is_consumable_by_all_on_skip_backup(self):
        """Mirrors HiCacheController._page_set_zero_copy, which does `all(...)`.

        A scalar return raises `TypeError: 'bool' object is not iterable` here.
        """
        store = self._make_store(rank=1)
        self.assertTrue(store.skip_backup)

        results = store.batch_set_v1(["page0", "page1"], torch.tensor([0, 1]))

        self.assertTrue(all(results))
        self.assertEqual(results, [True, True])

    def test_batch_set_v1_backup_rank_writes_and_returns_per_key_list(self):
        """Control: the rank that does back up keeps returning a per-key list."""
        store = self._make_store(rank=0)
        self.assertFalse(store.skip_backup)

        keys = ["page0", "page1"]
        results = store.batch_set_v1(keys, torch.tensor([0, 1]))

        self.assertEqual(results, [True, True])
        self.assertTrue(all(results))
        self.assertEqual(store.batch_exists(keys), len(keys))


if __name__ == "__main__":
    unittest.main()
