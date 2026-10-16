"""Parallel page reads in HiCacheFile (SGLANG_HICACHE_FILE_BACKEND_READ_WORKERS).

With read_workers > 1, batch_get and batch_get_v2 read the page files of one call
on a thread pool. The results must be identical to serial reads, in the same
order; a read that fails must become a miss, not fail the batch; and every
host-pool write (set_from_flat_data_page) must stay on the calling thread,
because host pools may hold thread-local state there (calling it from a pool
thread can deadlock with a unified host pool's layout lease).

CPU-only: a minimal flat-page host pool over a CPU tensor and the real
HiCacheFile backend.
"""

import tempfile
import threading
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE, PAGE_NUMEL, PAGES = 4, 1024, 24


class _FlatPagePool:
    """The flat-page subset of a host pool that HiCacheFile uses."""

    def __init__(self, fill: bool):
        self.page_size = PAGE_SIZE
        self.buffer = (
            torch.randn(PAGES, PAGE_NUMEL) if fill else torch.zeros(PAGES, PAGE_NUMEL)
        )
        self.write_threads = set()

    def get_data_page(self, index, flat: bool = True):
        return self.buffer[index // PAGE_SIZE].clone()

    def get_dummy_flat_data_page(self):
        return torch.zeros(PAGE_NUMEL)

    def set_from_flat_data_page(self, index, data_page):
        self.write_threads.add(threading.get_ident())
        self.buffer[index // PAGE_SIZE].copy_(data_page)


def _storage(path: str) -> HiCacheFile:
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test",
        extra_config={"file_storage_path": path},
    )
    return HiCacheFile(config)


def _transfer(keys, pages):
    host_indices = torch.tensor(
        [p * PAGE_SIZE + i for p in pages for i in range(PAGE_SIZE)]
    )
    return PoolTransfer(name=PoolName.KV, keys=keys, host_indices=host_indices)


class TestHiCacheFileParallelReads(CustomTestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        writer = _storage(self.path)
        self.src = _FlatPagePool(fill=True)
        writer.register_mem_host_pool_v2(self.src, PoolName.KV)
        self.keys = [f"page-{p}" for p in range(PAGES)]
        written = writer.batch_set_v2([_transfer(self.keys, range(PAGES))])
        self.assertEqual(written[PoolName.KV], [True] * PAGES)

    def _read_v2(self, workers: int, keys):
        storage = _storage(self.path)
        storage.read_workers = workers
        dst = _FlatPagePool(fill=False)
        storage.register_mem_host_pool_v2(dst, PoolName.KV)
        pages = list(reversed(range(len(keys))))  # restore into other slots
        result = storage.batch_get_v2([_transfer(keys, pages)])[PoolName.KV]
        return result, dst, pages

    def test_parallel_equals_serial(self):
        serial, serial_dst, pages = self._read_v2(1, self.keys)
        parallel, parallel_dst, _ = self._read_v2(8, self.keys)
        self.assertEqual(serial, [True] * PAGES)
        self.assertEqual(parallel, serial)
        self.assertTrue(torch.equal(parallel_dst.buffer, serial_dst.buffer))
        for i, page in enumerate(pages):
            self.assertTrue(torch.equal(parallel_dst.buffer[page], self.src.buffer[i]))

    def test_pool_writes_stay_on_calling_thread(self):
        _, dst, _ = self._read_v2(8, self.keys)
        self.assertEqual(dst.write_threads, {threading.get_ident()})

    def test_reads_run_on_several_threads(self):
        storage = _storage(self.path)
        storage.read_workers = 4
        storage.register_mem_host_pool_v2(_FlatPagePool(fill=False), PoolName.KV)
        read_threads = set()
        barrier = threading.Barrier(2, timeout=5)
        original_get = storage.get

        def get(key, target_location=None, target_sizes=None):
            read_threads.add(threading.get_ident())
            if len(read_threads) <= 2:
                try:
                    barrier.wait()  # two reads in flight at the same time
                except threading.BrokenBarrierError:
                    pass
            return original_get(key, target_location, target_sizes)

        storage.get = get
        storage.batch_get_v2([_transfer(self.keys, range(PAGES))])
        self.assertGreater(len(read_threads), 1)
        self.assertNotIn(threading.get_ident(), read_threads)

    def test_missing_page_is_a_miss(self):
        keys = list(self.keys)
        keys[5] = "not-stored"
        result, _, _ = self._read_v2(8, keys)
        self.assertFalse(result[5])
        self.assertEqual(sum(result), PAGES - 1)

    def test_batch_get_parallel_equals_serial(self):
        for workers in (1, 8):
            storage = _storage(self.path)
            storage.read_workers = workers
            targets = [torch.zeros(PAGE_NUMEL) for _ in self.keys]
            got = storage.batch_get(self.keys, targets)
            for i, page in enumerate(got):
                self.assertIsNotNone(page)
                self.assertTrue(torch.equal(page, self.src.buffer[i]))


if __name__ == "__main__":
    unittest.main()
