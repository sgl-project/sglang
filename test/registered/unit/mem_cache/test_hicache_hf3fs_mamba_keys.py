"""MLA TP ranks share one HF3FS file and metadata, but each backs up its own Mamba shard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.test.test_utils import CustomTestCase

PAGE_BYTES = 16
NUM_PAGES = 4


class _StatePool:
    page_size = 1

    def __init__(self):
        self.buffer = torch.zeros((NUM_PAGES, PAGE_BYTES), dtype=torch.uint8)

    def get_ksize_per_token(self):
        return PAGE_BYTES

    def get_dummy_flat_data_page(self):
        return torch.zeros(PAGE_BYTES, dtype=torch.uint8)

    def get_data_page(self, index, flat=True):
        return self.buffer[index].clone()

    def set_from_flat_data_page(self, index, data_page):
        self.buffer[index].copy_(data_page)


class TestHf3fsMlaMambaKeys(CustomTestCase):
    def test_mla_mamba_state_stays_per_rank(self):
        tmp = tempfile.mkdtemp()
        metadata = Hf3fsLocalMetadataClient()
        ranks = []
        for tp_rank in (0, 1):
            store = HiCacheHF3FS(
                rank=tp_rank,
                file_path=os.path.join(tmp, "hicache.0.bin"),
                file_size=PAGE_BYTES * 64,
                numjobs=1,
                bytes_per_page=PAGE_BYTES,
                entries=8,
                client_timeout=5,
                dtype=torch.uint8,
                metadata_client=metadata,
                is_mla_model=True,
                use_mock_client=True,
                tp_size=2,
            )
            self.addCleanup(store.close)
            pool = _StatePool()
            store.register_mem_host_pool_v2(pool, PoolName.MAMBA)
            pool.buffer.fill_(tp_rank + 1)
            ranks.append((store, pool))

        transfer = lambda: [
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=[f"page{i}" for i in range(NUM_PAGES)],
                host_indices=torch.arange(NUM_PAGES, dtype=torch.int64),
            )
        ]
        for store, _ in ranks:
            self.assertEqual(
                store.batch_set_v2(transfer())[PoolName.MAMBA], [True] * NUM_PAGES
            )
        for tp_rank, (store, pool) in enumerate(ranks):
            pool.buffer.zero_()
            self.assertEqual(
                store.batch_get_v2(transfer())[PoolName.MAMBA], [True] * NUM_PAGES
            )
            self.assertTrue(torch.all(pool.buffer == tp_rank + 1))


if __name__ == "__main__":
    unittest.main()
