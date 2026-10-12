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


class _StatePool:
    page_size = 1

    def __init__(self, value):
        self.buffer = torch.full((1, PAGE_BYTES), value, dtype=torch.uint8)

    def get_ksize_per_token(self):
        return PAGE_BYTES

    def get_dummy_flat_data_page(self):
        return torch.zeros(PAGE_BYTES, dtype=torch.uint8)

    def get_data_page(self, index, flat=True):
        return self.buffer[index].clone()

    def set_from_flat_data_page(self, index, data_page):
        self.buffer[index].copy_(data_page)


class TestHf3fsMambaKeys(CustomTestCase):
    def test_mla_mamba_round_trip_keeps_each_tp_rank(self):
        """MLA shares KV keys, but Mamba checkpoints must not overwrite another rank."""
        tmp = tempfile.mkdtemp()
        metadata = Hf3fsLocalMetadataClient()
        transfer = PoolTransfer(
            name=PoolName.MAMBA, keys=["page"], host_indices=torch.tensor([0])
        )
        ranks = []
        for rank in range(2):
            store = HiCacheHF3FS(
                rank=rank,
                file_path=os.path.join(tmp, "hicache.bin"),
                file_size=PAGE_BYTES * 16,
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
            pool = _StatePool(rank + 1)
            store.register_mem_host_pool_v2(pool, PoolName.MAMBA)
            self.assertEqual(store.batch_set_v2([transfer])[PoolName.MAMBA], [True])
            ranks.append((store, pool))

        for rank, (store, pool) in enumerate(ranks):
            pool.buffer.zero_()
            self.assertEqual(store.batch_get_v2([transfer])[PoolName.MAMBA], [True])
            self.assertTrue(torch.all(pool.buffer == rank + 1))


if __name__ == "__main__":
    unittest.main()
