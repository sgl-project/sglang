import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dspark.dspark_attn_metadata import (
    build_dspark_swa_page_indices,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4DraftSWASidecar(unittest.TestCase):
    def test_scratch_rows_are_outside_committed_sidecar(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.draft_swa_scratch_width = 5
        pool.draft_swa_committed_size = 1024
        pool.draft_swa_scratch_base = 1280

        locs = pool.get_draft_swa_scratch_locs(
            torch.tensor([0, 3], dtype=torch.int64), block_size=5
        )

        self.assertEqual(
            locs.tolist(),
            [1280, 1281, 1282, 1283, 1284, 1295, 1296, 1297, 1298, 1299],
        )
        self.assertTrue(torch.all(locs > pool.draft_swa_committed_size))
        with self.assertRaisesRegex(ValueError, "exceeds scratch width"):
            pool.get_draft_swa_scratch_locs(
                torch.tensor([0], dtype=torch.int64), block_size=6
            )

    def test_pd_buffer_range_excludes_scratch_pages(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.draft_swa_scratch_width = 5
        pool.draft_swa_committed_size = 1024
        pool.swa_page_size = 256
        pool.swa_kv_pool = SimpleNamespace(
            kv_buffer=[torch.empty((10, 8), dtype=torch.uint8)]
        )
        pool.compress_state_pools = []
        pool.indexer_compress_state_pools = []

        _, data_lens, item_lens = pool.get_state_buf_infos()

        self.assertEqual(item_lens, [8])
        self.assertEqual(data_lens, [5 * 8])
        self.assertLess(data_lens[0], pool.swa_kv_pool.kv_buffer[0].nbytes)

    def test_page_indices_read_direct_scratch_rows(self):
        req_to_token = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
        full_to_swa = torch.arange(64, dtype=torch.int64) + 20

        indices, lengths = build_dspark_swa_page_indices(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa,
            req_pool_indices_per_request=torch.tensor([0]),
            offsets=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
            invalid=torch.zeros((1, 4), dtype=torch.bool),
            out_loc=torch.tensor([50, 51], dtype=torch.int64),
            context_lens=torch.tensor([3], dtype=torch.int32),
            block_size=2,
            swa_window=4,
            page_index_aligned_size=8,
            block_swa_locs=torch.tensor([100, 101], dtype=torch.int64),
        )

        self.assertEqual(indices.shape, (2, 8))
        self.assertEqual(indices[0, :5].tolist(), [31, 32, 33, 100, 101])
        self.assertTrue(torch.equal(indices[0], indices[1]))
        self.assertEqual(lengths.tolist(), [5, 5])


if __name__ == "__main__":
    unittest.main()
