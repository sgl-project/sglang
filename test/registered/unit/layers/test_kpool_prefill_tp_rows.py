"""Query sharding must preserve global row order across memory-budget chunks."""

import unittest

from sglang.srt.layers.attention.dsa.prefill_tp_rows import shard_row_chunks
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillTPRows(CustomTestCase):
    def test_chunks_reassemble_without_missing_or_duplicate_rows(self):
        for num_rows in (8, 17, 4097, 16384):
            for world_size in (2, 4, 8):
                for chunk_size in (3, 1024, 16384):
                    chunks = tuple(
                        slice(i, min(i + chunk_size, num_rows))
                        for i in range(0, num_rows, chunk_size)
                    )
                    all_rows = []
                    for rank in range(world_size):
                        local, padded = shard_row_chunks(chunks, world_size, rank)
                        rows = [
                            i for chunk in local for i in range(chunk.start, chunk.stop)
                        ]
                        self.assertLessEqual(len(rows), padded // world_size)
                        self.assertEqual(padded % world_size, 0)
                        all_rows.extend(rows)
                    self.assertEqual(all_rows, list(range(num_rows)))


if __name__ == "__main__":
    unittest.main()
