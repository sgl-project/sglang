"""Grid sizing for the EAGLE verify / draft-extend KV-index launches.

The token-block count is derived from the batch's mean KV length (a host int),
not the req_to_token width, and is 1 on short-context servers: a rewrite that
sizes from the table width fans a batch of short prefixes out into idle
programs, and one that drops the context gate changes short-context launches.
"""

import unittest

import torch

from sglang.srt.speculative.eagle_info import _kv_index_blocks
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

LONG_CONTEXT_TABLE = 262_144 + 8


class TestEagleKvIndexBlocks(CustomTestCase):
    def test_short_context_server_keeps_historical_grid(self):
        for table_width in (4096, 32_767):
            self.assertEqual(
                _kv_index_blocks(
                    table_width=table_width,
                    paged_kernel_lens_sum=8 * table_width,
                    batch_size=8,
                ),
                1,
            )

    def test_blocks_follow_mean_length_not_table_width(self):
        cases = [
            # (batch_size, mean_len, expected): ceil(mean_len / 8192), capped
            # by 512 // batch_size; the table width alone would give 33.
            (4, 169_336, 21),
            (8, 32_768, 4),
            (2, 4_096, 1),
            (1, 8_193, 2),
            (256, 262_144, 2),
            (600, 262_144, 1),
        ]
        for batch_size, mean_len, expected in cases:
            self.assertEqual(
                _kv_index_blocks(
                    table_width=LONG_CONTEXT_TABLE,
                    paged_kernel_lens_sum=batch_size * mean_len,
                    batch_size=batch_size,
                ),
                expected,
                (batch_size, mean_len),
            )

    def test_device_length_sum_falls_back_to_table_width(self):
        # Draft extend may only have cum_kv_seq_len[-1]; reading it would sync.
        self.assertEqual(
            _kv_index_blocks(
                table_width=LONG_CONTEXT_TABLE,
                paged_kernel_lens_sum=torch.tensor(4 * 4096, dtype=torch.int32),
                batch_size=4,
            ),
            33,
        )

    def test_empty_batch(self):
        self.assertEqual(
            _kv_index_blocks(
                table_width=LONG_CONTEXT_TABLE, paged_kernel_lens_sum=0, batch_size=0
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
