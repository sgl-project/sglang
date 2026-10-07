"""Grid sizing for the EAGLE verify / draft-extend KV-index launches.

The token-block count is derived from the batch's mean KV length (a host int),
not the req_to_token width, and is 1 on short-context servers: a rewrite that
sizes from the table width fans a batch of short prefixes out into idle
programs, and one that drops the context gate changes short-context launches.
"""

import unittest

import torch

from sglang.srt.speculative.eagle_info import spec_kv_index_token_blocks
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

LONG_CONTEXT_TABLE = 262_144 + 8


class TestEagleKvIndexBlocks(CustomTestCase):
    def test_short_context_server_keeps_historical_grid(self):
        for table_width in (4096, 32_767):
            self.assertEqual(
                spec_kv_index_token_blocks(
                    table_width=table_width,
                    kv_lens_sum=8 * table_width,
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
                spec_kv_index_token_blocks(
                    table_width=LONG_CONTEXT_TABLE,
                    kv_lens_sum=batch_size * mean_len,
                    batch_size=batch_size,
                ),
                expected,
                (batch_size, mean_len),
            )

    def test_device_length_sum_falls_back_to_table_width(self):
        # Draft extend may only have cum_kv_seq_len[-1]; reading it would sync.
        self.assertEqual(
            spec_kv_index_token_blocks(
                table_width=LONG_CONTEXT_TABLE,
                kv_lens_sum=torch.tensor(4 * 4096, dtype=torch.int32),
                batch_size=4,
            ),
            33,
        )

    def test_empty_batch(self):
        self.assertEqual(
            spec_kv_index_token_blocks(
                table_width=LONG_CONTEXT_TABLE, kv_lens_sum=0, batch_size=0
            ),
            1,
        )

    def test_draft_launch_sizes_from_live_lengths_and_window(self):
        # Draft decode: base grid = steps * num_seqs * topk; the reviewer's
        # case (256K table, steps 3, bs 4, topk 1, 512 live tokens) must not
        # fan out into 33 blocks per base program.
        base = 3 * 4 * 1
        self.assertEqual(
            spec_kv_index_token_blocks(
                LONG_CONTEXT_TABLE, 4 * 512, 4, base_programs=base
            ),
            1,
        )
        # 4 x 169k live tokens: ceil(169k / 8192) = 21 blocks, under 512 // 12
        self.assertEqual(
            spec_kv_index_token_blocks(
                LONG_CONTEXT_TABLE, 4 * 169_336, 4, base_programs=base
            ),
            21,
        )
        # a draft window caps the copied length, so it caps the block count
        self.assertEqual(
            spec_kv_index_token_blocks(
                LONG_CONTEXT_TABLE,
                4 * 169_336,
                4,
                base_programs=base,
                length_cap=4032 + 64,
            ),
            1,
        )
        # a device-side or missing sum falls back to the table width
        self.assertEqual(
            spec_kv_index_token_blocks(LONG_CONTEXT_TABLE, None, 4, base_programs=base),
            33,
        )


if __name__ == "__main__":
    unittest.main()
