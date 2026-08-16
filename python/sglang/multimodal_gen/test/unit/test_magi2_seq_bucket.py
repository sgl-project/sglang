# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.layers.attention.magi2_block_grid_attention import (
    SEQ_BUCKET,
)
from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    pad_rows_to_multiple,
)

# 1080p refiner grid: 32 latent frames of 68x120.
_VIDEO_TOKENS = 32 * 68 * 120


class TestSequenceBucket(unittest.TestCase):
    def test_whole_prompt_range_collapses_to_a_handful_of_lengths(self):
        # Each distinct length costs a retained compiled graph, so what matters is
        # the count over every prompt the encoder accepts, not any one length.
        lengths = {
            total + -total % SEQ_BUCKET
            for total in (_VIDEO_TOKENS + text for text in range(7001))
        }
        self.assertLess(len(lengths), 16)

    def test_bucket_absorbs_the_shard_padding(self):
        # Only the refiner uses the block grid, and its 8 GQA key/value heads limit
        # it to these degrees. 3, 6 and 12 are preview-only and never reach here;
        # the bucket is not a multiple of those, which costs a little shard padding
        # rather than being wrong.
        for sp_size in (1, 2, 4):
            with self.subTest(sp_size=sp_size):
                self.assertEqual(SEQ_BUCKET % sp_size, 0)

    def test_padding_repeats_the_last_row(self):
        rows = torch.arange(SEQ_BUCKET + 5, dtype=torch.float32)[:, None]
        (padded,), num_pad = pad_rows_to_multiple(rows, multiple=SEQ_BUCKET)

        self.assertEqual(num_pad, SEQ_BUCKET - 5)
        self.assertEqual(padded.shape[0] % SEQ_BUCKET, 0)
        self.assertTrue(torch.equal(padded[: rows.shape[0]], rows))
        # Repeated, not zeroed: a zero coordinate row is a valid grid position.
        self.assertTrue(bool((padded[rows.shape[0] :] == rows[-1]).all()))


if __name__ == "__main__":
    unittest.main()
