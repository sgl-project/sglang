"""The Wan VAE attention block over a row range equals those rows of the
full-frame result (what spatial-parallel decode keeps on each rank)."""

import unittest

import torch

from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    WanAttentionBlock,
    attention_block_forward,
)


class TestWanVaeAttentionRows(unittest.TestCase):
    def test_row_range_matches_full_frame_rows(self):
        torch.manual_seed(0)
        block = WanAttentionBlock(16).eval()
        x = torch.randn(1, 16, 3, 7, 5)
        with torch.no_grad():
            full = attention_block_forward(block, x)
            for first, count in ((0, 2), (2, 3), (5, 2)):
                with self.subTest(rows=(first, count)):
                    part = attention_block_forward(block, x, rows=(first, count))
                    torch.testing.assert_close(
                        part,
                        full[..., first : first + count, :],
                        rtol=1e-5,
                        atol=1e-5,
                    )


if __name__ == "__main__":
    unittest.main()
