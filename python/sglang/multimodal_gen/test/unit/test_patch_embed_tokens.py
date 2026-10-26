"""PatchEmbed(tokens=True) is the token-major view of its [B, C, T, H, W] output."""

import unittest

import torch

from sglang.multimodal_gen.runtime.layers.visual_embedding import PatchEmbed


class TestPatchEmbedTokens(unittest.TestCase):
    def test_tokens_match_flattened_channel_first_output(self):
        torch.manual_seed(0)
        embed = PatchEmbed(
            patch_size=(1, 2, 2), in_chans=16, embed_dim=32, flatten=False
        )
        # divisible (patchify GEMM) and indivisible (Conv3d fallback) grids
        for shape in ((1, 16, 3, 8, 6), (2, 16, 2, 5, 6)):
            with self.subTest(shape=shape), torch.no_grad():
                x = torch.randn(shape)
                expected = embed(x).flatten(2).transpose(1, 2)
                actual = embed(x, tokens=True)
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
