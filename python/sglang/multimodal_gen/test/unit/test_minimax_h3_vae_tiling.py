# SPDX-License-Identifier: Apache-2.0
"""Reject tile settings that hang splitting or fail pixel/latent stitching."""

import unittest

import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    AutoencoderKL,
)
from sglang.test.test_utils import CustomTestCase


class TestMiniMaxH3VAETiling(CustomTestCase):
    @staticmethod
    def make_vae(**kwargs):
        # Exercise the real setup/split/assembly without allocating model weights.
        vae = AutoencoderKL()
        vae.vae_ratio = 16
        vae.vae_ratio_t = 4
        vae.transform = None
        vae.transform_rev = None
        vae.transform_rev_inplace = None
        vae.use_3d_conv = True
        vae.setup_forward(**kwargs)
        return vae

    def test_invalid_tiling_rejected_at_both_entry_points(self):
        for field, values in (
            ("tile_size", (0, -16, 64, 255, 256.5, True)),
            ("tile_overlap_min", (0, -16, 256, 300, 1, 63, 64.5, True)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    config = MiniMaxH3VideoVAEConfig()
                    setattr(config.arch_config, f"vae_{field}", value)
                    with self.assertRaisesRegex(
                        ValueError, f"vae_{field}|tile_overlap_min"
                    ):
                        config.post_init()
                    for prefix in ("", "decoder_"):
                        with self.subTest(prefix=prefix):
                            with self.assertRaisesRegex(
                                ValueError, field + "|tile_overlap_min"
                            ):
                                self.make_vae(**{prefix + field: value})

    def test_valid_tiles_reconstruct_pixel_and_latent_canvases(self):
        for kwargs in (
            {},
            {"tile_size": 64, "tile_overlap_min": 16},
            {"decoder_tile_size": 128, "decoder_tile_overlap_min": 32},
        ):
            vae = self.make_vae(**kwargs)
            for is_decoder in (False, True):
                for length in (128, 512, 528):
                    with self.subTest(
                        kwargs=kwargs, is_decoder=is_decoder, length=length
                    ):
                        starts, lengths, overlaps = vae.split_tiles(length, is_decoder)
                        scale = 1 if is_decoder else vae.vae_ratio
                        canvas = torch.arange(
                            (length // scale) ** 2, dtype=torch.float32
                        )
                        canvas = canvas.reshape(
                            1, 1, 1, length // scale, length // scale
                        )
                        rows = [
                            [
                                canvas[
                                    ...,
                                    y // scale : (y + h) // scale,
                                    x // scale : (x + w) // scale,
                                ].clone()
                                for x, w in zip(starts, lengths)
                            ]
                            for y, h in zip(starts, lengths)
                        ]
                        overlap = [value // scale for value in overlaps]
                        actual = vae._assemble_tiles(rows, overlap, overlap)
                        torch.testing.assert_close(actual, canvas)


if __name__ == "__main__":
    unittest.main()
