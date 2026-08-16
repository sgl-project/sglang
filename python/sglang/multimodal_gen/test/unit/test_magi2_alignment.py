# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.configs.models.encoders.magi2 import (
    Magi2TextEncoderArchConfig,
)
from sglang.multimodal_gen.configs.sample.magi2 import (
    MAGI2_NEGATIVE_PROMPT,
    Magi2SamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    build_layout,
    build_timesteps,
    ref_images,
)


class TestPerTokenTimesteps(unittest.TestCase):
    def _layout(self, ref_counts=()):
        return build_layout(
            video_latent_thw=(2, 4, 6),
            audio_tokens=16,
            text_tokens=8,
            device=torch.device("cpu"),
            ref_patch_counts=ref_counts,
        )

    def test_only_noised_modalities_carry_the_step(self):
        layout = self._layout(ref_counts=[15])
        t = build_timesteps(
            layout=layout,
            video_t=torch.tensor(500.0),
            audio_t=torch.tensor(500.0),
        )
        self.assertTrue(bool((t[layout.video_index] == 500.0).all()))
        self.assertTrue(bool((t[layout.audio_index] == 500.0).all()))
        # Text and reference images are clean conditioning: a non-zero step here
        # presents them to the DiT as partially noised.
        self.assertTrue(bool((t[layout.text_index] == 0.0).all()))
        self.assertTrue(bool((t[layout.ref_special_index] == 0.0).all()))
        self.assertTrue(bool((t[layout.ref_patch_index] == 0.0).all()))


class TestRefImageSegments(unittest.TestCase):
    def test_special_token_routes_as_text_and_patches_as_video(self):
        layout = build_layout(
            video_latent_thw=(2, 4, 6),
            audio_tokens=16,
            text_tokens=8,
            device=torch.device("cpu"),
            ref_patch_counts=[15],
        )
        special = int(layout.ref_special_index[0])
        self.assertEqual(int(layout.modality_ids[special]), 2)
        self.assertTrue(bool((layout.modality_ids[layout.ref_patch_index] == 0).all()))
        # The pooled prompt embedding immediately precedes its own patches.
        self.assertEqual(special + 1, int(layout.ref_patch_index[0]))


class TestFigurePhrase(unittest.TestCase):
    def test_plain_prompt_gains_the_phrase(self):
        out = ref_images.ensure_figure_phrase("a lake at dawn")
        self.assertIn(ref_images.FIGURE_ONE, out)


class TestResizePad(unittest.TestCase):
    def test_letterboxes_without_cropping(self):
        from PIL import Image

        source = Image.new("RGB", (200, 100), (10, 20, 30))
        out = ref_images.resize_pad(source, height=128, width=128)
        self.assertEqual(out.size, (128, 128))
        # A crop would fill the canvas; letterboxing leaves white bars.
        corner = out.getpixel((0, 0))
        self.assertEqual(corner, (255, 255, 255))

    def test_reference_asset_reaches_the_reference_latent_grid(self):
        from PIL import Image

        # The reference run logs `images: (1, 1, 48, 1, 30, 56)` for this asset
        # at the 512x896 preview tier, so the sizing chain has to land on 30x56
        # latent patches. Flooring to the compression ratio gives 31x56.
        source = Image.new("RGB", (2710, 1510))
        height, width = ref_images.target_size(
            source, generation_height=512, generation_width=896
        )
        self.assertEqual((height, width), (499, 896))

        height, width = ref_images.aligned_size(height, width, align=32)
        self.assertEqual((height, width), (480, 896))
        self.assertEqual((height // 16, width // 16), (30, 56))


class TestShippedDefaults(unittest.TestCase):
    def test_negative_prompt_covers_video_audio_and_voice(self):
        params = Magi2SamplingParams(prompt="x")
        self.assertEqual(params.negative_prompt, MAGI2_NEGATIVE_PROMPT)
        # Audio guidance runs at the highest scale in the model, so its negatives
        # have to be present.
        for marker in ("JPEG compression residue", "hiss", "monotone"):
            self.assertIn(marker, params.negative_prompt)

    def test_skimmed_guidance_is_off_by_default(self):
        # use_skimmed_cfg_linear is false in the shipped config.
        self.assertFalse(Magi2SamplingParams(prompt="x").use_skimmed_guidance)

    def test_text_conditioning_skips_two_layers(self):
        self.assertEqual(Magi2TextEncoderArchConfig().skip_layer, 2)


if __name__ == "__main__":
    unittest.main()
