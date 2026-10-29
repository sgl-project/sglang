"""Unit tests for HunyuanImage-3 RoPE image-info span/shape pairing.

The section shape stream follows sequence order -- each joint image
contributes [vae, vit] dims and the tokenizer emits VAE tokens then ViT
tokens inside every joint block -- while the cond slices arrive grouped by
type. ``_build_rope_image_info`` must pair every span with its own shape
even when consecutive cond images have different sizes.
"""

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    ar_stage,
)
from sglang.test.test_utils import CustomTestCase


def _tokenizer_output(vae_slices, vit_slices, gen_slices):
    return SimpleNamespace(
        cond_vae_image_slices=[vae_slices],
        cond_vit_image_slices=[vit_slices],
        gen_image_slices=[gen_slices],
    )


def _joint_section(vae_hw, vit_hw):
    return {
        "type": "joint_image",
        "token_height": [vae_hw[0], vit_hw[0]],
        "token_width": [vae_hw[1], vit_hw[1]],
    }


class TestHunyuanImage3RoPEPairing(CustomTestCase):
    def test_multi_image_different_sizes_pair_per_image(self):
        # Two cond images with different grids: image 0 = 24x32 vae / 14x14
        # vit, image 1 = 16x48 vae / 21x21 vit.
        vae0, vit0, vae1, vit1 = (24, 32), (14, 14), (16, 48), (21, 21)
        s_vae0, s_vit0 = slice(10, 20), slice(21, 30)
        s_vae1, s_vit1 = slice(40, 55), slice(56, 60)
        tok = _tokenizer_output(
            vae_slices=[s_vae0, s_vae1],
            vit_slices=[s_vit0, s_vit1],
            gen_slices=[],
        )
        sections = [_joint_section(vae0, vit0), _joint_section(vae1, vit1)]

        info = ar_stage._build_rope_image_info(
            tokenizer_output=tok,
            batch_size=1,
            token_h=99,
            token_w=99,
            image_info=SimpleNamespace(token_height=32, token_width=32),
            sections=sections,
        )[0]

        got = {s.start: shape for s, shape in info}
        self.assertEqual(
            got,
            {
                s_vae0.start: vae0,
                s_vit0.start: vit0,
                s_vae1.start: vae1,
                s_vit1.start: vit1,
            },
        )
        # Spans must stay in sequence order.
        spans = [s for s, _ in info]
        self.assertEqual(spans, sorted(spans, key=lambda s: s.start))

    def test_single_image_pairing_unchanged(self):
        vae, vit = (24, 32), (14, 14)
        s_vae, s_vit = slice(10, 20), slice(21, 30)
        tok = _tokenizer_output(vae_slices=[s_vae], vit_slices=[s_vit], gen_slices=[])
        sections = [_joint_section(vae, vit)]

        info = ar_stage._build_rope_image_info(
            tokenizer_output=tok,
            batch_size=1,
            token_h=99,
            token_w=99,
            image_info=SimpleNamespace(token_height=32, token_width=32),
            sections=sections,
        )[0]

        self.assertEqual(info, [(s_vae, vae), (s_vit, vit)])

    def test_missing_shapes_fall_back_to_gen_dims(self):
        # A cond span whose shape is absent from the section stream falls
        # back to the gen-image dims instead of consuming the wrong shape.
        vae, vit = (24, 32), (14, 14)
        s_vae, s_vit = slice(10, 20), slice(21, 30)
        tok = _tokenizer_output(vae_slices=[s_vae], vit_slices=[s_vit], gen_slices=[])
        sections = []  # no image sections -> empty shape stream

        info = ar_stage._build_rope_image_info(
            tokenizer_output=tok,
            batch_size=1,
            token_h=99,
            token_w=99,
            image_info=SimpleNamespace(token_height=32, token_width=32),
            sections=sections,
        )[0]

        self.assertEqual(info, [(s_vae, (99, 99)), (s_vit, (99, 99))])

    def test_gen_images_take_their_own_shapes(self):
        gen_hw = (40, 24)
        s_gen = slice(70, 90)
        tok = _tokenizer_output(vae_slices=[], vit_slices=[], gen_slices=[s_gen])
        sections = [{"type": "gen_image", "token_height": 40, "token_width": 24}]

        info = ar_stage._build_rope_image_info(
            tokenizer_output=tok,
            batch_size=1,
            token_h=99,
            token_w=99,
            image_info=SimpleNamespace(token_height=32, token_width=32),
            sections=sections,
        )[0]

        self.assertEqual(info, [(s_gen, gen_hw)])


if __name__ == "__main__":
    unittest.main()
