# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    Magi2FourierRope,
    apply_partial_rope,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    coords as magi2_coords,
)

NUM_BANDS = 16


class TestMagi2FourierRope(unittest.TestCase):
    def test_width_is_six_bands_and_splits_into_equal_halves(self):
        rope = Magi2FourierRope(NUM_BANDS)
        embedding = rope(magi2_coords.video_coords(latent_shape=(3, 4, 5)))

        self.assertEqual(embedding.shape, (3 * 4 * 5, 6 * NUM_BANDS))
        sin, cos = embedding.tensor_split(2, -1)
        self.assertEqual(sin.shape[-1], 3 * NUM_BANDS)
        self.assertEqual(cos.shape[-1], 3 * NUM_BANDS)
        self.assertTrue(torch.allclose(sin.pow(2) + cos.pow(2), torch.ones_like(sin)))

    def test_text_coordinates_are_exactly_unrotated(self):
        rope = Magi2FourierRope(NUM_BANDS)
        with torch.no_grad():
            rope.bands.copy_(torch.rand(NUM_BANDS) + 0.5)

        embedding = rope(magi2_coords.text_coords(num_tokens=11))
        sin, cos = embedding.tensor_split(2, -1)
        self.assertTrue(torch.equal(sin, torch.zeros_like(sin)))
        self.assertTrue(torch.equal(cos, torch.ones_like(cos)))

    def test_singleton_axis_does_not_divide_by_zero(self):
        rope = Magi2FourierRope(NUM_BANDS)
        embedding = rope(magi2_coords.video_coords(latent_shape=(1, 1, 1)))
        self.assertTrue(torch.isfinite(embedding).all())


class TestApplyPartialRope(unittest.TestCase):
    def _cos_sin(self, tokens, rotary_half):
        torch.manual_seed(0)
        angle = torch.randn(tokens, rotary_half) * 2.0
        return angle.cos(), angle.sin()

    def test_channels_beyond_the_rotary_span_are_bit_identical(self):
        tokens, heads, head_dim, rotary_dim = 9, 4, 128, 6 * NUM_BANDS
        x = torch.randn(tokens, heads, head_dim)
        cos, sin = self._cos_sin(tokens, rotary_dim // 2)

        out = apply_partial_rope(x, cos, sin)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.equal(out[..., rotary_dim:], x[..., rotary_dim:]))
        self.assertFalse(torch.equal(out[..., :rotary_dim], x[..., :rotary_dim]))


if __name__ == "__main__":
    unittest.main()
