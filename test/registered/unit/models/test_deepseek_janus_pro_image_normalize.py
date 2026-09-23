"""CPU coverage for Janus-Pro CLIPVisionTower image normalization."""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.deepseek_janus_pro import (  # noqa: E402
    CLIPVisionTower,
    ImageNormalize,
    Normalize,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestJanusProImageNormalize(CustomTestCase):
    def test_per_channel_mean_std(self):
        image = torch.zeros(2, 3, 4, 4)
        image[:, 0].fill_(1.0)
        image[:, 1].fill_(0.5)
        image[:, 2].fill_(0.0)
        original = image.clone()

        out = ImageNormalize(mean=[0.5, 0.4, 0.3], std=[0.5, 0.2, 0.1])(image)

        expected = torch.empty_like(image)
        expected[:, 0].fill_((1.0 - 0.5) / 0.5)
        expected[:, 1].fill_((0.5 - 0.4) / 0.2)
        expected[:, 2].fill_((0.0 - 0.3) / 0.1)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(image, original)

    def test_vae_normalize_factory_is_group_norm(self):
        self.assertTrue(callable(Normalize))
        layer = Normalize(64)
        self.assertIsInstance(layer, nn.GroupNorm)
        self.assertEqual(layer.num_channels, 64)
        self.assertEqual(layer.num_groups, 32)

    def test_clip_vision_tower_accepts_pixel_mean_std(self):
        with patch.object(
            CLIPVisionTower,
            "build_vision_tower",
            return_value=(MagicMock(), {}),
        ):
            tower = CLIPVisionTower(
                pixel_mean=[0.0, 0.0, 0.0],
                pixel_std=[0.5, 0.5, 0.5],
            )

        self.assertIsInstance(tower.image_norm, ImageNormalize)
        image = torch.ones(1, 3, 2, 2)
        torch.testing.assert_close(
            tower.image_norm(image), torch.full((1, 3, 2, 2), 2.0)
        )


if __name__ == "__main__":
    unittest.main()
