"""Step3-VL embeds an image that the processor sliced into patches."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.configs.step3_vl import Step3VisionEncoderConfig
from sglang.srt.models.step3_vl import (
    Step3VisionTransformer,
    Step3VLForConditionalGeneration,
)
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PATCH_SIZE = 14
THUMBNAIL_SIDE = 4 * PATCH_SIZE
SLICE_SIDE = 2 * PATCH_SIZE
TEXT_HIDDEN_SIZE = 24


def _build_vision_half():
    """The vision tower and projector of the entry class, without the text model."""
    vision_config = Step3VisionEncoderConfig(
        hidden_size=32,
        intermediate_size=64,
        output_hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=THUMBNAIL_SIDE,
        patch_size=PATCH_SIZE,
    )
    model = Step3VLForConditionalGeneration.__new__(Step3VLForConditionalGeneration)
    nn.Module.__init__(model)
    model.vision_model = Step3VisionTransformer(vision_config)
    model.vit_downsampler = nn.Conv2d(32, 16, kernel_size=2, stride=1)
    model.vit_downsampler2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
    model.vit_large_projector = nn.Linear(32, TEXT_HIDDEN_SIZE)
    for param in model.parameters():
        nn.init.normal_(param, std=0.02)
    return model.eval()


class TestStep3VLImageFeature(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", mm_attention_backend="sdpa"),
            role="test",
        )

    def test_sliced_image_yields_patch_and_thumbnail_tokens(self):
        """An image item that carries patch_pixel_values must embed; the patch
        tokens come first, followed by the thumbnail tokens."""
        model = _build_vision_half()
        thumbnail = torch.randn(1, 3, THUMBNAIL_SIDE, THUMBNAIL_SIDE)
        slices = torch.randn(2, 3, SLICE_SIDE, SLICE_SIDE)
        item = SimpleNamespace(
            feature=thumbnail,
            model_specific_data={"num_patches": [2], "patch_pixel_values": slices},
        )

        with torch.no_grad():
            features = model.get_image_feature([item])
            thumbnail_tokens = model._process_image_features(
                model._get_vision_model_output(thumbnail)
            )
            slice_tokens = model._process_image_features(
                model._get_vision_model_output(slices)
            )

        expected = torch.cat(
            [slice_tokens.flatten(0, 1), thumbnail_tokens.flatten(0, 1)]
        )
        self.assertEqual(features.shape[-1], TEXT_HIDDEN_SIZE)
        torch.testing.assert_close(features, expected)


if __name__ == "__main__":
    unittest.main()
