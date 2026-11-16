"""Step3 family models embed sliced images and several image items per call."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers.activations import ACT2FN

from sglang.srt.configs.step3_vl import Step3VisionEncoderConfig
from sglang.srt.configs.step3p7 import Step3p7VisionEncoderConfig
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.models.step3_vl import (
    Step3VisionTransformer,
    Step3VLForConditionalGeneration,
)
from sglang.srt.models.step3_vl_10b import (
    PerceptionEncoder,
    StepVLForConditionalGeneration,
)
from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PATCH_SIZE = 14
THUMBNAIL_SIDE = 4 * PATCH_SIZE
SLICE_SIDE = 2 * PATCH_SIZE
TEXT_HIDDEN_SIZE = 24


def _init(model):
    for param in model.parameters():
        nn.init.normal_(param, std=0.02)
    return model.eval()


def _build_step3_vision_half():
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
    return _init(model)


def _build_perception_vision_half(model_cls):
    """Same for the two entry classes built on the perception encoder."""
    vision_config = Step3p7VisionEncoderConfig(
        width=32,
        layers=1,
        heads=4,
        image_size=THUMBNAIL_SIDE,
        patch_size=PATCH_SIZE,
        mlp_ratio=2.0,
    )
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.vision_model = PerceptionEncoder(
        vision_config, ACT2FN[vision_config.hidden_act], prefix="vision_model"
    )
    model.vit_large_projector = ColumnParallelLinear(
        4 * vision_config.width, TEXT_HIDDEN_SIZE, bias=False, gather_output=True
    )
    return _init(model)


def _image_item(num_slices):
    data = {"num_patches": [num_slices]}
    if num_slices:
        data["patch_pixel_values"] = torch.randn(num_slices, 3, SLICE_SIDE, SLICE_SIDE)
    return SimpleNamespace(
        feature=torch.randn(1, 3, THUMBNAIL_SIDE, THUMBNAIL_SIDE),
        model_specific_data=data,
    )


class TestStep3ImageFeature(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", mm_attention_backend="sdpa"),
            role="test",
        )
        torch.manual_seed(5)

    def test_sliced_image_yields_patch_and_thumbnail_tokens(self):
        """An image item that carries patch_pixel_values must embed; the patch
        tokens come first, followed by the thumbnail tokens."""
        model = _build_step3_vision_half()
        item = _image_item(num_slices=2)

        with torch.no_grad():
            features = model.get_image_feature([item])
            thumbnail_tokens = model._process_image_features(
                model._get_vision_model_output(item.feature)
            )
            slice_tokens = model._process_image_features(
                model._get_vision_model_output(
                    item.model_specific_data["patch_pixel_values"]
                )
            )

        expected = torch.cat(
            [slice_tokens.flatten(0, 1), thumbnail_tokens.flatten(0, 1)]
        )
        self.assertEqual(features.shape[-1], TEXT_HIDDEN_SIZE)
        torch.testing.assert_close(features, expected)

    def test_several_image_items_embed_in_one_call(self):
        """The scheduler encodes the cache misses of a whole prefill batch in one
        call, so every Step3 entry class must accept more than one image item
        and return the items' embeddings in order."""
        builders = {
            "step3": _build_step3_vision_half,
            "step3_10b": lambda: _build_perception_vision_half(
                StepVLForConditionalGeneration
            ),
            "step3p7": lambda: _build_perception_vision_half(
                Step3p7ForConditionalGeneration
            ),
        }
        tp_group = SimpleNamespace(
            rank_in_group=0, world_size=1, all_gather=lambda x, dim: x
        )
        for name, build in builders.items():
            with self.subTest(model=name), parallel_scope(tp_group=tp_group):
                model = build()
                items = [_image_item(num_slices=2), _image_item(num_slices=0)]
                with torch.no_grad():
                    together = model.get_image_feature(items)
                    one_by_one = torch.cat(
                        [model.get_image_feature([item]) for item in items]
                    )
                torch.testing.assert_close(together, one_by_one, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
