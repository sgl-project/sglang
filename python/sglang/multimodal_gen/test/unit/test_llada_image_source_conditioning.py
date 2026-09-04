# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_source import (
    LLaDAImageSourceImageConditioningStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class _FakeImageProcessor:
    def __init__(self):
        self.calls = []
        self.output = torch.arange(3 * 8 * 12, dtype=torch.float32).reshape(1, 3, 8, 12)

    def preprocess(self, image, height, width, resize_mode):
        self.calls.append((image, height, width, resize_mode))
        return self.output.clone()


class _FakeSigVQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(patch_size=16)
        self.pixel_values = None

    def forward(self, pixel_values):
        self.pixel_values = pixel_values
        batch_size = pixel_values.shape[0]
        features = torch.arange(
            batch_size * 3 * 5,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).reshape(batch_size, 3, 5)
        return SimpleNamespace(semantic_features=features)


class _FakePosterior:
    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.bn = SimpleNamespace(
            running_mean=torch.arange(8, dtype=torch.float32),
            running_var=torch.full((8,), 3.75, dtype=torch.float32),
        )
        self.config = SimpleNamespace(arch_config=SimpleNamespace(batch_norm_eps=0.25))
        self.encoded_image = None
        self.raw_latents = None

    def encode(self, image):
        self.encoded_image = image
        batch_size = image.shape[0]
        self.raw_latents = torch.arange(
            batch_size * 2 * 4 * 6,
            device=image.device,
            dtype=image.dtype,
        ).reshape(batch_size, 2, 4, 6)
        return _FakePosterior(self.raw_latents)


class TestLLaDAImageSourceImageConditioningStage(unittest.TestCase):
    def setUp(self):
        self.processor = _FakeImageProcessor()
        self.sigvq = _FakeSigVQ()
        self.vae = _FakeVAE()
        with patch(_GLOBAL_ARGS_PATCH, return_value=SimpleNamespace()):
            self.stage = LLaDAImageSourceImageConditioningStage(
                sigvq=self.sigvq,
                vae=self.vae,
                image_processor=self.processor,
            )

    def test_text_generation_is_a_noop(self):
        batch = SimpleNamespace(
            condition_image=None,
            image_embeds=[],
            source_latents=None,
        )

        result = self.stage.forward(batch, server_args=SimpleNamespace())

        self.assertIs(result, batch)
        self.assertEqual(result.image_embeds, [])
        self.assertIsNone(result.source_latents)
        self.assertEqual(self.processor.calls, [])

    def test_edit_matches_source_preprocessing_and_repeats_outputs(self):
        image = Image.new("RGB", (40, 30), color="red")
        batch = SimpleNamespace(
            condition_image=image,
            height=64,
            width=80,
            batch_size=2,
            image_embeds=[],
            source_latents=None,
        )

        result = self.stage.forward(batch, server_args=SimpleNamespace())

        self.assertEqual(self.processor.calls, [(image, 64, 80, "crop")])
        self.assertEqual(tuple(self.sigvq.pixel_values.shape), (2, 3, 32, 48))
        self.assertTrue(
            torch.equal(self.vae.encoded_image[0], self.processor.output[0])
        )
        self.assertTrue(
            torch.equal(self.vae.encoded_image[0], self.vae.encoded_image[1])
        )
        self.assertEqual([tuple(x.shape) for x in result.image_embeds], [(3, 5)] * 2)
        self.assertEqual(
            [tuple(x.shape) for x in result.source_latents], [(8, 1, 2, 3)] * 2
        )

        raw = self.vae.raw_latents
        expected = raw.reshape(2, 2, 2, 2, 3, 2)
        expected = expected.permute(0, 1, 3, 5, 2, 4).reshape(2, 8, 2, 3)
        expected = (expected - self.vae.bn.running_mean.view(1, -1, 1, 1)) / 2
        expected = [sample.unsqueeze(1) for sample in expected]
        for actual, wanted in zip(result.source_latents, expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def test_edit_rejects_more_than_one_source_image(self):
        batch = SimpleNamespace(
            condition_image=[Image.new("RGB", (32, 32))] * 2,
            height=64,
            width=64,
            batch_size=1,
        )

        with self.assertRaisesRegex(ValueError, "exactly one source image"):
            self.stage.forward(batch, server_args=SimpleNamespace())

    def test_source_normalization_matches_official_std_cast_order(self):
        latents = torch.tensor([[[[0.5]]]], dtype=torch.bfloat16)
        vae = SimpleNamespace(
            bn=SimpleNamespace(
                running_mean=torch.tensor([0.01], dtype=torch.float32),
                running_var=torch.tensor([0.0129973], dtype=torch.float32),
            ),
            config=SimpleNamespace(arch_config=SimpleNamespace(batch_norm_eps=0.003)),
        )
        expected_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
        expected_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.arch_config.batch_norm_eps
        ).to(latents)

        actual = self.stage._normalize_latents(latents, vae)

        torch.testing.assert_close(
            actual,
            (latents - expected_mean) / expected_std,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
