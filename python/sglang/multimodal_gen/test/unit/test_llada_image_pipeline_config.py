# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)


class TestLLaDAImagePipelineConfig(unittest.TestCase):
    def setUp(self):
        self.config = LLaDAImagePipelineConfig()

    def test_edit_keeps_requested_default_output_size(self):
        image = Image.new("RGB", (768, 512))

        self.assertIsNone(
            self.config.calculate_condition_image_size(image, image.width, image.height)
        )
        self.assertIsNone(self.config.prepare_calculated_size(image))

    def test_decode_preprocessing_matches_official_vae_dtype_order(self):
        class FakeVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(
                    torch.zeros((), dtype=torch.bfloat16), requires_grad=False
                )
                self.bn = SimpleNamespace(
                    running_mean=torch.tensor(
                        [0.01, -0.02, 0.03, -0.04], dtype=torch.float32
                    ),
                    running_var=torch.full((4,), 0.0129973, dtype=torch.float32),
                )
                self.config = SimpleNamespace(
                    arch_config=SimpleNamespace(batch_norm_eps=0.003)
                )

        vae = FakeVAE()
        latents = torch.tensor(
            [[[[0.501]], [[-0.249]], [[0.126]], [[-0.751]]]],
            dtype=torch.float32,
        )
        official_latents = latents.to(torch.bfloat16)
        latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(official_latents)
        latent_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.arch_config.batch_norm_eps
        ).to(official_latents)
        expected = official_latents * latent_std + latent_mean
        expected = expected.reshape(1, 1, 2, 2, 1, 1)
        expected = expected.permute(0, 1, 4, 2, 5, 3).reshape(1, 1, 2, 2)

        actual = self.config.preprocess_decoding(latents, vae=vae)

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
