import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageLayeredPipelineConfig,
)


class TestQwenImageLayeredPipelineConfig(unittest.TestCase):
    def test_unpack_uses_layered_img_shapes_not_stale_request_size(self):
        config = QwenImageLayeredPipelineConfig()
        channels = config.dit_config.arch_config.in_channels
        generated_layers = 2
        latent_height = 40
        latent_width = 40
        latents = torch.empty(
            1,
            generated_layers * latent_height * latent_width,
            channels,
        )
        batch = SimpleNamespace(
            height=512,
            width=512,
            raw_latent_shape=latents.shape,
            img_shapes=[
                [
                    (1, latent_height, latent_width),
                    (1, latent_height, latent_width),
                    (1, latent_height, latent_width),
                ]
            ],
        )

        unpacked, batch_size, unpacked_channels, height, width = (
            config._unpad_and_unpack_latents(latents, batch)
        )

        self.assertEqual(batch_size, 1)
        self.assertEqual(unpacked_channels, channels)
        self.assertEqual((height, width), (80, 80))
        self.assertEqual(unpacked.shape, (1, channels // 4, generated_layers, 80, 80))


if __name__ == "__main__":
    unittest.main()
