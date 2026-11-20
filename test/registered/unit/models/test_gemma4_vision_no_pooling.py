import unittest

import torch
from transformers import Gemma4VisionConfig

from sglang.srt.models.gemma4_vision import Gemma4VisionEncoder
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestGemma4VisionWithoutPooling(CustomTestCase):
    def test_encoder_keeps_valid_patches(self):
        """With kernel size one, the mask must select real patches for embedding."""
        config = Gemma4VisionConfig(
            num_hidden_layers=0,
            hidden_size=4,
            patch_size=1,
            pooling_kernel_size=1,
            position_embedding_size=2,
        )
        encoder = Gemma4VisionEncoder(config)
        pixels = torch.zeros(1, 3, 3)
        positions = torch.tensor([[[0, 0], [1, 0], [-1, -1]]])

        with torch.inference_mode():
            _, valid = encoder(pixels, positions)

        self.assertEqual(valid.tolist(), [[True, True, False]])


if __name__ == "__main__":
    unittest.main()
