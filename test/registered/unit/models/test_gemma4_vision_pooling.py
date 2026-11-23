import unittest

import torch
from transformers import Gemma4VisionConfig

from sglang.srt.models.gemma4_vision import Gemma4VisionEncoder
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGemma4VisionPooling(CustomTestCase):
    def setUp(self):
        config = Gemma4VisionConfig(
            hidden_size=16,
            num_hidden_layers=0,
            patch_size=1,
            position_embedding_size=2,
            pooling_kernel_size=2,
            standardize=True,
        )
        self.encoder = Gemma4VisionEncoder(config)
        self.positions = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]])

    @torch.no_grad()
    def test_padding_does_not_change_first_image_token(self):
        """Padding positions map to the same pooling bin as the first image patch."""
        self.encoder.patch_embedder.input_proj.weight.fill_(1)
        pixels = torch.ones(1, 4, 3)
        expected, _ = self.encoder(pixels, self.positions)
        padded_pixels = torch.cat([pixels, torch.zeros_like(pixels)], dim=1)
        padded_positions = torch.cat(
            [self.positions, -torch.ones_like(self.positions)], dim=1
        )
        pooled, valid = self.encoder(padded_pixels, padded_positions)
        torch.testing.assert_close(pooled[valid], expected[0])

    @torch.no_grad()
    def test_fp16_scaling_before_standardization(self):
        """Standardization can bring a large scaled activation back into FP16 range."""
        self.encoder.half()
        self.encoder.patch_embedder.position_embedding_table.zero_()
        self.encoder.patch_embedder.input_proj.weight.fill_(8192)
        self.encoder.std_bias.fill_(32768)
        self.encoder.std_scale.fill_(0.125)
        pooled, _ = self.encoder(torch.ones(1, 4, 3), self.positions)
        # Each RGB patch projects to 3 * 8192. Scaling by sqrt(16) gives 98304.
        torch.testing.assert_close(
            pooled, torch.full((1, 1, 16), 8192, dtype=torch.float16)
        )


if __name__ == "__main__":
    unittest.main()
