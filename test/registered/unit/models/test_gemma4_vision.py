"""Unit tests for srt/models/gemma4_vision.py"""

import unittest
from unittest.mock import patch

import torch
from transformers import Gemma4VisionConfig

from sglang.srt.models import gemma4_vision
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The pooler scales by sqrt(hidden_size) = 4.
HIDDEN_SIZE = 16


class _PatchEmbedderStub(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, pixel_values, pixel_position_ids, padding_positions):
        return pixel_values


class _TransformerStub(torch.nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()

    def forward(self, inputs_embeds, attention_mask, patch_positions):
        return inputs_embeds


def _make_encoder(dtype, std_bias, std_scale):
    config = Gemma4VisionConfig(
        hidden_size=HIDDEN_SIZE, pooling_kernel_size=3, standardize=True
    )
    with (
        patch.object(gemma4_vision, "Gemma4VisionPatchEmbedder", _PatchEmbedderStub),
        patch.object(gemma4_vision, "Gemma4VisionTransformer", _TransformerStub),
    ):
        encoder = gemma4_vision.Gemma4VisionEncoder(config)
    encoder.std_bias.fill_(std_bias)
    encoder.std_scale.fill_(std_scale)
    return encoder.to(dtype)


def _make_patches(value_at, dtype, num_padding=0, padding_value=0.0):
    # 6x6 patch grid, pooled 3x3 into 4 tokens; token j = x // 3 + 2 * (y // 3).
    positions = [(x, y) for y in range(6) for x in range(6)]
    values = [value_at(x, y) for x, y in positions]
    positions += [(-1, -1)] * num_padding
    values += [padding_value] * num_padding
    hidden = torch.tensor(values, dtype=dtype)[None, :, None]
    return hidden.expand(1, -1, HIDDEN_SIZE).contiguous(), torch.tensor([positions])


def _per_token(values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)[:, None].expand(-1, HIDDEN_SIZE)


class TestGemma4VisionEncoderPooling(CustomTestCase):
    def test_padding_patches_do_not_leak_into_pooled_tokens(self):
        """Padding patches (position -1) must not change any pooled image token."""
        encoder = _make_encoder(torch.float32, std_bias=0.0, std_scale=1.0)
        hidden, positions = _make_patches(
            lambda x, y: [1.0, 2.0, 3.0, 4.0][x // 3 + 2 * (y // 3)],
            torch.float32,
            num_padding=9,
            padding_value=100.0,
        )

        pooled, pooler_mask = encoder(hidden, positions)

        torch.testing.assert_close(
            pooled[pooler_mask], _per_token([4.0, 8.0, 12.0, 16.0])
        )

    def test_float16_scaling_does_not_overflow(self):
        """Float16 image tokens stay finite when scaled pooled values exceed 65504."""
        encoder = _make_encoder(torch.float16, std_bias=60000.0, std_scale=2.0**-10)
        hidden, positions = _make_patches(
            lambda x, y: [20000.0, 18000.0, 17008.0, 16400.0][x // 3 + 2 * (y // 3)],
            torch.float16,
        )

        pooled, pooler_mask = encoder(hidden, positions)

        self.assertEqual(pooled.dtype, torch.float16)
        # (4 * v - 60000) / 1024, exact in float16.
        expected = _per_token([19.53125, 11.71875, 7.84375, 5.46875], torch.float16)
        torch.testing.assert_close(pooled[pooler_mask], expected)

    def test_bfloat16_pooling_is_the_rounded_mean(self):
        """Each bfloat16 pooled token is its patches' mean rounded once to bfloat16."""
        encoder = _make_encoder(torch.bfloat16, std_bias=0.0, std_scale=1.0)
        # Mean (8 * 1.0 + 1.09375) / 9 = 1.0104 rounds to 1.0078125 in bfloat16.
        hidden, positions = _make_patches(
            lambda x, y: 1.09375 if x % 3 == 0 and y % 3 == 0 else 1.0,
            torch.bfloat16,
        )

        pooled, pooler_mask = encoder(hidden, positions)

        expected = _per_token([4 * 1.0078125] * 4, torch.bfloat16)
        torch.testing.assert_close(pooled[pooler_mask], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
