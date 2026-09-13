"""Unit tests for sglang.srt.layers.quantization.quark.utils — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.quark.utils import (
    e8m0_to_f32,
    should_ignore_layer,
)
from sglang.test.test_utils import CustomTestCase


class TestShouldIgnoreLayer(CustomTestCase):
    FUSED_MAPPING = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    PREFIX = "visual.blocks.0.attn"

    def test_fused_checkpoint_exclusion_is_preserved(self):
        # GLM-5.3 exports BF16 vision QKV fused in the checkpoint. Expanding
        # its explicit exclusion into absent q/k/v names would allocate FP4.
        for exclusions in (
            [f"{self.PREFIX}.qkv_proj"],
            [r"re:visual\.blocks\.0\.attn\.qkv_proj$"],
        ):
            with self.subTest(exclusions=exclusions):
                self.assertTrue(
                    should_ignore_layer(
                        f"{self.PREFIX}.qkv_proj", exclusions, self.FUSED_MAPPING
                    )
                )

    def test_unfused_checkpoint_exclusions_are_preserved(self):
        exclusions = [f"{self.PREFIX}.{part}_proj" for part in ("q", "k", "v")]
        for targets in (exclusions, iter(exclusions)):
            self.assertTrue(
                should_ignore_layer(
                    f"{self.PREFIX}.qkv_proj", targets, self.FUSED_MAPPING
                )
            )

    def test_partial_shard_exclusion_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "different quantization schemes"):
            should_ignore_layer(
                f"{self.PREFIX}.qkv_proj",
                [f"{self.PREFIX}.q_proj"],
                self.FUSED_MAPPING,
            )

    def test_unexcluded_fused_layer_stays_quantized(self):
        self.assertFalse(
            should_ignore_layer(
                f"{self.PREFIX}.qkv_proj", ["other.qkv_proj"], self.FUSED_MAPPING
            )
        )

    def test_minimax_dsa_index_qkv_ignored(self):
        # MiniMax-M3 packs only q/k because its DSA value projection is absent.
        layer = "language_model.model.layers.3.self_attn.index_qkv_proj"
        exclusions = (
            "language_model.model.layers.3.self_attn.index_q_proj",
            "language_model.model.layers.3.self_attn.index_k_proj",
        )
        mapping = {"index_qkv_proj": ["index_q_proj", "index_k_proj"]}
        self.assertTrue(should_ignore_layer(layer, exclusions, mapping))


class TestE8M0ToF32(CustomTestCase):
    """Cover OCP MX-format v1.0 e8m0 decoding:
    encoded 0..254 -> 2^(x-127); encoded 255 -> NaN.
    """

    def test_scale_128_is_not_nan(self):
        x = torch.tensor([134], dtype=torch.uint8)
        out = e8m0_to_f32(x)
        self.assertEqual(out.item(), 128.0)
        self.assertFalse(torch.isnan(out).any().item())

    def test_nan_sentinel(self):
        x = torch.tensor([255], dtype=torch.uint8)
        self.assertTrue(torch.isnan(e8m0_to_f32(x)).all().item())

    def test_known_powers_of_two(self):
        x = torch.tensor([0, 125, 126, 127, 128, 129, 134, 254], dtype=torch.uint8)
        expected = torch.tensor(
            [2.0**-127, 0.25, 0.5, 1.0, 2.0, 4.0, 128.0, 2.0**127],
            dtype=torch.float32,
        )
        torch.testing.assert_close(e8m0_to_f32(x), expected)

    def test_only_255_is_nan(self):
        # Exactly one of 0..255 should be NaN, and it must be index 255.
        x = torch.arange(256).to(torch.uint8)
        out = e8m0_to_f32(x)
        nan_idx = torch.isnan(out).nonzero().flatten().tolist()
        self.assertEqual(nan_idx, [255])

    def test_shape_preserved(self):
        x = torch.zeros((3, 4, 5), dtype=torch.uint8)
        self.assertEqual(tuple(e8m0_to_f32(x).shape), (3, 4, 5))

    @unittest.skipUnless(torch.cuda.is_available(), "no GPU")
    def test_cuda_parity(self):
        x = torch.tensor([127, 134, 255], dtype=torch.uint8, device="cuda")
        out = e8m0_to_f32(x)
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out[0].item(), 1.0)
        self.assertEqual(out[1].item(), 128.0)
        self.assertTrue(torch.isnan(out[2]).item())


if __name__ == "__main__":
    unittest.main()
