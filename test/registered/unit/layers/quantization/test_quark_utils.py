"""Unit tests for sglang.srt.layers.quantization.quark.utils — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.quark.utils import (
    e8m0_to_f32,
    should_ignore_layer,
)
from sglang.test.test_utils import CustomTestCase


class TestE8M0ToF32(CustomTestCase):
    """Cover OCP MX-format v1.0 e8m0 decoding:
    encoded 0..254 -> 2^(x-127); encoded 255 -> NaN.
    """

    # ---- Bug-catchers: must FAIL on unfixed code ----------------------------

    def test_scale_128_is_not_nan(self):
        # Bug facet 1: legit scale 128.0 (x=134) was being poisoned to NaN.
        x = torch.tensor([134], dtype=torch.uint8)
        out = e8m0_to_f32(x)
        self.assertEqual(out.item(), 128.0)
        self.assertFalse(torch.isnan(out).any().item())

    def test_nan_sentinel(self):
        # Bug facet 2: x=255 is the OCP NaN sentinel; was passing through as +inf.
        x = torch.tensor([255], dtype=torch.uint8)
        self.assertTrue(torch.isnan(e8m0_to_f32(x)).all().item())

    def test_only_255_is_nan(self):
        # Exactly one of 0..255 should be NaN, and it must be index 255.
        # Build the range in the default int dtype then cast — passing the
        # uint8 dtype directly to `arange(0, 256, dtype=uint8)` raises on
        # PyTorch versions that bounds-check the end value (256 is out of
        # uint8 range).
        x = torch.arange(256).to(torch.uint8)
        out = e8m0_to_f32(x)
        nan_idx = torch.isnan(out).nonzero().flatten().tolist()
        self.assertEqual(nan_idx, [255])

    def test_known_powers_of_two(self):
        x = torch.tensor([0, 125, 126, 127, 128, 129, 134, 254], dtype=torch.uint8)
        expected = torch.tensor(
            [2.0**-127, 0.25, 0.5, 1.0, 2.0, 4.0, 128.0, 2.0**127],
            dtype=torch.float32,
        )
        torch.testing.assert_close(e8m0_to_f32(x), expected)

    # ---- Guardrails: pass on both buggy and fixed code ----------------------

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


QKV_MAPPING = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}


class TestShouldIgnoreLayerFusedNames(CustomTestCase):
    """`exclude` entries that name an ALREADY-FUSED module, or name MoE experts
    individually, must still exclude the module SGLang actually builds.

    Both shapes occur in OneNexus/GLM-5.3-Flash-MXFP4. Getting either wrong is
    not a silent accuracy loss -- the weights fail to load, because an
    MXFP4-packed parameter (2 values per byte) meets an unpacked BF16 weight:
    `param_data.shape=(128, 512), loaded_weight.shape=(128, 1024)`.
    """

    # ---- Bug-catchers: must FAIL on unfixed code ---------------------------

    def test_directly_excluded_fused_qkv_is_ignored(self):
        # The checkpoint serializes one fused qkv tensor and excludes that exact
        # name. Expanding to q/k/v_proj first finds nothing and quantizes it.
        name = "visual.blocks.0.attn.qkv_proj"
        self.assertTrue(
            should_ignore_layer(name, ignore=[name], fused_mapping=QKV_MAPPING)
        )

    def test_per_expert_excludes_ignore_the_fused_moe_module(self):
        # SGLang builds one FusedMoE named ...mlp.experts; the checkpoint
        # excludes each expert separately.
        layer = "model.layers.6.mlp.experts"
        ignore = [
            f"{layer}.{i}.{proj}"
            for i in range(3)
            for proj in ("down_proj", "gate_proj", "up_proj")
        ]
        self.assertTrue(
            should_ignore_layer(layer, ignore=ignore, fused_mapping=QKV_MAPPING)
        )

    # ---- Guards: behavior that must NOT change -----------------------------

    def test_unexcluded_fused_qkv_still_expands(self):
        # With no direct match, the packed expansion must still decide, and an
        # empty ignore list must not start excluding everything.
        self.assertFalse(
            should_ignore_layer(
                "model.layers.0.self_attn.qkv_proj",
                ignore=[],
                fused_mapping=QKV_MAPPING,
            )
        )

    def test_shard_level_excludes_still_work(self):
        # The original path: shards named individually, module named fused.
        name = "model.layers.0.self_attn.qkv_proj"
        ignore = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        ]
        self.assertTrue(
            should_ignore_layer(name, ignore=ignore, fused_mapping=QKV_MAPPING)
        )

    def test_unrelated_moe_layer_is_not_ignored(self):
        # Only the layer whose experts are excluded may be excluded: a prefix
        # test must not match a different layer index.
        ignore = ["model.layers.6.mlp.experts.0.down_proj"]
        self.assertFalse(
            should_ignore_layer(
                "model.layers.7.mlp.experts",
                ignore=ignore,
                fused_mapping=QKV_MAPPING,
            )
        )


if __name__ == "__main__":
    unittest.main()
