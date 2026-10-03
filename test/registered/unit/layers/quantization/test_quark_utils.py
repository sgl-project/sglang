"""Unit tests for sglang.srt.layers.quantization.quark.utils — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import os
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    QuarkW4A4MXFP4,
    _resolve_dequant_linear_to_bf16,
)
from sglang.srt.layers.quantization.quark.utils import (
    e8m0_to_f32,
    should_ignore_layer,
)
from sglang.test.test_utils import CustomTestCase


class TestShouldIgnoreLayer(CustomTestCase):
    """MiniMax-M3 MXFP4: sparse index_qkv_proj packs only q/k (the DSA value
    projection is disabled, so index_v_proj is absent on disk)."""

    _LAYER = "language_model.model.layers.3.self_attn.index_qkv_proj"
    _IGNORE = (
        "language_model.model.layers.3.self_attn.index_q_proj",
        "language_model.model.layers.3.self_attn.index_k_proj",
    )
    # The fix lives in the model: index_qkv_proj maps to only q/k (no v).
    _MAPPING = {
        "index_qkv_proj": ["index_q_proj", "index_k_proj"],
    }

    def test_minimax_dsa_index_qkv_ignored(self):
        # Both present shards are excluded -> fused module stays bf16, no raise.
        self.assertTrue(should_ignore_layer(self._LAYER, self._IGNORE, self._MAPPING))

    def test_all_shards_agree_still_works(self):
        layer = "model.layers.0.self_attn.qkv_proj"
        ignore = (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        )
        mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        self.assertTrue(should_ignore_layer(layer, ignore, mapping))

    def test_no_shards_ignored(self):
        layer = "model.layers.0.self_attn.qkv_proj"
        mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        self.assertFalse(should_ignore_layer(layer, (), mapping))

    def test_mixed_schemes_raise(self):
        # Safety net preserved: if a fused module genuinely mixes excluded and
        # quantized shards, the loader must fail loudly rather than guess.
        layer = "model.layers.0.self_attn.qkv_proj"
        ignore = (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
        )  # v_proj NOT excluded -> inconsistent with q/k
        mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        with self.assertRaises(ValueError):
            should_ignore_layer(layer, ignore, mapping)


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
    """An `exclude` entry naming an already-fused module, or naming experts
    individually, must exclude the fused module SGLang builds; otherwise an
    MXFP4-packed parameter is allocated for a BF16 tensor and loading aborts."""

    # ---- Bug-catchers: must FAIL on unfixed code ---------------------------

    def test_directly_excluded_fused_qkv_is_ignored(self):
        name = "visual.blocks.0.attn.qkv_proj"
        self.assertTrue(
            should_ignore_layer(name, ignore=[name], fused_mapping=QKV_MAPPING)
        )

    def test_per_expert_excludes_ignore_the_fused_moe_module(self):
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

    def test_unrelated_moe_layer_is_not_ignored(self):
        # a prefix match must not bleed into a neighboring layer index
        ignore = ["model.layers.6.mlp.experts.0.down_proj"]
        self.assertFalse(
            should_ignore_layer(
                "model.layers.7.mlp.experts",
                ignore=ignore,
                fused_mapping=QKV_MAPPING,
            )
        )


class TestMXFP4WeightMaterialization(CustomTestCase):
    def test_materializes_packed_weight_without_mutating_layer(self):
        # Packed low/high nibbles 1/2 decode to 0.5/1.0. An e8m0 scale of
        # 127 is exactly one.
        packed = torch.full((2, 16), 0x21, dtype=torch.uint8)
        scales = torch.full((2, 1), 127, dtype=torch.uint8)
        layer = SimpleNamespace(weight=packed, weight_scale=scales)
        scheme = object.__new__(QuarkW4A4MXFP4)

        dense = scheme.materialize_bf16_weight(layer)

        expected = torch.tensor([0.5, 1.0], dtype=torch.bfloat16).repeat(2, 16)
        torch.testing.assert_close(dense, expected)
        self.assertIs(layer.weight, packed)
        self.assertIs(layer.weight_scale, scales)

    def test_reuses_already_materialized_weight(self):
        weight = torch.randn(2, 32, dtype=torch.bfloat16)
        layer = SimpleNamespace(weight=weight, dequantized_bf16=True)
        scheme = object.__new__(QuarkW4A4MXFP4)

        self.assertIs(scheme.materialize_bf16_weight(layer), weight)


class TestMXFP4LinearActResolution(CustomTestCase):
    """The activation knob applies on ROCm and is inert everywhere else."""

    def test_rocm_defaults_to_bf16(self):
        self.assertTrue(_resolve_dequant_linear_to_bf16("bf16", is_hip=True))

    def test_rocm_fp4_keeps_packed_weights(self):
        self.assertFalse(_resolve_dequant_linear_to_bf16("fp4", is_hip=True))

    def test_rocm_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            _resolve_dequant_linear_to_bf16("fp8", is_hip=True)

    def test_non_rocm_ignores_every_value(self):
        # Including values ROCm would reject.
        for act in ("bf16", "fp4", "nonsense"):
            with self.subTest(act=act):
                self.assertFalse(_resolve_dequant_linear_to_bf16(act, is_hip=False))

    def test_non_rocm_warns_only_when_explicitly_set(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(envs.SGLANG_ROCM_QUARK_MXFP4_LINEAR_ACT.name, None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _resolve_dequant_linear_to_bf16("bf16", is_hip=False)
            self.assertEqual(list(caught), [])

        with envs.SGLANG_ROCM_QUARK_MXFP4_LINEAR_ACT.override("fp4"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _resolve_dequant_linear_to_bf16("fp4", is_hip=False)
        self.assertEqual(len(caught), 1)
        self.assertIn("ROCm-only", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
