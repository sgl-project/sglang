"""CPU unit tests for maybe_remap_kv_scale_name.

Covers the FP8 KV cache scale naming families the helper must bridge to the
RadixAttention parameters (``attn.k_scale`` / ``attn.v_scale``):

  - llm-compressor fused names (``self_attn.qkv_proj.k_scale``), including
    the ``qkqkv_proj`` form produced by stacked-mapping name mangling
  - ModelOpt per-projection names (``self_attn.k_proj.k_scale``, ``mixer``)
  - the deprecated single ``.kv_scale`` tensor
  - quark ``output_scale`` names
  - pass-through of names that are not KV scales
  - ``None`` plus warning when the model has no matching parameter

Run: python3 test/srt/models/test_kv_scale_name_remap.py
"""

import unittest

from sglang.srt.model_loader.weight_utils import maybe_remap_kv_scale_name


class TestMaybeRemapKVScaleName(unittest.TestCase):
    def test_remap_families(self):
        params = {
            "model.layers.0.self_attn.attn.k_scale",
            "model.layers.0.self_attn.attn.v_scale",
            "model.layers.0.mixer.attn.v_scale",
            "model.layers.0.self_attn.o_proj.weight",
        }
        cases = [
            # llm-compressor fused naming
            ("model.layers.0.self_attn.qkv_proj.k_scale",
             "model.layers.0.self_attn.attn.k_scale"),
            ("model.layers.0.self_attn.qkv_proj.v_scale",
             "model.layers.0.self_attn.attn.v_scale"),
            # same, after name.replace("v_proj", "qkv_proj") mangling
            ("model.layers.0.self_attn.qkqkv_proj.k_scale",
             "model.layers.0.self_attn.attn.k_scale"),
            # ModelOpt per-projection naming
            ("model.layers.0.self_attn.k_proj.k_scale",
             "model.layers.0.self_attn.attn.k_scale"),
            ("model.layers.0.mixer.v_proj.v_scale",
             "model.layers.0.mixer.attn.v_scale"),
            # deprecated single tensor
            ("model.layers.0.self_attn.kv_scale",
             "model.layers.0.self_attn.attn.k_scale"),
            # quark naming
            ("model.layers.0.self_attn.k_proj.output_scale",
             "model.layers.0.self_attn.attn.k_scale"),
            # not a KV scale: untouched
            ("model.layers.0.self_attn.o_proj.weight",
             "model.layers.0.self_attn.o_proj.weight"),
        ]
        for name, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    maybe_remap_kv_scale_name(name, params), expected
                )

    def test_missing_param_returns_none(self):
        # Fused and per-projection names on a model without attn scales.
        for name in (
            "model.layers.0.self_attn.qkv_proj.k_scale",
            "model.layers.0.self_attn.k_proj.k_scale",
        ):
            with self.subTest(name=name):
                self.assertIsNone(maybe_remap_kv_scale_name(name, set()))


if __name__ == "__main__":
    unittest.main()
