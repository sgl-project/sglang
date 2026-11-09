"""ModelSlim name resolution and weight dtypes, using small CPU layers."""

import unittest
from copy import copy, deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.modelslim.modelslim import (
    ModelSlimConfig,
    ModelSlimLinearMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.utils import WeightsMapper

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestModelSlimMapping(CustomTestCase):
    def make_linear(self, config, prefix):
        """Build a small CPU layer: input width 4, output width 6."""
        with torch.device("cpu"):
            return ReplicatedLinear(
                4,
                6,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=prefix,
            )

    def test_qwen_qkvz_uses_model_mapping_and_creates_int8_weights(self):
        """Use qkv + z descriptions to create an INT8 qkvz weight."""
        parent = "model.layers.0.linear_attn."
        mapping = {"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"]}
        config = ModelSlimConfig(
            {parent + part + ".weight": "W8A8" for part in mapping["in_proj_qkvz"]}
        )
        config.update_packed_modules_mapping(mapping)
        layer = self.make_linear(config, parent + "in_proj_qkvz")
        self.assertIsInstance(layer.quant_method, ModelSlimLinearMethod)
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(layer.weight.shape, (6, 4))
        self.assertEqual(
            config._resolve_linear_quant_prefixes(parent + "in_proj_qkvz"),
            [parent + "in_proj_qkv", parent + "in_proj_z"],
        )

    def test_qkv_and_gate_up_defaults_work_for_target_and_draft(self):
        """Under model/mtp/decoder/stages, QKV and gate/up use INT8 or BF16."""
        for parent in (
            "model.layers.0.",
            "mtp.layers.0.",
            "model.decoder.",
            "stages.0.",
        ):
            for fused, parts in (
                (
                    "self_attn.qkv_proj",
                    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                ),
                ("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
            ):
                for scheme, dtype in (("W8A8", torch.int8), ("FLOAT", torch.bfloat16)):
                    with self.subTest(parent=parent, fused=fused, scheme=scheme):
                        config = ModelSlimConfig(
                            {parent + part + ".weight": scheme for part in parts}
                        )
                        layer = self.make_linear(config, parent + fused)
                        self.assertEqual(layer.weight.dtype, dtype)
                        self.assertEqual(config.packed_modules_mapping, {})

    def test_model_rules_override_defaults_and_accept_kimi_group(self):
        """Use flat or Kimi 'model' rules first; [] disables a fusion rule."""
        for mapping in ({"qkv_proj": ["packed"]}, {"model": {"qkv_proj": ["packed"]}}):
            with self.subTest(mapping=mapping):
                config = ModelSlimConfig({"model.attn.packed.weight": "W8A8"})
                config.update_packed_modules_mapping(mapping)
                self.assertEqual(
                    self.make_linear(config, "model.attn.qkv_proj").weight.dtype,
                    torch.int8,
                )
        config = ModelSlimConfig(
            {
                "packed_modules_mapping": {"qkv_proj": []},
                "model.attn.q_proj.weight": "W8A8",
            }
        )
        self.assertEqual(
            config._resolve_linear_quant_prefixes("model.attn.qkv_proj"),
            ["model.attn.qkv_proj"],
        )
        # Other nested groups, e.g. 'visual', must raise instead of being ignored.
        config.update_packed_modules_mapping({"visual": {"qkv_proj": ["qkv"]}})
        with self.assertRaisesRegex(ValueError, "supports flat packed mappings"):
            config._resolve_linear_quant_prefixes("visual.attn.qkv_proj")

    def test_mla_fallback_does_not_change_model_table(self):
        """Fuse q_a_proj + kv_a_proj_with_mqa without editing the model's table."""
        parent = "model.layers.0.self_attn."
        parts = ("q_a_proj", "kv_a_proj_with_mqa")
        shared = {}
        config = ModelSlimConfig(
            {
                "packed_modules_mapping": shared,
                **{parent + part + ".weight": "W8A8" for part in parts},
            }
        )
        layer = self.make_linear(config, parent + "fused_qkv_a_proj_with_mqa")
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(shared, {})

    def test_legacy_vision_names_do_not_change_text_scheme(self):
        """Resolve vision names, e.g. qkv_proj -> qkv, while text stays BF16."""
        cases = (
            ("model.visual.blocks.0.attn.qkv_proj", "model.visual.blocks.0.attn.qkv"),
            (
                "vision_model.encoder.layers.0.self_attn.proj",
                "vision_model.encoder.layers.0.self_attn.out_proj",
            ),
            ("vision_tower.blocks.0.attn.qkv_proj", "vision_tower.blocks.0.wqkv"),
            ("mm_projector.blocks.0.attn.proj", "mm_projector.blocks.0.wo"),
        )
        for runtime, checkpoint in cases:
            with self.subTest(runtime=runtime):
                config = ModelSlimConfig(
                    {
                        checkpoint + ".weight": "W8A8",
                        **{
                            "model.attn." + part + ".weight": "FLOAT"
                            for part in ("q_proj", "k_proj", "v_proj")
                        },
                    }
                )
                self.assertEqual(
                    self.make_linear(config, runtime).weight.dtype, torch.int8
                )
                self.assertEqual(
                    self.make_linear(config, "model.attn.qkv_proj").weight.dtype,
                    torch.bfloat16,
                )

    def test_vision_alias_requires_path_boundaries_and_existing_description(self):
        """Match whole path parts: visual must not match visualizer.

        Also reject self_attn.proj, attn.projection, and missing descriptions.
        """
        for runtime, checkpoint in (
            ("model.attn.qkv_proj", "model.attn.qkv"),
            ("visualizer.attn.qkv_proj", "visualizer.attn.qkv"),
            ("vision_tower.self_attn.proj", "vision_tower.wo"),
            ("vision_tower.attn.projection", "vision_tower.woection"),
        ):
            with self.subTest(runtime=runtime):
                config = ModelSlimConfig({checkpoint + ".weight": "W8A8"})
                self.assertEqual(
                    config._resolve_linear_quant_prefixes(runtime), [runtime]
                )
        config = ModelSlimConfig()
        self.assertEqual(
            config._resolve_linear_quant_prefixes("visual.attn.qkv_proj"),
            ["visual.attn.qkv_proj"],
        )

    def test_mapper_preserves_auxiliary_entries_and_is_repeatable(self):
        """Rename qkv.weight, qkv.weight_scale and qkv.weight_offset together.

        Keep global settings; repeating the mapper must change nothing.
        """
        fields = {
            "weight": "W8A8_DYNAMIC",
            "weight_scale": "FLOAT",
            "weight_offset": "FLOAT",
        }
        original = {
            "model.visual.attn.qkv." + field: value for field, value in fields.items()
        }
        original.update(
            {
                "quant_method": "modelslim",
                "packed_modules_mapping": {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
            }
        )
        config = ModelSlimConfig(original)
        mapper = WeightsMapper(
            orig_to_new_prefix={"model.visual.": "visual."},
            orig_to_new_suffix={".qkv": ".qkv_proj"},
        )
        before_mapper = deepcopy(mapper)
        config.apply_weight_name_mapper(mapper)
        expected = {
            **original,
            **{
                "visual.attn.qkv_proj." + field: value
                for field, value in fields.items()
            },
        }
        self.assertEqual(config.quant_description, expected)
        config.apply_weight_name_mapper(mapper)
        self.assertEqual(config.quant_description, expected)
        self.assertEqual(mapper, before_mapper)
        layer = self.make_linear(config, "visual.attn.qkv_proj")
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertTrue(layer.scheme.is_dynamic)
        self.assertTrue(hasattr(layer, "weight_scale"))

    def test_full_key_suffix_precedes_layer_suffix_and_none_adds_no_alias(self):
        """Try .qkv.weight before .qkv; None adds no alias and keeps the original."""
        for full_suffix, expected in (
            (".packed_weight", {"mapped.attn.packed_weight": "W8A8"}),
            (None, {}),
        ):
            with self.subTest(full_suffix=full_suffix):
                original = {"model.attn.qkv.weight": "W8A8"}
                config = ModelSlimConfig(original)
                config.apply_weight_name_mapper(
                    WeightsMapper(
                        orig_to_new_prefix={"model.": "mapped."},
                        orig_to_new_suffix={
                            ".qkv.weight": full_suffix,
                            ".qkv": ".qkv_proj",
                        },
                    )
                )
                self.assertEqual(config.quant_description, {**original, **expected})

    def test_mapper_conflict_leaves_description_unchanged(self):
        """If qkv -> qkv_proj would mix W8A8 and FLOAT, raise and keep the config."""
        config = ModelSlimConfig(
            {"attn.qkv.weight": "W8A8", "attn.qkv_proj.weight": "FLOAT"}
        )
        before = deepcopy(config.quant_description)
        with self.assertRaisesRegex(ValueError, "Conflicting ModelSlim descriptions"):
            config.apply_weight_name_mapper(
                WeightsMapper(orig_to_new_substr={"attn.qkv": "attn.qkv_proj"})
            )
        self.assertEqual(config.quant_description, before)

    def test_vision_copy_does_not_change_language_config(self):
        """Renaming a vision copy twice adds only qkv_proj, not qkv_proj_proj.

        The text config must stay unchanged.
        """
        config = ModelSlimConfig({"model.visual.attn.qkv.weight": "W8A8"})
        before = deepcopy(config.quant_description)
        vision = copy(config)
        mapper = WeightsMapper(orig_to_new_substr={"attn.qkv": "attn.qkv_proj"})
        vision.apply_weight_name_mapper(mapper)
        vision.apply_weight_name_mapper(mapper)
        self.assertEqual(
            vision.quant_description,
            {
                **before,
                "model.visual.attn.qkv_proj.weight": "W8A8",
            },
        )
        self.assertEqual(
            self.make_linear(vision, "model.visual.attn.qkv_proj").weight.dtype,
            torch.int8,
        )
        self.assertEqual(config.quant_description, before)

    def test_update_replaces_nested_group_without_mutating_inputs(self):
        """Replace the 'model' group; adding 'v' must not change either input table."""
        original = {
            "packed_modules_mapping": {"flat": ["keep"], "model": {"old": ["a"]}},
            "model.block_sparse_moe.experts.0.gate_proj.weight": "FLOAT",
        }
        incoming = {"model": {"fused": ["q", "k"]}}
        before, before_incoming = deepcopy(original), deepcopy(incoming)
        config = ModelSlimConfig(original)
        config.update_packed_modules_mapping(incoming)
        config.update_packed_modules_mapping(incoming)
        self.assertEqual(config.packed_modules_mapping, {"flat": ["keep"], **incoming})
        config.packed_modules_mapping["model"]["fused"].append("v")
        config.packed_modules_mapping["flat"].append("local")
        self.assertEqual(original, before)
        self.assertEqual(incoming, before_incoming)

    def test_incomplete_mixed_and_ambiguous_layouts_raise(self):
        """Require all q/k/v parts with one scheme; reject whole + split layouts."""
        parts = {
            "visual.attn." + part + ".weight": "W8A8"
            for part in ("q_proj", "k_proj", "v_proj")
        }
        cases = (
            ({"visual.attn.q_proj.weight": "W8A8"}, "Missing ModelSlim"),
            (
                {**parts, "visual.attn.v_proj.weight": "FLOAT"},
                "same quantization scheme",
            ),
            ({**parts, "visual.attn.qkv_proj.weight": "W8A8"}, "Ambiguous ModelSlim"),
            ({**parts, "visual.attn.qkv.weight": "W8A8"}, "Ambiguous ModelSlim"),
        )
        for entries, error in cases:
            with (
                self.subTest(entries=entries),
                self.assertRaisesRegex(ValueError, error),
            ):
                self.make_linear(ModelSlimConfig(entries), "visual.attn.qkv_proj")

    def test_zero_width_components_are_optional_only_with_matching_sizes(self):
        """Allow missing V for sizes [8, 4, 0], but not for a shorter size list."""
        parent = "model.layers.0.self_attn."
        config = ModelSlimConfig(
            {
                parent + part + ".weight": "W8A8"
                for part in ("index_q_proj", "index_k_proj")
            }
        )
        config.update_packed_modules_mapping(
            {"index_qkv_proj": ["index_q_proj", "index_k_proj", "index_v_proj"]}
        )
        scheme = config.get_linear_scheme(
            SimpleNamespace(output_sizes=[8, 4, 0]), parent + "index_qkv_proj"
        )
        self.assertIsNotNone(scheme)
        with self.assertRaisesRegex(ValueError, "Missing ModelSlim"):
            config.get_linear_scheme(
                SimpleNamespace(output_sizes=[8, 4]), parent + "index_qkv_proj"
            )

    def test_float_missing_and_unsupported_descriptions_are_distinct(self):
        """FLOAT stays BF16; missing names warn; unsupported W16A16S raises."""
        config = ModelSlimConfig(
            {"model.float.weight": "FLOAT", "model.unsupported.weight": "W16A16S"}
        )
        self.assertIsInstance(
            self.make_linear(config, "model.float").quant_method,
            UnquantizedLinearMethod,
        )
        with self.assertLogs(
            "sglang.srt.layers.quantization.modelslim.modelslim", level="WARNING"
        ) as logs:
            self.assertEqual(
                self.make_linear(config, "model.missing").weight.dtype, torch.bfloat16
            )
        self.assertIn("model.missing.weight", " ".join(logs.output))
        with self.assertRaisesRegex(
            ValueError, "Unsupported ModelSlim Linear scheme.*W16A16S"
        ):
            self.make_linear(config, "model.unsupported")

    def test_loader_keeps_model_mapping_clean_and_applies_draft_mapper(self):
        """Map layers.45 -> decoder in the real loader; leave model rules alone."""
        from sglang.srt.model_loader import loader

        mapping = {"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"]}
        before = deepcopy(mapping)
        model_class = SimpleNamespace(
            packed_modules_mapping=mapping,
            hf_to_sglang_mapper=WeightsMapper(
                orig_to_new_prefix={"model.layers.45.": "model.decoder."}
            ),
        )

        # Supply a small config instead of reading a file; keep the real loader.
        def read_config(model_config, load_config, packed, remap):
            return ModelSlimConfig(
                {
                    "packed_modules_mapping": packed,
                    "model.layers.45.self_attn.q_proj.weight": "W8A8",
                }
            )

        with (
            patch.object(loader, "_is_npu", True),
            patch.object(
                loader, "get_model_architecture", return_value=(model_class, None)
            ),
            patch.object(loader, "get_quant_config", side_effect=read_config),
            patch.object(
                loader,
                "get_spec",
                return_value=SimpleNamespace(speculative_draft_kv_cache_dtype=None),
            ),
            patch.object(
                loader, "get_model", return_value=SimpleNamespace(kv_cache_dtype="auto")
            ),
        ):
            config = loader._get_quantization_config(
                SimpleNamespace(
                    quantization="modelslim", dtype=torch.bfloat16, is_draft_model=True
                ),
                None,
            )
            self.assertEqual(
                config.quant_description["model.decoder.self_attn.q_proj.weight"],
                "W8A8",
            )
            self.assertEqual(mapping, before)
            config.update_packed_modules_mapping(
                {"gate_up_proj": ["gate_proj", "up_proj"]}
            )
            self.assertEqual(mapping, before)
            self.assertIsNone(
                loader._get_quantization_config(
                    SimpleNamespace(quantization=None), None
                )
            )
            self.assertEqual(mapping, before)

    def test_qwen_vision_constructor_passes_isolated_modelslim_config(self):
        """Give Qwen vision its own ModelSlim config; other configs keep BF16."""
        from sglang.srt.models import qwen3_vl

        def small_vision(vision_config, *, quant_config, prefix, **kwargs):
            return self.make_linear(quant_config, prefix + ".blocks.0.attn.qkv_proj")

        model_config = SimpleNamespace(
            encoder_only=True,
            rope_scaling={},
            vision_config=SimpleNamespace(deepstack_visual_indexes=[]),
        )
        quant_config = ModelSlimConfig(
            {"model.visual.blocks.0.attn.qkv.weight": "W8A8"}
        )
        original = deepcopy(quant_config.quant_description)
        # Keep the real model constructor; replace the large tower with one linear.
        with (
            patch.object(
                qwen3_vl,
                "get_parallel",
                return_value=SimpleNamespace(pp_group=SimpleNamespace()),
            ),
            patch.object(
                qwen3_vl,
                "get_mm",
                return_value=SimpleNamespace(mm_enable_dp_encoder=False),
            ),
            patch.object(qwen3_vl, "Qwen3VLMoeVisionModel", side_effect=small_vision),
            patch.object(qwen3_vl, "LogitsProcessor", return_value=torch.nn.Identity()),
        ):
            for config, dtype in (
                (quant_config, torch.int8),
                (None, torch.bfloat16),
                (SimpleNamespace(), torch.bfloat16),
            ):
                with self.subTest(dtype=dtype, config=config):
                    model = qwen3_vl.Qwen3VLForConditionalGeneration(
                        model_config, quant_config=config
                    )
                    self.assertEqual(model.visual.weight.dtype, dtype)
                    if config is quant_config:
                        self.assertIsNot(model.visual.quant_config, quant_config)
                        self.assertIn(
                            "visual.blocks.0.attn.qkv_proj.weight",
                            model.visual.quant_config.quant_description,
                        )
        self.assertEqual(quant_config.quant_description, original)


if __name__ == "__main__":
    unittest.main()
