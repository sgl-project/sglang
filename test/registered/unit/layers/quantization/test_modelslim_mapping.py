"""ModelSlim name resolution and weight dtypes, using small CPU layers."""

import json
import unittest
from copy import copy, deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
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
    def test_sidecar_uses_model_context_without_heuristic_markers(self):
        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.model_loader.weight_utils import get_quant_config

        raw = "layers.0.attn.wq_a.weight"
        # Exercise the loader's sidecar path without depending on the CI CPU's
        # device-specific quantization registry.
        with (
            TemporaryDirectory() as path,
            patch(
                "sglang.srt.model_loader.weight_utils.get_quantization_config",
                return_value=ModelSlimConfig,
            ),
        ):
            Path(path, "quant_model_description.json").write_text(
                json.dumps({raw: "W8A8_DYNAMIC"})
            )
            for arch, expected in (
                ("DeepseekV4ForCausalLM", "model.layers.0.self_attn.wq_a.weight"),
                ("OtherModel", raw),
            ):
                with self.subTest(arch=arch):
                    model = SimpleNamespace(
                        quantization="modelslim",
                        hf_config={"architectures": [arch]},
                        model_path=path,
                        is_draft_model=False,
                    )
                    cfg = get_quant_config(model, LoadConfig(), {})
                    self.assertEqual(cfg.quant_description[expected], "W8A8_DYNAMIC")

    def test_v4_export_names_follow_the_model_not_the_quantization_dtype(self):
        weight = "layers.0.ffn.experts.0.w1.weight"
        target = "model.layers.0.mlp.experts.0.gate_proj.weight"
        draft = "mtp.0.ffn.experts.0.w1.weight"
        for scheme in ("W4A8_DYNAMIC", "W8A8_DYNAMIC"):
            desc = {weight: scheme, draft: scheme}
            for arch in ("DeepseekV4ForCausalLM", "DeepseekV4ForCausalLMDSpark"):
                with self.subTest(scheme=scheme, arch=arch):
                    cfg = ModelSlimConfig.from_config(
                        {**desc, "hf_config": SimpleNamespace(architectures=[arch])}
                    )
                    self.assertEqual(cfg.quant_description[target], scheme)
                    self.assertEqual(
                        cfg.quant_description[
                            "stages.0.mlp.experts.0.gate_proj.weight"
                        ],
                        scheme,
                    )
            # Explicit non-V4 models must not be inferred from the spelling of a key.
            other = ModelSlimConfig.from_config(
                {**desc, "hf_config": SimpleNamespace(architectures=["OtherModel"])}
            )
            self.assertIn(weight, other.quant_description)
            self.assertNotIn(target, other.quant_description)
            self.assertIn(target, ModelSlimConfig.from_config(desc).quant_description)

    def make_linear(self, config, prefix, output_size=6):
        """Build a small CPU layer with input width 4."""
        with torch.device("cpu"):
            return ReplicatedLinear(
                4,
                output_size,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=prefix,
            )

    def test_engram_table_follows_its_own_checkpoint_entry(self):
        table = {
            "format": "int8_sym",
            "weight_dtype": "I8",
            "layout": "row_major",
            "group_size": 32,
            "axis": 1,
            "scale_suffix": "scale",
            "scale_dtype": "F8_E8M0",
            "weight_shape": [8, 64],
            "scale_shape": [8, 2],
        }
        modules = {"layers.1.engram.embed": table}
        for scheme in ("W4A8_DYNAMIC", "W8A8_DYNAMIC"):
            cfg = ModelSlimConfig(
                {
                    "model.layers.0.mlp.experts.0.gate_proj.weight": scheme,
                    "optional": {"embedding_storage": {"modules": modules}},
                }
            )
            native = torch.float8_e4m3fn
            with patch(
                "sglang.srt.layers.quantization.modelslim.modelslim.is_npu",
                return_value=True,
            ):
                self.assertEqual(
                    cfg.get_engram_table_dtype(1, 8, 64, native), torch.int8
                )
                self.assertEqual(cfg.get_engram_table_dtype(2, 8, 64, native), native)
                with self.assertRaisesRegex(ValueError, "storage"):
                    cfg.get_engram_table_dtype(1, 8, 96, native)
                with self.assertRaisesRegex(ValueError, "MXFP4"):
                    cfg.get_engram_table_dtype(1, 8, 64, torch.uint8)

    def test_split_int8_wqkv_requires_separate_projections(self):
        prefix = "model.layers.0.self_attn."

        def desc(parts):
            return ModelSlimConfig(
                {
                    prefix + part + field: "W8A8_DYNAMIC"
                    for part in parts
                    for field in (".weight", ".weight_scale")
                }
            )

        split = desc(("wq_a", "wkv"))
        for part in ("wq_a", "wkv"):
            self.assertEqual(
                self.make_linear(split, prefix + part).weight.dtype, torch.int8
            )
        with self.assertRaisesRegex(ValueError, "SGLANG_OPT_FUSE_WQA_WKV=0"):
            self.make_linear(split, prefix + "wqkv_a")
        self.assertEqual(
            self.make_linear(desc(("wqkv_a",)), prefix + "wqkv_a").weight_scale.shape,
            (6, 1),
        )

    def test_v4_loads_split_float_weights_with_fusion_on_and_off(self):
        from sglang.srt.environ import envs
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        # The 1017 W8A8 export has FLOAT attention inputs despite its INT8 experts.
        q, kv = torch.arange(24, dtype=torch.bfloat16).view(6, 4).split((2, 4))
        source = [("layers.0.attn.wkv.weight", kv), ("layers.0.attn.wq_a.weight", q)]
        x = torch.eye(4, dtype=q.dtype)
        prefix = "model.layers.0.self_attn."
        cfg = ModelSlimConfig(
            {prefix + part + ".weight": "FLOAT" for part in ("wq_a", "wkv")}
        )
        for fused in (False, True):
            parts = {"wqkv_a": 6} if fused else {"wq_a": 2, "wkv": 4}
            weights = {
                part: self.make_linear(cfg, prefix + part, rows).weight
                for part, rows in parts.items()
            }
            model = SimpleNamespace(
                named_parameters=lambda: [
                    (prefix + p + ".weight", w) for p, w in weights.items()
                ],
                config=SimpleNamespace(num_hidden_layers=1, n_routed_experts=0),
                model=SimpleNamespace(layers=[]),
                pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
                quant_config=cfg,
                wo_a_fp8=False,
                use_npu_arch35_mxfp8_wo_a=False,
                num_fused_shared_experts=0,
                remap_weight_name_to_dpsk_hf_format=DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format,
                post_load_weights=lambda **_: None,
                _prewarm_mhc_kernels=lambda: None,
            )
            with (
                self.subTest(fused=fused),
                envs.SGLANG_OPT_FUSE_WQA_WKV.override(fused),
            ):
                DeepseekV4ForCausalLM.load_weights(model, source)
                actual = torch.cat([x @ w.T for w in weights.values()], dim=-1)
                torch.testing.assert_close(
                    actual, torch.cat((x @ q.T, x @ kv.T), dim=-1)
                )

    def test_engram_gate_basis_comes_from_weight_snapshot(self):
        from safetensors.torch import save_file

        from sglang.srt.layers.quantization.modelslim import engram

        net = torch.nn.Module()
        net.layers = [SimpleNamespace(engram=SimpleNamespace(q_weight=torch.ones(2)))]
        cfg = SimpleNamespace(hidden_size=2, _name_or_path="stale/config/path")
        engram.load_gate_basis(cfg, net)
        self.assertFalse(hasattr(net, "engram_gate_basis"))
        cfg.engram_rotation_config = dict(
            value_projection_rotated=True,
            value_basis="quarot_global",
            key_and_gate_basis="original",
            runtime_delta_rotation=False,
        )
        with (
            TemporaryDirectory() as snapshot,
            patch.object(engram, "get_context") as ctx,
            patch.object(
                engram,
                "get_model",
                return_value=SimpleNamespace(model_path="org/model", revision="rev"),
            ),
            patch(
                "sglang.srt.model_loader.loader.loaded_checkpoint_files",
                return_value=(snapshot, ()),
            ) as resolve,
        ):
            ctx.return_value.is_config_namespace_published.return_value = True
            with self.assertRaises(FileNotFoundError):
                engram.load_gate_basis(cfg, net)
            resolve.assert_called_once_with("org/model", "rev")
            path = Path(snapshot) / "optional" / "quarot.safetensors"
            path.parent.mkdir()
            basis = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
            save_file({"global_rotation": basis.bfloat16()}, path)
            engram.load_gate_basis(cfg, net)
            torch.testing.assert_close(net.engram_gate_basis, basis)
            self.assertNotIn("engram_gate_basis", net.state_dict())
            # Repeated post-load on this rank reuses the buffer.
            loaded = net.engram_gate_basis
            engram.load_gate_basis(cfg, net)
            self.assertIs(net.engram_gate_basis, loaded)

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
