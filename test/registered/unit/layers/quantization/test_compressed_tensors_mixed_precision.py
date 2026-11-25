"""CPU regression tests for multi-group ("mixed-precision") compressed-tensors configs.

Such a checkpoint used to load completely unquantized. ``ignore`` was matched by
substring, so a parent module entry swallowed its quantized children, and the
activation-quantization gate read the top-level format -- which compressed-tensors
sets to ``mixed-precision`` when groups disagree -- dropping ``input_activations``
for every group. Both are config-parsing paths, so these tests run on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import ast
import re
import unittest
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
    should_ignore_layer,
)
from sglang.test.test_utils import CustomTestCase

EXPERTS_LAYER = "model.language_model.layers.0.mlp.experts"
GATE_PROJ = f"{EXPERTS_LAYER}.0.gate_proj"
MLP_LAYER = "model.language_model.layers.0.mlp.gate_proj"

# A vision tower whose ignore list is spelled against the CompressedLinear
# wrapper. The names below are the prefixes the matcher actually receives:
# ClippableQKVParallelLinear and ClippableGateUpParallelLinear hand their own
# prefix to the inner fused linear, while ClippableRowParallelLinear nests its
# inner linear at ".linear".
VISION_PREFIX = "model.vision_tower.encoder.layers"
VISION_QKV_LAYER = f"{VISION_PREFIX}.0.self_attn.qkv_proj"
VISION_GATE_UP_LAYER = f"{VISION_PREFIX}.0.mlp.gate_up_proj"
VISION_O_PROJ_LAYER = f"{VISION_PREFIX}.0.self_attn.o_proj.linear"
VISION_DOWN_PROJ_LAYER = f"{VISION_PREFIX}.0.mlp.down_proj.linear"
VISION_FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
VISION_NUM_LAYERS = 27
TEXT_PREFIX = "model.language_model.layers.0"


def _wrapped_tower_ignore_list():
    """27 layers x 7 projections, every entry spelled ``<module>.linear``.

    135 name fused shards (q/k/v, gate/up) and 54 name unfused layers
    (o_proj, down_proj).
    """
    entries = []
    for idx in range(VISION_NUM_LAYERS):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            entries.append(f"{VISION_PREFIX}.{idx}.self_attn.{proj}.linear")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            entries.append(f"{VISION_PREFIX}.{idx}.mlp.{proj}.linear")
    return entries


FP8_TARGET = "re:.*self_attn\\.(q|k|v|o)_proj$"
NVFP4_TARGET = "re:.*mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$"
WNA16_TARGET = "re:.*mlp\\.(gate|up|down)_proj$"

# FP8 W8A8 attention projections.
FP8_GROUP = {
    "format": "float-quantized",
    "targets": [FP8_TARGET],
    "weights": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "channel",
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "token",
        "dynamic": True,
    },
}

# NVFP4 W4A4 expert projections.
NVFP4_GROUP = {
    "format": "nvfp4-pack-quantized",
    "targets": [NVFP4_TARGET],
    "weights": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": 16,
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": 16,
        "dynamic": "local",
    },
}

# Weight-only INT4: a group whose format is not an activation format.
WNA16_GROUP = {
    "format": "pack-quantized",
    "targets": [WNA16_TARGET],
    "weights": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
        "dynamic": False,
    },
    "input_activations": None,
}


def _mixed_precision_config(*groups, ignore=()):
    """Groups disagree, so compressed-tensors writes format="mixed-precision"."""
    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {f"group_{i}": g for i, g in enumerate(groups)},
        "ignore": list(ignore),
    }


class TestIgnoreListPrefixMatching(CustomTestCase):
    """A parent module in `ignore` must not de-quantize its children."""

    def test_parent_module_does_not_swallow_children(self):
        ignore = [f"{EXPERTS_LAYER}.0", "model.language_model.layers.0.linear_attn"]

        # The parent itself is still ignored: semantics are unchanged.
        self.assertTrue(should_ignore_layer(f"{EXPERTS_LAYER}.0", ignore=ignore))
        self.assertTrue(
            should_ignore_layer(
                "model.language_model.layers.0.linear_attn", ignore=ignore
            )
        )

        # Its quantized children are not.
        self.assertFalse(should_ignore_layer(GATE_PROJ, ignore=ignore))
        self.assertFalse(
            should_ignore_layer(
                "model.language_model.layers.0.linear_attn.out_proj", ignore=ignore
            )
        )

    def test_module_suffix_target_still_matches(self):
        # Targets written as a module suffix keep working (dotted-path boundary).
        self.assertTrue(
            check_equal_or_regex_match(
                "model.layers.0.self_attn.kv_b_proj", ["self_attn.kv_b_proj"]
            )
        )
        self.assertTrue(check_equal_or_regex_match("model.lm_head", ["lm_head"]))
        # ... but only on a boundary, never mid-token.
        self.assertFalse(
            check_equal_or_regex_match("model.layers.0.gate_proj", ["ate"])
        )
        self.assertFalse(
            check_equal_or_regex_match(
                "model.layers.0.mlp.shared_expert_gate", ["gate"]
            )
        )

    def test_exact_and_regex_targets_unchanged(self):
        self.assertTrue(check_equal_or_regex_match(GATE_PROJ, [GATE_PROJ]))
        self.assertTrue(check_equal_or_regex_match(GATE_PROJ, ["re:.*gate_proj$"]))
        self.assertFalse(check_equal_or_regex_match(GATE_PROJ, ["re:.*down_proj$"]))


class TestFusedIgnoreLinearWrapper(CustomTestCase):
    """Fused shard names must honor ignores written for the CT wrapper."""

    def test_all_wrapped_shards_are_ignored(self):
        ignore = [
            f"{VISION_PREFIX}.0.self_attn.{proj}.linear"
            for proj in ("q_proj", "k_proj", "v_proj")
        ]

        self.assertTrue(
            should_ignore_layer(
                VISION_QKV_LAYER,
                ignore=ignore,
                fused_mapping=VISION_FUSED_MAPPING,
            )
        )

    def test_partial_wrapped_shards_extend_inconsistent_scheme_error(self):
        """Wrapped shards join the consistency check, so a partial list now raises.

        Before this change the wrapped entry matched nothing, every shard agreed
        on "not ignored", and the layer loaded quantized. The error boundary is
        extended to the wrapper spelling, not preserved unchanged.
        """
        ignore = [f"{VISION_PREFIX}.0.mlp.gate_proj.linear"]

        with self.assertRaisesRegex(ValueError, "different quantization schemes"):
            should_ignore_layer(
                VISION_GATE_UP_LAYER,
                ignore=ignore,
                fused_mapping=VISION_FUSED_MAPPING,
            )

    def test_mixed_spelling_shards_resolve_to_ignored(self):
        """A list mixing plain and wrapped shard spellings is now consistent.

        Before this change the plain entry matched and the wrapped ones did not,
        which raised the inconsistent-scheme error. Both spellings now match, so
        the layer is ignored instead of aborting the load.
        """
        ignore = [f"{VISION_PREFIX}.0.self_attn.q_proj"] + [
            f"{VISION_PREFIX}.0.self_attn.{proj}.linear"
            for proj in ("k_proj", "v_proj")
        ]

        self.assertTrue(
            should_ignore_layer(
                VISION_QKV_LAYER,
                ignore=ignore,
                fused_mapping=VISION_FUSED_MAPPING,
            )
        )

    def test_wrapped_tower_list_is_honored_at_runtime_names(self):
        """The whole wrapped list, queried at the names the tower really has.

        The two fused names need the retry; the two unfused ones are already
        nested at ".linear" by ClippableRowParallelLinear and match directly.
        """
        ignore = _wrapped_tower_ignore_list()
        self.assertEqual(len(ignore), VISION_NUM_LAYERS * 7)

        for idx in range(VISION_NUM_LAYERS):
            for layer_name in (
                f"{VISION_PREFIX}.{idx}.self_attn.qkv_proj",
                f"{VISION_PREFIX}.{idx}.mlp.gate_up_proj",
                f"{VISION_PREFIX}.{idx}.self_attn.o_proj.linear",
                f"{VISION_PREFIX}.{idx}.mlp.down_proj.linear",
            ):
                self.assertTrue(
                    should_ignore_layer(
                        layer_name,
                        ignore=ignore,
                        fused_mapping=VISION_FUSED_MAPPING,
                    ),
                    layer_name,
                )

    def test_wrapped_tower_list_leaves_the_text_model_quantized(self):
        """A tower-only ignore list must not reach the language model."""
        ignore = _wrapped_tower_ignore_list()

        for layer_name in (
            f"{TEXT_PREFIX}.self_attn.qkv_proj",
            f"{TEXT_PREFIX}.mlp.gate_up_proj",
            f"{TEXT_PREFIX}.self_attn.o_proj.linear",
            f"{TEXT_PREFIX}.mlp.down_proj.linear",
            "lm_head",
        ):
            self.assertFalse(
                should_ignore_layer(
                    layer_name,
                    ignore=ignore,
                    fused_mapping=VISION_FUSED_MAPPING,
                ),
                layer_name,
            )

    def test_intentionally_quantized_tower_keeps_its_scheme(self):
        config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {"group_0": WNA16_GROUP},
                "ignore": [],
            }
        )
        config.update_packed_modules_mapping(VISION_FUSED_MAPPING)

        scheme = config.get_scheme_dict(torch.nn.Linear(1, 1), VISION_GATE_UP_LAYER)

        self.assertIsNotNone(
            scheme, "an empty ignore list must not force the tower to bf16"
        )

    def test_parent_ignore_does_not_swallow_fused_children(self):
        """The no-prefix invariant, in both spellings of the parent entry."""
        for entry in (f"{VISION_PREFIX}.0.mlp", f"{VISION_PREFIX}.0.mlp.linear"):
            self.assertFalse(
                should_ignore_layer(
                    VISION_GATE_UP_LAYER,
                    ignore=[entry],
                    fused_mapping=VISION_FUSED_MAPPING,
                ),
                entry,
            )

    def test_retry_stays_keyed_on_the_ignore_list(self):
        """Only the wrapper suffix, only one level, only listed modules."""
        for entry in (
            "some.other.module.linear",
            f"{VISION_PREFIX}.0.self_attn.q_proj.linear.linear",
            f"{VISION_PREFIX}.0.self_attn.q_proj.weight",
        ):
            self.assertFalse(
                should_ignore_layer(
                    VISION_QKV_LAYER,
                    ignore=[entry],
                    fused_mapping=VISION_FUSED_MAPPING,
                ),
                entry,
            )

    def test_class_name_target_spelling_stays_inert(self):
        """Configs name the class as "Linear", and matching is case-sensitive."""
        for layer_name in (
            VISION_QKV_LAYER,
            VISION_O_PROJ_LAYER,
            VISION_DOWN_PROJ_LAYER,
        ):
            self.assertFalse(
                should_ignore_layer(
                    layer_name,
                    ignore=["Linear"],
                    fused_mapping=VISION_FUSED_MAPPING,
                ),
                layer_name,
            )

    def test_bare_linear_token_reaches_fused_names_only(self):
        """A bare "linear" entry is a suffix target, so it now reaches fused names.

        It cannot reach further than that: unfused names that are not already
        nested at ".linear" are matched exactly as before.
        """
        self.assertTrue(
            should_ignore_layer(
                VISION_QKV_LAYER,
                ignore=["linear"],
                fused_mapping=VISION_FUSED_MAPPING,
            )
        )
        for layer_name in (f"{VISION_PREFIX}.0.self_attn.o_proj", "lm_head"):
            self.assertFalse(
                should_ignore_layer(
                    layer_name,
                    ignore=["linear"],
                    fused_mapping=VISION_FUSED_MAPPING,
                ),
                layer_name,
            )


class TestMixedPrecisionFormat(CustomTestCase):
    """The per-group `format` must win over a top-level "mixed-precision"."""

    def test_input_activations_survive_mixed_precision(self):
        config = _mixed_precision_config(FP8_GROUP, NVFP4_GROUP)
        quant_config = CompressedTensorsConfig.from_config(config)

        for target, scheme in quant_config.target_scheme_map.items():
            self.assertIsNotNone(
                scheme["input_activations"],
                f"input_activations dropped for target {target}",
            )

        expert_scheme = quant_config.target_scheme_map[NVFP4_TARGET]
        self.assertEqual(expert_scheme["format"], "nvfp4-pack-quantized")
        self.assertEqual(expert_scheme["weights"].num_bits, 4)
        self.assertEqual(expert_scheme["input_activations"].num_bits, 4)

    def test_linear_scheme_uses_per_group_format(self):
        # WNA16 is selected only when the format is "pack-quantized". Reading the
        # top-level format instead of the matched group's would see
        # "mixed-precision" and resolve no scheme at all.
        config = _mixed_precision_config(WNA16_GROUP, NVFP4_GROUP)
        quant_config = CompressedTensorsConfig.from_config(config)

        with mock.patch.object(
            CompressedTensorsConfig, "_check_scheme_supported", return_value=True
        ):
            scheme = quant_config.get_linear_scheme(
                torch.nn.Module(), layer_name=MLP_LAYER
            )

        self.assertIsInstance(scheme, CompressedTensorsWNA16)
        self.assertEqual(scheme.pack_factor, 32 // 4)
        self.assertEqual(scheme.strategy, "group")
        self.assertEqual(scheme.group_size, 128)


# Layer-0 excerpt of cyankiwi/GLM-4.5-Air-AWQ-4bit's ignore list. Glm4Moe declares
# no packed_modules_mapping; AFMoE and DeepSeek-V3 AWQ repos use the same names.
GLM_IGNORE = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
]

# Layer-0/2 excerpt of cyankiwi/LFM2-8B-A1B-AWQ-4bit's ignore list.
LFM2_IGNORE = [
    "model.layers.0.conv.in_proj",
    "model.layers.0.feed_forward.w1",
    "model.layers.0.feed_forward.w3",
    "model.layers.0.feed_forward.w2",
    "model.layers.2.self_attn.q_proj",
    "model.layers.2.self_attn.k_proj",
    "model.layers.2.self_attn.v_proj",
    "model.layers.2.self_attn.out_proj",
]


class TestFusedIgnoreWithoutModelMapping(CustomTestCase):
    """Ignore lists name unfused shards even when the model declares no mapping."""

    def test_fused_layers_follow_their_ignored_shards(self):
        """qkv_proj / gate_up_proj are ignored exactly when all their shards are."""
        for name in ("self_attn.qkv_proj", "mlp.gate_up_proj"):
            with self.subTest(name=name):
                self.assertTrue(
                    should_ignore_layer(f"model.layers.0.{name}", ignore=GLM_IGNORE)
                )
                self.assertFalse(
                    should_ignore_layer(f"model.layers.1.{name}", ignore=GLM_IGNORE)
                )

    def test_direct_match_on_a_real_fused_tensor_wins(self):
        """A checkpoint that stores a real qkv_proj tensor can ignore it by name."""
        for ignore in (
            ["model.layers.0.self_attn.qkv_proj"],
            ["re:.*self_attn\\.qkv_proj$"],
        ):
            with self.subTest(ignore=ignore):
                self.assertTrue(
                    should_ignore_layer(
                        "model.layers.0.self_attn.qkv_proj", ignore=ignore
                    )
                )

    def test_partially_ignored_fused_layer_raises(self):
        """Ignoring only some shards of a fused layer is rejected, not half-loaded."""
        with self.assertRaisesRegex(ValueError, "same scheme"):
            should_ignore_layer(
                "model.layers.0.self_attn.qkv_proj", ignore=GLM_IGNORE[:2]
            )

    def test_fused_regex_matching_some_shards_raises(self):
        """`re:.*v_proj$` also matches qkv_proj, but q/k stay quantized, so it raises."""
        with self.assertRaisesRegex(ValueError, "same scheme"):
            should_ignore_layer(
                "model.layers.0.self_attn.qkv_proj",
                ignore=["re:.*v_proj$"],
                fused_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
            )

    def test_fused_regex_decides_when_the_model_declares_no_fusion(self):
        """MiMoV2 Pro stores one fused qkv_proj, so `re:.*v_proj$` ignores it whole."""
        self.assertTrue(
            should_ignore_layer(
                "model.layers.0.self_attn.qkv_proj", ignore=["re:.*v_proj$"]
            )
        )

    def test_literal_fused_name_wins_over_a_partial_regex(self):
        """A real fused tensor named literally stays ignored even if a regex hits one shard."""
        self.assertTrue(
            should_ignore_layer(
                "visual.blocks.0.attn.qkv_proj",
                ignore=["visual.blocks.0.attn.qkv_proj", "re:.*v_proj$"],
            )
        )

    @staticmethod
    def _lfm2_scheme(ignore):
        """Scheme lookup with LFM2-MoE's class mapping and mapper, as the loader applies them."""
        from sglang.srt.models.lfm2_moe import Lfm2MoeForCausalLM

        quant_config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {"group_0": {**WNA16_GROUP, "targets": ["Linear"]}},
                "ignore": list(ignore),
            }
        )
        quant_config.update_packed_modules_mapping(
            Lfm2MoeForCausalLM.packed_modules_mapping
        )
        quant_config.apply_weight_name_mapper(Lfm2MoeForCausalLM.hf_to_sglang_mapper)
        return lambda name: quant_config.get_scheme_dict(torch.nn.Linear(1, 1), name)

    def test_lfm2_moe_ignore_list_reaches_its_dense_layers(self):
        """LFM2-MoE's w1/w3/w2 names leave its ignored dense layers unquantized."""
        scheme = self._lfm2_scheme(LFM2_IGNORE)
        for name in (
            "model.layers.0.feed_forward.gate_up_proj",
            "model.layers.0.feed_forward.down_proj",
            "model.layers.2.self_attn.qkv_proj",
        ):
            with self.subTest(name=name):
                self.assertIsNone(scheme(name))
        for name in (
            "model.layers.1.feed_forward.gate_up_proj",
            "model.layers.6.self_attn.qkv_proj",
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(scheme(name))

    def test_lfm2_moe_partial_qkv_regex_raises(self):
        """LFM2-MoE stores split q/k/v, so `re:.*v_proj$` alone is a mixed scheme."""
        scheme = self._lfm2_scheme(["re:.*v_proj$"])
        with self.assertRaisesRegex(ValueError, "same scheme"):
            scheme("model.layers.2.self_attn.qkv_proj")

    @staticmethod
    def _lfm2_qkv_scheme(*groups, quant_format="mixed-precision"):
        """Scheme of layer 2's qkv_proj with per-shard exact targets."""
        from sglang.srt.models.lfm2_moe import Lfm2MoeForCausalLM

        quant_config = CompressedTensorsConfig.from_config(
            {**_mixed_precision_config(*groups), "format": quant_format}
        )
        quant_config.update_packed_modules_mapping(
            Lfm2MoeForCausalLM.packed_modules_mapping
        )
        return quant_config.get_scheme_dict(
            torch.nn.Linear(1, 1), "model.layers.2.self_attn.qkv_proj"
        )

    def test_lfm2_moe_qkv_shards_with_different_schemes_raise(self):
        """FP8 q/k with INT4 v is rejected instead of building v with q's scheme."""
        attn = "model.layers.2.self_attn"
        with self.assertRaisesRegex(ValueError, "same scheme"):
            self._lfm2_qkv_scheme(
                {**FP8_GROUP, "targets": [f"{attn}.q_proj", f"{attn}.k_proj"]},
                {**WNA16_GROUP, "targets": [f"{attn}.v_proj"]},
            )

    def test_lfm2_moe_qkv_shards_listed_separately_share_a_scheme(self):
        """One group naming q, k and v as three exact targets still resolves."""
        attn = "model.layers.2.self_attn"
        scheme = self._lfm2_qkv_scheme(
            {
                **WNA16_GROUP,
                "targets": [f"{attn}.{p}" for p in ("q_proj", "k_proj", "v_proj")],
            },
            FP8_GROUP,
        )
        self.assertEqual(scheme["weights"].num_bits, 4)

    def test_lfm2_moe_qkv_shards_with_an_inherited_format_share_a_scheme(self):
        """q/k inheriting pack-quantized match a v that sets it explicitly."""
        attn = "model.layers.2.self_attn"
        inherited = {k: v for k, v in WNA16_GROUP.items() if k != "format"}
        scheme = self._lfm2_qkv_scheme(
            {**inherited, "targets": [f"{attn}.q_proj", f"{attn}.k_proj"]},
            {**WNA16_GROUP, "targets": [f"{attn}.v_proj"]},
            quant_format="pack-quantized",
        )
        self.assertEqual(scheme["weights"].num_bits, 4)

    def test_lfm2_moe_qkv_shards_differing_only_in_observer_share_a_scheme(self):
        """The calibration observer is not loaded, so it does not split a scheme."""
        attn = "model.layers.2.self_attn"
        mse = {**WNA16_GROUP["weights"], "observer": "mse"}
        scheme = self._lfm2_qkv_scheme(
            {**WNA16_GROUP, "targets": [f"{attn}.q_proj", f"{attn}.k_proj"]},
            {**WNA16_GROUP, "weights": mse, "targets": [f"{attn}.v_proj"]},
        )
        self.assertEqual(scheme["weights"].num_bits, 4)

    def test_lfm2_moe_fused_name_entry_still_wins_over_its_shards(self):
        """An entry naming the fused gate_up_proj keeps working once w1/w3 are declared."""
        scheme = self._lfm2_scheme(["layers.0.feed_forward.gate_up_proj"])
        self.assertIsNone(scheme("model.layers.0.feed_forward.gate_up_proj"))
        self.assertIsNotNone(scheme("model.layers.1.feed_forward.gate_up_proj"))

    def test_lfm2_moe_regex_w2_entry_reaches_down_proj(self):
        """A regex written against w2 matches the renamed dense down_proj."""
        scheme = self._lfm2_scheme(["re:.*layers\\.0\\.feed_forward\\.w2$"])
        self.assertIsNone(scheme("model.layers.0.feed_forward.down_proj"))
        self.assertIsNotNone(scheme("model.layers.1.feed_forward.down_proj"))


def _declared_fused_mappings():
    """(model, fused, shards) for every literal packed_modules_mapping entry.

    Read with ast so the sweep needs no model imports. Mappings built at
    runtime (Qwen3.5, Glm5Next, ...) are left out, and so are entries whose
    shards are not plain leaf names: Solar's (name, shard_id) tuples are read
    only by its own load_weights, and dotted keys (Gemma3n's ".qkv_proj") are
    never found by the leaf-name lookup, so the fallback table covers them.
    """
    import sglang

    models_dir = Path(sglang.__file__).parent / "srt" / "models"
    entries = []
    for path in sorted(models_dir.rglob("*.py")):
        for cls in ast.walk(ast.parse(path.read_text())):
            if not isinstance(cls, ast.ClassDef):
                continue
            for stmt in cls.body:
                if not (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "packed_modules_mapping"
                ):
                    continue
                try:
                    mapping = ast.literal_eval(stmt.value)
                except ValueError:
                    continue
                for fused, shards in mapping.items():
                    names = [fused, *shards]
                    if all(isinstance(n, str) and "." not in n for n in names):
                        entries.append((f"{path.stem}.{cls.name}", fused, shards))
    return entries


class TestDeclaredFusedMappingSweep(CustomTestCase):
    """Every model's declared fusion resolves ignore lists and schemes the same way."""

    PREFIX = "model.layers.0.blk"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.entries = _declared_fused_mappings()

    def _cases(self):
        for model, fused, shards in self.entries:
            with self.subTest(model=model, fused=fused):
                yield (
                    f"{self.PREFIX}.{fused}",
                    [f"{self.PREFIX}.{shard}" for shard in shards],
                    {fused: list(shards)},
                )

    def test_the_sweep_sees_the_common_fusions(self):
        """Guard against the ast reader silently finding nothing."""
        fused_names = {fused for _, fused, _ in self.entries}
        self.assertGreater(len(self.entries), 50)
        self.assertLessEqual({"qkv_proj", "gate_up_proj"}, fused_names)

    def test_ignore_list_follows_the_shards(self):
        """All shards listed: ignored. None: kept. The fused name itself: ignored."""
        for layer, shard_names, mapping in self._cases():
            self.assertTrue(
                should_ignore_layer(layer, ignore=shard_names, fused_mapping=mapping)
            )
            self.assertFalse(
                should_ignore_layer(layer, ignore=["lm_head"], fused_mapping=mapping)
            )
            self.assertTrue(
                should_ignore_layer(layer, ignore=[layer], fused_mapping=mapping)
            )

    def test_some_shards_ignored_raises(self):
        """A literal entry or a regex on one shard only is a mixed scheme, even
        when the regex also matches the fused name (`re:.*v_proj$` on qkv_proj)."""
        for layer, shard_names, mapping in self._cases():
            if len(shard_names) < 2:
                continue
            last = re.escape(shard_names[-1].rsplit(".", 1)[1])
            for ignore in (
                shard_names[:1],
                [f"re:.*\\.{last}$"],
                [f"re:.*{last}$"],
            ):
                with self.assertRaisesRegex(ValueError, "same scheme"):
                    should_ignore_layer(layer, ignore=ignore, fused_mapping=mapping)

    @staticmethod
    def _scheme(mapping, *groups):
        quant_config = CompressedTensorsConfig.from_config(
            _mixed_precision_config(*groups)
        )
        quant_config.update_packed_modules_mapping(mapping)
        return quant_config

    def test_shard_targets_resolve_to_one_scheme(self):
        """Per-shard exact targets: one scheme loads, two different schemes raise."""
        for layer, shard_names, mapping in self._cases():
            linear = torch.nn.Linear(1, 1)
            mse = {**WNA16_GROUP["weights"], "observer": "mse"}
            same = self._scheme(
                mapping,
                {**WNA16_GROUP, "targets": shard_names[:1]},
                {**WNA16_GROUP, "weights": mse, "targets": shard_names[1:]},
            )
            self.assertEqual(same.get_scheme_dict(linear, layer)["weights"].num_bits, 4)
            if len(shard_names) < 2:
                continue
            mixed = self._scheme(
                mapping,
                {**FP8_GROUP, "targets": shard_names[:-1]},
                {**WNA16_GROUP, "targets": shard_names[-1:]},
            )
            with self.assertRaisesRegex(ValueError, "same scheme"):
                mixed.get_scheme_dict(linear, layer)


class TestCheckSchemeSupportedError(CustomTestCase):
    def test_error_is_one_message(self):
        """The unsupported-capability error reads as a sentence, not a tuple repr."""
        quant_config = CompressedTensorsConfig.__new__(CompressedTensorsConfig)
        with (
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 0)),
            self.assertRaises(RuntimeError) as ctx,
        ):
            quant_config._check_scheme_supported(min_capability=90)
        self.assertEqual(len(ctx.exception.args), 1, ctx.exception.args)
        self.assertIn("Min capability: 90. Current capability: 70.", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
