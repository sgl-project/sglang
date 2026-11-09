import re
import unittest

from sglang.srt.layers.quantization.utils import (
    is_layer_skipped,
    rename_excluded_modules,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Config:
    """Stands in for a quantization config: only the exclusion lists matter."""

    def __init__(self, **lists):
        for name, value in lists.items():
            setattr(self, name, value)


FUSED = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}


class TestRenameExcludedModules(CustomTestCase):
    def test_none_stays_none(self):
        self.assertIsNone(rename_excluded_modules(None, str.upper))

    def test_renames_a_copy(self):
        cfg = _Config(ignored_layers=["midlayer.self_attn.q_proj"], other=["midlayer"])
        out = rename_excluded_modules(cfg, lambda n: n.replace("midlayer", "layers.0"))
        self.assertEqual(out.ignored_layers, ["layers.0.self_attn.q_proj"])
        self.assertEqual(out.other, ["midlayer"])  # not an exclusion list
        self.assertEqual(cfg.ignored_layers, ["midlayer.self_attn.q_proj"])

    def test_every_exclusion_attribute(self):
        names = ["midlayer.mlp.down_proj"]
        cfg = _Config(
            ignore=list(names),
            exclude_modules=list(names),
            modules_to_not_convert=list(names),
        )
        out = rename_excluded_modules(cfg, lambda n: n.replace("midlayer", "layers.0"))
        for attr in ("ignore", "exclude_modules", "modules_to_not_convert"):
            self.assertEqual(getattr(out, attr), ["layers.0.mlp.down_proj"])

    def test_regex_entries_are_kept(self):
        cfg = _Config(ignore=["re:.*midlayer.*", "midlayer.mlp.down_proj"])
        out = rename_excluded_modules(cfg, lambda n: n.replace("midlayer", "layers.0"))
        self.assertEqual(out.ignore, ["re:.*midlayer.*", "layers.0.mlp.down_proj"])

    def test_eagle3_draft_layer_is_skipped(self):
        # An EAGLE-3 checkpoint that names its draft layer `midlayer` excludes it
        # under that name; SGLang builds it as model.layers.0.
        cfg = _Config(
            ignored_layers=[
                f"midlayer.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj")
            ]
        )
        out = rename_excluded_modules(cfg, lambda n: n.replace("midlayer", "layers.0"))
        prefix = "model.layers.0.self_attn.qkv_proj"
        self.assertFalse(is_layer_skipped(prefix, cfg.ignored_layers, FUSED))
        self.assertTrue(is_layer_skipped(prefix, out.ignored_layers, FUSED))

    def test_mla_eagle3_draft_layer_is_skipped(self):
        # The MLA EAGLE-3 draft is the reverse: the checkpoint says layers.0 and
        # SGLang builds model.midlayer.
        cfg = _Config(ignored_layers=["layers.0.mlp.down_proj"])
        out = rename_excluded_modules(
            cfg, lambda n: re.sub(r"^layers\.0\.", "midlayer.", n)
        )
        self.assertTrue(
            is_layer_skipped("model.midlayer.mlp.down_proj", out.ignored_layers, {})
        )
        self.assertFalse(
            is_layer_skipped("model.midlayer.mlp.down_proj", cfg.ignored_layers, {})
        )


if __name__ == "__main__":
    unittest.main()
