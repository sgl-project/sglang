import unittest

import torch

from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.srt.models.utils import WeightsMapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAutoRoundConfigMapping(unittest.TestCase):
    def config(self, **kwargs):
        return AutoRoundConfig.from_config(
            dict(bits=4, group_size=128, sym=True, **kwargs)
        )

    def test_packed_module_bit_widths(self):
        prefix = "model.layers.0.linear_attn."
        config = self.config(
            packed_modules_mapping={
                "in_proj_ba": ["in_proj_b", "in_proj_a"],
                "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
            },
            extra_config={
                prefix + "in_proj_a": {"bits": 16},
                prefix + "in_proj_b": {"bits": 16},
                prefix + "in_proj_qkv": {"bits": 8},
                prefix + "in_proj_z": {"bits": 8},
            },
        )
        layer = torch.nn.Linear(1, 1)
        self.assertEqual(config.get_layer_config(layer, prefix + "in_proj_ba")[0], 16)
        self.assertEqual(config.get_layer_config(layer, prefix + "in_proj_qkvz")[0], 8)

    def test_inconsistent_packed_shards_raise(self):
        config = self.config(
            packed_modules_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
            extra_config={"model.q_proj": {"bits": 8}},
        )
        with self.assertRaisesRegex(ValueError, "consistent quant config"):
            config.get_layer_config(torch.nn.Linear(1, 1), "model.qkv_proj")

    def test_block_prefix_is_mapped(self):
        config = self.config(block_name_to_quantize=["model.language_model.layers"])
        config.apply_weight_name_mapper(
            WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
        )
        self.assertEqual(config.block_name_to_quantize, ["model.layers"])
        self.assertEqual(
            config.get_layer_config(
                torch.nn.Linear(1, 1), "model.layers.3.mlp.experts"
            )[0],
            4,
        )
        self.assertEqual(
            config.get_layer_config(torch.nn.Linear(1, 1), "other.layers.0.mlp")[0],
            16,
        )

    def test_extra_config_and_fusion_are_mapped_together(self):
        config = self.config(
            block_name_to_quantize="model.language_model.layers",
            packed_modules_mapping={"gate_up_proj": ["gate_proj", "up_proj"]},
            extra_config={
                "model.language_model.layers.0.mlp.gate_proj": {"bits": 8},
                "model.language_model.layers.0.mlp.up_proj": {"bits": 8},
            },
        )
        config.apply_weight_name_mapper(
            WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
        )
        self.assertEqual(
            config.get_layer_config(
                torch.nn.Linear(1, 1), "model.layers.0.mlp.gate_up_proj"
            )[0],
            8,
        )

    def test_defaults_and_empty_fields(self):
        for extra_config in (None, {}):
            with self.subTest(extra_config=extra_config):
                config = self.config(extra_config=extra_config)
                config.apply_weight_name_mapper(WeightsMapper())
                self.assertIsNone(config.block_name_to_quantize)
                self.assertEqual(config.extra_config, extra_config)
                self.assertEqual(config.packed_modules_mapping, {})
                self.assertEqual(
                    config.get_layer_config(torch.nn.Linear(1, 1), "model.q_proj"),
                    (4, 128, True),
                )

    def test_mapper_deletion_and_values_preserved(self):
        settings = {"bits": 8, "group_size": 32, "sym": False}
        config = self.config(
            block_name_to_quantize=["drop", "old.layers"],
            extra_config={"drop": {"bits": 16}, "old.layers.proj": settings},
        )
        config.apply_weight_name_mapper(
            WeightsMapper(orig_to_new_prefix={"drop": None, "old.": "new."})
        )
        self.assertEqual(config.block_name_to_quantize, ["new.layers"])
        self.assertEqual(list(config.extra_config), ["new.layers.proj"])
        self.assertIs(config.extra_config["new.layers.proj"], settings)


if __name__ == "__main__":
    unittest.main()
