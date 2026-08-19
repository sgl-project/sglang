# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config


class TestAutoRoundMXFPConfig(unittest.TestCase):
    def test_autoround_mxfp8_maps_to_native_fp8_config(self):
        config = {
            "bits": 8,
            "act_bits": 8,
            "data_type": "mx_fp",
            "act_data_type": "mx_fp",
            "group_size": 32,
            "act_group_size": 32,
            "sym": True,
            "act_sym": True,
            "act_dynamic": True,
            "quant_method": "auto-round",
            "packing_format": "auto_round:llm_compressor",
        }

        self.assertEqual(
            AutoRoundConfig.override_quantization_method(config, None), "mxfp8"
        )

        native_config = AutoRoundConfig.to_native_mxfp_config(config)
        quant_config = Fp8Config.from_config(native_config)

        self.assertEqual(native_config["quant_method"], "mxfp8")
        self.assertEqual(native_config["weight_block_size"], [1, 32])
        self.assertTrue(quant_config.use_mxfp8)
        self.assertTrue(quant_config.is_checkpoint_fp8_serialized)

    def test_autoround_mxfp4_maps_to_native_mxfp4_config(self):
        config = {
            "bits": 4,
            "data_type": "mx_fp",
            "group_size": 32,
            "sym": True,
            "quant_method": "auto-round",
            "packing_format": "auto_round:llm_compressor",
        }

        self.assertEqual(
            AutoRoundConfig.override_quantization_method(config, None), "mxfp4"
        )

        native_config = AutoRoundConfig.to_native_mxfp_config(config)
        quant_config = Mxfp4Config.from_config(native_config)

        self.assertEqual(native_config["quant_method"], "mxfp4")
        self.assertTrue(quant_config.is_checkpoint_mxfp4_serialized)

    def test_autoround_mxfp_rejects_non_native_group_size(self):
        config = {
            "bits": 8,
            "data_type": "mx_fp",
            "group_size": 64,
            "sym": True,
            "quant_method": "auto-round",
            "packing_format": "auto_round:llm_compressor",
        }

        with self.assertRaisesRegex(ValueError, "group_size=32"):
            AutoRoundConfig.from_config(config)


if __name__ == "__main__":
    unittest.main()
