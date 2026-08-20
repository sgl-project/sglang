# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.quantization.auto_round import (
    AutoRoundConfig,
    AutoRoundMxfp4MoEWNA16Method,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod


class TestAutoRoundMXFPConfig(unittest.TestCase):
    def _mixed_mxfp4_moe_config(self):
        return AutoRoundConfig.from_config(
            {
                "bits": 8,
                "data_type": "mx_fp",
                "group_size": 32,
                "sym": True,
                "quant_method": "auto-round",
                "packing_format": "auto_round:llm_compressor",
                "extra_config": {
                    "model.layers.1.mlp.experts.0.gate_proj": {"bits": 4},
                    "model.layers.1.mlp.experts.0.up_proj": {"bits": 4},
                    "model.layers.1.mlp.experts.0.down_proj": {"bits": 4},
                },
            }
        )

    def test_autoround_mxfp8_keeps_autoround_dispatch(self):
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
            AutoRoundConfig.override_quantization_method(config, None), "auto-round"
        )

        native_config = AutoRoundConfig.to_native_mxfp_config(config)
        quant_config = AutoRoundConfig.from_config(native_config)

        self.assertEqual(native_config["quant_method"], "auto-round")
        self.assertTrue(native_config["_auto_round_mxfp"])
        self.assertFalse(native_config["_auto_round_mxfp_mixed"])
        self.assertTrue(quant_config.use_mxfp8)
        self.assertEqual(quant_config.weight_block_size, [1, 32])
        self.assertEqual(quant_config.activation_scheme, "dynamic")

    def test_autoround_mxfp4_keeps_autoround_dispatch(self):
        config = {
            "bits": 4,
            "data_type": "mx_fp",
            "group_size": 32,
            "sym": True,
            "quant_method": "auto-round",
            "packing_format": "auto_round:llm_compressor",
        }

        self.assertEqual(
            AutoRoundConfig.override_quantization_method(config, None), "auto-round"
        )

        native_config = AutoRoundConfig.to_native_mxfp_config(config)
        quant_config = AutoRoundConfig.from_config(native_config)

        self.assertEqual(native_config["quant_method"], "auto-round")
        self.assertTrue(native_config["_auto_round_mxfp"])
        self.assertFalse(native_config["_auto_round_mxfp_mixed"])
        self.assertFalse(quant_config.use_mxfp8)
        self.assertIsNone(quant_config.weight_block_size)

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

    def test_autoround_mixed_mxfp_keeps_autoround_dispatch(self):
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
            "extra_config": {
                "model.layers.1.mlp.experts.0.gate_proj": {"bits": 4},
                "model.layers.1.mlp.experts.0.up_proj": {"bits": 4},
                "model.layers.1.mlp.experts.0.down_proj": {"bits": 4},
                ".*model\\.layers\\.1\\.mlp\\.gate.*": {
                    "bits": 16,
                    "data_type": "float",
                    "act_bits": 16,
                    "act_data_type": "float",
                },
            },
        }

        self.assertTrue(AutoRoundConfig.is_mixed_mxfp_config(config))
        self.assertEqual(
            AutoRoundConfig.override_quantization_method(config, None), "auto-round"
        )

        native_config = AutoRoundConfig.to_native_mxfp_config(config)
        self.assertEqual(native_config["quant_method"], "auto-round")
        self.assertTrue(native_config["_auto_round_mxfp_mixed"])

    def test_autoround_mixed_mxfp_layer_config(self):
        class DummyFusedMoE:
            pass

        class DummyLinear:
            pass

        config = {
            "bits": 8,
            "data_type": "mx_fp",
            "group_size": 32,
            "sym": True,
            "quant_method": "auto-round",
            "packing_format": "auto_round:llm_compressor",
            "block_name_to_quantize": "model.layers",
            "extra_config": {
                "model.layers.1.mlp.experts.0.gate_proj": {"bits": 4},
                "model.layers.1.mlp.experts.0.up_proj": {"bits": 4},
                "model.layers.1.mlp.experts.0.down_proj": {"bits": 4},
                "model.layers.1.mlp.shared_experts.gate_proj": {"bits": 4},
                "model.layers.1.mlp.shared_experts.up_proj": {"bits": 4},
                "model.layers.1.mlp.shared_experts.down_proj": {"bits": 4},
                ".*model\\.layers\\.1\\.mlp\\.gate.*": {
                    "bits": 16,
                    "data_type": "float",
                    "act_bits": 16,
                    "act_data_type": "float",
                },
            },
        }
        quant_config = AutoRoundConfig.from_config(config)

        self.assertEqual(
            quant_config.get_layer_config(
                DummyFusedMoE(), "model.layers.1.mlp.experts"
            ),
            (4, 32, True),
        )
        self.assertEqual(
            quant_config.get_layer_config(
                DummyLinear(), "model.layers.1.mlp.shared_experts.gate_up_proj"
            ),
            (4, 32, True),
        )
        self.assertEqual(
            quant_config.get_layer_config(
                DummyLinear(), "model.layers.1.mlp.gate"
            ),
            (16, -1, True),
        )

    def test_autoround_mixed_mxfp4_moe_uses_native_w4a8_method(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        quant_config = self._mixed_mxfp4_moe_config()
        moe_layer = FusedMoE.__new__(FusedMoE)

        exec_cfg = SimpleNamespace(
            moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
        )
        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_MXFP4,
            ),
            patch("sglang.srt.utils.is_sm90_supported", return_value=False),
            patch("sglang.srt.utils.is_sm100_supported", return_value=True),
            patch("sglang.srt.utils.is_sm120_supported", return_value=False),
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_MXFP4,
            ),
            patch("sglang.srt.layers.quantization.mxfp4.get_exec") as get_exec,
        ):
            get_exec.return_value = exec_cfg
            quant_method = quant_config.get_quant_method(
                moe_layer, "model.layers.1.mlp.experts"
            )

        self.assertIsInstance(quant_method, Mxfp4MoEMethod)

    def test_autoround_mxfp4_moe_falls_back_on_sm90_flashinfer(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        quant_config = self._mixed_mxfp4_moe_config()
        moe_layer = FusedMoE.__new__(FusedMoE)

        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_MXFP4,
            ),
            patch("sglang.srt.utils.is_sm90_supported", return_value=True),
            patch("sglang.srt.utils.is_sm100_supported", return_value=False),
            patch("sglang.srt.utils.is_sm120_supported", return_value=False),
        ):
            quant_method = quant_config.get_quant_method(
                moe_layer, "model.layers.1.mlp.experts"
            )

        self.assertIsInstance(quant_method, AutoRoundMxfp4MoEWNA16Method)

    def test_autoround_mxfp4_moe_falls_back_for_weight_only_backend(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        quant_config = self._mixed_mxfp4_moe_config()
        moe_layer = FusedMoE.__new__(FusedMoE)

        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=MoeRunnerBackend.MARLIN,
            ),
            patch("sglang.srt.utils.is_sm90_supported", return_value=False),
            patch("sglang.srt.utils.is_sm100_supported", return_value=True),
            patch("sglang.srt.utils.is_sm120_supported", return_value=False),
        ):
            quant_method = quant_config.get_quant_method(
                moe_layer, "model.layers.1.mlp.experts"
            )

        self.assertIsInstance(quant_method, AutoRoundMxfp4MoEWNA16Method)

    def test_autoround_mxfp4_moe_falls_back_on_unsupported_arch(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        quant_config = self._mixed_mxfp4_moe_config()
        moe_layer = FusedMoE.__new__(FusedMoE)

        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_MXFP4,
            ),
            patch("sglang.srt.utils.is_sm90_supported", return_value=False),
            patch("sglang.srt.utils.is_sm100_supported", return_value=False),
            patch("sglang.srt.utils.is_sm120_supported", return_value=False),
        ):
            quant_method = quant_config.get_quant_method(
                moe_layer, "model.layers.1.mlp.experts"
            )

        self.assertIsInstance(quant_method, AutoRoundMxfp4MoEWNA16Method)


if __name__ == "__main__":
    unittest.main()
