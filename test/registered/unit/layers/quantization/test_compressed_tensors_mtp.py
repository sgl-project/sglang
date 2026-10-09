"""Unit tests for compressed-tensors MTP/NEXTN draft layer handling (Issue #38574)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from torch import nn

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import is_mtp_layer_name
from sglang.test.test_utils import CustomTestCase

_W4_WEIGHTS = {
    "num_bits": 4,
    "type": "int",
    "strategy": "group",
    "group_size": 128,
    "symmetric": True,
    "dynamic": False,
}
_FP8_DYNAMIC_ACTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "token",
    "symmetric": True,
    "dynamic": True,
}
_FP8_WEIGHTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "channel",
    "symmetric": True,
    "dynamic": False,
}


def _w4afp8_linear_target_config(ignore=()):
    return CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
            "ignore": list(ignore),
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": _W4_WEIGHTS,
                    "input_activations": _FP8_DYNAMIC_ACTS,
                }
            },
        }
    )


class TestCompressedTensorsMtpHandling(CustomTestCase):
    def test_is_mtp_layer_name(self):
        self.assertTrue(is_mtp_layer_name("mtp.layers.0.self_attn.qkv_proj"))
        self.assertTrue(is_mtp_layer_name("model.mtp.layers.0.mlp.gate_proj"))
        self.assertTrue(is_mtp_layer_name("model.mtp_layers.0.self_attn.o_proj"))
        self.assertTrue(is_mtp_layer_name("mtp_block.layers.0.fc"))
        self.assertTrue(is_mtp_layer_name("llm.mtp.layers.0.attn"))
        self.assertFalse(is_mtp_layer_name("model.layers.0.self_attn.qkv_proj"))
        self.assertFalse(is_mtp_layer_name("lm_head"))
        self.assertFalse(is_mtp_layer_name(None))

    def test_unmentioned_mtp_linear_layer_falls_back_to_unquantized(self):
        """MTP draft layers not explicitly named in targets or ignore must
        fall back to unquantized (None) instead of matching generic 'Linear'
        and raising NotImplementedError on W4AFP8 checkpoints."""
        config = _w4afp8_linear_target_config()
        layer = nn.Linear(16, 16)
        scheme = config.get_linear_scheme(
            layer, layer_name="mtp.layers.0.self_attn.qkv_proj"
        )
        self.assertIsNone(scheme)

    def test_unmentioned_mtp_moe_layer_falls_back_to_unquantized(self):
        config = _w4afp8_linear_target_config()
        layer = nn.Module()
        scheme = config.get_moe_scheme(layer, layer_name="mtp.layers.0.mlp.experts")
        self.assertIsNone(scheme)

    def test_explicitly_targeted_mtp_layer_resolves_scheme(self):
        """When a config_group explicitly targets mtp.* by regex or layer name,
        the MTP layer must resolve its quantization scheme."""
        config = CompressedTensorsConfig.from_config(
            {
                "format": "float-quantized",
                "quant_method": "compressed-tensors",
                "ignore": [],
                "config_groups": {
                    "group_0": {
                        "targets": ["re:^mtp\\..*"],
                        "weights": _FP8_WEIGHTS,
                        "input_activations": _FP8_DYNAMIC_ACTS,
                    }
                },
            }
        )
        layer = nn.Linear(16, 16)
        with patch.object(
            CompressedTensorsConfig, "_check_scheme_supported", return_value=True
        ):
            scheme = config.get_linear_scheme(
                layer, layer_name="mtp.layers.0.self_attn.qkv_proj"
            )
        self.assertIsInstance(scheme, CompressedTensorsW8A8Fp8)

    def test_unsupported_scheme_error_includes_layer_name_and_ignore_hint(self):
        """When _get_scheme_from_parts raises NotImplementedError, the message
        must name the failing layer and suggest quantization_config.ignore."""
        config = _w4afp8_linear_target_config()
        layer = nn.Linear(16, 16)
        with self.assertRaises(NotImplementedError) as ctx:
            config.get_linear_scheme(
                layer, layer_name="model.layers.0.self_attn.qkv_proj"
            )
        err_msg = str(ctx.exception)
        self.assertIn("model.layers.0.self_attn.qkv_proj", err_msg)
        self.assertIn("quantization_config.ignore", err_msg)


if __name__ == "__main__":
    unittest.main()
