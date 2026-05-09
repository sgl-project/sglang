"""Unit tests for compressed-tensors quantization scheme selection."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_asymmetric_int4_pack_quantized_config():
    return {
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": None,
                "output_activations": None,
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": 32,
                    "num_bits": 4,
                    "observer": "mse",
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "group",
                    "symmetric": False,
                    "type": "int",
                    "zp_dtype": "torch.int8",
                },
            }
        },
    }


class TestCompressedTensorsWNA16SchemeSelection(CustomTestCase):
    def _get_linear_scheme_dict(self):
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        quant_config = CompressedTensorsConfig.from_config(
            _make_asymmetric_int4_pack_quantized_config()
        )
        return quant_config, quant_config.target_scheme_map["Linear"]

    def test_asymmetric_int4_pack_quantized_uses_wna16(self):
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            CompressedTensorsWNA16,
        )

        quant_config, scheme_dict = self._get_linear_scheme_dict()

        scheme = quant_config._get_scheme_from_parts(
            weight_quant=scheme_dict["weights"],
            input_quant=scheme_dict["input_activations"],
        )

        self.assertIsInstance(scheme, CompressedTensorsWNA16)
        self.assertFalse(scheme.symmetric)
        self.assertEqual(scheme.group_size, 32)
        self.assertEqual(scheme.pack_factor, 8)

    def test_asymmetric_int4_wna16_is_not_enabled_by_default(self):
        quant_config, scheme_dict = self._get_linear_scheme_dict()

        self.assertFalse(
            quant_config._is_wNa16_group_channel(
                weight_quant=scheme_dict["weights"],
                input_quant=scheme_dict["input_activations"],
            )
        )


if __name__ == "__main__":
    unittest.main()
