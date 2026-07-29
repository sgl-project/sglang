"""CPU unit tests for wiring a model's packed_modules_mapping into its quant config.

Quantized checkpoints name the unfused projections (``q_proj``/``k_proj``/
``v_proj``) while sglang builds fused modules (``qkv_proj``). Quantization
configs bridge the two through ``packed_modules_mapping``, which
``QuantizationConfig`` documents as "updated by models as they initialize" --
but only 2 of the 39 models declaring the attribute actually assigned it, so for
everything else the mapping stayed empty and a fused module could not be matched
against the checkpoint's targets ("Unable to find matching target ...").

``_initialize_model`` now calls ``configure_quant_config`` for every model and
every loader. These tests are pure attribute plumbing plus one end-to-end match
through the compressed-tensors matcher, so they run on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import find_matched_target
from sglang.srt.model_loader.utils import configure_quant_config
from sglang.test.test_utils import CustomTestCase

QKV_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


class _ModelWithMapping:
    packed_modules_mapping = QKV_MAPPING


class _ModelWithoutMapping:
    pass


def _make_fp8_config(packed_modules_mapping=None):
    """FP8 W8A8 config whose targets name the *unfused* attention projections."""
    config = {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["re:.*self_attn\\.(q|k|v)_proj$"],
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
        },
        "ignore": [],
    }
    if packed_modules_mapping is not None:
        config["packed_modules_mapping"] = packed_modules_mapping
    return config


class TestConfigureQuantConfig(CustomTestCase):
    def test_mapping_is_copied_from_model_class(self):
        quant_config = CompressedTensorsConfig.from_config(_make_fp8_config())
        self.assertEqual(quant_config.packed_modules_mapping, {})

        configure_quant_config(quant_config, _ModelWithMapping)
        self.assertEqual(quant_config.packed_modules_mapping, QKV_MAPPING)

    def test_checkpoint_mapping_wins_and_is_extended(self):
        # An override supplied through config.json must survive, while keys it
        # does not mention are still filled in from the model class.
        override = {"qkv_proj": ["q_proj"]}
        quant_config = CompressedTensorsConfig.from_config(_make_fp8_config(override))

        configure_quant_config(quant_config, _ModelWithMapping)
        self.assertEqual(quant_config.packed_modules_mapping["qkv_proj"], ["q_proj"])
        self.assertEqual(
            quant_config.packed_modules_mapping["gate_up_proj"],
            ["gate_proj", "up_proj"],
        )

    def test_no_quant_config_or_no_mapping_is_a_noop(self):
        configure_quant_config(None, _ModelWithMapping)  # must not raise

        quant_config = CompressedTensorsConfig.from_config(_make_fp8_config())
        configure_quant_config(quant_config, _ModelWithoutMapping)
        self.assertEqual(quant_config.packed_modules_mapping, {})

    def test_fused_layer_resolves_against_unfused_targets(self):
        quant_config = CompressedTensorsConfig.from_config(_make_fp8_config())
        fused_layer_name = "model.layers.0.self_attn.qkv_proj"
        targets = quant_config.target_scheme_map.keys()

        # Without the mapping the fused module matches nothing.
        with self.assertRaises(ValueError):
            find_matched_target(
                layer_name=fused_layer_name,
                module=torch.nn.Module(),
                targets=targets,
                fused_mapping=quant_config.packed_modules_mapping,
            )

        configure_quant_config(quant_config, _ModelWithMapping)
        matched = find_matched_target(
            layer_name=fused_layer_name,
            module=torch.nn.Module(),
            targets=targets,
            fused_mapping=quant_config.packed_modules_mapping,
        )
        self.assertEqual(matched, "re:.*self_attn\\.(q|k|v)_proj$")


if __name__ == "__main__":
    unittest.main()
