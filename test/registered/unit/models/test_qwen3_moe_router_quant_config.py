"""CPU tests: the Qwen3-MoE router (``mlp.gate``) is built with the quantization
config whenever the config says, per layer, how the checkpoint stores it.

A compressed-tensors checkpoint may store the router quantized
(RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16 ignores only ``lm_head``
and stores ``mlp.gate.weight_packed``/``weight_scale``/``weight_shape``) or in
BF16, with the router named in ``ignore`` (as most AWQ exports do). A router
built without the config has only ``weight``, so a quantized router's tensors
have no parameter, are skipped, and the router keeps its initial weight.

The tests repeat the decision the model loader and ``CompressedTensorsConfig``
make: the loader passes the class's ``packed_modules_mapping`` into the config,
and the config resolves a scheme (or none, for an ignored layer) by prefix.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16,
)
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM, _router_quant_config
from sglang.test.test_utils import CustomTestCase

LAYERS = 2
ROUTER = "model.layers.{}.mlp.gate"
ATTN_OUT = "model.layers.{}.self_attn.o_proj"


def _w4a16_config(ignore):
    """A W4A16 group-128 compressed-tensors config, as the loader builds it."""
    return CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "targets": ["Linear"],
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
            },
            "ignore": list(ignore),
            "packed_modules_mapping": dict(Qwen3MoeForCausalLM.packed_modules_mapping),
        }
    )


class ReplicatedLinear(torch.nn.Module):
    """Stands in for the layer class: the "Linear" target matches by class name."""


def _scheme(config, layer_name):
    with mock.patch.object(
        CompressedTensorsConfig, "_check_scheme_supported", return_value=True
    ):
        return config.get_linear_scheme(ReplicatedLinear(), layer_name=layer_name)


class _FakeQuantConfig:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class TestQwen3MoeRouterQuantConfig(CustomTestCase):
    def test_quantized_router_is_built_quantized(self):
        # As in RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16.
        config = _w4a16_config(["lm_head"])

        self.assertIs(_router_quant_config(config), config)
        for i in range(LAYERS):
            with self.subTest(layer=i):
                scheme = _scheme(config, ROUTER.format(i))
                self.assertIsInstance(scheme, CompressedTensorsWNA16)
                self.assertEqual(scheme.group_size, 128)

    def test_ignored_router_stays_unquantized(self):
        # As in cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit: every layer's
        # router is listed in ignore and stored as a BF16 weight.
        config = _w4a16_config([ROUTER.format(i) for i in range(LAYERS)] + ["lm_head"])

        self.assertIs(_router_quant_config(config), config)
        for i in range(LAYERS):
            with self.subTest(layer=i):
                self.assertIsNone(_scheme(config, ROUTER.format(i)))
                self.assertIsInstance(
                    _scheme(config, ATTN_OUT.format(i)), CompressedTensorsWNA16
                )

    def test_modelslim_router_keeps_the_config(self):
        config = _FakeQuantConfig("modelslim")

        self.assertIs(_router_quant_config(config), config)

    def test_other_formats_keep_the_router_unquantized(self):
        for name in ("fp8", "mxfp8", "awq", "gptq", "modelopt_fp4"):
            with self.subTest(format=name):
                self.assertIsNone(_router_quant_config(_FakeQuantConfig(name)))
        self.assertIsNone(_router_quant_config(None))


if __name__ == "__main__":
    unittest.main()
