"""CPU tests: a MoE router (Qwen3-MoE ``mlp.gate``, Mixtral ``block_sparse_moe.gate``) is built as the checkpoint
stores it.

A checkpoint may store the router quantized (RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16: W4A16
``weight_packed``/``weight_scale``; anyisalin/Mixtral-8x7B-Instruct-v0.1-FP8-D: FP8 codes with a ``weight_scale``)
or in BF16, either listed in the config's ``ignore`` (most AWQ exports) or not (AngelSlim/Qwen3-30B-A3B_fp8_static,
amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV). ``router_quant_config`` builds the router with the config for the formats
that state per layer how it is stored, and ``unquantize_router_if_stored_so`` rebuilds it unquantized in
``load_weights`` when the checkpoint stores a 16- or 32-bit float ``weight`` for a router built quantized.

The scheme tests repeat the decision the model loader and ``CompressedTensorsConfig`` make; the rebuild tests replace
the parallel Linear layer and the unquantized method with stand-ins, so they need no distributed setup.
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
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.srt.models.utils import router_quant_config, unquantize_router_if_stored_so
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
    """Stands in for the layer class: the "Linear" target matches by class name. Built with a config, its
    weight is FP8 (as a quantized method allocates it); without one, BF16."""

    def __init__(
        self,
        input_size,
        output_size,
        bias=False,
        params_dtype=None,
        quant_config=None,
        prefix="",
    ):
        super().__init__()
        self.input_size, self.output_size, self.params_dtype = (
            input_size,
            output_size,
            params_dtype,
        )
        self.quant_method = _Quantized() if quant_config is not None else _Unquantized()
        dtype = torch.float8_e4m3fn if quant_config is not None else params_dtype
        self.weight = torch.nn.Parameter(
            torch.zeros(output_size, input_size, dtype=dtype), requires_grad=False
        )
        self.bias = None


class _Quantized:
    pass


class _Unquantized:
    pass


def _scheme(config, layer_name):
    with mock.patch.object(
        CompressedTensorsConfig, "_check_scheme_supported", return_value=True
    ):
        return config.get_linear_scheme(ReplicatedLinear(4, 8), layer_name=layer_name)


class _FakeQuantConfig:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class TestRouterQuantConfig(CustomTestCase):
    def test_quantized_router_is_built_quantized(self):
        # As in RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16.
        config = _w4a16_config(["lm_head"])

        self.assertIs(router_quant_config(config), config)
        for i in range(LAYERS):
            with self.subTest(layer=i):
                scheme = _scheme(config, ROUTER.format(i))
                self.assertIsInstance(scheme, CompressedTensorsWNA16)
                self.assertEqual(scheme.group_size, 128)

    def test_ignored_router_stays_unquantized(self):
        # As in cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit: every layer's
        # router is listed in ignore and stored as a BF16 weight.
        config = _w4a16_config([ROUTER.format(i) for i in range(LAYERS)] + ["lm_head"])

        self.assertIs(router_quant_config(config), config)
        for i in range(LAYERS):
            with self.subTest(layer=i):
                self.assertIsNone(_scheme(config, ROUTER.format(i)))
                self.assertIsInstance(
                    _scheme(config, ATTN_OUT.format(i)), CompressedTensorsWNA16
                )

    def test_formats_that_state_the_router(self):
        for name in ("modelslim", "compressed_tensors", "fp8"):
            with self.subTest(format=name):
                config = _FakeQuantConfig(name)
                self.assertIs(router_quant_config(config), config)

    def test_other_formats_keep_the_router_unquantized(self):
        for name in ("mxfp8", "awq", "gptq", "gguf", "modelopt_fp4"):
            with self.subTest(format=name):
                self.assertIsNone(router_quant_config(_FakeQuantConfig(name)))
        self.assertIsNone(router_quant_config(None))


class TestUnquantizeRouterIfStoredSo(CustomTestCase):
    def _model(self, quantized=True):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.model.layers[0].mlp = torch.nn.Module()
        model.model.layers[0].mlp.gate = ReplicatedLinear(
            16,
            8,
            params_dtype=torch.bfloat16,
            quant_config=object() if quantized else None,
        )
        return model, dict(model.named_parameters())

    def _call(self, model, params_dict, name, weight):
        with (
            mock.patch("sglang.srt.layers.linear.ReplicatedLinear", ReplicatedLinear),
            mock.patch(
                "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod",
                _Unquantized,
            ),
        ):
            unquantize_router_if_stored_so(
                model, name, weight, params_dict, (".mlp.gate",)
            )

    def test_router_stored_unquantized_is_rebuilt_unquantized(self):
        # As in AngelSlim/Qwen3-30B-A3B_fp8_static: the config quantizes every Linear,
        # but the checkpoint stores the router in BF16.
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                model, params_dict = self._model()
                name = ROUTER.format(0) + ".weight"
                self._call(model, params_dict, name, torch.ones(8, 16, dtype=dtype))
                gate = model.model.layers[0].mlp.gate
                self.assertIsInstance(gate.quant_method, _Unquantized)
                self.assertEqual(gate.weight.dtype, torch.bfloat16)
                self.assertIs(params_dict[name], gate.weight)

    def test_router_stored_quantized_is_kept(self):
        model, params_dict = self._model()
        gate = model.model.layers[0].mlp.gate
        name = ROUTER.format(0) + ".weight"
        self._call(
            model, params_dict, name, torch.zeros(8, 16, dtype=torch.float8_e4m3fn)
        )
        self.assertIs(model.model.layers[0].mlp.gate, gate)

    def test_other_weights_and_unquantized_routers_are_untouched(self):
        model, params_dict = self._model(quantized=False)
        gate = model.model.layers[0].mlp.gate
        self._call(
            model,
            params_dict,
            ROUTER.format(0) + ".weight",
            torch.ones(8, 16, dtype=torch.bfloat16),
        )
        self.assertIs(model.model.layers[0].mlp.gate, gate)
        model, params_dict = self._model()
        gate = model.model.layers[0].mlp.gate
        self._call(
            model,
            params_dict,
            ATTN_OUT.format(0) + ".weight",
            torch.ones(8, 16, dtype=torch.bfloat16),
        )
        self._call(
            model, params_dict, ROUTER.format(0) + ".weight_scale", torch.ones(8, 1)
        )
        self.assertIs(model.model.layers[0].mlp.gate, gate)


if __name__ == "__main__":
    unittest.main()
