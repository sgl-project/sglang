"""Unit tests for dequantize_layers_to_bf16 in srt/layers/quantization/dequantization.py."""

import unittest

import torch

from sglang.srt.layers.quantization.dequantization import (
    dequantize_layers_to_bf16,
    dequantize_nvfp4,
)
from sglang.srt.layers.quantization.mxfp8_block_convert import (
    dequant_mxfp8_2d_to_bf16,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The checkpoint prefix the step dequantizes, as Qwen4-Exp names its vision tower
VISION = "model.visual."

# Layers of each recipe, keyed as FP4-Local's quantized_layers keys them
ATTENTION_QKV = "model.visual.blocks.0.attn.qkv"
MLP_FC1 = "model.visual.blocks.0.mlp.linear_fc1"
MTP_EXPERT = "mtp.layers.0.mlp.experts.0.gate_proj"

# The recipe of each layer above
RECIPES = {
    ATTENTION_QKV: "MXFP8",
    MLP_FC1: "W4A16_NVFP4",
    MTP_EXPERT: "W4A16_NVFP4",
}


def _quant_algo_of(prefix: str):
    """The recipe RECIPES gives PREFIX, or None for a BF16 layer."""
    return RECIPES.get(prefix)


def _mxfp8_parts(rows: int = 8, columns: int = 64) -> dict:
    """An MXFP8 weight and its UE8M0 scale, one byte per 32 weights."""
    generator = torch.Generator().manual_seed(rows * columns)
    weight = torch.randn(rows, columns, generator=generator).to(torch.float8_e4m3fn)
    scale = torch.randint(
        120, 134, (rows, columns // 32), dtype=torch.uint8, generator=generator
    )
    return {"weight": weight, "weight_scale": scale}


def _nvfp4_parts(rows: int = 8, columns: int = 48) -> dict:
    """A W4A16_NVFP4 weight, its E4M3 scale per 16 weights and its global scale."""
    generator = torch.Generator().manual_seed(rows * columns)
    weight = torch.randint(
        0, 256, (rows, columns // 2), dtype=torch.uint8, generator=generator
    )
    scale = (torch.rand(rows, columns // 16, generator=generator) + 0.5).to(
        torch.float8_e4m3fn
    )
    return {
        "weight": weight,
        "weight_scale": scale,
        "weight_scale_2": torch.tensor(0.25),
    }


def _named(layer: str, parts: dict) -> list:
    """The checkpoint (name, tensor) pairs of LAYER's PARTS."""
    return [(f"{layer}.{part}", tensor) for part, tensor in parts.items()]


def _dequantized(weights: list) -> dict:
    """The step's output for WEIGHTS, by name."""
    return dict(
        dequantize_layers_to_bf16(weights, prefix=VISION, quant_algo_of=_quant_algo_of)
    )


class TestDequantizeLayersToBf16(CustomTestCase):
    def test_mxfp8_layer_becomes_its_bf16_weight(self):
        parts = _mxfp8_parts()
        bias = torch.randn(8, dtype=torch.bfloat16)
        expected = dequant_mxfp8_2d_to_bf16(parts["weight"], parts["weight_scale"])
        # #41624's rename hands the scale over as weight_scale_inv.
        for scale_name in ("weight_scale", "weight_scale_inv"):
            with self.subTest(scale_name=scale_name):
                weights = [
                    (f"{ATTENTION_QKV}.bias", bias),
                    (f"{ATTENTION_QKV}.weight", parts["weight"]),
                    (f"{ATTENTION_QKV}.{scale_name}", parts["weight_scale"]),
                ]
                result = _dequantized(weights)
                self.assertEqual(
                    sorted(result), [f"{ATTENTION_QKV}.bias", f"{ATTENTION_QKV}.weight"]
                )
                self.assertEqual(
                    result[f"{ATTENTION_QKV}.weight"].dtype, torch.bfloat16
                )
                self.assertTrue(
                    torch.equal(result[f"{ATTENTION_QKV}.weight"], expected)
                )
                self.assertIs(result[f"{ATTENTION_QKV}.bias"], bias)

    def test_mxfp8_scale_is_a_power_of_two(self):
        weight = torch.full((1, 32), 1.5).to(torch.float8_e4m3fn)
        scale = torch.full((1, 1), 129, dtype=torch.uint8)
        weights = _named(ATTENTION_QKV, {"weight": weight, "weight_scale": scale})
        result = _dequantized(weights)[f"{ATTENTION_QKV}.weight"]
        self.assertTrue(torch.equal(result, torch.full((1, 32), 6.0).bfloat16()))

    def test_w4a16_nvfp4_layer_becomes_its_bf16_weight(self):
        parts = _nvfp4_parts()
        expected = dequantize_nvfp4(
            parts["weight"], parts["weight_scale"], parts["weight_scale_2"]
        )
        # The parts arrive in any order; the layer is yielded once all are in.
        weights = list(reversed(_named(MLP_FC1, parts)))
        result = _dequantized(weights)
        self.assertEqual(list(result), [f"{MLP_FC1}.weight"])
        self.assertEqual(result[f"{MLP_FC1}.weight"].shape, (8, 48))
        self.assertTrue(torch.equal(result[f"{MLP_FC1}.weight"], expected))

    def test_tensors_outside_the_prefix_and_bf16_tensors_pass_through(self):
        norm = torch.randn(8, dtype=torch.bfloat16)
        weights = _named(MTP_EXPERT, _nvfp4_parts()) + [
            ("model.visual.blocks.0.norm1.weight", norm)
        ]
        result = list(
            dequantize_layers_to_bf16(
                weights, prefix=VISION, quant_algo_of=_quant_algo_of
            )
        )
        self.assertEqual(len(result), len(weights))
        for (name, tensor), (result_name, result_tensor) in zip(weights, result):
            self.assertEqual(result_name, name)
            self.assertIs(result_tensor, tensor)

    def test_incomplete_layer_raises(self):
        parts = _nvfp4_parts()
        del parts["weight_scale_2"]
        with self.assertRaisesRegex(ValueError, "missing parts"):
            _dequantized(_named(MLP_FC1, parts))


if __name__ == "__main__":
    unittest.main()
