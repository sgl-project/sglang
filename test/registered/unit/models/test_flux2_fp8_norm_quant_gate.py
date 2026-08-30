"""Unit tests for FLUX.2 ModelOpt FP8 norm+quant activation gates."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8LinearMethod,
)
from sglang.multimodal_gen.runtime.models.dits.flux_2 import (
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _fp8_linear(input_scale: float) -> nn.Module:
    linear = nn.Module()
    linear.quant_method = object.__new__(ModelOptFp8LinearMethod)
    linear.register_parameter(
        "input_scale",
        nn.Parameter(
            torch.tensor(input_scale, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        ),
    )
    return linear


def _double_block(scales: tuple[float, float, float]) -> Flux2TransformerBlock:
    block = object.__new__(Flux2TransformerBlock)
    nn.Module.__init__(block)
    block.attn = SimpleNamespace(
        use_fused_qkv=False,
        use_fused_added_qkv=False,
        to_q=_fp8_linear(scales[0]),
        to_k=_fp8_linear(scales[1]),
        to_v=_fp8_linear(scales[2]),
        add_q_proj=_fp8_linear(scales[0]),
        add_k_proj=_fp8_linear(scales[1]),
        add_v_proj=_fp8_linear(scales[2]),
    )
    block.ff = SimpleNamespace(linear_in=_fp8_linear(scales[0]))
    block.ff_context = SimpleNamespace(linear_in=_fp8_linear(scales[0]))
    block._fp8_img_attn_norm_quant = False
    block._fp8_txt_attn_norm_quant = False
    block._fp8_img_ff_norm_quant = False
    block._fp8_txt_ff_norm_quant = False
    return block


class TestFlux2Fp8NormQuantGate(CustomTestCase):
    def test_qkv_requires_identical_input_scales(self) -> None:
        matching = _double_block((0.25, 0.25, 0.25))
        mismatched = _double_block((0.25, 0.5, 0.25))

        matching.configure_fp8_norm_quant()
        mismatched.configure_fp8_norm_quant()

        self.assertTrue(matching._fp8_img_attn_norm_quant)
        self.assertTrue(matching._fp8_txt_attn_norm_quant)
        self.assertFalse(mismatched._fp8_img_attn_norm_quant)
        self.assertFalse(mismatched._fp8_txt_attn_norm_quant)

    def test_single_block_uses_merged_projection_scale(self) -> None:
        block = object.__new__(Flux2SingleTransformerBlock)
        nn.Module.__init__(block)
        block.attn = SimpleNamespace(to_qkv_mlp_proj=_fp8_linear(0.25))
        block._fp8_norm_quant = False

        block.configure_fp8_norm_quant()

        self.assertTrue(block._fp8_norm_quant)

    def test_nonpositive_scale_keeps_fusion_disabled(self) -> None:
        block = _double_block((0.0, 0.0, 0.0))

        block.configure_fp8_norm_quant()

        self.assertFalse(block._fp8_img_attn_norm_quant)
        self.assertFalse(block._fp8_img_ff_norm_quant)


if __name__ == "__main__":
    unittest.main()
