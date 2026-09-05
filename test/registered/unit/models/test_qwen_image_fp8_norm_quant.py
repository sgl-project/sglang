"""Unit tests for Qwen-Image ModelOpt FP8 norm+quant activation gates."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8LinearMethod,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformerBlock,
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


def _attention(*, fused: bool, scales: tuple[float, ...]) -> SimpleNamespace:
    if fused:
        return SimpleNamespace(
            use_fused_qkv=True,
            to_qkv=_fp8_linear(scales[0]),
            added_kv_proj_dim=None,
        )
    return SimpleNamespace(
        use_fused_qkv=False,
        to_q=_fp8_linear(scales[0]),
        to_k=_fp8_linear(scales[1]),
        to_v=_fp8_linear(scales[2]),
        added_kv_proj_dim=None,
    )


def _block(attn: SimpleNamespace) -> QwenImageTransformerBlock:
    block = object.__new__(QwenImageTransformerBlock)
    nn.Module.__init__(block)
    block.dim = 3072
    block.zero_cond_t = False
    block.attn = attn
    block.img_mlp = None
    block.txt_mlp = None
    block._fp8_img_attn_norm_quant = False
    block._fp8_txt_attn_norm_quant = False
    block._fp8_img_mlp_norm_quant = False
    block._fp8_txt_mlp_norm_quant = False
    return block


@patch("torch.cuda.get_device_capability", return_value=(10, 0))
@patch("torch.cuda.is_available", return_value=True)
class TestQwenImageFp8NormQuantGate(CustomTestCase):
    def test_separate_qkv_requires_identical_input_scales(
        self, _is_available, _capability
    ) -> None:
        matching = _block(_attention(fused=False, scales=(0.25, 0.25, 0.25)))
        mismatched = _block(_attention(fused=False, scales=(0.25, 0.5, 0.25)))

        matching.configure_fp8_norm_quant()
        mismatched.configure_fp8_norm_quant()

        self.assertTrue(matching._fp8_img_attn_norm_quant)
        self.assertFalse(mismatched._fp8_img_attn_norm_quant)

    def test_merged_qkv_uses_its_materialized_input_scale(
        self, _is_available, _capability
    ) -> None:
        block = _block(_attention(fused=True, scales=(0.25,)))

        block.configure_fp8_norm_quant()

        self.assertTrue(block._fp8_img_attn_norm_quant)

    def test_nonpositive_scale_keeps_fusion_disabled(
        self, _is_available, _capability
    ) -> None:
        block = _block(_attention(fused=True, scales=(0.0,)))

        block.configure_fp8_norm_quant()

        self.assertFalse(block._fp8_img_attn_norm_quant)


if __name__ == "__main__":
    unittest.main()
