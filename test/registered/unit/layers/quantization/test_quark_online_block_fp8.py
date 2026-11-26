"""Unit tests for Quark's load-time block FP8 of excluded BF16 linears — CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.quark import online_block_fp8
from sglang.srt.layers.quantization.quark.online_block_fp8 import (
    QuarkOnlineBlockFp8LinearMethod,
    online_block_fp8_applies,
)
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.test_utils import CustomTestCase

_KIMI_EXCLUDES = [
    "model.layers.1.self_attn.q_a_proj",
    "model.layers.1.self_attn.kv_a_proj_with_mqa",
    "model.layers.1.self_attn.q_b_proj",
    "model.layers.1.self_attn.kv_b_proj",
    "model.layers.1.self_attn.o_proj",
    "model.layers.1.mlp.gate",
    "model.layers.1.mlp.shared_experts.gate_proj",
    "model.layers.1.mlp.shared_experts.up_proj",
    "model.layers.1.mlp.shared_experts.down_proj",
]


def _bare_config() -> QuarkConfig:
    config = QuarkConfig.__new__(QuarkConfig)
    config.exclude_layers = list(_KIMI_EXCLUDES)
    config.packed_modules_mapping = {
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    config.excluded_fp8_config = None
    config.kv_cache_group = []
    config._online_quantized_layers = set()
    return config


def _dequant(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return qweight.float() * expanded[: qweight.shape[0], : qweight.shape[1]]


def _fake_prepare(weight, weight_scale, block_size):
    return weight, weight_scale


def _fake_linear(x, weight, block_size, weight_scale, *args, **kwargs):
    return (x.float() @ _dequant(weight, weight_scale).T).to(x.dtype)


class TestOnlineBlockFp8Selection(CustomTestCase):
    def _applies(self, prefix: str, enabled: bool = True, aiter: bool = True) -> bool:
        with (
            envs.SGLANG_QUARK_DENSE_FP8.override(enabled),
            patch.object(fp8_utils, "_use_aiter_mxfp8_gfx1250", aiter),
        ):
            return online_block_fp8_applies(prefix)

    def test_default_targets(self):
        for prefix in (
            "model.layers.1.self_attn.fused_qkv_a_proj_with_mqa",
            "model.layers.1.self_attn.q_b_proj",
            "model.layers.1.self_attn.o_proj",
            "model.layers.1.mlp.shared_experts.gate_up_proj",
            "model.layers.1.mlp.shared_experts.down_proj",
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.0.mlp.down_proj",
        ):
            self.assertTrue(self._applies(prefix), prefix)

    def test_router_kv_b_and_lm_head_stay_bf16(self):
        for prefix in (
            "model.layers.1.mlp.gate",
            "model.layers.1.self_attn.kv_b_proj",
            "lm_head",
        ):
            self.assertFalse(self._applies(prefix), prefix)

    def test_off_by_default_and_without_gfx1250_aiter(self):
        prefix = "model.layers.1.self_attn.o_proj"
        self.assertFalse(self._applies(prefix, enabled=False))
        self.assertFalse(self._applies(prefix, aiter=False))

    def test_quark_routes_excluded_linears(self):
        config = _bare_config()
        layer = LinearBase.__new__(LinearBase)
        with (
            envs.SGLANG_QUARK_DENSE_FP8.override(True),
            patch.object(fp8_utils, "_use_aiter_mxfp8_gfx1250", True),
        ):
            fused = config.get_quant_method(
                layer, "model.layers.1.self_attn.fused_qkv_a_proj_with_mqa"
            )
            kv_b = config.get_quant_method(layer, "model.layers.1.self_attn.kv_b_proj")
        self.assertIsInstance(fused, QuarkOnlineBlockFp8LinearMethod)
        self.assertIs(type(kv_b), UnquantizedLinearMethod)


class TestOnlineBlockFp8Weights(CustomTestCase):
    def _convert(self, n: int, k: int) -> tuple[torch.nn.Module, torch.Tensor]:
        torch.manual_seed(0)
        layer = torch.nn.Module()
        weight = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
        layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        with patch.object(
            online_block_fp8, "gfx1250_mxfp8_prepare_weight", _fake_prepare
        ):
            QuarkOnlineBlockFp8LinearMethod().process_weights_after_loading(layer)
        return layer, weight

    def test_n_is_padded_to_whole_blocks_and_output_narrowed(self):
        layer, weight = self._convert(n=2112, k=256)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(layer.weight.shape), (2176, 256))
        self.assertEqual(tuple(layer.weight_scale_inv.shape), (17, 2))
        self.assertEqual(layer.dense_fp8_out_features, 2112)
        mantissa, _ = torch.frexp(layer.weight_scale_inv)
        self.assertTrue(bool((mantissa == 0.5).all()))

        x = torch.randn(5, 256, dtype=torch.bfloat16)
        with patch.object(
            online_block_fp8, "aiter_mxfp8_gfx1250_block_fp8_linear", _fake_linear
        ):
            out = QuarkOnlineBlockFp8LinearMethod().apply(layer, x)
        ref = (x.float() @ weight.float().T).to(torch.bfloat16)
        self.assertEqual(tuple(out.shape), (5, 2112))
        torch.testing.assert_close(out.float(), ref.float(), atol=0.1, rtol=0.1)

    def test_k_not_multiple_of_128_stays_bf16(self):
        layer, weight = self._convert(n=256, k=200)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertFalse(hasattr(layer, "dense_fp8_out_features"))


if __name__ == "__main__":
    unittest.main()
