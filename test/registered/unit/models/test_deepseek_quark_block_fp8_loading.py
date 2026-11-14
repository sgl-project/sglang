"""Regression tests for Quark block-FP8 weights in the DeepSeek loader."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
    _load_fused_indexer_wk,
    _resolve_layer_weight_block_size,
    _resolve_quark_fp8_block_scale_name,
)
from sglang.test.test_utils import CustomTestCase


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight):
        self.loaded = loaded_weight


class _FakeDeepseekLoader(DeepseekV2WeightLoaderMixin):
    def __init__(self, params):
        self.config = SimpleNamespace(
            n_routed_experts=0,
            q_lora_rank=2048,
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
        )
        self.model = SimpleNamespace()
        self.quant_config = SimpleNamespace(get_name=lambda: "quark")
        self.num_fused_shared_experts = 0
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self._params = params
        self.post_load_called = False

    def named_parameters(self):
        return iter(self._params.items())

    def _maybe_quant_weights_to_fp8_ue8m0(
        self, weights, attn_quant_modules, nextn_conf
    ):
        return weights

    def post_load_weights(self, is_nextn=False, weight_names=None):
        self.post_load_called = True


class TestQuarkBlockFp8WeightLoading(CustomTestCase):
    def test_scale_name_maps_only_when_scale_inv_is_registered(self):
        scale_inv_name = (
            "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv"
        )
        params_dict = {scale_inv_name: torch.empty(1)}
        checkpoint_name = scale_inv_name.removesuffix("_inv")

        self.assertEqual(
            _resolve_quark_fp8_block_scale_name(checkpoint_name, params_dict),
            scale_inv_name,
        )

        mxfp4_name = "model.layers.0.mlp.experts.w13_weight_scale"
        params_dict = {mxfp4_name: torch.empty(1)}
        self.assertEqual(
            _resolve_quark_fp8_block_scale_name(mxfp4_name, params_dict),
            mxfp4_name,
        )

    def test_layer_quant_method_precedes_config_level_summary(self):
        layer = SimpleNamespace(
            quant_method=SimpleNamespace(weight_block_size=[64, 128])
        )
        quant_config = SimpleNamespace(
            linear_fp8_config=SimpleNamespace(weight_block_size=[128, 128]),
            weight_block_size=None,
        )

        self.assertEqual(
            _resolve_layer_weight_block_size(layer, quant_config),
            [64, 128],
        )

        layer.quant_method.weight_block_size = None
        self.assertIsNone(_resolve_layer_weight_block_size(layer, quant_config))

    def test_fused_qkv_a_quark_scales_load_into_scale_inv(self):
        target_name = (
            "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv"
        )
        target = _FakeParam()
        loader = _FakeDeepseekLoader({target_name: target})
        q_scale = torch.ones((2, 3), dtype=torch.float32)
        kv_scale = torch.full((1, 3), 2.0, dtype=torch.float32)

        loader.do_load_weights(
            [
                (
                    "model.layers.0.self_attn.q_a_proj.weight_scale",
                    q_scale,
                ),
                (
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale",
                    kv_scale,
                ),
            ]
        )

        torch.testing.assert_close(
            target.loaded,
            torch.cat([q_scale, kv_scale], dim=0),
        )
        self.assertTrue(loader.post_load_called)

    def test_unfused_quark_scale_loads_into_scale_inv(self):
        target_name = "model.layers.0.self_attn.o_proj.weight_scale_inv"
        target = _FakeParam()
        loader = _FakeDeepseekLoader({target_name: target})
        loaded_scale = torch.ones((2, 3), dtype=torch.float32)

        loader.do_load_weights(
            [
                (
                    "model.layers.0.self_attn.o_proj.weight_scale",
                    loaded_scale,
                ),
            ]
        )

        self.assertIs(target.loaded, loaded_scale)
        self.assertTrue(loader.post_load_called)

    def test_quark_weight_scale_is_not_misclassified_as_indexer_weight(self):
        fused_name = "model.layers.0.self_attn.indexer.wk_weights_proj.weight"
        fused_param = torch.nn.Parameter(
            torch.zeros((4, 4), dtype=torch.bfloat16),
            requires_grad=False,
        )
        params_dict = {fused_name: fused_param}
        pending = {}
        loaded_weight = torch.ones((2, 4), dtype=torch.float8_e4m3fn)
        loaded_scale = torch.ones((1, 1), dtype=torch.float32)
        dequantized = torch.ones((2, 4), dtype=torch.bfloat16)

        with patch(
            "sglang.srt.models.deepseek_common.deepseek_weight_loader."
            "block_quant_dequant",
            return_value=dequantized,
        ) as block_quant_dequant:
            self.assertTrue(
                _load_fused_indexer_wk(
                    "model.layers.0.self_attn.indexer.wk.weight",
                    loaded_weight,
                    params_dict,
                    pending,
                    None,
                )
            )
            self.assertTrue(
                _load_fused_indexer_wk(
                    "model.layers.0.self_attn.indexer.wk.weight_scale",
                    loaded_scale,
                    params_dict,
                    pending,
                    None,
                )
            )

        block_quant_dequant.assert_called_once_with(
            loaded_weight,
            loaded_scale,
            [128, 128],
            torch.bfloat16,
        )
        torch.testing.assert_close(fused_param[:2], dequantized)
        self.assertEqual(pending, {})


if __name__ == "__main__":
    unittest.main()
