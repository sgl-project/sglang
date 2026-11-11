"""Regression tests for quantized weights in the DeepSeek loader."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import ReplicatedLinear
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
    def __init__(self, params, q_lora_rank=2048):
        self.config = SimpleNamespace(
            n_routed_experts=0,
            q_lora_rank=q_lora_rank,
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


def _fp8(values):
    return torch.tensor(values, dtype=torch.float32).to(torch.float8_e4m3fn)


class TestFusedQkvAPerTensorScale(CustomTestCase):
    """fused_qkv_a_proj_with_mqa holds one per-tensor scale, so the loader
    loads the max of the q_a/kv_a scales and requantizes the other rows."""

    prefix = "model.layers.0.self_attn"
    fused = f"{prefix}.fused_qkv_a_proj_with_mqa"
    # Every value stays exactly representable after scaling by 1/4.
    q_weight = _fp8([[1.0, 2.0, -4.0, 0.5], [8.0, -1.0, 0.0, 4.0]])
    kv_weight = _fp8(
        [[1.0, -2.0, 3.0, 0.5], [0.25, 6.0, -8.0, 1.0], [2.0, 2.0, -0.5, 0.0]]
    )

    def _load(self, q_scale, kv_scale, q_input_scale, kv_input_scale):
        def param(*shape, dtype):
            p = torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
            # The fused layer is a ReplicatedLinear; keep its strict shape check.
            p.weight_loader = partial(ReplicatedLinear.weight_loader, None)
            return p

        params = {
            f"{self.fused}.weight": param(5, 4, dtype=torch.float8_e4m3fn),
            f"{self.fused}.weight_scale": param(1, dtype=torch.float32),
            f"{self.fused}.input_scale": param(1, dtype=torch.float32),
        }
        # Scales arrive before the weights to show the order does not matter.
        _FakeDeepseekLoader(params, q_lora_rank=2).do_load_weights(
            [
                (f"{self.prefix}.q_a_proj.weight_scale", q_scale),
                (f"{self.prefix}.kv_a_proj_with_mqa.weight_scale", kv_scale),
                (f"{self.prefix}.q_a_proj.input_scale", q_input_scale),
                (f"{self.prefix}.kv_a_proj_with_mqa.input_scale", kv_input_scale),
                (f"{self.prefix}.q_a_proj.weight", self.q_weight.clone()),
                (f"{self.prefix}.kv_a_proj_with_mqa.weight", self.kv_weight.clone()),
            ]
        )
        return params

    def test_per_tensor_scales_take_the_max_and_requantize(self):
        # compressed-tensors stores per-tensor scales as [1], others as 0-d.
        for wrap in (lambda v: torch.tensor([v]), torch.tensor):
            for q_scale, kv_scale in ((0.5, 2.0), (4.0, 1.0)):
                with self.subTest(
                    shape=tuple(wrap(0.0).shape), q_scale=q_scale, kv_scale=kv_scale
                ):
                    params = self._load(
                        wrap(q_scale), wrap(kv_scale), wrap(1.0), wrap(3.0)
                    )
                    fused_scale = params[f"{self.fused}.weight_scale"].data
                    torch.testing.assert_close(
                        fused_scale, torch.tensor([max(q_scale, kv_scale)])
                    )
                    torch.testing.assert_close(
                        params[f"{self.fused}.input_scale"].data, torch.tensor([3.0])
                    )
                    torch.testing.assert_close(
                        params[f"{self.fused}.weight"].float() * fused_scale,
                        torch.cat(
                            [
                                self.q_weight.float() * q_scale,
                                self.kv_weight.float() * kv_scale,
                            ]
                        ),
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
