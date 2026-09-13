"""Unit tests for GLM-5.3 NextN weight-name mapping."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.models.glm5_next_nextn import (
    Glm5NextForConditionalGenerationNextN,
)
from sglang.test.test_utils import CustomTestCase


class TestGlm5NextNextNWeightNameMapper(CustomTestCase):
    _NEXTN_PREFIXES = ("model.layers.45", "model.language_model.layers.45")

    def _mapper(self):
        config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=45))
        return Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper(config)

    def test_special_nextn_tensors_map_to_model(self):
        for prefix in self._NEXTN_PREFIXES:
            for tail, expected in (
                ("eh_proj.weight", "model.eh_proj.weight"),
                ("enorm.weight", "model.enorm.weight"),
                ("hnorm.weight", "model.hnorm.weight"),
                ("shared_head.norm.weight", "model.shared_head.norm.weight"),
            ):
                with self.subTest(prefix=prefix, tail=tail):
                    self.assertEqual(
                        self._mapper()._map_name(f"{prefix}.{tail}"), expected
                    )

    def test_decoder_tensors_map_to_model_decoder(self):
        mapper = self._mapper()
        cases = (
            (
                "model.layers.45.self_attn.q_b_proj.weight",
                "model.decoder.self_attn.q_b_proj.weight",
            ),
            (
                "model.language_model.layers.45.mlp.experts.0.up_proj.weight",
                "model.decoder.mlp.experts.0.up_proj.weight",
            ),
            ("model.layers.45.mlp.gate.weight", "model.decoder.mlp.gate.weight"),
            (
                "model.layers.45.mlp.shared_experts.down_proj.weight",
                "model.decoder.mlp.shared_experts.down_proj.weight",
            ),
            (
                "model.language_model.layers.45.mlp.shared_experts.down_proj.weight",
                "model.decoder.mlp.shared_experts.down_proj.weight",
            ),
        )
        self.assertEqual(
            mapper.apply_list([c[0] for c in cases]), [c[1] for c in cases]
        )

    def test_other_layer_names_are_unchanged(self):
        for name in (
            "model.layers.0.self_attn.q_b_proj.weight",
            "model.language_model.layers.0.self_attn.q_b_proj.weight",
            "model.layers.450.self_attn.q_b_proj.weight",
            "model.language_model.layers.450.self_attn.q_b_proj.weight",
            "model.embed_tokens.weight",
        ):
            with self.subTest(name=name):
                self.assertEqual(self._mapper()._map_name(name), name)

    def test_get_hf_to_sglang_mapper_uses_text_config(self):
        config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=45))
        mapper = Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper(config)
        self.assertEqual(
            mapper._map_name("model.language_model.layers.45.eh_proj.weight"),
            "model.eh_proj.weight",
        )


class TestGlm5NextNextNQuantConfigNameMapping(CustomTestCase):
    def _nextn_fp8_config(self):
        from sglang.srt.layers.quantization.quark.quark import QuarkConfig

        fp8 = {
            "weight": {"dtype": "fp8_e4m3", "qscheme": "per_block"},
            "input_tensors": {"dtype": "fp8_e4m3", "qscheme": "per_group"},
        }
        quant_config = QuarkConfig(
            quant_config={
                "packed_modules_mapping": {},
                "exclude": [
                    "model.language_model.layers.45.eh_proj",
                    "model.layers.45.mlp.gate",
                ],
                "layer_quant_config": {
                    "model.language_model.layers.45.mlp.experts.0.gate_proj": fp8,
                    "model.layers.45.self_attn.q_b_proj": fp8,
                },
            },
            is_prequantized=True,
        )
        return quant_config, fp8

    def test_quark_config_maps_both_checkpoint_prefix_forms(self):
        quant_config, fp8 = self._nextn_fp8_config()
        hf_config = SimpleNamespace(
            num_hidden_layers=45,
            num_nextn_predict_layers=1,
            quantization_config={},
        )
        mapper = Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper(
            hf_config,
        )
        quant_config.apply_weight_name_mapper(mapper)
        nextn = object.__new__(Glm5NextForConditionalGenerationNextN)
        resolved = nextn._resolve_nextn_quant_config(hf_config, quant_config)
        self.assertIs(resolved, quant_config)
        self.assertIn("model.eh_proj", resolved.exclude_layers)
        self.assertIn("model.decoder.mlp.gate", resolved.exclude_layers)
        self.assertNotIn("model.decoder.mlp.experts", resolved.exclude_layers)
        self.assertIn(
            "model.decoder.mlp.experts.0.gate_proj",
            resolved.quant_config["layer_quant_config"],
        )
        self.assertIn(
            "model.decoder.self_attn.q_b_proj",
            resolved.quant_config["layer_quant_config"],
        )
        self.assertIs(
            resolved.quant_config["layer_quant_config"][
                "model.decoder.mlp.experts.0.gate_proj"
            ],
            fp8,
        )


class TestGlm5NextNextNScaleLoading(CustomTestCase):
    def _load(self, weights, params, *, fused_shared=0, fuse_qkv=False):
        model = SimpleNamespace(
            config=SimpleNamespace(
                num_nextn_predict_layers=1, num_hidden_layers=45, n_routed_experts=2
            ),
            num_fused_shared_experts=fused_shared,
            fuse_qkv_a_proj=fuse_qkv,
            quant_config=None,
            named_parameters=lambda: params.items(),
            model=SimpleNamespace(decoder=SimpleNamespace(self_attn=None)),
        )
        weights = [(f"model.language_model.layers.45.{k}", v) for k, v in weights]
        Glm5NextForConditionalGenerationNextN.load_weights(model, weights)

    def _param(self, shape=(1, 2)):
        param = torch.nn.Parameter(torch.full(shape, -1.0), requires_grad=False)
        param.weight_loader = Mock(
            side_effect=lambda p, w, *args, **kwargs: p.data.copy_(w)
        )
        return param

    def test_block_scales_reach_all_projection_loaders(self):
        cases = [
            ("self_attn.q_b_proj", "self_attn.q_b_proj.", (), {}, 0),
            (
                "mlp.shared_experts.gate_proj",
                "mlp.shared_experts.gate_up_proj.",
                (0,),
                {},
                0,
            ),
            (
                "mlp.shared_experts.up_proj",
                "mlp.shared_experts.gate_up_proj.",
                (1,),
                {},
                0,
            ),
            (
                "mlp.experts.0.gate_proj",
                "mlp.experts.w13_",
                (),
                {"shard_id": "w1", "expert_id": 0},
                0,
            ),
            (
                "mlp.experts.1.up_proj",
                "mlp.experts.w13_",
                (),
                {"shard_id": "w3", "expert_id": 1},
                0,
            ),
            (
                "mlp.experts.1.down_proj",
                "mlp.experts.w2_",
                (),
                {"shard_id": "w2", "expert_id": 1},
                0,
            ),
            (
                "mlp.shared_experts.gate_proj",
                "mlp.experts.w13_",
                (),
                {"shard_id": "w1", "expert_id": 2},
                1,
            ),
        ]
        scale = torch.tensor([[0.25, 0.5]])
        for source, target, args, kwargs, fused_shared in cases:
            for suffix in ("weight_scale", "weight_scale_inv"):
                with self.subTest(source=source, suffix=suffix):
                    name = f"model.decoder.{target}weight_scale_inv"
                    param = self._param()
                    self._load(
                        [(f"{source}.{suffix}", scale)],
                        {name: param},
                        fused_shared=fused_shared,
                    )
                    torch.testing.assert_close(param, scale)
                    param.weight_loader.assert_called_once()
                    call = param.weight_loader.call_args
                    self.assertEqual(call.args[2:], (name,) if kwargs else args)
                    self.assertEqual(call.kwargs, kwargs)

    def test_fused_latent_projection_scales_are_concatenated(self):
        param = self._param((2, 2))
        self._load(
            [
                (
                    "self_attn.kv_a_proj_with_mqa.weight_scale",
                    torch.tensor([[0.5, 1.0]]),
                ),
                ("self_attn.q_a_proj.weight_scale", torch.tensor([[0.125, 0.25]])),
            ],
            {
                "model.decoder.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv": param
            },
            fuse_qkv=True,
        )
        torch.testing.assert_close(param, torch.tensor([[0.125, 0.25], [0.5, 1.0]]))
        param.weight_loader.assert_called_once()

    def test_existing_scale_parameter_takes_priority(self):
        for source, target in (
            ("self_attn.q_b_proj", "self_attn.q_b_proj."),
            ("mlp.experts.0.gate_proj", "mlp.experts.w13_"),
        ):
            with self.subTest(source=source):
                name = f"model.decoder.{target}weight_scale"
                param, unused = self._param(), self._param()
                scale = torch.tensor([[0.25, 0.5]])
                self._load(
                    [(f"{source}.weight_scale", scale)],
                    {name: param, f"{name}_inv": unused},
                )
                torch.testing.assert_close(param, scale)
                param.weight_loader.assert_called_once()
                unused.weight_loader.assert_not_called()


if __name__ == "__main__":
    sys.exit(unittest.main())
