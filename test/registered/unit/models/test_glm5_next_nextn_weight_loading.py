"""
Unit tests for Glm5NextForConditionalGenerationNextN.load_weights.

Regression tests ensuring NextN draft model weight loading delegates cleanly to
Glm5NextForConditionalGeneration.load_weights without raising AttributeError on
instances that lack visual attributes or multimodal configurations.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        self.loaded = (param, loaded_weight, args, kwargs)


class TestGlm5NextNextNWeightLoading(unittest.TestCase):
    def _make_minimal_model(self, named_parameters=(), mm_config=None):
        model = object.__new__(Glm5NextForConditionalGenerationNextN)
        model.config = SimpleNamespace(
            num_hidden_layers=80,
            num_nextn_predict_layers=1,
            n_routed_experts=0,
            q_lora_rank=None,
        )
        model.num_fused_shared_experts = 0
        model.quant_config = None
        model.model = SimpleNamespace(decoder=SimpleNamespace(self_attn=None))
        if mm_config is not None:
            model.mm_config = mm_config
        model.named_parameters = lambda: iter(named_parameters)
        return model

    def test_nextn_loads_weights_without_visual_attribute(self):
        param = _FakeParam()
        model = self._make_minimal_model([("model.shared_head.norm.weight", param)])
        self.assertFalse(hasattr(model, "visual"))

        loaded_weight = torch.ones(1)
        model.load_weights([("model.layers.80.shared_head.norm.weight", loaded_weight)])

        self.assertEqual(param.loaded, (param, loaded_weight, (), {}))

    def test_nextn_spec_weights_map_to_model_prefix(self):
        params = {
            "model.enorm.weight": _FakeParam(),
            "model.hnorm.weight": _FakeParam(),
            "model.eh_proj.weight": _FakeParam(),
            "model.shared_head.norm.weight": _FakeParam(),
        }
        model = self._make_minimal_model(list(params.items()))
        weights = [
            ("model.layers.80.enorm.weight", torch.ones(1)),
            ("model.layers.80.hnorm.weight", torch.ones(1)),
            ("model.layers.80.eh_proj.weight", torch.ones(1)),
            ("model.layers.80.shared_head.norm.weight", torch.ones(1)),
        ]

        model.load_weights(weights)

        for name, param in params.items():
            self.assertIsNotNone(param.loaded, f"{name} was not loaded")

    def test_nextn_decoder_layer_weights_map_to_decoder_prefix(self):
        param = _FakeParam()
        model = self._make_minimal_model(
            [("model.decoder.input_layernorm.weight", param)]
        )
        loaded_weight = torch.ones(1)

        model.load_weights([("model.layers.80.input_layernorm.weight", loaded_weight)])

        self.assertEqual(param.loaded, (param, loaded_weight, (), {}))

    def test_nextn_skips_visual_weights(self):
        head_param = _FakeParam()
        visual_decoder_param = _FakeParam()
        visual_qkv_param = _FakeParam()
        params = {
            "model.shared_head.norm.weight": head_param,
            "model.decoder.visual.weight": visual_decoder_param,
            "model.decoder.visual.attn.qkv_proj.weight": visual_qkv_param,
        }
        mm_config = SimpleNamespace(
            vision_config=SimpleNamespace(num_dummy_heads=0, head_dim=64)
        )
        model = self._make_minimal_model(list(params.items()), mm_config=mm_config)
        weights = [
            ("visual.attn.qkv.weight", torch.ones(4, 4)),
            ("model.visual.patch_embed.proj.weight", torch.ones(4, 4)),
            ("model.layers.80.visual.weight", torch.ones(4, 4)),
            (
                "model.language_model.layers.80.visual.attn.qkv.weight",
                torch.ones(4, 4),
            ),
            ("model.layers.80.shared_head.norm.weight", torch.ones(1)),
        ]

        with patch(
            "sglang.srt.models.glm5_next.vision_utils.pad_vit_attn_dummy_heads"
        ) as mock_pad_helper:
            model.load_weights(weights)

        self.assertIsNotNone(head_param.loaded)
        self.assertIsNone(
            visual_decoder_param.loaded,
            "model.layers.80.visual.weight should have been skipped",
        )
        self.assertIsNone(
            visual_qkv_param.loaded,
            "model.language_model.layers.80.visual.attn.qkv.weight should have been skipped",
        )
        mock_pad_helper.assert_not_called()

    def test_nextn_embed_tokens_and_shared_head_head_skipped(self):
        params = {
            "model.embed_tokens.weight": _FakeParam(),
            "model.shared_head.head.weight": _FakeParam(),
        }
        model = self._make_minimal_model(list(params.items()))
        weights = [
            ("model.layers.80.embed_tokens.weight", torch.ones(1)),
            ("model.layers.80.shared_head.head.weight", torch.ones(1)),
        ]

        model.load_weights(weights)

        for name, param in params.items():
            self.assertIsNone(param.loaded, f"{name} should have been skipped")

    def test_delegated_load_weights_direct_call_on_no_visual_self(self):
        head_param = _FakeParam()
        visual_param_1 = _FakeParam()
        visual_param_2 = _FakeParam()
        params = {
            "model.shared_head.norm.weight": head_param,
            "visual.blocks.0.weight": visual_param_1,
            "model.decoder.visual.weight": visual_param_2,
        }
        # Bare object without visual or mm_config
        no_visual_self = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=1,
                num_nextn_predict_layers=1,
                n_routed_experts=0,
            ),
            num_fused_shared_experts=0,
            quant_config=None,
            model=SimpleNamespace(decoder=SimpleNamespace(self_attn=None)),
            named_parameters=lambda: list(params.items()),
        )

        weights = [
            ("model.layers.0.shared_head.norm.weight", torch.ones(1)),
            ("model.visual.blocks.0.weight", torch.ones(1)),
            ("model.layers.0.visual.weight", torch.ones(1)),
        ]

        with patch(
            "sglang.srt.models.glm5_next.vision_utils.pad_vit_attn_dummy_heads"
        ) as mock_pad_helper:
            Glm5NextForConditionalGeneration.load_weights(
                no_visual_self, weights, is_nextn=True
            )

        self.assertIsNotNone(head_param.loaded)
        self.assertIsNone(
            visual_param_1.loaded,
            "visual.blocks.0.weight should have been skipped on no-visual model",
        )
        self.assertIsNone(
            visual_param_2.loaded,
            "model.layers.0.visual.weight should have been skipped on no-visual model",
        )
        mock_pad_helper.assert_not_called()

    def test_delegated_load_weights_direct_call_language_only(self):
        head_param = _FakeParam()
        visual_param = _FakeParam()
        params = {
            "model.shared_head.norm.weight": head_param,
            "visual.attn.qkv_proj.weight": visual_param,
        }
        language_only_self = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=1,
                num_nextn_predict_layers=1,
                n_routed_experts=0,
            ),
            language_only=True,
            visual=None,
            mm_config=SimpleNamespace(
                vision_config=SimpleNamespace(num_dummy_heads=2, head_dim=64)
            ),
            num_fused_shared_experts=0,
            quant_config=None,
            model=SimpleNamespace(decoder=SimpleNamespace(self_attn=None)),
            named_parameters=lambda: list(params.items()),
        )

        weights = [
            ("model.layers.0.shared_head.norm.weight", torch.ones(1)),
            ("visual.attn.qkv.weight", torch.ones(4, 4)),
        ]

        with patch(
            "sglang.srt.models.glm5_next.vision_utils.pad_vit_attn_dummy_heads"
        ) as mock_pad_helper:
            Glm5NextForConditionalGeneration.load_weights(
                language_only_self, weights, is_nextn=True
            )

        self.assertIsNotNone(head_param.loaded)
        self.assertIsNone(
            visual_param.loaded,
            "visual.attn.qkv.weight should have been skipped on language-only model",
        )
        mock_pad_helper.assert_not_called()

    def test_delegated_load_weights_direct_call_vision_present(self):
        head_param = _FakeParam()
        visual_param = _FakeParam()
        params = {
            "model.shared_head.norm.weight": head_param,
            "visual.blocks.0.attn.qkv_proj.weight": visual_param,
        }
        vision_present_self = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=1,
                num_nextn_predict_layers=0,
                n_routed_experts=0,
                q_lora_rank=None,
            ),
            encoder_only=True,
            language_only=False,
            visual=object(),
            mm_config=SimpleNamespace(
                vision_config=SimpleNamespace(num_dummy_heads=2, head_dim=64)
            ),
            num_fused_shared_experts=0,
            quant_config=None,
            named_parameters=lambda: list(params.items()),
        )

        raw_weight = torch.randn(6, 64)
        padded_weight = torch.randn(8, 64)
        weights = [("visual.blocks.0.attn.qkv.weight", raw_weight)]

        with patch(
            "sglang.srt.models.glm5_next.vision_utils.pad_vit_attn_dummy_heads",
            return_value=padded_weight,
        ) as mock_pad_helper:
            Glm5NextForConditionalGeneration.load_weights(
                vision_present_self, weights, is_nextn=False
            )

        mock_pad_helper.assert_called_once_with(
            vision_present_self.mm_config,
            "visual.blocks.0.attn.qkv_proj.weight",
            raw_weight,
        )
        self.assertIsNotNone(visual_param.loaded)
        self.assertIs(visual_param.loaded[1], padded_weight)


if __name__ == "__main__":
    unittest.main()
