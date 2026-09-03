import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQwen3_5PipelineParallel(CustomTestCase):
    @staticmethod
    def _make_mtp_weight_loader_stub():
        model = Qwen3_5ForCausalLMMTP.__new__(Qwen3_5ForCausalLMMTP)
        torch.nn.Module.__init__(model)
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(4, 3)
        model.config = SimpleNamespace(num_experts=None)
        model.quant_config = None
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(torch.nan)
        return model

    @staticmethod
    def _get_num_fused_shared_experts(layers, start_layer, end_layer):
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=layers,
                start_layer=start_layer,
                end_layer=end_layer,
            )
        )
        return Qwen3_5MoeForConditionalGeneration._get_num_fused_shared_experts(model)

    def test_get_num_fused_shared_experts_returns_zero_without_layers(self):
        model = SimpleNamespace(model=SimpleNamespace())

        num_fused_shared_experts = (
            Qwen3_5MoeForConditionalGeneration._get_num_fused_shared_experts(model)
        )

        self.assertEqual(num_fused_shared_experts, 0)

    def test_get_num_fused_shared_experts_uses_local_pp_layers(self):
        layers = [
            PPMissingLayer(),
            PPMissingLayer(),
            SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=1),
            ),
            SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=1),
            ),
        ]

        num_fused_shared_experts = self._get_num_fused_shared_experts(
            layers,
            start_layer=2,
            end_layer=4,
        )

        self.assertEqual(num_fused_shared_experts, 1)

    def test_get_num_fused_shared_experts_returns_zero_without_local_fusion(self):
        layers = [
            PPMissingLayer(),
            SimpleNamespace(mlp=SimpleNamespace()),
        ]

        num_fused_shared_experts = self._get_num_fused_shared_experts(
            layers,
            start_layer=1,
            end_layer=2,
        )

        self.assertEqual(num_fused_shared_experts, 0)

    def test_load_kv_cache_scales_skips_non_attention_layers(self):
        attention_layer = Qwen3_5AttentionDecoderLayer.__new__(
            Qwen3_5AttentionDecoderLayer
        )
        torch.nn.Module.__init__(attention_layer)
        attention_layer.attn = SimpleNamespace(
            k_scale=None,
            v_scale=None,
            k_scale_float=None,
            v_scale_float=None,
        )
        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=3),
            layers=[PPMissingLayer(), attention_layer, PPMissingLayer()],
        )
        parallel = SimpleNamespace(tp_size=2, tp_rank=1)

        with (
            patch("sglang.srt.models.qwen3_5.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.models.qwen3_5.kv_cache_scales_loader",
                return_value=[(0, 0.25), (1, 0.5), (2, 0.75)],
            ) as loader,
        ):
            Qwen3_5ForCausalLM.load_kv_cache_scales(model, "scales.json")

        loader.assert_called_once_with("scales.json", 1, 2, 3, None)
        self.assertIsInstance(attention_layer.attn.k_scale, torch.Tensor)
        self.assertIsInstance(attention_layer.attn.v_scale, torch.Tensor)
        self.assertEqual(attention_layer.attn.k_scale.item(), 0.5)
        self.assertEqual(attention_layer.attn.v_scale.item(), 0.5)
        self.assertEqual(attention_layer.attn.k_scale_float, 0.5)
        self.assertEqual(attention_layer.attn.v_scale_float, 0.5)

    def test_conditional_generation_delegates_kv_cache_scales(self):
        delegate = MagicMock()
        model = SimpleNamespace(model=delegate)

        Qwen3_5ForConditionalGeneration.load_kv_cache_scales(model, "scales.json")

        delegate.load_kv_cache_scales.assert_called_once_with("scales.json")

        Qwen3_5MoeForConditionalGeneration.load_kv_cache_scales(model, "scales.json")
        self.assertEqual(delegate.load_kv_cache_scales.call_count, 2)

    def test_mtp_loads_vl_target_embedding_for_last_pp_stage(self):
        model = self._make_mtp_weight_loader_stub()
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)

        loaded = model.load_weights(
            [("model.language_model.embed_tokens.weight", expected)]
        )

        self.assertEqual(loaded, {"model.embed_tokens.weight"})
        torch.testing.assert_close(model.model.embed_tokens.weight, expected)

    def test_mtp_loads_text_target_embedding_for_last_pp_stage(self):
        model = self._make_mtp_weight_loader_stub()
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)

        loaded = model.load_weights([("model.embed_tokens.weight", expected)])

        self.assertEqual(loaded, {"model.embed_tokens.weight"})
        torch.testing.assert_close(model.model.embed_tokens.weight, expected)


if __name__ == "__main__":
    unittest.main()
