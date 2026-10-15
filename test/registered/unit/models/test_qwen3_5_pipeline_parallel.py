import unittest
from itertools import product
from types import SimpleNamespace

import torch

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _RecordingParam:
    def __init__(self):
        self.loaded_weight = None

    def weight_loader(self, _param, loaded_weight, *args, **kwargs):
        self.loaded_weight = loaded_weight


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

    def test_pp_filter_only_applies_to_language_model_layers(self):
        for model_cls, visual_namespace in product(
            (Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration),
            ("layers", "blocks"),
        ):
            with self.subTest(
                model_cls=model_cls.__name__, visual_namespace=visual_namespace
            ):
                decoder_weight = _RecordingParam()
                skipped_weight = _RecordingParam()
                visual_weight = _RecordingParam()
                params = {
                    "model.layers.0.norm.weight": decoder_weight,
                    "model.layers.5.norm.weight": skipped_weight,
                    f"visual.{visual_namespace}.5.norm.weight": visual_weight,
                }
                model = object.__new__(model_cls)
                torch.nn.Module.__init__(model)
                model.config = SimpleNamespace(
                    tie_word_embeddings=False,
                    num_experts=1,
                    encoder_only=False,
                )
                model.model = SimpleNamespace(start_layer=0, end_layer=1)
                model.pp_group = SimpleNamespace(
                    is_first_rank=True,
                    is_last_rank=False,
                )
                model.enable_shared_expert_fusion = False
                model.pooler = None
                model.named_parameters = lambda remove_duplicate=False: iter(
                    params.items()
                )

                decoder_loaded = torch.ones(1)
                skipped_loaded = torch.full((1,), 2)
                visual_loaded = torch.full((1,), 3)
                loaded_params = model_cls.load_weights(
                    model,
                    [
                        ("model.layers.0.norm.weight", decoder_loaded),
                        ("model.layers.5.norm.weight", skipped_loaded),
                        (
                            f"model.visual.{visual_namespace}.5.norm.weight",
                            visual_loaded,
                        ),
                    ],
                )

                self.assertIs(decoder_weight.loaded_weight, decoder_loaded)
                self.assertIsNone(skipped_weight.loaded_weight)
                self.assertIs(visual_weight.loaded_weight, visual_loaded)
                self.assertIn("model.layers.0.norm.weight", loaded_params)
                self.assertIn(f"visual.{visual_namespace}.5.norm.weight", loaded_params)
                self.assertNotIn("model.layers.5.norm.weight", loaded_params)

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
