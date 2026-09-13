"""Unit tests for native Nemotron-H Omni model integration."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.models.nano_nemotron_vl import (
    NemotronH_Nano_VL_V2,
    NemotronH_Omni_Reasoning_V3,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNemotronHOmniModel(CustomTestCase):
    def test_existing_nano_model_keeps_ignoring_unrecognized_weights(self):
        model = object.__new__(NemotronH_Nano_VL_V2)
        nn.Module.__init__(model)
        model.mlp1 = nn.Sequential()
        model.language_model = SimpleNamespace(
            load_weights=lambda weights: list(weights)
        )
        model.vision_model = SimpleNamespace(load_weights=lambda weights: None)
        model.sound_encoder = None

        model.load_weights([("unrecognized.weight", torch.ones(1))])

    def test_model_registry_resolves_new_architecture(self):
        from sglang.srt.models.registry import ModelRegistry

        model_class, architecture = ModelRegistry.resolve_model_cls(
            "NemotronH_Omni_Reasoning_V3"
        )

        self.assertIs(model_class, NemotronH_Omni_Reasoning_V3)
        self.assertEqual(architecture, "NemotronH_Omni_Reasoning_V3")

    def test_exposes_language_embed_and_head(self):
        model = object.__new__(NemotronH_Omni_Reasoning_V3)
        nn.Module.__init__(model)
        embed = object()
        head = object()
        model.language_model = SimpleNamespace(
            get_embed_and_head=lambda: (embed, head),
            lm_head=head,
        )

        self.assertEqual(model.get_embed_and_head(), (embed, head))
        self.assertIs(model.lm_head, head)

    def test_delegates_dflash_capture_to_language_model(self):
        model = object.__new__(NemotronH_Omni_Reasoning_V3)
        nn.Module.__init__(model)
        captured_layer_ids = []
        model.language_model = SimpleNamespace(
            set_dflash_layers_to_capture=captured_layer_ids.extend
        )

        model.set_dflash_layers_to_capture([1, 22, 43, 64, 85])

        self.assertEqual(captured_layer_ids, [1, 22, 43, 64, 85])

    def test_vision_final_layernorm_is_loaded_and_applied(self):
        model = object.__new__(NemotronH_Omni_Reasoning_V3)
        nn.Module.__init__(model)
        model.mlp1 = nn.Sequential()
        model.vision_final_layernorm = nn.LayerNorm(2)
        model.language_model = SimpleNamespace(load_weights=lambda weights: None)
        model.vision_model = SimpleNamespace(load_weights=lambda weights: None)
        model.sound_encoder = None

        weight = torch.tensor([2.0, 3.0])
        bias = torch.tensor([0.5, -0.5])
        model.load_weights(
            [
                ("vision_projector.vision_final_layernorm.weight", weight),
                ("vision_projector.vision_final_layernorm.bias", bias),
            ]
        )

        features = torch.tensor([[1.0, 3.0]])
        expected = nn.functional.layer_norm(features, (2,), weight, bias)
        torch.testing.assert_close(model._normalize_vision_features(features), expected)

    def test_hf_vision_and_projector_names_are_remapped(self):
        remap = NemotronH_Omni_Reasoning_V3._remap_checkpoint_weight_name

        self.assertEqual(
            remap("vision_model.embeddings.position_embedding"),
            "vision_model.radio_model.hf_model.embeddings.position_embedding",
        )
        self.assertEqual(
            remap("vision_model.embeddings.video_patch_projection.weight"),
            (
                "vision_model.radio_model.hf_model.embeddings."
                "video_patch_projection.weight"
            ),
        )
        self.assertEqual(
            remap("vision_projector.mlp1.linear1.weight"),
            "mlp1.1.weight",
        )
        self.assertEqual(
            remap("vision_model.radio_model.model.patch_generator.pos_embed"),
            "vision_model.radio_model.model.patch_generator.pos_embed",
        )

    def test_unexpected_checkpoint_weight_raises(self):
        model = object.__new__(NemotronH_Omni_Reasoning_V3)
        nn.Module.__init__(model)
        model.mlp1 = nn.Sequential()
        model.vision_final_layernorm = nn.LayerNorm(2)
        model.language_model = SimpleNamespace(load_weights=lambda weights: None)
        model.vision_model = SimpleNamespace(load_weights=lambda weights: None)
        model.sound_encoder = None

        cases = (
            ("vision_projector.unknown.weight", "Unexpected Nemotron-H Omni"),
            (
                "vision_projector.vision_final_layernorm.running_mean",
                "Unexpected vision projector weight",
            ),
        )
        for name, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                model.load_weights([(name, torch.ones(1))])

    def test_language_weights_are_streamed_and_remaining_components_are_routed(self):
        model = object.__new__(NemotronH_Omni_Reasoning_V3)
        nn.Module.__init__(model)
        model.mlp1 = nn.Sequential()
        model.vision_final_layernorm = None
        source_exhausted = False
        loaded_language_weights = []
        loaded_vision_weights = []
        loaded_sound_weights = []

        def source_weights():
            nonlocal source_exhausted
            yield "language_model.model.layer.weight", torch.ones(1)
            yield "vision_model.radio_model.encoder.weight", torch.ones(1)
            yield "sound_encoder.projection.weight", torch.ones(1)
            source_exhausted = True

        def load_language_weights(weights):
            self.assertFalse(source_exhausted)
            loaded_language_weights.append(next(weights))

        def load_vision_weights(weights):
            self.assertFalse(source_exhausted)
            loaded_vision_weights.extend(weights)

        def load_sound_weights(weights):
            self.assertFalse(source_exhausted)
            loaded_sound_weights.extend(weights)

        model.language_model = SimpleNamespace(load_weights=load_language_weights)
        model.vision_model = SimpleNamespace(load_weights=load_vision_weights)
        model.sound_encoder = SimpleNamespace(load_weights=load_sound_weights)

        model.load_weights(source_weights())

        self.assertTrue(source_exhausted)
        self.assertEqual(
            [name for name, _ in loaded_language_weights], ["model.layer.weight"]
        )
        self.assertEqual(
            [name for name, _ in loaded_vision_weights],
            ["radio_model.encoder.weight"],
        )
        self.assertEqual(
            [name for name, _ in loaded_sound_weights],
            ["sound_encoder.projection.weight"],
        )


if __name__ == "__main__":
    unittest.main()
