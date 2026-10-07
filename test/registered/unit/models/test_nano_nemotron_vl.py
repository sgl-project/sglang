"""Unit tests for native Nemotron-H Omni model integration."""

import math
import unittest
from array import array
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.models.nano_nemotron_vl import (
    NemotronH_Nano_VL_V2,
    NemotronH_Omni_Reasoning_V3,
)
from sglang.srt.multimodal.evs import EVS, EVSEmbeddingResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

TOKENS_PER_TUBELET = 4
HIDDEN_SIZE = 8


def _temporal_video_model(video_pruning_rate: float):
    model = object.__new__(NemotronH_Omni_Reasoning_V3)
    config = SimpleNamespace(
        video_temporal_patch_size=2, video_pruning_rate=video_pruning_rate
    )
    EVS.__init__(model, config)
    model.config = config
    model.encoded_num_frames = []

    def extract_video_feature_temporal(pixel_values, num_frames):
        model.encoded_num_frames.append(num_frames)
        num_tubelets = math.ceil(num_frames / 2)
        return torch.randn(num_tubelets, TOKENS_PER_TUBELET, HIDDEN_SIZE)

    model.extract_video_feature_temporal = extract_video_feature_temporal
    return model


def _video_item(frames_per_video: list[int], **model_specific_data):
    return MultimodalDataItem(
        modality=Modality.VIDEO,
        feature=torch.zeros(sum(frames_per_video), 3, 4, 4),
        offsets=[],
        model_specific_data={"frames_per_video": frames_per_video}
        | model_specific_data,
    )


class TestNemotronHOmniVideo(CustomTestCase):
    def test_temporal_tubelets_do_not_span_videos(self):
        """Two odd-length videos in one request must be grouped into tubelets per video."""
        model = _temporal_video_model(video_pruning_rate=0.0)

        features = model.get_video_feature([_video_item([3, 1])])

        self.assertEqual(model.encoded_num_frames, [3, 1])
        self.assertEqual(features.shape[0], 3)

    def test_evs_prunes_temporal_tubelets_per_video(self):
        """EVS on 2-frame tubelets keeps each video's first tubelet and prunes per video."""
        model = _temporal_video_model(video_pruning_rate=0.25)
        rows = cols = int(math.sqrt(TOKENS_PER_TUBELET))
        item = _video_item([3, 1], thw_grids=[(2, rows, cols), (1, rows, cols)])

        result = model.get_video_feature([item])

        self.assertIsInstance(result, EVSEmbeddingResult)
        self.assertEqual(result.num_tokens_per_frame, [4, 2, 4])
        self.assertEqual(result.embedding.shape, (10, HIDDEN_SIZE))

    def test_evs_keeps_image_pad_values_for_redistribution(self):
        """EVS placeholder redistribution must not restore raw image tokens next to a video."""
        start, end, image_token = 1, 2, 18
        image = MultimodalDataItem(modality=Modality.IMAGE, offsets=[], hash=11)
        video = _video_item([2], pre_chunked_input_ids=[])
        video.hash = 22
        for item in (image, video):
            item.set_pad_value()
        input_ids = array("q", [7, start, image_token, end, start, image_token, end])
        video.pre_chunked_input_ids = input_ids.tolist()
        mm_inputs = MultimodalInputs(
            mm_items=[image, video], im_start_id=start, im_end_id=end
        )

        model = object.__new__(NemotronH_Omni_Reasoning_V3)
        padded = model.pad_input_ids(input_ids, mm_inputs)

        self.assertEqual(list(video.pre_chunked_input_ids), list(padded))
        self.assertIn(image.pad_value, video.pre_chunked_input_ids)


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
