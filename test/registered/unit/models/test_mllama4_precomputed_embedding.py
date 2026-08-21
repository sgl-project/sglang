"""CPU-only coverage for Llama 4 precomputed embedding inputs."""

import pytest
import torch
import torch.nn as nn

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
)
from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, pixel_values):
        self.calls.append(pixel_values)
        return pixel_values.unsqueeze(0)


class _Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, vision_flat):
        self.calls.append(vision_flat)
        return vision_flat


def _bare_model():
    model = Llama4ForConditionalGeneration.__new__(Llama4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.has_vision = True
    model.vision_model = _VisionTower()
    model.multi_modal_projector = _Projector()
    return model


def _precomputed_item(feature):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, feature.shape[0] - 1)],
        feature=feature,
        format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
    )


class TestLlama4PrecomputedEmbedding(CustomTestCase):
    def test_returns_embedding_without_entering_vision_tower(self):
        model = _bare_model()
        embedding = torch.arange(24, dtype=torch.float32).reshape(3, 8)

        out = model.get_image_feature([_precomputed_item(embedding)])

        self.assertTrue(torch.equal(out, embedding))
        self.assertEqual(model.vision_model.calls, [])
        self.assertEqual(model.multi_modal_projector.calls, [])

    def test_concatenates_multiple_items(self):
        model = _bare_model()
        first = torch.ones(2, 8)
        second = torch.full((3, 8), 2.0)

        out = model.get_image_feature(
            [_precomputed_item(first), _precomputed_item(second)]
        )

        self.assertEqual(out.shape, (5, 8))
        self.assertTrue(torch.equal(out, torch.cat([first, second])))
        self.assertEqual(model.vision_model.calls, [])

    def test_flattens_batched_embedding_to_two_dims(self):
        model = _bare_model()
        embedding = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)

        out = model.get_image_feature([_precomputed_item(embedding)])

        self.assertEqual(out.shape, (6, 8))
        self.assertTrue(torch.equal(out, embedding.reshape(6, 8)))
        self.assertEqual(model.vision_model.calls, [])

    def test_pixel_values_still_go_through_vision_tower(self):
        model = _bare_model()
        pixel_values = torch.ones(2, 8)
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 1)],
            feature=pixel_values,
        )

        out = model.get_image_feature([item])

        self.assertEqual(len(model.vision_model.calls), 1)
        self.assertEqual(len(model.multi_modal_projector.calls), 1)
        self.assertEqual(out.shape, (2, 8))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
