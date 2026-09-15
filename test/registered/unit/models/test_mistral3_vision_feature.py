"""Mistral3 should not ask the vision tower for every layer to read one.

The tower materialises one hidden-state tensor per layer when hidden states are
requested and holds them for the whole item loop, which is ~49x the tensor the
model actually consumes. These tests pin that the final-layer case takes the
cheap path and that both paths agree.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_LAYERS = 4
TOKENS = 3
HIDDEN = 8

get_image_feature = Mistral3ForConditionalGeneration.get_image_feature


class RecordingTower:
    """Stands in for PixtralVisionModel, mirroring how it answers.

    ``all_hidden_states`` is the input embedding followed by one tensor per
    layer, and the value returned when hidden states are *not* requested is the
    last entry of that list -- the property the fix relies on.
    """

    def __init__(self):
        torch.manual_seed(0)
        self.layers = [torch.randn(1, TOKENS, HIDDEN) for _ in range(NUM_LAYERS + 1)]
        self.calls = []

    def __call__(self, pixel_values, image_sizes, output_hidden_states=False):
        self.calls.append(output_hidden_states)
        if output_hidden_states:
            return SimpleNamespace(
                last_hidden_state=self.layers[-1], hidden_states=list(self.layers)
            )
        return self.layers[-1]


def _model(vision_feature_layer):
    return SimpleNamespace(
        vision_tower=RecordingTower(),
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy="full",
        multi_modal_projector=lambda feature, image_sizes: feature,
    )


def _items(n):
    return [
        SimpleNamespace(feature=torch.zeros(1, 3, 4, 4), image_sizes=[(4, 4)])
        for _ in range(n)
    ]


class TestMistral3VisionFeature(unittest.TestCase):
    def test_final_layer_never_requests_hidden_states(self):
        model = _model(-1)
        get_image_feature(model, _items(3))
        self.assertEqual(model.vision_tower.calls, [False, False, False])

    def test_non_final_layer_still_requests_hidden_states(self):
        model = _model(-2)
        get_image_feature(model, _items(2))
        self.assertEqual(model.vision_tower.calls, [True, True])

    def test_both_paths_agree_on_the_final_layer(self):
        # -1 through the cheap path vs NUM_LAYERS (the same tensor, reached by
        # indexing the hidden-state list) must produce identical features.
        cheap = get_image_feature(_model(-1), _items(2))
        listed = get_image_feature(_model(NUM_LAYERS), _items(2))
        self.assertTrue(torch.equal(cheap, listed))

    def test_non_final_layer_selects_that_layer(self):
        model = _model(1)
        out = get_image_feature(model, _items(1))
        self.assertTrue(torch.equal(out, model.vision_tower.layers[1].squeeze(0)))

    def test_features_are_concatenated_per_item(self):
        out = get_image_feature(_model(-1), _items(3))
        self.assertEqual(out.shape, (3 * TOKENS, HIDDEN))


if __name__ == "__main__":
    unittest.main()
