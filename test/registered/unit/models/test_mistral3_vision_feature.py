"""Pixtral adaptors should not ask the vision tower for every layer to read one.

The tower materialises one hidden-state tensor per layer when hidden states are
requested, ~49x the tensor the model actually consumes. These tests pin that the
final-layer case takes the cheap path and that both paths agree, for each
adaptor that reads the Pixtral tower this way.
"""

import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.llava import LlavaForConditionalGeneration
from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

NUM_LAYERS = 4
TOKENS = 3
HIDDEN = 8

ADAPTORS = {
    "mistral3": Mistral3ForConditionalGeneration.get_image_feature,
    "llava": LlavaForConditionalGeneration.get_image_feature,
}


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
        multi_modal_projector=lambda feature, image_sizes=None: feature,
    )


def _items(n):
    return [
        SimpleNamespace(feature=torch.zeros(1, 3, 4, 4), image_sizes=[(4, 4)])
        for _ in range(n)
    ]


class TestPixtralAdaptorVisionFeature(unittest.TestCase):
    def test_final_layer_never_requests_hidden_states(self):
        for name, get_image_feature in ADAPTORS.items():
            with self.subTest(adaptor=name):
                model = _model(-1)
                get_image_feature(model, _items(3))
                self.assertEqual(model.vision_tower.calls, [False, False, False])

    def test_non_final_layer_still_requests_hidden_states(self):
        for name, get_image_feature in ADAPTORS.items():
            with self.subTest(adaptor=name):
                model = _model(-2)
                get_image_feature(model, _items(2))
                self.assertEqual(model.vision_tower.calls, [True, True])

    def test_both_paths_agree_on_the_final_layer(self):
        # -1 through the cheap path vs NUM_LAYERS (the same tensor, reached by
        # indexing the hidden-state list) must produce identical features.
        for name, get_image_feature in ADAPTORS.items():
            with self.subTest(adaptor=name):
                cheap = get_image_feature(_model(-1), _items(2))
                listed = get_image_feature(_model(NUM_LAYERS), _items(2))
                self.assertTrue(torch.equal(cheap, listed))

    def test_non_final_layer_selects_that_layer(self):
        for name, get_image_feature in ADAPTORS.items():
            with self.subTest(adaptor=name):
                model = _model(1)
                out = get_image_feature(model, _items(1))
                self.assertTrue(
                    torch.equal(out, model.vision_tower.layers[1].squeeze(0))
                )

    def test_features_are_concatenated_per_item(self):
        for name, get_image_feature in ADAPTORS.items():
            with self.subTest(adaptor=name):
                out = get_image_feature(_model(-1), _items(3))
                self.assertEqual(out.shape, (3 * TOKENS, HIDDEN))


def _bare_mistral3():
    # Skip LlavaForConditionalGeneration.__init__, which builds the language and
    # vision models from an HF config.
    model = Mistral3ForConditionalGeneration.__new__(Mistral3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    return model


class RecordingLoader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded = []

    def load_weights(self, weights):
        self.loaded.extend(name for name, _ in weights)


class RecordingProjector(torch.nn.Module):
    # Mistral3's projector takes image_sizes; Llava's takes the feature alone.
    def __init__(self):
        super().__init__()
        self.image_sizes = []

    def forward(self, feature, image_sizes):
        self.image_sizes.append(image_sizes)
        return feature


class TestMistral3ModelSurvivesEval(unittest.TestCase):
    def test_eval_keeps_the_mistral3_model(self):
        """Loaders keep what ``model.eval()`` returns, so it must be the model the
        loader built: same ``shared_experts_config``, children in eval mode, and
        Mistral3's image-feature path."""
        model = _bare_mistral3()
        sentinel = object()
        model.shared_experts_config = sentinel
        model.vision_tower = RecordingTower()
        model.vision_feature_layer = -1
        model.vision_feature_select_strategy = "full"
        model.multi_modal_projector = RecordingProjector()

        loaded = model.eval()

        self.assertIs(loaded, model)
        self.assertIs(loaded.shared_experts_config, sentinel)
        self.assertFalse(loaded.multi_modal_projector.training)
        loaded.get_image_feature(_items(1))
        self.assertEqual(loaded.multi_modal_projector.image_sizes, [[(4, 4)]])

    def test_v5_checkpoint_names_load_the_same_before_and_after_eval(self):
        """Weight updates call ``load_weights`` on the model kept after ``eval()``,
        so transformers v5 checkpoint names must reach the same destinations as
        the initial load."""
        model = _bare_mistral3()
        model.vision_feature_select_strategy = "full"
        model.language_model = RecordingLoader()
        model.vision_tower = RecordingLoader()
        model.multi_modal_projector = torch.nn.Linear(2, 2, bias=False)
        projector_weight = torch.ones(2, 2)
        v5_weights = [
            ("model.language_model.embed_tokens.weight", torch.zeros(1)),
            ("lm_head.weight", torch.zeros(1)),
            ("model.vision_tower.patch_conv.weight", torch.zeros(1)),
            ("model.multi_modal_projector.weight", projector_weight),
        ]

        def load_into(target):
            model.language_model.loaded.clear()
            model.vision_tower.loaded.clear()
            torch.nn.init.zeros_(model.multi_modal_projector.weight)
            target.load_weights(v5_weights)
            return (
                list(model.language_model.loaded),
                list(model.vision_tower.loaded),
                torch.equal(model.multi_modal_projector.weight, projector_weight),
            )

        expected = (
            ["model.embed_tokens.weight", "lm_head.weight"],
            ["patch_conv.weight"],
            True,
        )
        self.assertEqual(load_into(model), expected)
        self.assertEqual(load_into(model.eval()), expected)


class TestMistralLlavaImportOrder(unittest.TestCase):
    def test_either_module_can_be_imported_first(self):
        """mistral.py subclasses LlavaForConditionalGeneration and llava.py builds
        MistralForCausalLM, so neither module may depend on import order."""
        for first, second in (("mistral", "llava"), ("llava", "mistral")):
            with self.subTest(first=first):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"import sglang.srt.models.{first}, sglang.srt.models.{second}",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
