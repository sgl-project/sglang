"""Regression for GLM-5-Next MTP weight filtering across checkpoint prefixes.

The MTP/NextN block (layer index == num_hidden_layers) must never reach the main
model's loader, whichever prefix the checkpoint uses for it, and must always reach
the draft loader. Matching a literal `model.layers.N.` prefix silently dropped the
`layers.N.*` and `language_model.layers.N.*` forms, which is invisible at load time.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import glm5_next
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_HIDDEN_LAYERS = 45
MTP_LAYER_ID = NUM_HIDDEN_LAYERS

# The four prefix forms published GLM-5.3 checkpoints use for the same tensor.
MTP_TENSOR_NAMES = [
    f"model.language_model.layers.{MTP_LAYER_ID}.input_layernorm.weight",
    f"language_model.layers.{MTP_LAYER_ID}.input_layernorm.weight",
    f"model.layers.{MTP_LAYER_ID}.input_layernorm.weight",
    f"layers.{MTP_LAYER_ID}.input_layernorm.weight",
]
BACKBONE_TENSOR_NAME = (
    f"model.language_model.layers.{NUM_HIDDEN_LAYERS - 1}.input_layernorm.weight"
)


class _RecordingParam:
    """Stand-in for an nn.Parameter that records what the loader wrote into it."""

    def __init__(self):
        self.loaded = []

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        self.loaded.append(loaded_weight)


def _make_main_model(params_dict):
    model = object.__new__(Glm5NextForConditionalGeneration)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_nextn_predict_layers=1,
            n_routed_experts=0,
        ),
    )
    object.__setattr__(model, "quant_config", None)
    object.__setattr__(model, "num_fused_shared_experts", 0)
    object.__setattr__(model, "encoder_only", False)
    object.__setattr__(model, "named_parameters", lambda: iter(params_dict.items()))
    return model


class TestGlm5NextMTPWeightLoading(CustomTestCase):
    def test_main_loader_skips_mtp_layer_for_every_prefix_form(self):
        """No layer-45 tensor is written when is_nextn=False, whatever its prefix."""
        # Keyed on the post-`language_model.`-strip names so that a tensor slipping
        # past the layer guard is observable as a write rather than a dict miss.
        mtp_params = {
            f"model.layers.{MTP_LAYER_ID}.input_layernorm.weight": _RecordingParam(),
            f"layers.{MTP_LAYER_ID}.input_layernorm.weight": _RecordingParam(),
        }
        backbone_param = _RecordingParam()
        params_dict = {
            **mtp_params,
            f"model.layers.{NUM_HIDDEN_LAYERS - 1}.input_layernorm.weight": backbone_param,
        }
        model = _make_main_model(params_dict)

        weights = [(name, torch.zeros(1)) for name in MTP_TENSOR_NAMES]
        weights.append((BACKBONE_TENSOR_NAME, torch.ones(1)))

        with patch.object(
            glm5_next.DeepseekV2WeightLoaderMixin, "post_load_weights"
        ) as post_load:
            model.load_weights(weights, is_nextn=False)

        for name, param in mtp_params.items():
            self.assertEqual(param.loaded, [], f"MTP tensor {name} was loaded")
        self.assertEqual(len(backbone_param.loaded), 1)
        self.assertEqual(backbone_param.loaded[0].item(), 1.0)
        post_load.assert_called_once()

    def test_draft_loader_keeps_and_normalizes_every_prefix_form(self):
        """The NextN selector keeps all four prefix forms and canonicalizes them."""
        weights = [(name, torch.zeros(1)) for name in MTP_TENSOR_NAMES]
        weights.append((BACKBONE_TENSOR_NAME, torch.ones(1)))

        selected = list(
            Glm5NextForConditionalGenerationNextN._select_nextn_weights(
                weights=weights, layer_id=MTP_LAYER_ID
            )
        )

        self.assertEqual(
            [name for name, _ in selected],
            [f"model.layers.{MTP_LAYER_ID}.input_layernorm.weight"]
            * len(MTP_TENSOR_NAMES),
        )


if __name__ == "__main__":
    unittest.main()
