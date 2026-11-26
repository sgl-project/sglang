"""Unit tests for srt/models/yivl.py"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel

from sglang.srt.models.yivl import YiVLForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestYiVLLoadWeights(CustomTestCase):
    def test_unsupported_select_feature_is_named_in_the_error(self):
        """A checkpoint whose mm_vision_select_feature is neither "patch" nor
        "cls_patch" is rejected with a ValueError that names the value."""
        model = YiVLForCausalLM.__new__(YiVLForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            _name_or_path="dummy",
            mm_vision_select_layer=-2,
            mm_vision_select_feature="full",
        )
        model.vision_tower_subfolder = "vit"
        model.language_model = nn.Linear(2, 2)
        vision_tower = CLIPVisionModel(
            CLIPVisionConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
            )
        )

        with (
            patch.object(CLIPVisionModel, "from_pretrained", return_value=vision_tower),
            self.assertRaisesRegex(ValueError, "Unexpected select feature: full"),
        ):
            model.load_weights([])


if __name__ == "__main__":
    unittest.main()
