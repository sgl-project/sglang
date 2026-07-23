from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from torch import nn
from transformers import PreTrainedModel

from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_language_only_does_not_construct_multimodal_encoders():
    config = SimpleNamespace(
        language_only=True,
        text_config=SimpleNamespace(vocab_size=32, tie_word_embeddings=True),
        vision_config=SimpleNamespace(),
        audio_config=SimpleNamespace(),
    )
    text_model = SimpleNamespace(embed_tokens=MagicMock())
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True, world_size=1)

    with (
        patch.object(
            PreTrainedModel,
            "__init__",
            lambda self, config: nn.Module.__init__(self),
        ),
        patch.object(PreTrainedModel, "post_init"),
        patch("sglang.srt.models.gemma4_mm.get_pp_group", return_value=pp_group),
        patch(
            "sglang.srt.models.gemma4_mm.Gemma4VisionEncoder",
            side_effect=AssertionError("vision encoder constructed"),
        ),
        patch(
            "sglang.srt.models.gemma4_mm.Gemma4AudioEncoder",
            side_effect=AssertionError("audio encoder constructed"),
        ),
        patch(
            "sglang.srt.models.gemma4_mm.Gemma4TextModel",
            return_value=text_model,
        ),
        patch("sglang.srt.models.gemma4_mm.LogitsProcessor"),
    ):
        model = Gemma4ForConditionalGeneration(config)

    assert model.language_model is text_model
