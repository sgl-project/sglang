from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class DummyTextEncodingStage(TextEncodingStage):
    def __init__(self):
        with patch(_GLOBAL_ARGS_PATCH) as mock_global_args:
            mock_global_args.return_value = MagicMock()
            super().__init__(text_encoders=[], tokenizers=[])
        self.calls = 0

    def encode_text(self, *args, **kwargs):
        self.calls += 1
        embeds = torch.full((1, 1, 1), float(self.calls))
        mask = torch.ones((1, 1), dtype=torch.int64)
        return [embeds], [mask], [], [mask], [[1]]


def make_req(**kwargs):
    defaults = {
        "negative_prompt": "bad quality",
        "prompt_template": {"template": "{}"},
        "max_sequence_length": 1024,
        "is_warmup": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_negative_text_cache_key_tracks_encode_options():
    stage = DummyTextEncodingStage()
    server_args = SimpleNamespace(pipeline_class_name="LTX2TwoStagePipeline")

    stage.get_or_compute_negative_text_embedding(make_req(), server_args, [0])
    stage.get_or_compute_negative_text_embedding(make_req(), server_args, [0])
    assert stage.calls == 1

    stage.get_or_compute_negative_text_embedding(
        make_req(max_sequence_length=512), server_args, [0]
    )
    assert stage.calls == 2

    stage.get_or_compute_negative_text_embedding(
        make_req(prompt_template={"template": "negative: {}"}), server_args, [0]
    )
    assert stage.calls == 3


def test_negative_text_cache_skips_warmup():
    stage = DummyTextEncodingStage()
    server_args = SimpleNamespace(pipeline_class_name="LTX2TwoStagePipeline")

    stage.get_or_compute_negative_text_embedding(
        make_req(is_warmup=True), server_args, [0]
    )
    stage.get_or_compute_negative_text_embedding(make_req(), server_args, [0])

    assert stage.calls == 2
