from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
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
        text = args[0]
        batch_size = len(text) if isinstance(text, list) else 1
        embeds = torch.full((batch_size, 1, 1), float(self.calls))
        mask = torch.ones((batch_size, 1), dtype=torch.int64)
        return [embeds], [mask], [], [mask], [[1] * batch_size]


def make_req(**kwargs):
    defaults = {
        "prompt": "hello",
        "negative_prompt": "bad quality",
        "do_classifier_free_guidance": True,
        "prompt_template": {"template": "{}"},
        "max_sequence_length": 1024,
        "is_warmup": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_server_args(**kwargs):
    defaults = {
        "pipeline_class_name": "LTX2TwoStagePipeline",
        "model_path": "dummy-model",
        "backend": "auto",
        "model_id": None,
        "pipeline_config": SimpleNamespace(text_encoder_configs=[]),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def get_negative_embedding_twice(stage, server_args, first_req, second_req=None):
    stage.get_or_compute_negative_text_embedding(first_req, server_args, [0])
    stage.get_or_compute_negative_text_embedding(
        second_req if second_req is not None else make_req(), server_args, [0]
    )


def test_negative_text_cache_key_tracks_encode_options():
    stage = DummyTextEncodingStage()
    server_args = make_server_args()

    get_negative_embedding_twice(stage, server_args, make_req())
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
    server_args = make_server_args()

    with patch.object(
        stage, "_get_model_default_negative_prompt", return_value="default negative"
    ):
        get_negative_embedding_twice(stage, server_args, make_req(is_warmup=True))

    assert stage.calls == 2


def test_negative_text_cache_keeps_default_warmup():
    stage = DummyTextEncodingStage()
    server_args = make_server_args()

    with patch.object(
        stage, "_get_model_default_negative_prompt", return_value="bad quality"
    ):
        get_negative_embedding_twice(stage, server_args, make_req(is_warmup=True))

    assert stage.calls == 1


def test_forward_reuses_preencoded_negative_prompt_embeds():
    stage = DummyTextEncodingStage()
    server_args = make_server_args()
    negative_prompt_embeds = torch.zeros(1, 1, 1)
    req = Req(
        prompt=["hello", "world"],
        negative_prompt=None,
        negative_prompt_embeds=[negative_prompt_embeds],
        guidance_scale=4.0,
    )

    out = stage.forward(req, server_args)

    assert stage.calls == 1
    assert out.negative_prompt_embeds[0].shape[0] == 2
    assert torch.equal(out.negative_prompt_embeds[0][0], negative_prompt_embeds[0])
    assert torch.equal(out.negative_prompt_embeds[0][1], negative_prompt_embeds[0])


def test_dedup_fingerprint_tracks_preencoded_negative_prompt_embed_identity():
    stage = DummyTextEncodingStage()
    first = Req(
        prompt="hello",
        negative_prompt=None,
        negative_prompt_embeds=[torch.zeros(1, 1, 1)],
        guidance_scale=4.0,
    )
    second = Req(
        prompt="hello",
        negative_prompt=None,
        negative_prompt_embeds=[torch.zeros(1, 1, 1)],
        guidance_scale=4.0,
    )

    assert stage.build_dedup_fingerprint(first, make_server_args()) != (
        stage.build_dedup_fingerprint(second, make_server_args())
    )
