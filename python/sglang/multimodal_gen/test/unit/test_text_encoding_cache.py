from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages import text_encoding
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


def make_forward_req(**kwargs):
    defaults = {
        "prompt_embeds": [],
        "pooled_embeds": [],
        "prompt_attention_mask": None,
        "prompt_embeds_mask": None,
        "prompt_seq_lens": None,
        "negative_prompt_embeds": [],
        "neg_pooled_embeds": [],
        "negative_attention_mask": None,
        "negative_prompt_embeds_mask": None,
        "negative_prompt_seq_lens": None,
    }
    defaults.update(make_req(**kwargs).__dict__)
    return SimpleNamespace(**defaults)


def make_server_args(**kwargs):
    defaults = {
        "pipeline_config": SimpleNamespace(text_encoder_configs=[]),
        "enable_request_warmup_text_cache": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(
        **defaults,
    )


def test_negative_text_cache_key_tracks_encode_options():
    stage = DummyTextEncodingStage()
    server_args = make_server_args()

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
    server_args = make_server_args()

    stage.get_or_compute_negative_text_embedding(
        make_req(is_warmup=True), server_args, [0]
    )
    stage.get_or_compute_negative_text_embedding(make_req(), server_args, [0])

    assert stage.calls == 2


def test_negative_text_cache_reuses_serve_warmup():
    stage = DummyTextEncodingStage()
    server_args = make_server_args(enable_request_warmup_text_cache=True)

    stage.get_or_compute_negative_text_embedding(
        make_req(is_warmup=True), server_args, [0]
    )
    stage.get_or_compute_negative_text_embedding(make_req(), server_args, [0])

    assert stage.calls == 1


def test_cfg_text_batch_encodes_positive_and_negative_once():
    stage = DummyTextEncodingStage()
    server_args = make_server_args()
    batch = make_forward_req()

    stage.forward(batch, server_args)

    assert stage.calls == 1
    assert batch.prompt_embeds[0].shape[0] == 1
    assert batch.negative_prompt_embeds[0].shape[0] == 1


def test_cfg_text_batch_skips_dmd_pipeline():
    stage = DummyTextEncodingStage()
    server_args = make_server_args(
        pipeline_config=SimpleNamespace(
            text_encoder_configs=[],
            dmd_denoising_steps=1,
        )
    )

    stage.forward(make_forward_req(), server_args)

    assert stage.calls == 2


def test_text_outputs_cache_skips_matching_generate_warmup_request():
    stage = DummyTextEncodingStage()
    server_args = make_server_args()

    stage.forward(make_forward_req(is_warmup=True), server_args)
    stage.forward(make_forward_req(), server_args)
    assert stage.calls == 2


def test_text_outputs_cache_reuses_matching_serve_warmup_request():
    stage = DummyTextEncodingStage()
    server_args = make_server_args(enable_request_warmup_text_cache=True)

    with patch.object(text_encoding.logger, "info") as mock_log:
        stage.forward(make_forward_req(is_warmup=True), server_args)
        stage.forward(make_forward_req(), server_args)

    assert stage.calls == 1
    assert mock_log.call_count == 1

    stage.forward(make_forward_req(prompt="different"), server_args)
    assert stage.calls == 2


def test_text_outputs_cache_does_not_store_real_requests():
    stage = DummyTextEncodingStage()
    server_args = make_server_args(enable_request_warmup_text_cache=True)

    stage.forward(make_forward_req(), server_args)
    stage.forward(make_forward_req(), server_args)

    assert stage.calls == 2
