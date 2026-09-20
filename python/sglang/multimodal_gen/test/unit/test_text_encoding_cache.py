from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import BatchEncoding

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class FullHiddenStateEncoder(TextEncoder):
    uses_sglang_forward_context = False

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        states = tuple(input_ids[..., None].float() + layer for layer in range(32))
        return BaseEncoderOutput(
            last_hidden_state=states[-1],
            hidden_states=states,
            pooler_output=states[-1][:, 0].clone(),
        )


@pytest.mark.parametrize("output_type", ["tensor", "tuple", "structured"])
@torch.no_grad()
def test_cache_stores_only_consumed_text_conditioning(output_type):
    encoder = FullHiddenStateEncoder().eval()

    def postprocess(output, text_inputs, return_attention_mask=False):
        embedding = output.hidden_states[9]
        mask = text_inputs["attention_mask"].bool()
        if output_type == "tuple":
            return embedding, mask
        if output_type == "structured":
            return TextConditioningOutput(embedding, mask, [2])
        return embedding

    def tokenize(texts, _tokenizer, _kwargs):
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[len(text), 2] for text in texts]),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }
        )

    config = SimpleNamespace(
        text_encoder_configs=[SimpleNamespace(tokenizer_kwargs={})],
        preprocess_text_funcs=[None],
        postprocess_text_funcs=[postprocess],
        text_encoder_extra_args=[],
        is_flux_v1=lambda: False,
        tokenize_prompt=tokenize,
        get_text_encoder_attention_mask=lambda inputs, _: inputs["attention_mask"],
        get_text_encoder_pooler_output=lambda outputs, _: outputs.pooler_output,
        build_text_conditioning_mask=lambda inputs, mask, embeds, _: mask.bool(),
        seq_lens_from_text_conditioning_mask=lambda mask: mask.sum(-1).tolist(),
    )
    args = make_server_args(pipeline_config=config)

    def make_stage():
        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock()):
            stage = TextEncodingStage(text_encoders=[encoder], tokenizers=[object()])
        stage._begin_text_encoder_use = MagicMock()
        stage._text_encode_dp_group = MagicMock(return_value=None)
        return stage

    stage = make_stage()
    cache = ConditioningCache(4096)
    with cache.scope():
        first = stage.encode_text(
            "hello", args, device="cpu", return_attention_mask=True
        )
        expected = first[0][0].clone()
        expected_pooled = first[2][0].clone()
        first[0][0].zero_()
        first[2][0].zero_()
        restored = stage.encode_text(
            "hello", args, device="cpu", return_attention_mask=True
        )
        torch.testing.assert_close(restored[0][0], expected, rtol=0, atol=0)
        torch.testing.assert_close(restored[2][0], expected_pooled, rtol=0, atol=0)
        assert restored[4] == [[2]]
        assert encoder.calls == 1
        assert cache.stats()["entries"] == 1
        assert cache.bytes < 32  # two embeddings, one pooled value, optional mask
        stage.encode_text("changed", args, device="cpu", return_attention_mask=True)
        assert encoder.calls == 2
        # The same encoder can serve different pipeline postprocessing contracts.
        make_stage().encode_text(
            "hello", args, device="cpu", return_attention_mask=True
        )
        assert encoder.calls == 3


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


def test_negative_text_encoding_has_no_separate_gpu_cache():
    stage = DummyTextEncodingStage()
    server_args = make_server_args()
    get_negative_embedding_twice(stage, server_args, make_req())
    assert stage.calls == 2


def test_component_uses_exact_encoder_precision():
    with patch(_GLOBAL_ARGS_PATCH) as mock_global_args:
        mock_global_args.return_value = MagicMock()
        stage = TextEncodingStage(text_encoders=[object(), object()], tokenizers=[])
    server_args = make_server_args(
        component_precisions={"text_encoder_2": "fp32"},
        pipeline_config=SimpleNamespace(
            text_encoder_configs=[], text_encoder_precisions=["bf16", "bf16"]
        ),
    )

    uses = stage.component_uses(server_args)

    assert [(use.component_name, use.target_dtype) for use in uses] == [
        ("text_encoder", None),
        ("text_encoder_2", torch.float32),
    ]


def test_negative_text_encoding_warmup_does_not_seed_a_private_cache():
    stage = DummyTextEncodingStage()
    get_negative_embedding_twice(stage, make_server_args(), make_req(is_warmup=True))
    assert stage.calls == 2
