import types

import pytest
import torch

from sgl_diffusion.api.configs.models.encoders.base import (
    BaseEncoderOutput,
    TextEncoderArchConfig,
    TextEncoderConfig,
)
from sgl_diffusion.api.configs.pipelines.base import PipelineConfig
from sgl_diffusion.api.configs.sample.base import DataType
from sgl_diffusion.runtime.pipelines.pipeline_batch_info import Req
from sgl_diffusion.runtime.pipelines.stages.text_encoding import (
    TextEncodingStage,
)
from sgl_diffusion.runtime.server_args import ServerArgs


class TensorDict(dict):
    def to(self, device):
        return TensorDict({k: v.to(device) for k, v in self.items()})


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        B = len(texts)
        seq_len = int(kwargs.get("max_length", 4))
        return TensorDict(
            {
                "input_ids": torch.arange(B * seq_len).view(B, seq_len),
                "attention_mask": torch.ones(B, seq_len, dtype=torch.long),
            }
        )


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        B, T = input_ids.shape
        last_hidden_state = torch.arange(
            B * T * self.hidden_size, dtype=torch.float32
        ).view(B, T, self.hidden_size)
        return types.SimpleNamespace(last_hidden_state=last_hidden_state)


def id_preprocess(x: str) -> str:
    return x


def take_mean_postprocess(outputs: BaseEncoderOutput) -> torch.Tensor:
    # [B, T, H] -> [B, H]
    return outputs.last_hidden_state.mean(dim=1)


def make_args(num_encoders=2, text_len=4, hidden_size=8):
    enc_cfgs = []
    preprocess_fns = []
    postprocess_fns = []
    for _ in range(num_encoders):
        arch = TextEncoderArchConfig(text_len=text_len)
        enc_cfgs.append(TextEncoderConfig(arch_config=arch))
        preprocess_fns.append(id_preprocess)
        postprocess_fns.append(take_mean_postprocess)
    pipe_cfg = PipelineConfig(
        text_encoder_configs=tuple(enc_cfgs),
        text_encoder_precisions=tuple(["fp32"] * num_encoders),
        preprocess_text_funcs=tuple(preprocess_fns),
        postprocess_text_funcs=tuple(postprocess_fns),
    )
    return ServerArgs(model_path="", pipeline_config=pipe_cfg), hidden_size


def make_stage(num_encoders=2, hidden_size=8):
    tokenizers = [FakeTokenizer() for _ in range(num_encoders)]
    encoders = [FakeTextEncoder(hidden_size=hidden_size) for _ in range(num_encoders)]
    return TextEncodingStage(text_encoders=encoders, tokenizers=tokenizers)


def test_encode_text_selection_and_shapes():
    server_args, hidden = make_args(num_encoders=2, text_len=4, hidden_size=8)
    stage = make_stage(num_encoders=2, hidden_size=hidden)

    # list return, two encoders
    embeds = stage.encode_text(["a", "b"], server_args, encoder_index=[0, 1])
    assert isinstance(embeds, list) and len(embeds) == 2
    for e in embeds:
        assert e.shape == (2, hidden)

    # with masks
    embeds2, masks2 = stage.encode_text(
        "a", server_args, encoder_index=[1], return_attention_mask=True
    )
    assert len(embeds2) == 1 and len(masks2) == 1
    assert embeds2[0].shape == (1, hidden)
    assert masks2[0].shape == (1, 4)

    # dict return
    d = stage.encode_text(
        ["a", "b"], server_args, encoder_index=[0, 1], return_type="dict"
    )
    assert set(d.keys()) == {"0", "1"}
    assert d["0"].shape == (2, hidden)

    # stack return
    s = stage.encode_text(
        ["a", "b"], server_args, encoder_index=[0, 1], return_type="stack"
    )
    assert s.shape == (2, 2, hidden)  # [encoders, batch, hidden]

    # overrides: dtype + max_length
    e3, m3 = stage.encode_text(
        ["a"],
        server_args,
        encoder_index=[0],
        dtype=torch.float16,
        return_attention_mask=True,
        max_length=3,
    )
    assert e3[0].dtype == torch.float16
    assert m3[0].shape[1] == 3


def test_forward_integration_cfg_off_and_on():
    server_args, hidden = make_args(num_encoders=2, text_len=4, hidden_size=8)
    stage = make_stage(num_encoders=2, hidden_size=hidden)

    # CFG off
    batch = Req(
        data_type=DataType.VIDEO,
        prompt="a cat",
        negative_prompt="",
        do_classifier_free_guidance=False,
        prompt_embeds=[],
        negative_prompt_embeds=None,
        prompt_attention_mask=[],
        negative_attention_mask=None,
    )
    out = stage.forward(batch, server_args)
    assert len(out.prompt_embeds) == 2
    for e in out.prompt_embeds:
        assert e.shape[1] == hidden

    # CFG on
    batch2 = Req(
        data_type=DataType.VIDEO,
        prompt=["a cat", "a dog"],
        negative_prompt="bad picture",
        do_classifier_free_guidance=True,
        prompt_embeds=[],
        negative_prompt_embeds=[],
        prompt_attention_mask=[],
        negative_attention_mask=[],
    )
    out2 = stage.forward(batch2, server_args)
    assert len(out2.prompt_embeds) == 2
    assert len(out2.negative_prompt_embeds) == 2
    assert len(out2.prompt_attention_mask) == 2
    assert len(out2.negative_attention_mask) == 2
