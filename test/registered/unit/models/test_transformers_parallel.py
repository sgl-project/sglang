# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import copy
from types import SimpleNamespace

import pytest
import torch
import transformers
from torch import nn

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.transformers.base import TransformersBase
from sglang.srt.models.transformers.layers import (
    HFCompatibleColumnParallelLinear,
    HFCompatibleRowParallelLinear,
    replace_linear_class,
)
from sglang.srt.models.transformers.parallel import ParallelMixin
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr("sglang.srt.models.transformers.base.get_device", lambda: "cpu")
    group = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        is_first_rank=True,
        is_last_rank=True,
        all_gather=lambda tensor, dim: tensor,
    )
    with get_parallel().override(
        tp_size=1,
        tp_rank=0,
        attn_tp_size=1,
        attn_tp_rank=0,
        pp_group=group,
        tp_group=group,
    ):
        yield


class ReferenceAttention:
    def forward(self, q, k, v, layer, batch, save_cache, **kwargs):
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        outputs, start = [], 0
        for length in batch.extend_seq_lens.tolist():
            query = q[start : start + length].transpose(0, 1)
            key = k[start : start + length].transpose(0, 1)
            value = v[start : start + length].transpose(0, 1)
            repeats = query.shape[0] // key.shape[0]
            key, value = (
                tensor.repeat_interleave(repeats, dim=0) for tensor in (key, value)
            )
            logits = query @ key.transpose(-1, -2) * layer.scaling
            if layer.logit_cap:
                logits = (logits / layer.logit_cap).tanh() * layer.logit_cap
            positions = torch.arange(length)
            distance = positions[:, None] - positions[None, :]
            causal = layer.attn_type == AttentionType.DECODER
            allowed = distance >= 0 if causal else torch.ones_like(distance).bool()
            if layer.sliding_window_size >= 0:
                allowed &= distance.abs() <= layer.sliding_window_size
            logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
            output = logits.softmax(-1, dtype=torch.float32).to(query.dtype) @ value
            outputs.append(output.transpose(0, 1).flatten(1))
            start += length
        return torch.cat(outputs)


def tiny_config(name):
    kwargs = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=1,
    )
    if name == "Phi3":
        kwargs["original_max_position_embeddings"] = 128
    if name.startswith("Gemma"):
        kwargs.update(query_pre_attn_scalar=4, sliding_window=64)
    return getattr(transformers, name + "Config")(**kwargs)


@pytest.mark.parametrize(
    "name", ["Qwen3", "Gemma2", "Gemma3Text", "Cohere2", "Phi3", "Olmo2"]
)
def test_hf_parallel_plans_allow_complete_tp1_backbone(runtime, name):
    torch.manual_seed(42)
    config = tiny_config(name)
    reference = transformers.AutoModel.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    ).eval()
    config.architectures = [type(reference).__name__]
    wrapper = TransformersBase(config)
    wrapper.load_weights(reference.state_dict().items())
    ids = torch.tensor([3, 4, 5, 6, 7, 8, 9])
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    batch = SimpleNamespace(
        input_ids=ids,
        extend_seq_lens=torch.tensor([4, 3]),
        extend_seq_lens_cpu=[4, 3],
        forward_mode=ForwardMode.EXTEND,
        token_type_ids=None,
    )
    with torch.no_grad():
        expected = torch.cat(
            [
                reference(input_ids=part[None]).last_hidden_state[0]
                for part in ids.split([4, 3])
            ]
        )
        with forward_context(ForwardContext(ReferenceAttention())):
            actual = wrapper._forward_hidden_states(ids, positions, batch)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


class PlanOwner(ParallelMixin):
    def __init__(self):
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 16)
        self.model.norm = nn.LayerNorm(16)
        self.model.proj = nn.Linear(16, 16)


def test_non_linear_and_unmatched_plan_entries_are_not_linear_contracts(runtime):
    owner = PlanOwner()
    plan = {
        "embed_tokens": "embedding_rowwise",
        "norm": "sequence_parallel",
        "unused.*": "future_style",
        "proj": "colwise_gather_output",
    }
    assert owner._normalize_tp_plan(plan) == {"proj": "colwise_rep"}


@pytest.mark.parametrize(
    "style,expected,attribute",
    [
        ("colwise_gather_output", HFCompatibleColumnParallelLinear, "gather_output"),
        ("rowwise_split_input", HFCompatibleRowParallelLinear, "input_is_parallel"),
    ],
)
def test_hf_gather_and_split_aliases_preserve_native_linear_contract(
    runtime, style, expected, attribute
):
    reference = nn.Linear(16, 24)
    layer = replace_linear_class(reference, style)
    assert isinstance(layer, expected)
    assert getattr(layer, attribute) == (attribute == "gather_output")
    layer.weight.weight_loader(layer.weight, reference.weight)
    layer.bias.weight_loader(layer.bias, reference.bias)
    inputs = torch.randn(3, 16)
    torch.testing.assert_close(layer(inputs), reference(inputs))


@pytest.mark.parametrize("name", ["Phi3", "Olmo2"])
def test_gathered_attention_tp_requires_a_head_layout_adapter(runtime, name):
    model = transformers.AutoModel.from_config(tiny_config(name))
    owner = PlanOwner()
    owner.model = model
    with get_parallel().override(
        tp_size=2, tp_rank=0, attn_tp_size=2, attn_tp_rank=0, tp_group=None
    ):
        with pytest.raises(ValueError, match="tensor-parallel head adapter"):
            owner._normalize_tp_plan(owner._get_model_tp_plan())


def test_ordinary_qwen3_tp_plan_is_preserved(runtime):
    owner = PlanOwner()
    owner.model = transformers.AutoModel.from_config(tiny_config("Qwen3"))
    with get_parallel().override(
        tp_size=2, tp_rank=0, attn_tp_size=2, attn_tp_rank=0, tp_group=None
    ):
        plan = owner._normalize_tp_plan(owner._get_model_tp_plan())
    assert plan["layers.*.self_attn.q_proj"] == "colwise"
    assert plan["layers.*.self_attn.o_proj"] == "rowwise"
    assert plan["layers.*.mlp.gate_proj"] == "colwise"


def test_prepacked_linear_tp_requires_a_native_checkpoint_contract(runtime):
    owner = PlanOwner()
    with get_parallel().override(
        tp_size=2, tp_rank=0, attn_tp_size=2, attn_tp_rank=0, tp_group=None
    ):
        with pytest.raises(ValueError, match="packed weight adapter"):
            owner._normalize_tp_plan({"proj": "packed_colwise"})


def test_mla_latent_projection_style_is_replicated(runtime):
    assert PlanOwner()._normalize_tp_plan({"proj": "mla_kv_a_proj"}) == {
        "proj": "replicate"
    }


def test_unknown_linear_parallel_styles_are_not_silently_replicated(runtime):
    with pytest.raises(ValueError, match="Unsupported TP style"):
        PlanOwner()._normalize_tp_plan({"proj": "future_style"})


def test_anchored_plan_can_be_prefixed_for_the_wrapper(runtime):
    owner = PlanOwner()
    assert owner._normalize_tp_plan({r"^model\.proj$": "colwise"}) == {
        "proj": "colwise"
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
