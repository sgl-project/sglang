"""Unit coverage for Qwen3.5 dense GGUF GDN load-time transforms."""

from types import SimpleNamespace
from unittest.mock import patch

import gguf
import torch
import torch.nn as nn

from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _model(*, key_heads=2, value_heads=4, value_head_dim=2, key_head_dim=2):
    """Build only the module tree used by the transform helpers."""
    model = object.__new__(Qwen3_5ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            linear_num_key_heads=key_heads,
            linear_num_value_heads=value_heads,
            linear_value_head_dim=value_head_dim,
            linear_key_head_dim=key_head_dim,
        )
    )
    model.add_module("model", nn.Module())
    model.model.config = SimpleNamespace(num_hidden_layers=1)
    model.model.add_module("layers", nn.ModuleList([nn.Module()]))
    linear_attn = nn.Module()
    model.model.layers[0].add_module("linear_attn", linear_attn)
    linear_attn.add_module("out_proj", nn.Module())
    return model


def _value_row_expected(weight, *, ratio=2, key_heads=2):
    per_head = weight.shape[0] // (ratio * key_heads)
    return (
        weight.reshape(ratio, key_heads, per_head, *weight.shape[1:])
        .transpose(0, 1)
        .reshape_as(weight)
    )


def test_embedding_builder_forwards_gguf_quantization_config():
    model = object.__new__(Qwen3_5ForCausalLM)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_first_rank=True)
    model.quant_config = object()
    model._embedding_prefix = "model.embed_tokens"
    config = SimpleNamespace(vocab_size=32, hidden_size=16)

    with (
        patch(
            "sglang.srt.models.qwen3_5.is_dp_attention_enabled",
            return_value=False,
        ),
        patch("sglang.srt.models.qwen3_5.VocabParallelEmbedding") as embedding,
    ):
        model._build_embed_tokens(config)

    embedding.assert_called_once_with(
        32,
        16,
        org_num_embeddings=32,
        quant_config=model.quant_config,
        prefix="model.embed_tokens",
        enable_tp=True,
    )


def test_gdn_value_rows_a_log_qkv_and_conv_layout():
    model = _model()
    rows = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)

    actual = model._gguf_gdn_transform(
        "model.layers.0.linear_attn.in_proj_b.qweight", rows
    )
    torch.testing.assert_close(actual, _value_row_expected(rows))

    a = -torch.arange(1, 9, dtype=torch.float32)
    actual = model._gguf_gdn_transform("model.layers.0.linear_attn.A_log", a)
    torch.testing.assert_close(actual, _value_row_expected(torch.log(-a)))

    qkv = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    actual = model._gguf_gdn_transform(
        "model.layers.0.linear_attn.in_proj_qkv.qweight", qkv
    )
    expected = torch.cat((qkv[:8], _value_row_expected(qkv[8:])))
    torch.testing.assert_close(actual, expected)

    conv = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    actual = model._gguf_gdn_transform("model.layers.0.linear_attn.conv1d.weight", conv)
    expected = torch.cat((conv[:8], _value_row_expected(conv[8:])))
    torch.testing.assert_close(actual, expected)


def test_q5_k_out_proj_uses_block_safe_coarse_permutation_and_metadata():
    # Q5_K blocks encode 256 elements. With 4 key heads, ratio 2, 128-wide
    # value heads and TP2, only a two-stage permutation can avoid splitting a
    # packed block: each rank owns 2 * 128 elements (= one whole block).
    model = _model(key_heads=4, value_heads=8, value_head_dim=128, key_head_dim=2)
    out_proj = model.model.layers[0].linear_attn.out_proj
    out_proj.tp_size = 2
    block_elems, block_bytes = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q5_K]
    assert (block_elems, block_bytes) == (256, 176)

    raw = torch.arange(2 * 4 * block_bytes, dtype=torch.int64).reshape(2, -1)
    raw = raw.to(torch.uint8)
    actual = model._gguf_gdn_transform(
        "model.layers.0.linear_attn.out_proj.qweight", raw
    )
    expected = (
        raw.reshape(2, 2, 2, 2 * (raw.shape[1] // 8)).transpose(1, 2).reshape_as(raw)
    )
    torch.testing.assert_close(actual, expected)
    assert out_proj._gguf_gdn_col_perm == (2, 2, 128)


def test_dense_loader_redirects_f32_ba_shard_to_qweight_with_shard_id():
    model = _model(key_heads=1, value_heads=1)
    model.quant_config = SimpleNamespace(get_name=lambda: "gguf")
    model.config.tie_word_embeddings = False
    model.pp_group = SimpleNamespace(is_last_rank=False)

    ba = nn.Module()
    model.model.layers[0].linear_attn.add_module("in_proj_ba", ba)
    qweight = nn.Parameter(torch.empty(1))
    calls = []

    def loader(param, weight, shard_id):
        calls.append((weight.clone(), shard_id))

    qweight.weight_loader = loader
    ba.register_parameter("qweight", qweight)

    source = torch.tensor([[1.0, 2.0]])
    loaded = model.load_weights(
        [("model.layers.0.linear_attn.in_proj_b.weight", source)]
    )

    assert "model.layers.0.linear_attn.in_proj_ba.qweight" in loaded
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], source)
    assert calls[0][1] == 0


def test_text_only_loader_applies_the_same_f32_ba_redirect():
    # The text-only GGUF entrypoint calls Qwen3_5ForCausalLM.load_weights
    # directly, so it must not rely on the conditional-generation wrapper.
    model = object.__new__(Qwen3_5ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_value_head_dim=2,
        linear_key_head_dim=2,
    )
    model.quant_config = SimpleNamespace(get_name=lambda: "gguf")
    model.add_module("layers", nn.ModuleList([nn.Module()]))
    linear_attn = nn.Module()
    model.layers[0].add_module("linear_attn", linear_attn)
    ba = nn.Module()
    linear_attn.add_module("in_proj_ba", ba)
    qweight = nn.Parameter(torch.empty(1))
    calls = []
    qweight.weight_loader = lambda param, weight, shard_id: calls.append(
        (weight.clone(), shard_id)
    )
    ba.register_parameter("qweight", qweight)

    source = torch.tensor([[3.0, 4.0]])
    loaded = model.load_weights([("layers.0.linear_attn.in_proj_b.weight", source)])

    assert "layers.0.linear_attn.in_proj_ba.qweight" in loaded
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], source)
    assert calls[0][1] == 0


def test_text_only_loader_converts_wrapper_stripped_final_gemma_norm():
    model = object.__new__(Qwen3_5ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_value_head_dim=2,
        linear_key_head_dim=2,
    )
    model.quant_config = SimpleNamespace(get_name=lambda: "gguf")
    norm = nn.Module()
    model.add_module("norm", norm)
    weight = nn.Parameter(torch.empty(2))
    calls = []
    weight.weight_loader = lambda param, loaded: calls.append(loaded.clone())
    norm.register_parameter("weight", weight)

    model.load_weights([("norm.weight", torch.tensor([1.25, 0.75]))])

    torch.testing.assert_close(calls[0], torch.tensor([0.25, -0.25]))
