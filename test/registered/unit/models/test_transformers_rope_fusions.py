# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import copy

import pytest
import torch
from transformers import LlamaConfig, Qwen3Config
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from sglang.srt.models.transformers.fusers import fuse_module, load_fused_weights
from sglang.srt.models.transformers.fusers.rope import (
    PairedRotaryEmbedding,
    StaticRotaryEmbedding,
    _matches_pair,
    fuse_rotary_embedding,
    replace_rotary_embedding,
)
from sglang.srt.models.utils import AutoWeightsLoader
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _config(config_type, rope_type="default"):
    scaling = {"rope_type": rope_type, "rope_theta": 10000.0}
    if rope_type in {"linear", "llama3", "dynamic"}:
        scaling["factor"] = 4.0
    if rope_type == "llama3":
        scaling.update(
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=64,
        )
    config = config_type(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_parameters=scaling,
    )
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize(
    "model_type,config_type",
    [(Qwen3RotaryEmbedding, Qwen3Config), (LlamaRotaryEmbedding, LlamaConfig)],
)
@pytest.mark.parametrize("rope_type", ["default", "linear", "llama3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cached_rotary_matches_hf_at_repeated_and_out_of_range_positions(
    model_type, config_type, rope_type, dtype
):
    original = model_type(_config(config_type, rope_type))
    fused = replace_rotary_embedding(original)
    assert isinstance(fused, StaticRotaryEmbedding)
    for positions in (
        torch.tensor([[0, 7, 7, 96], [12, 17, 35, 127]]),
        torch.tensor([[-1, 128, 256, 2048]]),
    ):
        x = torch.zeros(*positions.shape, 32, dtype=dtype)
        expected, actual = original(x, positions), fused(x, positions)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_rotary_cache_preserves_fp32_buffers_on_dtype_conversion():
    original = Qwen3RotaryEmbedding(_config(Qwen3Config))
    module = replace_rotary_embedding(original)
    cache = module.cos_sin_cache.clone()
    module.to(dtype=torch.bfloat16)
    assert module.cos_sin_cache.dtype == torch.float32
    torch.testing.assert_close(module.cos_sin_cache, cache, rtol=0, atol=0)


def test_dynamic_rotary_keeps_original_behavior():
    original = Qwen3RotaryEmbedding(_config(Qwen3Config, "dynamic"))
    assert replace_rotary_embedding(original) is original


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_paired_rotary_preserves_noncontiguous_queries_and_broadcast_angles(dtype):
    torch.manual_seed(73)
    q, k = (
        torch.randn(2, 7, 4, 8, dtype=dtype).transpose(1, 2),
        torch.randn(2, 7, 2, 8, dtype=dtype).transpose(1, 2),
    )
    angles = torch.randn(1, 7, 8, dtype=dtype)
    cosine, sine = angles.cos(), angles.sin()
    q_before, k_before = q.clone(), k.clone()
    actual = PairedRotaryEmbedding()(q, k, cosine, sine)
    expected = apply_rotary_pos_emb(q, k, cosine, sine)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    torch.testing.assert_close(q, q_before, rtol=0, atol=0)
    torch.testing.assert_close(k, k_before, rtol=0, atol=0)


def test_pair_match_requires_the_complete_arithmetic():
    def extra_scale(q, k, cos, sin, unsqueeze_dim=1):
        query, key = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)
        return query * 2, key

    assert _matches_pair(apply_rotary_pos_emb)
    assert not _matches_pair(extra_scale)


def test_qwen3_attention_composes_qkv_and_rotary_lowering():
    config = _config(Qwen3Config)
    original = Qwen3Attention(config, layer_idx=0).eval()
    fused = copy.deepcopy(original)
    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0):
        result = fuse_module(fused, "model")
        assert result is not None
        assert fuse_rotary_embedding(fused)
        wrapper = torch.nn.Module()
        wrapper.model = fused
        weights = (
            (f"model.{name}", weight) for name, weight in original.state_dict().items()
        )
        weights = load_fused_weights(wrapper, weights, result.stacked_mapping, set())
        AutoWeightsLoader(wrapper).load_weights(weights)
        x = torch.randn(2, 7, 32)
        positions = torch.arange(7).expand(2, -1)
        angles = Qwen3RotaryEmbedding(config)(x, positions)
        with torch.no_grad():
            actual, expected = fused(x, angles, None), original(x, angles, None)
    torch.testing.assert_close(actual[0], expected[0])


def test_partial_and_dynamic_attention_rotary_are_preserved():
    for rope_type in ("default", "dynamic"):
        config = _config(Qwen3Config, rope_type)
        if rope_type == "default":
            config.partial_rotary_factor = 0.5
        module = Qwen3Attention(config, layer_idx=0)
        assert not fuse_rotary_embedding(module)


def test_cached_rotary_compiles_with_dynamic_token_shapes():
    module = replace_rotary_embedding(Qwen3RotaryEmbedding(_config(Qwen3Config)))
    compiled = torch.compile(module, backend="eager", dynamic=True, fullgraph=True)
    for length in (1, 7):
        x = torch.randn(1, length, 32)
        positions = torch.arange(length)[None, :]
        actual, expected = compiled(x, positions), module(x, positions)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
