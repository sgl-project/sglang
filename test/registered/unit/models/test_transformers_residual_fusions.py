# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import copy

import pytest
import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from sglang.srt.models.transformers.fusers import fuse_module, load_fused_weights
from sglang.srt.models.transformers.fusers.residual import fuse_residual_norm
from sglang.srt.models.transformers.layers import replace_rms_norm_class
from sglang.srt.models.utils import AutoWeightsLoader
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Decoder(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, hidden_states, residual):
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        return hidden_states, residual


class ObservedSum(Decoder):
    def forward(self, hidden_states, residual):
        hidden_states = residual + hidden_states
        observed = hidden_states.square()
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        return hidden_states, residual, observed


@pytest.mark.parametrize("norm_type", [Qwen3RMSNorm, Gemma3RMSNorm])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_residual_norm_preserves_low_precision_rounding_and_input_aliases(
    norm_type, dtype
):
    torch.manual_seed(92)
    norm = norm_type(32).to(dtype=dtype)
    with torch.no_grad():
        norm.weight.normal_()
    original = Decoder(norm)
    fused_norm = replace_rms_norm_class(norm, 32).to(dtype=dtype)
    with torch.no_grad():
        fused_norm.weight.copy_(norm.weight)
    fused = Decoder(fused_norm)
    assert fuse_residual_norm(fused)
    x, residual = torch.randn(2, 3, 32, dtype=dtype), torch.randn(2, 3, 32, dtype=dtype)
    x_before, residual_before = x.clone(), residual.clone()
    with torch.no_grad():
        actual = fused(x, residual)
        expected = original(x, residual)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
    torch.testing.assert_close(x, x_before, atol=0, rtol=0)
    torch.testing.assert_close(residual, residual_before, atol=0, rtol=0)
    assert actual[1].data_ptr() != residual.data_ptr()


def test_residual_norm_preserves_observed_intermediate():
    norm = replace_rms_norm_class(Qwen3RMSNorm(32), 32)
    module = ObservedSum(norm)
    assert not fuse_residual_norm(module)


def test_qwen3_decoder_composes_dense_and_residual_fusions():
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )
    config._attn_implementation = "eager"
    original = Qwen3DecoderLayer(config, layer_idx=0).eval()
    fused = copy.deepcopy(original)
    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0):
        attention = fuse_module(fused.self_attn, "model.self_attn")
        mlp = fuse_module(fused.mlp, "model.mlp")
        fused.input_layernorm = replace_rms_norm_class(fused.input_layernorm, 32)
        fused.post_attention_layernorm = replace_rms_norm_class(
            fused.post_attention_layernorm, 32
        )
        assert fuse_residual_norm(fused)
        wrapper = nn.Module()
        wrapper.model = fused
        weights = (
            (f"model.{name}", weight) for name, weight in original.state_dict().items()
        )
        weights = load_fused_weights(
            wrapper, weights, attention.stacked_mapping | mlp.stacked_mapping, set()
        )
        AutoWeightsLoader(wrapper).load_weights(weights)
        x = torch.randn(2, 5, 32)
        positions = (torch.ones(2, 5, 4), torch.zeros(2, 5, 4))
        with torch.no_grad():
            actual = fused(x, position_embeddings=positions)
            expected = original(x, position_embeddings=positions)
    torch.testing.assert_close(actual, expected)


def test_residual_fusion_compiles_with_dynamic_token_shapes():
    norm = replace_rms_norm_class(Qwen3RMSNorm(32), 32)
    model = Decoder(norm)
    assert fuse_residual_norm(model)
    norm.enter_torch_compile(num_tokens=7)
    compiled = torch.compile(model, backend="eager", dynamic=True, fullgraph=True)
    for length in (1, 7):
        x, residual = torch.randn(1, length, 32), torch.randn(1, length, 32)
        with torch.no_grad():
            actual, expected = compiled(x, residual), model(x, residual)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
