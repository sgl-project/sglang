# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from sglang.srt.models.transformers.fusers.rope import (
    PairedRotaryEmbedding,
    replace_rotary_embedding,
)
from sglang.srt.models.transformers.layers import replace_rms_norm_class
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel validation"
)


def _rotary_config(rope_type):
    scaling = {"rope_type": rope_type, "rope_theta": 10000.0}
    if rope_type in {"linear", "llama3"}:
        scaling["factor"] = 4.0
    if rope_type == "llama3":
        scaling.update(
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=64,
        )
    return Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_parameters=scaling,
    )


@pytest.mark.parametrize("norm_type", [Qwen3RMSNorm, Gemma3RMSNorm])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [80, 512, 4096])
def test_residual_norm_cuda_kernel_matches_hf(norm_type, dtype, width):
    torch.manual_seed(73)
    norm = norm_type(width).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.weight.normal_()
    fused_norm = replace_rms_norm_class(norm, width).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        fused_norm.weight.copy_(norm.weight)
    x, residual = (
        torch.randn(2, 7, width, device="cuda", dtype=dtype),
        torch.randn(2, 7, width, device="cuda", dtype=dtype),
    )
    expected_residual = x + residual
    with torch.no_grad():
        expected = norm(expected_residual)
        output, summed = fused_norm.forward_add(x, residual)
    torch.testing.assert_close(summed, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [(8, 2), (4, 4)])
def test_paired_rotary_cuda_kernel(dtype, heads):
    q = torch.randn(2, 17, heads[0], 128, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(2, 17, heads[1], 128, device="cuda", dtype=dtype).transpose(1, 2)
    angles = torch.randn(2, 17, 128, device="cuda", dtype=dtype)
    cosine, sine = angles.cos(), angles.sin()
    actual = PairedRotaryEmbedding()(q, k, cosine, sine)
    expected = apply_rotary_pos_emb(q, k, cosine, sine)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert value.transpose(1, 2).is_contiguous()


@pytest.mark.parametrize("norm_type", [Qwen3RMSNorm, Gemma3RMSNorm])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_head_sliced_norm_kernel_matches_contiguous_path(norm_type, dtype):
    torch.manual_seed(73)
    norm = norm_type(128).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.weight.normal_()
    fused_norm = replace_rms_norm_class(norm, 128).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        fused_norm.weight.copy_(norm.weight)
    packed = torch.randn(1, 9, 48 * 128, device="cuda", dtype=dtype)
    sliced = packed[..., : 32 * 128].view(1, 9, 32, 128)
    assert not sliced.is_contiguous()
    with torch.no_grad():
        expected = norm(sliced.contiguous())
        output = fused_norm(sliced)
    assert output.is_contiguous() and output.shape == sliced.shape
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("rope_type", ["default", "linear", "llama3"])
def test_cached_rotary_cuda_kernel_including_cache_misses(rope_type):
    original = Qwen3RotaryEmbedding(_rotary_config(rope_type)).cuda()
    fused = replace_rotary_embedding(original)
    positions = torch.tensor([[0, 7, 64, 96, 128, 16384]], device="cuda")
    x = torch.empty(1, 6, 32, device="cuda", dtype=torch.bfloat16)
    actual, expected = fused(x, positions), original(x, positions)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0.005, atol=0.005)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
