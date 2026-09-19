# SPDX-License-Identifier: Apache-2.0
"""SenseNova RoPE must preserve the eager bf16 rounding and sliced-head layout."""

import pytest
import torch

from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify import modeling_qwen3
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def eager_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)

    def rotate(x):
        left, right = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-right, left), dim=-1) * sin

    return rotate(q), rotate(k)


def inputs(batch, seq, axis, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(2026)
    q = torch.randn(batch, seq, 32, 64, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(batch, seq, 8, 64, device="cuda", dtype=dtype, generator=generator)
    if axis != "t":
        start = 0 if axis == "h" else 32
        q, k = q[..., start : start + 32], k[..., start : start + 32]
    angles = torch.randn(batch, seq, q.shape[-1], device="cuda", generator=generator)
    return (
        q.transpose(1, 2),
        k.transpose(1, 2),
        angles.cos().to(dtype),
        angles.sin().to(dtype),
    )


@pytest.mark.parametrize("batch,seq", [(1, 128), (1, 4096), (2, 129)])
@pytest.mark.parametrize("axis", ["t", "h", "w"])
@torch.no_grad()
def test_sensenova_rope_matches_eager_on_strided_heads(batch, seq, axis):
    q, k, cos, sin = inputs(batch, seq, axis)
    q_before, k_before = q.clone(), k.clone()
    expected = eager_rope(q, k, cos, sin)
    actual = modeling_qwen3.apply_rotary_pos_emb(q, k, cos, sin)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(q, q_before)
    assert torch.equal(k, k_before)
    assert actual[0].data_ptr() != q.data_ptr()
    assert actual[1].data_ptr() != k.data_ptr()


@pytest.mark.parametrize(
    "case", ["short", "fp16", "fp32", "mixed_tables", "broadcast", "seq_first", "grad"]
)
def test_sensenova_rope_fallback_preserves_dtype_broadcast_and_grad(monkeypatch, case):
    from sglang.kernels.ops.diffusion.rope import rope_rotate_half_bitexact

    def unexpected_fused_call(*args, **kwargs):
        raise AssertionError("Unsupported input reached the fused kernel")

    monkeypatch.setattr(
        rope_rotate_half_bitexact,
        "fused_rope_rotate_half_bitexact",
        unexpected_fused_call,
    )
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(case, torch.bfloat16)
    q, k, cos, sin = inputs(2, 37 if case == "short" else 128, "w", dtype)
    unsqueeze_dim = 1
    if case == "mixed_tables":
        cos, sin = cos.float(), sin.float()
    if case == "broadcast":
        cos, sin = cos[:1], sin[:1]
    if case == "seq_first":
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        unsqueeze_dim = 2
    with torch.set_grad_enabled(case == "grad"):
        q.requires_grad_(case == "grad")
        actual = modeling_qwen3.apply_rotary_pos_emb(
            q, k, cos, sin, unsqueeze_dim=unsqueeze_dim
        )
        expected = eager_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        if case == "grad":
            actual[0].sum().backward()
            assert q.grad is not None and torch.isfinite(q.grad).all()


@torch.no_grad()
def test_sensenova_attention_with_fused_rope_matches_eager(monkeypatch):
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
        NEOLLMConfig,
    )

    config = NEOLLMConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=1024,
    )
    config._attn_implementation = "eager"
    attention = (
        modeling_qwen3.Qwen3Attention(config, layer_idx=0)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    hidden = torch.randn(1, 129, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(129, device="cuda")
    indexes = torch.stack((positions, positions * 2, positions * 3))
    actual, _ = attention.forward_gen(hidden, indexes, None)
    monkeypatch.setattr(modeling_qwen3, "apply_rotary_pos_emb", eager_rope)
    expected, _ = attention.forward_gen(hidden, indexes, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
