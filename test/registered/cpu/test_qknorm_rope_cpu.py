import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import sgl_kernel  # noqa: F401
from sglang.kernels.ops.diffusion.rope.qknorm_rope_jit import (
    can_use_fused_inplace_qknorm_rope_cpu,
    fused_inplace_qknorm_rope,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-b-test-cpu")

torch.manual_seed(1234)

DTYPE = torch.bfloat16
NUM_HEADS = 56
HEAD_DIM = 128
ROPE_DIM = 96
EPS = 1e-6
MINIMAX_SOURCE = Path(__file__).parents[3] / "python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"


def create_cos_sin_cache(rotary_dim: int, max_position: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(DTYPE)


def rmsnorm_baseline(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    norm = nn.RMSNorm(x.shape[-1], eps=eps, dtype=x.dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_baseline(x: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
    half = cos_sin_cache.shape[-1] // 2
    cos_half, sin_half = cos_sin_cache.split(half, dim=-1)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
    x_rot, x_pass = x[..., : cos.shape[-1]], x[..., cos.shape[-1] :]
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


def baseline_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_norm = rmsnorm_baseline(q, q_weight, EPS)
    k_norm = rmsnorm_baseline(k, k_weight, EPS)
    cache = cos_sin_cache.index_select(0, positions)
    return apply_rope_baseline(q_norm, cache), apply_rope_baseline(k_norm, cache)


@pytest.mark.parametrize("num_tokens", [1, 17, 257])
def test_fused_qknorm_rope_cpu_matches_baseline(num_tokens: int) -> None:
    q = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE)
    k = torch.randn_like(q)
    q_weight = torch.randn(HEAD_DIM, dtype=DTYPE)
    k_weight = torch.randn(HEAD_DIM, dtype=DTYPE)
    positions = torch.arange(num_tokens, dtype=torch.int64)
    cos_sin_cache = create_cos_sin_cache(ROPE_DIM, max_position=num_tokens)

    q_ref, k_ref = baseline_qknorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
    )
    q_fused, k_fused = q.clone(), k.clone()
    fused_inplace_qknorm_rope(
        q_fused,
        k_fused,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=True,
        eps=EPS,
        head_dim=HEAD_DIM,
        rope_dim=ROPE_DIM,
        round_norm_before_rope=True,
    )

    torch.testing.assert_close(q_ref, q_fused, atol=6.25e-2, rtol=5e-2)
    torch.testing.assert_close(k_ref, k_fused, atol=6.25e-2, rtol=5e-2)


def test_fused_qknorm_rope_cpu_mutates_inputs_in_place() -> None:
    q = torch.randn(8, NUM_HEADS, HEAD_DIM, dtype=DTYPE)
    k = torch.randn_like(q)
    q_weight = torch.randn(HEAD_DIM, dtype=DTYPE)
    k_weight = torch.randn(HEAD_DIM, dtype=DTYPE)
    positions = torch.arange(8, dtype=torch.int64)
    cos_sin_cache = create_cos_sin_cache(ROPE_DIM, max_position=8)
    q_ptr = q.data_ptr()
    k_ptr = k.data_ptr()

    fused_inplace_qknorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=True,
        eps=EPS,
        head_dim=HEAD_DIM,
        rope_dim=ROPE_DIM,
        round_norm_before_rope=True,
    )

    assert q.data_ptr() == q_ptr
    assert k.data_ptr() == k_ptr


def test_can_use_fused_qknorm_rope_cpu_release_slice() -> None:
    assert can_use_fused_inplace_qknorm_rope_cpu(
        HEAD_DIM,
        ROPE_DIM,
        True,
        DTYPE,
        cache_dtype=DTYPE,
        round_norm_before_rope=True,
    )
    assert not can_use_fused_inplace_qknorm_rope_cpu(
        HEAD_DIM,
        ROPE_DIM,
        False,
        DTYPE,
        cache_dtype=DTYPE,
        round_norm_before_rope=True,
    )
    assert not can_use_fused_inplace_qknorm_rope_cpu(
        HEAD_DIM,
        ROPE_DIM,
        True,
        torch.float16,
        cache_dtype=torch.float16,
        round_norm_before_rope=True,
    )
    assert not can_use_fused_inplace_qknorm_rope_cpu(
        64,
        ROPE_DIM,
        True,
        DTYPE,
        cache_dtype=DTYPE,
        round_norm_before_rope=True,
    )
    assert not can_use_fused_inplace_qknorm_rope_cpu(
        HEAD_DIM,
        ROPE_DIM,
        True,
        DTYPE,
        cache_dtype=DTYPE,
        round_norm_before_rope=False,
    )


def test_minimax_attention_runtime_uses_fused_cpu_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as minimax_h3_mod

    attn = object.__new__(minimax_h3_mod.MiniMaxH3Attention)
    nn.Module.__init__(attn)
    attn.local_inner_dim = NUM_HEADS * HEAD_DIM
    attn.num_heads = NUM_HEADS
    attn.head_dim = HEAD_DIM
    attn._use_fused_qknorm_rope = True
    attn.q_norm = nn.RMSNorm(HEAD_DIM, eps=EPS, dtype=DTYPE)
    attn.k_norm = nn.RMSNorm(HEAD_DIM, eps=EPS, dtype=DTYPE)
    attn.bcg_breakpoint = False

    num_tokens = 4
    q = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    qkv = torch.cat([q.reshape(num_tokens, -1), k.reshape(num_tokens, -1), v.reshape(num_tokens, -1)], dim=-1)

    class QKVProj:
        def __call__(self, x):
            return qkv.clone(), None

    class OutProj:
        def __call__(self, x):
            return x, None

    attn.qkv_proj = QKVProj()
    attn.out_proj = OutProj()

    calls = {"fused": 0}
    real_fused = minimax_h3_mod.fused_inplace_qknorm_rope

    def fused_spy(*args, **kwargs):
        calls["fused"] += 1
        return real_fused(*args, **kwargs)

    def should_not_run(*args, **kwargs):
        raise AssertionError("fallback path should not run")

    monkeypatch.setattr(minimax_h3_mod, "fused_inplace_qknorm_rope", fused_spy)
    monkeypatch.setattr(minimax_h3_mod, "_apply_qk_norm", should_not_run)
    monkeypatch.setattr(minimax_h3_mod, "_apply_rope_qk", should_not_run)
    monkeypatch.setattr(minimax_h3_mod, "_minimax_h3_attention_core_impl", lambda self, q, k, v, **kwargs: q)
    monkeypatch.setattr(minimax_h3_mod, "_minimax_h3_attention_core_bcg", lambda self, q, k, v, **kwargs: q)

    out = minimax_h3_mod.MiniMaxH3Attention.forward(
        attn,
        torch.empty(num_tokens, 1, dtype=DTYPE),
        rope_cache=(create_cos_sin_cache(ROPE_DIM, num_tokens), torch.arange(num_tokens, dtype=torch.int64)),
        cu_seqlens=torch.tensor([0, num_tokens], dtype=torch.int32),
        cu_seqlens_host=None,
        max_seqlen=num_tokens,
    )

    assert out.shape == (num_tokens, NUM_HEADS * HEAD_DIM)
    assert calls["fused"] == 1


def test_minimax_cpu_dispatch_gate_present_in_source() -> None:
    text = MINIMAX_SOURCE.read_text()
    assert "can_use_fused_inplace_qknorm_rope_cpu" in text
    assert "current_platform.is_cpu()" in text
    assert "fused_inplace_qknorm_rope(" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
