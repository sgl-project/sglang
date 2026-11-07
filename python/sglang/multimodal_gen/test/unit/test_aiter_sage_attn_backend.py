# SPDX-License-Identifier: Apache-2.0
"""AITER Sage backend: kernel arguments, gfx1201 kernel selection and packed varlen (no GPU or aiter needed)."""

import sys
import types

import pytest
import torch

MODULE = "sglang.multimodal_gen.runtime.layers.attention.backends.aiter_sage"


class _Calls(list):
    kwargs: list
    contiguous: list


@pytest.fixture
def calls(monkeypatch):
    """Stub aiter's kernels; record (kernel, sequence length[, softmax_scale]) and fav3 kwargs."""
    calls = _Calls()
    calls.kwargs = []
    calls.contiguous = []

    def gfx1201_sage_attention(q, k, v, softmax_scale=None):
        calls.append(("gfx1201", q.shape[1], softmax_scale))
        return q + 1

    def fav3_sage_wrapper_func(q, k, v, softmax_scale=None, causal=False):
        calls.append(("fav3", q.shape[1]))
        calls.kwargs.append({"softmax_scale": softmax_scale, "causal": causal})
        calls.contiguous.append(all(t.is_contiguous() for t in (q, k, v)))
        return q + 2

    aiter = types.ModuleType("aiter")
    aiter.gfx1201_sage_attention = gfx1201_sage_attention
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    fav3 = types.ModuleType("aiter.ops.triton.attention.fav3_sage")
    fav3.fav3_sage_wrapper_func = fav3_sage_wrapper_func
    for name in ("aiter.ops", "aiter.ops.triton", "aiter.ops.triton.attention"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, fav3.__name__, fav3)
    return calls


def _impl(monkeypatch, arch="gfx942", *, head_size=128, causal=False, **extra):
    from sglang.multimodal_gen.runtime.layers.attention.backends.aiter_sage import (
        AITERSageImpl,
    )

    monkeypatch.setattr(f"{MODULE}.current_platform.is_rocm", lambda: True)
    monkeypatch.setattr(
        f"{MODULE}.current_platform.get_gcn_arch_name",
        lambda: arch,
        raising=False,
    )
    return AITERSageImpl(
        num_heads=4,
        head_size=head_size,
        softmax_scale=0.125,
        causal=causal,
        **extra,
    )


def test_forward_passes_softmax_scale_and_causal(monkeypatch, calls):
    q, k, v = (torch.randn(1, 16, 4, 128, dtype=torch.bfloat16) for _ in range(3))
    _impl(monkeypatch).forward(q, k, v)
    _impl(monkeypatch, "gfx1201", causal=True).forward(q, k, v)
    assert calls == [("fav3", 16), ("fav3", 16)]
    assert calls.kwargs == [
        {"softmax_scale": 0.125, "causal": False},
        {"softmax_scale": 0.125, "causal": True},
    ]


def test_forward_makes_strided_inputs_contiguous(calls, monkeypatch):
    qkv = torch.randn(1, 16, 3 * 4 * 128, dtype=torch.bfloat16)
    q, k, v = (t.view(1, 16, 4, 128) for t in qkv.split(4 * 128, dim=-1))
    assert not v.is_contiguous()
    out = _impl(monkeypatch).forward(q, k, v)
    assert calls.contiguous == [True]
    assert torch.equal(out, q + 2)


def _packed_qkv(tokens):
    return [torch.randn(tokens, 4, 128, dtype=torch.bfloat16) for _ in range(3)]


def test_backend_supports_packed_varlen():
    from sglang.multimodal_gen.runtime.layers.attention.backends.aiter_sage import (
        AITERSageBackend,
    )

    assert AITERSageBackend.supports_packed_varlen()


def test_forward_varlen_runs_each_segment(monkeypatch, calls):
    q, k, v = _packed_qkv(10)
    out = _impl(monkeypatch).forward_varlen(
        q, k, v, cu_seqlens=torch.tensor([0, 3, 3, 10]), max_seqlen=7
    )
    assert calls == [("fav3", 3), ("fav3", 7)]
    assert torch.equal(out, q + 2)


def test_forward_varlen_single_segment(monkeypatch, calls):
    q, k, v = _packed_qkv(10)
    out = _impl(monkeypatch).forward_varlen(
        q,
        k,
        v,
        cu_seqlens=torch.tensor([0, 10]),
        max_seqlen=10,
        cu_seqlens_host=[0, 10],
    )
    assert calls == [("fav3", 10)]
    assert torch.equal(out, q + 2)


def test_forward_varlen_skips_trailing_padding(monkeypatch, calls):
    q, k, v = _packed_qkv(10)
    out = _impl(monkeypatch, packed_trailing_padding=True).forward_varlen(
        q, k, v, cu_seqlens=torch.tensor([0, 6, 10]), max_seqlen=6
    )
    assert calls == [("fav3", 6)]
    assert torch.equal(out[:6], q[:6] + 2)
    assert not out[6:].any()


def test_forward_varlen_without_padding_flag_runs_the_tail_segment(monkeypatch, calls):
    q, k, v = _packed_qkv(10)
    _impl(monkeypatch).forward_varlen(
        q, k, v, cu_seqlens=torch.tensor([0, 6, 10]), max_seqlen=6
    )
    assert calls == [("fav3", 6), ("fav3", 4)]


def test_gfx1201_uses_the_gfx1201_kernel_with_the_softmax_scale(monkeypatch, calls):
    q, k, v = (torch.randn(1, 16, 4, 128, dtype=torch.bfloat16) for _ in range(3))
    out = _impl(monkeypatch, "gfx1201").forward(q, k, v)
    assert calls == [("gfx1201", 16, 0.125)]
    assert torch.equal(out, q + 1)


def test_gfx1201_packed_varlen_uses_the_gfx1201_kernel(monkeypatch, calls):
    q, k, v = _packed_qkv(10)
    impl = _impl(monkeypatch, "gfx1201", packed_trailing_padding=True)
    impl.forward_varlen(q, k, v, cu_seqlens=torch.tensor([0, 6, 10]), max_seqlen=6)
    assert calls == [("gfx1201", 6, 0.125)]


@pytest.mark.parametrize(
    "arch, head_size, causal, num_kv_heads, dtype, kv_tokens",
    [
        ("gfx942", 128, False, None, torch.bfloat16, 16),
        ("gfx950", 128, False, None, torch.bfloat16, 16),
        ("gfx1201", 64, False, None, torch.bfloat16, 16),
        ("gfx1201", 128, True, None, torch.bfloat16, 16),
        ("gfx1201", 128, False, 2, torch.bfloat16, 16),
        ("gfx1201", 128, False, None, torch.float16, 16),
        ("gfx1201", 128, False, None, torch.bfloat16, 24),
    ],
)
def test_other_cases_keep_the_triton_kernel(
    monkeypatch, calls, arch, head_size, causal, num_kv_heads, dtype, kv_tokens
):
    q = torch.randn(1, 16, 4, head_size).to(dtype)
    k, v = (torch.randn(1, kv_tokens, 4, head_size).to(dtype) for _ in range(2))
    impl = _impl(
        monkeypatch, arch, head_size=head_size, causal=causal, num_kv_heads=num_kv_heads
    )
    impl.forward(q, k, v)
    assert calls == [("fav3", 16)]


def test_gfx1201_without_the_aiter_op_keeps_the_triton_kernel(monkeypatch, calls):
    del sys.modules["aiter"].gfx1201_sage_attention
    q, k, v = (torch.randn(1, 16, 4, 128, dtype=torch.bfloat16) for _ in range(3))
    _impl(monkeypatch, "gfx1201").forward(q, k, v)
    assert calls == [("fav3", 16)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
