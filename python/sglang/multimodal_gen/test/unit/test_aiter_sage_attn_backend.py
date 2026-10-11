# SPDX-License-Identifier: Apache-2.0
"""AITER Sage backend: packed varlen attention (no GPU or aiter needed)."""

import sys
import types

import pytest
import torch


@pytest.fixture
def calls(monkeypatch):
    """Stub aiter's fav3_sage kernel; record the sequence length of each call."""
    calls = []

    def fav3_sage_wrapper_func(q, k, v):
        calls.append(q.shape[1])
        return q + 2

    fav3 = types.ModuleType("aiter.ops.triton.attention.fav3_sage")
    fav3.fav3_sage_wrapper_func = fav3_sage_wrapper_func
    for name in (
        "aiter",
        "aiter.ops",
        "aiter.ops.triton",
        "aiter.ops.triton.attention",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, fav3.__name__, fav3)
    return calls


def _impl(**extra):
    from sglang.multimodal_gen.runtime.layers.attention.backends.aiter_sage import (
        AITERSageImpl,
    )

    return AITERSageImpl(num_heads=4, head_size=128, softmax_scale=0.125, **extra)


def _packed_qkv(tokens):
    return [torch.randn(tokens, 4, 128, dtype=torch.bfloat16) for _ in range(3)]


def test_backend_supports_packed_varlen():
    from sglang.multimodal_gen.runtime.layers.attention.backends.aiter_sage import (
        AITERSageBackend,
    )

    assert AITERSageBackend.supports_packed_varlen()


def test_forward_varlen_runs_each_segment(calls):
    q, k, v = _packed_qkv(10)
    out = _impl().forward_varlen(
        q, k, v, cu_seqlens=torch.tensor([0, 3, 3, 10]), max_seqlen=7
    )
    assert calls == [3, 7]
    assert torch.equal(out, q + 2)


def test_forward_varlen_single_segment(calls):
    q, k, v = _packed_qkv(10)
    out = _impl().forward_varlen(
        q,
        k,
        v,
        cu_seqlens=torch.tensor([0, 10]),
        max_seqlen=10,
        cu_seqlens_host=[0, 10],
    )
    assert calls == [10]
    assert torch.equal(out, q + 2)


def test_forward_varlen_skips_trailing_padding(calls):
    q, k, v = _packed_qkv(10)
    out = _impl(packed_trailing_padding=True).forward_varlen(
        q, k, v, cu_seqlens=torch.tensor([0, 6, 10]), max_seqlen=6
    )
    assert calls == [6]
    assert torch.equal(out[:6], q[:6] + 2)
    assert not out[6:].any()


def test_forward_varlen_without_padding_flag_runs_the_tail_segment(calls):
    q, k, v = _packed_qkv(10)
    _impl().forward_varlen(q, k, v, cu_seqlens=torch.tensor([0, 6, 10]), max_seqlen=6)
    assert calls == [6, 4]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
