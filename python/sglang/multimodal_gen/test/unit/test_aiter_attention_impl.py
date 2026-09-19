# SPDX-License-Identifier: Apache-2.0
"""AITer attention impl construction guards (ROCm-only; skipped elsewhere)."""

import pytest

HEAD_SIZE = 128


def _impl_cls():
    # `aiter` ships with ROCm only, and importing the backend module needs it.
    pytest.importorskip("aiter", reason="AITer is a ROCm-only dependency")
    from sglang.multimodal_gen.runtime.layers.attention.backends.aiter import AITerImpl

    return AITerImpl


def _build(num_heads: int, num_kv_heads: int | None):
    return _impl_cls()(
        num_heads=num_heads,
        head_size=HEAD_SIZE,
        softmax_scale=HEAD_SIZE**-0.5,
        num_kv_heads=num_kv_heads,
    )


@pytest.mark.parametrize("num_kv_heads", [32, 8, 1, None])
def test_accepts_grouped_and_multi_query_kv_heads(num_kv_heads):
    # aiter's mha entry points broadcast each KV head across its group of
    # query heads, so Cosmos3-style GQA cross-attention is supported.
    assert _build(32, num_kv_heads).softmax_scale == pytest.approx(HEAD_SIZE**-0.5)


def test_rejects_kv_heads_that_do_not_divide_the_query_heads():
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        _build(32, 5)


def test_advertises_a_native_varlen_kernel():
    # USPAttention's masked path keys off this to pick packed varlen over the
    # SDPA fallback; clearing it silently drops AITer back to SDPA.
    assert _impl_cls().has_native_varlen_kernel()


def test_key_bounds_reach_the_aiter_kernel(monkeypatch):
    # The kv-gather masked site attends queries and keys with different
    # segment lengths; dropping cu_seqlens_k attends over the wrong ranges.
    import torch

    from sglang.multimodal_gen.runtime.layers.attention.backends import aiter as mod

    seen = {}

    def fake_varlen(**kwargs):
        seen.update(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(mod.aiter, "flash_attn_varlen_func", fake_varlen)
    monkeypatch.setattr(mod, "USE_AITER_GFX942", False)

    impl = _build(32, 32)
    packed = torch.randn(6, 32, HEAD_SIZE, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 4, 9], dtype=torch.int32, device="cuda")

    impl.forward_varlen(
        packed,
        packed,
        packed,
        cu_seqlens=cu_q,
        max_seqlen=3,
        cu_seqlens_k=cu_k,
        max_seqlen_k=5,
    )

    assert seen["cu_seqlens_q"].tolist() == [0, 3, 6]
    assert seen["cu_seqlens_k"].tolist() == [0, 4, 9]
    assert seen["max_seqlen_q"] == 3
    assert seen["max_seqlen_k"] == 5


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
