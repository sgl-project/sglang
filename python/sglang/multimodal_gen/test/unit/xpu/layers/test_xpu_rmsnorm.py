# SPDX-License-Identifier: Apache-2.0
"""Tests for the strided head-view path of RMSNorm.forward_xpu."""

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    _is_packed_head_view,
)

_xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
requires_xpu = pytest.mark.skipif(not _xpu_available, reason="XPU not available")

# Qwen3 text-encoder q/k: q is [B, S, 32, 128] and k [B, S, 8, 128] of a 6144-wide QKV row.
_Q_HEADS, _KV_HEADS, _HEAD_DIM = 32, 8, 128


def _qkv_views(
    batch: int, seq: int, device: str = "cpu", dtype: torch.dtype = torch.bfloat16
):
    q_size, kv_size = _Q_HEADS * _HEAD_DIM, _KV_HEADS * _HEAD_DIM
    qkv = torch.randn(batch, seq, q_size + 2 * kv_size, dtype=dtype, device=device)
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    return (
        q.reshape(batch, seq, _Q_HEADS, _HEAD_DIM),
        k.reshape(batch, seq, _KV_HEADS, _HEAD_DIM),
    )


def test_packed_head_view_predicate():
    """Only views that x.view(-1, heads, head_dim) can express without a copy qualify."""
    for batch in (1, 2):
        q, k = _qkv_views(batch=batch, seq=5)
        assert _is_packed_head_view(q) and _is_packed_head_view(k)
        q.view(-1, _Q_HEADS, _HEAD_DIM)  # must not raise

    # contiguous input keeps the plain 2-D path
    assert not _is_packed_head_view(torch.empty(2, 5, _Q_HEADS, _HEAD_DIM))
    # heads not dense within a row: the kernel's head stride would be wrong
    assert not _is_packed_head_view(
        torch.empty(2, 5, _HEAD_DIM, _Q_HEADS).transpose(-1, -2)
    )
    # batch and sequence not mergeable into one row axis
    assert not _is_packed_head_view(
        torch.empty(5, 2, _Q_HEADS, _HEAD_DIM).transpose(0, 1)
    )


@requires_xpu
@pytest.mark.parametrize("batch", [1, 2])
def test_strided_head_view_is_bitwise_equal_to_contiguous(batch):
    torch.manual_seed(42)
    norm = RMSNorm(_HEAD_DIM, eps=1e-6).to(device="xpu", dtype=torch.bfloat16)
    q, k = _qkv_views(batch=batch, seq=512, device="xpu")
    for x in (q, k):
        out = norm.forward_xpu(x)
        ref = norm.forward_xpu(x.contiguous())
        assert out.shape == x.shape
        assert torch.equal(out, ref)
