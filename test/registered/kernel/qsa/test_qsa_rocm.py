"""ROCm coverage for the QSA packed-varlen decode fallback.

On HIP there is neither flash_attn (FA2) nor flash-attn-4, so
``_resolve_flash_attn_varlen_func`` resolves to aiter's FA2-compatible varlen
kernel. The first two tests are device-independent (fake modules); the last one
runs the real aiter kernel against a torch reference and only executes on ROCm.
"""

import sys
from types import ModuleType

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend_module
from sglang.srt.layers.attention.qsa.mqa import HAS_TILELANG, qsa_mqa_decode
from sglang.srt.layers.attention.qsa.qsa_indexer import _mask_padded_group_locs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")


def _fake_aiter(varlen_func):
    module = ModuleType("aiter")
    module.flash_attn_varlen_func = varlen_func
    return module


def test_qsa_padded_group_reads_use_reserved_slot():
    source = torch.arange(6).reshape(3, 1, 2)
    group_locs = torch.tensor([[0, 1], [2, 3]])
    write_locs = torch.tensor([7, 0], dtype=torch.int32)

    safe_locs = _mask_padded_group_locs(group_locs, write_locs)

    assert safe_locs.tolist() == [[0, 1], [0, 0]]
    assert source[safe_locs].shape == (2, 2, 1, 2)


@pytest.mark.parametrize("returns_tuple", [False, True], ids=["tensor", "tuple"])
def test_qsa_hip_resolves_aiter_varlen_kernel(monkeypatch, returns_tuple):
    resolver = qsa_backend_module._resolve_flash_attn_varlen_func
    resolver.cache_clear()
    sentinel = object()
    calls = []

    def aiter_varlen(*args, **kwargs):
        calls.append(kwargs)
        return (sentinel, None) if returns_tuple else sentinel

    monkeypatch.setattr("sglang.srt.utils.is_sm121", lambda: False)
    monkeypatch.setattr("sglang.srt.utils.is_hip", lambda: True)
    monkeypatch.setitem(sys.modules, "aiter", _fake_aiter(aiter_varlen))

    try:
        func = resolver()
        assert func(q=1, causal=True) is sentinel
        assert calls == [{"q": 1, "causal": True}]
    finally:
        resolver.cache_clear()


def test_qsa_hip_without_aiter_falls_through_to_flash_attn(monkeypatch):
    resolver = qsa_backend_module._resolve_flash_attn_varlen_func
    resolver.cache_clear()
    fa2_func = object()
    flash_attn = ModuleType("flash_attn")
    flash_attn.flash_attn_varlen_func = fa2_func

    monkeypatch.setattr("sglang.srt.utils.is_sm121", lambda: False)
    monkeypatch.setattr("sglang.srt.utils.is_hip", lambda: True)
    monkeypatch.setitem(sys.modules, "aiter", None)  # makes `import aiter` fail
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    try:
        assert resolver() is fa2_func
    finally:
        resolver.cache_clear()


def test_qsa_hip_aiter_varlen_matches_torch_reference():
    if not (is_hip() and torch.cuda.is_available()):
        pytest.skip("ROCm-only kernel")
    pytest.importorskip("aiter")

    resolver = qsa_backend_module._resolve_flash_attn_varlen_func
    resolver.cache_clear()
    torch.manual_seed(2028)
    device = torch.device("cuda")
    # Qwen3.8-Flash-Next full-attention shape: 24 query heads, 2 KV heads, d=256.
    batch, topk = 3, 2048
    num_q_heads, num_kv_heads, head_dim = 24, 2, 256
    valid_counts = torch.tensor([5, topk, 700], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.cumsum(valid_counts, 0)
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    packed_k = torch.randn(
        int(valid_counts.sum()),
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    packed_v = torch.randn_like(packed_k)
    scale = head_dim**-0.5

    try:
        # Mirrors the decode call in QwenSparseAttnBackend._forward_paged_attention.
        out = resolver()(
            q=q,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=topk,
            softmax_scale=scale,
            causal=True,
        )
    finally:
        resolver.cache_clear()

    assert out.shape == (batch, num_q_heads, head_dim)
    repeats = num_q_heads // num_kv_heads
    for row in range(batch):
        start, end = int(cu_seqlens_k[row]), int(cu_seqlens_k[row + 1])
        keys = packed_k[start:end].float().repeat_interleave(repeats, dim=1)
        values = packed_v[start:end].float().repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys) * scale
        expected = torch.einsum("hk,khd->hd", scores.softmax(dim=-1), values)
        torch.testing.assert_close(out[row].float(), expected, atol=2e-2, rtol=2e-2)


def test_qsa_hip_tilelang_decode_matches_production_shape():
    """Exercise Qwen3.8's 4x128 indexer shape through the real ROCm MFMA path."""

    if not (is_hip() and torch.cuda.is_available()):
        pytest.skip("ROCm-only kernel")
    if not HAS_TILELANG:
        pytest.skip("TileLang is unavailable")

    torch.manual_seed(2029)
    device = torch.device("cuda")
    batch, num_q_heads, head_dim = 4, 4, 128
    page_size, max_pages = 64, 4
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    cache = torch.randn(12, page_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    page_table = torch.tensor(
        [[3, 1, 5, 8], [4, 2, 0, 9], [7, 6, 11, 10], [1, 8, 3, 5]],
        device=device,
        dtype=torch.int32,
    )
    context_lens = torch.tensor([1, 64, 130, 255], device=device, dtype=torch.int32)
    max_model_len = page_size * max_pages

    actual = qsa_mqa_decode(
        q,
        cache,
        page_table,
        context_lens,
        max_model_len=max_model_len,
    )
    torch.cuda.synchronize()

    gathered = cache[page_table.long(), :, 0].reshape(batch, max_model_len, head_dim)
    expected = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    expected = torch.relu(expected).sum(-1) / (head_dim**0.5)
    positions = torch.arange(max_model_len, device=device).unsqueeze(0)
    expected.masked_fill_(positions >= context_lens[:, None], -float("inf"))

    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite)
    torch.testing.assert_close(actual[finite], expected[finite], atol=5e-2, rtol=2e-2)
