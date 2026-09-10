"""Regression test for the QSA strided sparse-decode scratch zero-fill.

Poison the packed scratch with NaN, gather with the strided layout used by
`_forward_trtllm_sparse`, and require that (a) valid rows are copied exactly and
(b) every slot in [valid_count, stride) is zero, so the paged decode kernel can never
multiply masked probabilities into stale NaN/Inf bytes. Also checks the compact
(FA2 fallback) layout is unchanged. Intended for test/registered/kernel/qsa/.
"""

import pytest
import torch

from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_strided_gather_zero_fills_tail(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, topk, page, heads, dim = 3, 2051, 64, 2, 256
    pages_per_row = (topk + page - 1) // page
    stride = pages_per_row * page
    pool_rows = 8192
    k_pool = torch.randn(pool_rows, heads, dim, device=device, dtype=torch.bfloat16).to(
        dtype
    )
    v_pool = torch.randn(pool_rows, heads, dim, device=device, dtype=torch.bfloat16).to(
        dtype
    )
    seq_lens = torch.tensor([733, 109, 2500], device=device, dtype=torch.int32)
    req_to_token = (
        torch.randperm(pool_rows, device=device)[: batch * 2600]
        .reshape(batch, 2600)
        .to(torch.int32)
    )
    req_indices = torch.arange(batch, device=device, dtype=torch.int32)
    # top-k rows: the first min(seq_len, topk) logical positions, then -1 padding
    indices = torch.full((batch, topk), -1, device=device, dtype=torch.int32)
    for b in range(batch):
        n = min(int(seq_lens[b]), topk)
        indices[b, :n] = torch.arange(n, device=device, dtype=torch.int32)
    cu_strided = torch.arange(batch + 1, device=device, dtype=torch.int32) * stride
    # the scratch is always in the compute dtype (bf16); an FP8 pool is dequantized on the way in
    k_scale, v_scale = (1.0, 1.0) if dtype == torch.bfloat16 else (0.5, 2.0)
    packed_k = torch.full(
        (batch * stride, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    packed_v = packed_k.clone()

    qwen_sparse_kv_extraction_compact_triton(
        k_pool,
        v_pool,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_strided,
        packed_k,
        packed_v,
        batch,
        topk,
        zero_fill_cols=stride,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    pk, pv = (
        packed_k.float().view(batch, stride, heads, dim),
        packed_v.float().view(batch, stride, heads, dim),
    )
    assert torch.isfinite(pk).all() and torch.isfinite(pv).all()
    for b in range(batch):
        n = min(int(seq_lens[b]), topk)
        slots = req_to_token[b, :n].long()
        torch.testing.assert_close(
            pk[b, :n], (k_pool[slots].float() * k_scale).to(torch.bfloat16).float()
        )
        torch.testing.assert_close(
            pv[b, :n], (v_pool[slots].float() * v_scale).to(torch.bfloat16).float()
        )
        assert (pk[b, n:] == 0).all() and (pv[b, n:] == 0).all()


def test_compact_gather_unchanged():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, topk, heads, dim = 2, 2051, 2, 256
    k_pool = torch.randn(4096, heads, dim, device=device, dtype=torch.bfloat16)
    v_pool = torch.randn(4096, heads, dim, device=device, dtype=torch.bfloat16)
    seq_lens = torch.tensor([300, 50], device=device, dtype=torch.int32)
    req_to_token = torch.arange(batch * 512, device=device, dtype=torch.int32).reshape(
        batch, 512
    )
    req_indices = torch.arange(batch, device=device, dtype=torch.int32)
    indices = torch.full((batch, topk), -1, device=device, dtype=torch.int32)
    for b in range(batch):
        indices[b, : int(seq_lens[b])] = torch.arange(
            int(seq_lens[b]), device=device, dtype=torch.int32
        )
    counts = torch.empty(batch, device=device, dtype=torch.int32)
    cu_k = torch.empty(batch + 1, device=device, dtype=torch.int32)
    qwen_sparse_fa2_cu_seqlens_triton(seq_lens, indices, counts, cu_k, batch, topk)
    assert cu_k.tolist() == [0, 300, 350]
    packed_k = torch.full(
        (batch * topk, heads, dim), float("nan"), device=device, dtype=torch.bfloat16
    )
    packed_v = packed_k.clone()
    qwen_sparse_kv_extraction_compact_triton(
        k_pool,
        v_pool,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        packed_k,
        packed_v,
        batch,
        topk,
    )
    torch.testing.assert_close(packed_k[:300], k_pool[req_to_token[0, :300].long()])
    torch.testing.assert_close(packed_k[300:350], k_pool[req_to_token[1, :50].long()])
    # compact layout leaves the region past the packed rows untouched (still NaN)
    assert torch.isnan(packed_k[350:]).all()
