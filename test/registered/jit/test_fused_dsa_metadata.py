import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")


@pytest.mark.parametrize("real_page_size", [1, 4])
@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
def test_fused_dsa_decode_metadata(real_page_size, seq_dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from sglang.srt.layers.attention.triton_ops.dsa_metadata import (
        fused_dsa_decode_metadata,
    )

    device = "cuda"
    bs = 7
    req_to_token_cols = 80
    max_len = 37
    topk = 23

    seq_lens = torch.tensor([37, 12, 23, 4, 31, 18, 9], dtype=seq_dtype, device=device)
    req_pool_indices = torch.tensor(
        [3, 0, 5, 2, 7, 1, 6], dtype=torch.int64, device=device
    )
    req_to_token = (
        torch.arange(8 * req_to_token_cols, dtype=torch.int32, device=device)
        .view(8, req_to_token_cols)
        .mul_(3)
    )

    cache_seqlens = torch.empty(bs, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=device)
    page_table_1 = torch.full((bs, max_len + 11), -1, dtype=torch.int32, device=device)
    dsa_cache_seqlens = torch.empty(bs, dtype=torch.int32, device=device)
    dsa_cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=device)

    if real_page_size > 1:
        real_cols = (max_len + real_page_size - 1) // real_page_size
        real_page_table = torch.full(
            (bs, real_cols + 3), -1, dtype=torch.int32, device=device
        )
    else:
        real_page_table = page_table_1

    fused_dsa_decode_metadata(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_table_1=page_table_1,
        dsa_cache_seqlens=dsa_cache_seqlens,
        dsa_cu_seqlens_k=dsa_cu_seqlens_k,
        real_page_table=real_page_table,
        bs=bs,
        max_len=max_len,
        dsa_index_topk=topk,
        real_page_size=real_page_size,
    )
    torch.cuda.synchronize()

    ref_cache = seq_lens.to(torch.int32)
    ref_dsa = ref_cache.clamp(max=topk)
    ref_page = req_to_token[req_pool_indices, :max_len]

    assert torch.equal(cache_seqlens, ref_cache)
    assert torch.equal(
        cu_seqlens_k, torch.nn.functional.pad(torch.cumsum(ref_cache, 0), (1, 0))
    )
    assert torch.equal(dsa_cache_seqlens, ref_dsa)
    assert torch.equal(
        dsa_cu_seqlens_k, torch.nn.functional.pad(torch.cumsum(ref_dsa, 0), (1, 0))
    )
    assert torch.equal(page_table_1[:, :max_len], ref_page)
    assert torch.all(page_table_1[:, max_len:] == -1)

    if real_page_size > 1:
        ref_real = ref_page[:, torch.arange(0, max_len, real_page_size, device=device)]
        ref_real = ref_real // real_page_size
        assert torch.equal(real_page_table[:, : ref_real.shape[1]], ref_real)
        assert torch.all(real_page_table[:, ref_real.shape[1] :] == -1)


@pytest.mark.parametrize("real_page_size", [1, 4])
@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("next_n", [1, 3])
def test_fused_dsa_target_verify_metadata(real_page_size, seq_dtype, next_n):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from sglang.srt.layers.attention.triton_ops.dsa_metadata import (
        fused_dsa_target_verify_metadata,
    )

    device = "cuda"
    bs = 5
    req_to_token_cols = 80
    max_seq_len = 37
    max_seqlen_k = max_seq_len + next_n
    expanded_size = bs * next_n
    topk = 23

    seq_lens = torch.tensor([37, 12, 23, 4, 31], dtype=seq_dtype, device=device)
    req_pool_indices = torch.tensor([3, 0, 5, 2, 7], dtype=torch.int64, device=device)
    req_to_token = (
        torch.arange(8 * req_to_token_cols, dtype=torch.int32, device=device)
        .view(8, req_to_token_cols)
        .mul_(3)
    )

    cache_seqlens = torch.empty(bs, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=device)
    page_table_1 = torch.full(
        (expanded_size, max_seqlen_k + 11), -1, dtype=torch.int32, device=device
    )
    seqlens_expanded = torch.empty(expanded_size, dtype=torch.int32, device=device)
    dsa_cache_seqlens = torch.empty(expanded_size, dtype=torch.int32, device=device)
    dsa_cu_seqlens_k = torch.empty(expanded_size + 1, dtype=torch.int32, device=device)

    if real_page_size > 1:
        real_cols = (max_seqlen_k + real_page_size - 1) // real_page_size
        real_page_table = torch.full(
            (expanded_size, real_cols + 3), -1, dtype=torch.int32, device=device
        )
    else:
        real_page_table = page_table_1

    fused_dsa_target_verify_metadata(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_table_1=page_table_1,
        seqlens_expanded=seqlens_expanded,
        dsa_cache_seqlens=dsa_cache_seqlens,
        dsa_cu_seqlens_k=dsa_cu_seqlens_k,
        real_page_table=real_page_table,
        bs=bs,
        max_seqlen_k=max_seqlen_k,
        dsa_index_topk=topk,
        real_page_size=real_page_size,
        next_n=next_n,
    )
    torch.cuda.synchronize()

    ref_cache = seq_lens.to(torch.int32) + next_n
    ref_page = req_to_token[req_pool_indices, :max_seqlen_k]
    ref_page = torch.repeat_interleave(ref_page, repeats=next_n, dim=0)
    ref_expanded = (
        seq_lens.to(torch.int32).view(bs, 1)
        + torch.arange(1, next_n + 1, dtype=torch.int32, device=device).view(1, next_n)
    ).reshape(-1)
    ref_dsa = ref_expanded.clamp(max=topk)

    assert torch.equal(cache_seqlens, ref_cache)
    assert torch.equal(
        cu_seqlens_k, torch.nn.functional.pad(torch.cumsum(ref_cache, 0), (1, 0))
    )
    assert torch.equal(seqlens_expanded, ref_expanded)
    assert torch.equal(dsa_cache_seqlens, ref_dsa)
    assert torch.equal(
        dsa_cu_seqlens_k, torch.nn.functional.pad(torch.cumsum(ref_dsa, 0), (1, 0))
    )
    assert torch.equal(page_table_1[:, :max_seqlen_k], ref_page)
    assert torch.all(page_table_1[:, max_seqlen_k:] == -1)

    if real_page_size > 1:
        ref_real = ref_page[
            :, torch.arange(0, max_seqlen_k, real_page_size, device=device)
        ]
        ref_real = ref_real // real_page_size
        assert torch.equal(real_page_table[:, : ref_real.shape[1]], ref_real)
        assert torch.all(real_page_table[:, ref_real.shape[1] :] == -1)
