import flashinfer
import pytest
import torch
from sglang.srt.layers.extend_attention import extend_attention_fwd
from sglang.srt.layers.token_attention import token_attention_fwd


@pytest.mark.parametrize("batch_size", [12, 37, 67])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [37, 17])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("use_wrapper", [True, False])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    use_wrapper,
):
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    total_tokens = kv_len * batch_size
    kv_data = torch.randn(total_tokens, 2, num_kv_heads, 1, head_dim).to(0).half()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
    kv_indices = torch.arange(0, total_tokens).to(0).int()
    kv_last_page_len = torch.full((batch_size,), 1, dtype=torch.int32).to(0)

    # init args for triton kernel
    k_extend = (
        kv_data.view(batch_size, kv_len, 2, -1)[:, -qo_len:, 0]
        .contiguous()
        .view(-1, num_kv_heads, head_dim)
    )
    v_extend = (
        kv_data.view(batch_size, kv_len, 2, -1)[:, -qo_len:, 1]
        .contiguous()
        .view(-1, num_kv_heads, head_dim)
    )
    o_triton = torch.empty_like(q)
    k_buffer = kv_data[:, 0].view(-1, num_kv_heads, head_dim).contiguous()
    v_buffer = kv_data[:, 1].view(-1, num_kv_heads, head_dim).contiguous()
    req_to_token = torch.arange(0, total_tokens).to(0).int().view(batch_size, kv_len)
    b_req_idx = torch.arange(0, batch_size).to(0).int()
    b_seq_len = torch.full((batch_size,), kv_len, dtype=torch.int32).to(0)
    b_start_loc_extend = torch.arange(0, batch_size).to(0).int() * qo_len
    b_seq_len_extend = torch.full((batch_size,), qo_len, dtype=torch.int32).to(0)
    max_len_in_batch = kv_len
    max_len_extend = qo_len

    extend_attention_fwd(
        q,
        k_extend,
        v_extend,
        o_triton,
        k_buffer,
        v_buffer,
        req_to_token,
        b_req_idx,
        None,  # b_start_loc = None
        b_seq_len,
        None,  # b_seq_len_prefix = None
        b_start_loc_extend,
        b_seq_len_extend,
        max_len_in_batch,
        max_len_extend,
    )

    if use_wrapper:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper()
        wrapper.begin_forward(q_indptr, batch_size, num_qo_heads, num_kv_heads)
        o = wrapper.forward(
            q, q_indptr, kv_data, kv_indptr, kv_indices, kv_last_page_len
        )
    else:
        o = flashinfer.batch_prefill_with_paged_kv_cache(
            q,
            q_indptr,
            kv_data,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
        )

    print("Mean: ", torch.mean(torch.abs(o - o_triton)))
    print("Max: ", torch.max(torch.abs(o - o_triton)))
    assert torch.allclose(o, o_triton, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 37])
@pytest.mark.parametrize("kv_len", [54, 127, 537])
@pytest.mark.parametrize("num_kv_heads", [32])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_decode_with_paged_kv_cache(
    batch_size,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    # note(lsyin): when pytest, the number of heads cannot change, because triton kernel has a cache
    # to test different shape of decode, change the parameters in the __main__, and run decode only once

    q = torch.randn(batch_size, num_qo_heads, head_dim).to(0).half()
    total_tokens = kv_len * batch_size
    kv_data = torch.randn(total_tokens, 2, num_kv_heads, 1, head_dim).to(0).half()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
    kv_indices = torch.arange(0, total_tokens).to(0).int()
    kv_last_page_len = torch.full((batch_size,), 1, dtype=torch.int32).to(0)

    # init args for triton kernel
    k_buffer = kv_data[:, 0].view(-1, num_kv_heads, head_dim).contiguous()
    v_buffer = kv_data[:, 1].view(-1, num_kv_heads, head_dim).contiguous()
    o_triton = torch.empty_like(q)
    req_to_token = (
        torch.arange(0, kv_len * batch_size).to(0).int().view(batch_size, kv_len)
    )
    b_req_idx = torch.arange(0, batch_size).to(0).int()
    b_start_loc = torch.arange(0, batch_size).to(0).int() * kv_len
    b_seq_len = torch.full((batch_size,), kv_len, dtype=torch.int32).to(0)
    max_len_in_batch = kv_len
    other_kv_index = 0
    token_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o_triton,
        req_to_token,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        max_len_in_batch,
        other_kv_index,
        total_tokens,
    )

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper()
    wrapper.begin_forward(
        kv_indptr,
        kv_last_page_len,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
        "NONE",
        "float16",
    )
    o = wrapper.forward(q, kv_data, kv_indptr, kv_indices, kv_last_page_len)

    print("Mean: ", torch.mean(torch.abs(o - o_triton)))
    print("Max: ", torch.max(torch.abs(o - o_triton)))
    assert torch.allclose(o, o_triton, rtol=1e-2, atol=2e-3)


if __name__ == "__main__":
    test_batch_prefill_with_paged_kv_cache(12, 54, 37, 8, 8, 128, False)
    test_batch_prefill_with_paged_kv_cache(37, 1111, 456, 32, 32, 128, True)
    test_batch_decode_with_paged_kv_cache(12, 54, 4, 32, 128)
