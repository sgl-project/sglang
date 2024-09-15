import pytest
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

from sglang.srt.layers.token_attention import token_attention_fwd
from sglang.srt.layers.triton_attention.extend_attention import (
    extend_attention_fwd,
    redundant_attention,
)

flashinfer_prefill_wrapper = None
flashinfer_decode_wrapper = None


@pytest.mark.parametrize("batch_size", [12, 37, 67])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [37, 17])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [32, 4])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    init_flashinfer(num_qo_heads, num_kv_heads)

    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    total_tokens = kv_len * batch_size
    kv_data = torch.randn(total_tokens, 2, num_kv_heads, head_dim).to(0).half()
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

    o_redundant = torch.empty_like(q)
    b_start_loc = torch.zeros((batch_size,), dtype=torch.int32).to(0)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)
    b_seq_len_prefix = b_seq_len - b_seq_len_extend

    redundant_attention(
        q,
        k_extend,
        v_extend,
        o_redundant,
        k_buffer,
        v_buffer,
        req_to_token,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_seq_len_prefix,
        max_len_in_batch,
    )
    print("Mean: ", torch.mean(torch.abs(o_redundant - o_triton)))
    print("Max: ", torch.max(torch.abs(o_redundant - o_triton)))
    assert torch.allclose(o_redundant, o_triton, rtol=1e-2, atol=1e-3)

    flashinfer_prefill_wrapper.end_forward()

    flashinfer_prefill_wrapper.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
    )
    o = flashinfer_prefill_wrapper.forward(
        q.contiguous().view(-1, num_qo_heads, head_dim), kv_data
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
    init_flashinfer(num_qo_heads, num_kv_heads)

    q = torch.randn(batch_size, num_qo_heads, head_dim).to(0).half()
    total_tokens = kv_len * batch_size
    kv_data = torch.randn(total_tokens, 2, num_kv_heads, head_dim).to(0).half()
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

    flashinfer_decode_wrapper.end_forward()
    flashinfer_decode_wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
        pos_encoding_mode="NONE",
        data_type="float16",
    )
    o = flashinfer_decode_wrapper.forward(
        q.contiguous().view(-1, num_qo_heads, head_dim), kv_data
    )

    print("Mean: ", torch.mean(torch.abs(o - o_triton)))
    print("Max: ", torch.max(torch.abs(o - o_triton)))
    assert torch.allclose(o, o_triton, rtol=1e-2, atol=2e-3)


def init_flashinfer(num_attention_heads, num_kv_heads):
    if not _grouped_size_compiled_for_decode_kernels(num_attention_heads, num_kv_heads):
        use_tensor_cores = True
    else:
        use_tensor_cores = False

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")

    global flashinfer_prefill_wrapper, flashinfer_decode_wrapper

    flashinfer_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=use_tensor_cores
    )


if __name__ == "__main__":
    test_batch_prefill_with_paged_kv_cache(12, 54, 37, 8, 8, 128)
    test_batch_prefill_with_paged_kv_cache(37, 1111, 456, 32, 32, 128)
    test_batch_decode_with_paged_kv_cache(12, 54, 4, 32, 128)
