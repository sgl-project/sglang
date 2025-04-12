import math
from typing import List, Optional, Tuple

import pytest
import torch
from einops import rearrange, repeat
from sgl_kernel.sparse_flash_attn import sparse_attn_func, sparse_attn_varlen_func


def ref_attn(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        lse: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))

    lse_ref = scores.logsumexp(dim=-1)

    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)

    return output.to(dtype=dtype_og), lse_ref


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # clone to avoid clobbering the query tensor
        q = query[start_idx : start_idx + query_len].clone()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "seq_lens",
    [
        (1, 1),
        (1, 1024),
        (1, 2048),
        (1023, 2049),
        (1023, 1023),
        (32, 32),
        (65, 65),
        (129, 129),
    ],
)
@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("NNZ_S", [0, 1, 2, 3, 7, 15, 32])
@torch.inference_mode()
def test_sparse_attention(
    batch_size,
    seq_lens,
    num_heads,
    head_size,
    dtype,
    NNZ_S,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    block_size_M = 64
    block_size_N = 64
    seqlen_q, seqlen_k = seq_lens
    q = torch.randn(
        batch_size, seqlen_q, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        batch_size, seqlen_k, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    v = torch.randn(
        batch_size, seqlen_k, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    NUM_ROWS = (seqlen_q + block_size_M - 1) // block_size_M
    if NNZ_S * block_size_N > seqlen_k:
        return
    NNZ_V = seqlen_k - NNZ_S * block_size_N
    block_count = torch.tensor(
        [NNZ_S] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32
    ).reshape(batch_size, num_heads, NUM_ROWS)
    column_count = torch.tensor(
        [NNZ_V] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32
    ).reshape(batch_size, num_heads, NUM_ROWS)
    block_offset = torch.tensor(
        [[i * block_size_N for i in range(NNZ_S)]] * batch_size * NUM_ROWS * num_heads,
        dtype=torch.int32,
    ).reshape(batch_size, num_heads, NUM_ROWS, NNZ_S)
    column_index = torch.tensor(
        [[NNZ_S * block_size_N + i for i in range(NNZ_V)]]
        * batch_size
        * NUM_ROWS
        * num_heads,
        dtype=torch.int32,
    ).reshape(batch_size, num_heads, NUM_ROWS, NNZ_V)
    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        return_softmax_lse=True,
    )

    ref_out, ref_lse = ref_attn(q, k, v)

    torch.testing.assert_close(
        out, ref_out, atol=2e-2, rtol=1e-2
    ), f"{torch.max(torch.abs(out - ref_out))}"
    torch.testing.assert_close(
        lse, ref_lse, atol=2e-2, rtol=1e-2
    ), f"{torch.max(torch.abs(lse - ref_lse))}"


# @pytest.mark.parametrize("seq_lens", [[(1024, 1328)],
#                                     [(1024, 1328), (1, 2048)],
#                                     [(1025, 1328), (2, 2048)],
#                                     [(1025, 2049), (2, 1281)],
#                                     ])
# @pytest.mark.parametrize("head_size", [128])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @torch.inference_mode()
# def test_sparse_attention_varlen(
#         seq_lens,
#         head_size,
#         dtype,
# ) -> None:
#     torch.set_default_device("cuda")
#     torch.cuda.manual_seed_all(0)
#     block_size_M = 64
#     block_size_N = 64
#     num_seqs = len(seq_lens)
#     query_lens = [x[0] for x in seq_lens]
#     kv_lens = [x[1] for x in seq_lens]
#     num_heads = 1
#     query = torch.randn(sum(query_lens),
#                         num_heads,
#                         head_size,
#                         dtype=dtype)
#     key = torch.randn(sum(kv_lens),
#                     num_heads,
#                     head_size,
#                     dtype=dtype)
#     value = torch.randn_like(key)
#     cu_query_lens = torch.tensor([0] + query_lens,
#                                 dtype=torch.int32).cumsum(dim=0,
#                                                         dtype=torch.int32)
#     cu_kv_lens = torch.tensor([0] + kv_lens,
#                                 dtype=torch.int32).cumsum(dim=0,
#                                                         dtype=torch.int32)
#     max_query_len = max(query_lens)
#     max_kv_len = max(kv_lens)

#     NUM_ROWS = (max_query_len + block_size_M - 1) // block_size_M
#     NNZ_S = 20
#     NNZ_V = 2048
#     batch_size = len(query_lens)

#     block_counts = []
#     column_counts = []
#     block_offsets = []
#     column_indices = []
#     for b in range(batch_size):
#         block_counts.append(torch.tensor([NNZ_S] * NUM_ROWS * num_heads, dtype=torch.int32).reshape(num_heads, NUM_ROWS))
#         columns = kv_lens[b] - NNZ_S * block_size_N
#         column_counts.append(torch.tensor([columns] * NUM_ROWS * num_heads, dtype=torch.int32).reshape(num_heads, NUM_ROWS))
#         block_offsets.append(torch.tensor([[i * block_size_N for i in range(NNZ_S)]] * NUM_ROWS * num_heads, dtype=torch.int32).reshape(num_heads, NUM_ROWS, NNZ_S))
#         column_indices.append(torch.tensor([[NNZ_S * block_size_N + i for i in range(NNZ_V)]] * NUM_ROWS * num_heads, dtype=torch.int32).reshape(num_heads, NUM_ROWS, NNZ_V))
#     block_count = torch.concat(block_counts).reshape(batch_size, num_heads, NUM_ROWS)
#     column_count = torch.concat(column_counts).reshape(batch_size, num_heads, NUM_ROWS)
#     block_offset = torch.concat(block_offsets).reshape(batch_size, num_heads, NUM_ROWS, NNZ_S)
#     column_index = torch.concat(column_indices).reshape(batch_size, num_heads, NUM_ROWS, NNZ_V)
#     out, lse = sparse_attn_varlen_func(
#         query,
#         key,
#         value,
#         block_count,
#         block_offset,
#         column_count,
#         column_index,
#         cu_seqlens_q=cu_query_lens,
#         cu_seqlens_k=cu_kv_lens,
#         max_seqlen_q=max_query_len,
#         max_seqlen_k=max_kv_len,
#         return_softmax_lse=True,
#     )

#     max_num_blocks_per_seq = (max_kv_len + 2048 - 1) // 2048
#     block_tables = torch.randint(0,
#                                  2048,
#                                  (len(query_lens), max_num_blocks_per_seq),
#                                  dtype=torch.int32)
#     scale = head_size**-0.5

#     ref_out, ref_lse, _ = ref_paged_attn(
#         query,
#         key,
#         value,
#         query_lens=query_lens,
#         kv_lens=kv_lens,
#         block_tables=block_tables,
#         scale=scale
#     )

#     torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=1e-2), \
#         f"{torch.max(torch.abs(out - ref_out))}"
#     torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=1e-2), \
#         f"{torch.max(torch.abs(lse - ref_lse))}"

if __name__ == "__main__":
    pytest.main([__file__])
