import torch
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    merge_state,
)


class MHATokenToKVPool:
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
    ):
        # [size, head_num, head_dim] for each layer
        self.k_buffer = [
            torch.empty((size + 1, head_num, head_dim), dtype=dtype, device="cuda")
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.empty((size + 1, head_num, head_dim), dtype=dtype, device="cuda")
            for _ in range(layer_num)
        ]

    def get_kv_buffer(self, layer_id: int):
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id]


flashinfer_workspace_size = 192 * 1024 * 1024
flashinfer_workspace_buffers = torch.empty(
    3, flashinfer_workspace_size, dtype=torch.uint8, device="cuda"
)
flashinfer_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
    flashinfer_workspace_buffers[0], "NHD"
)
flashinfer_prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
    flashinfer_workspace_buffers[1], "NHD"
)
flashinfer_prefill_wrapper_paged2 = BatchPrefillWithPagedKVCacheWrapper(
    flashinfer_workspace_buffers[2], "NHD"
)

num_qo_heads, num_kv_heads, head_dim = 32, 32, 128
req_pool_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32, device="cuda")
req_to_token_pool = torch.empty((10, 1024), dtype=torch.int32, device="cuda")
token_to_kv_pool = MHATokenToKVPool(65536, torch.bfloat16, num_kv_heads, head_dim, 1)


def init_ragged(batch_size, prefix_lens, seq_lens):
    # Ragged part
    paged_kernel_lens = prefix_lens
    kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
    req_pool_indices_cpu = req_pool_indices.cpu().numpy()
    paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
    kv_indices = torch.cat(
        [
            req_to_token_pool[req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]]
            for i in range(batch_size)
        ],
        dim=0,
    ).contiguous()
    kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

    qo_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
    qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

    flashinfer_prefill_wrapper_ragged.end_forward()
    flashinfer_prefill_wrapper_ragged.begin_forward(
        qo_indptr,
        qo_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
    )

    # cached part
    flashinfer_prefill_wrapper_paged2.end_forward()
    flashinfer_prefill_wrapper_paged2.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
    )


def init_pagged(batch_size, seq_lens, prefix_lens):
    # Paged part
    paged_kernel_lens = seq_lens
    kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
    req_pool_indices_cpu = req_pool_indices.cpu().numpy()
    paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
    kv_indices = torch.cat(
        [
            req_to_token_pool[req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]]
            for i in range(batch_size)
        ],
        dim=0,
    ).contiguous()
    kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

    qo_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
    qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

    flashinfer_prefill_wrapper_paged.end_forward()
    flashinfer_prefill_wrapper_paged.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
    )


def run_triton(
    prefix_lens, seq_lens, q, k, v, num_qo_heads, head_dim, scaling, logit_cap
):
    """Init auxiliary variables for triton attention backend."""
    triton_max_seq_len = int(torch.max(seq_lens))
    triton_prefix_lens = prefix_lens
    triton_start_loc = torch.zeros_like(seq_lens, dtype=torch.int32)
    triton_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    extend_seq_lens = seq_lens - prefix_lens
    extend_start_loc = torch.zeros_like(seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    triton_max_extend_len = int(torch.max(extend_seq_lens))

    o = torch.empty_like(q)

    from sglang.srt.layers.extend_attention import extend_attention_fwd

    extend_attention_fwd(
        q.view(-1, num_qo_heads, head_dim),
        k.contiguous(),
        v.contiguous(),
        o.view(-1, num_qo_heads, head_dim),
        token_to_kv_pool.get_key_buffer(0),
        token_to_kv_pool.get_value_buffer(0),
        req_to_token_pool,
        req_pool_indices,
        triton_start_loc,
        seq_lens,
        triton_prefix_lens,
        extend_start_loc,
        extend_seq_lens,
        triton_max_seq_len,
        triton_max_extend_len,
        sm_scale=scaling,
        logit_cap=logit_cap,
    )

    return o


def main():
    # batch_size = 5
    # prefix_lens = torch.tensor([128, 156, 127, 176, 243], dtype=torch.int32, device="cuda")
    # seq_lens = torch.tensor([370, 390, 410, 234, 279], dtype=torch.int32, device="cuda")

    batch_size = 2
    prefix_lens = torch.tensor([261, 0], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([367, 22], dtype=torch.int32, device="cuda")

    # batch_size = 1
    # prefix_lens = torch.tensor([24], dtype=torch.int32, device="cuda")
    # seq_lens = torch.tensor([73], dtype=torch.int32, device="cuda")

    # Init Data
    input_len = torch.sum(seq_lens - prefix_lens)
    q = torch.randn(
        (input_len, num_qo_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        (input_len, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        (input_len, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )

    pt, pt_ragged = 0, 0
    for i in range(batch_size):
        req_to_token_pool[i, : seq_lens[i]] = torch.arange(
            pt, pt + seq_lens[i], device="cuda"
        )
        k_buffer, v_buffer = token_to_kv_pool.get_kv_buffer(0)
        k_buffer[pt : pt + prefix_lens[i]] = torch.randn(
            (prefix_lens[i], num_kv_heads, head_dim), device="cuda"
        )
        v_buffer[pt : pt + prefix_lens[i]] = torch.randn(
            (prefix_lens[i], num_kv_heads, head_dim), device="cuda"
        )
        k_buffer[pt + prefix_lens[i] : pt + seq_lens[i]] = k[
            pt_ragged : pt_ragged + seq_lens[i] - prefix_lens[i]
        ]
        v_buffer[pt + prefix_lens[i] : pt + seq_lens[i]] = v[
            pt_ragged : pt_ragged + seq_lens[i] - prefix_lens[i]
        ]
        pt += seq_lens[i]
        pt_ragged += seq_lens[i] - prefix_lens[i]

    assert pt == torch.sum(seq_lens)
    assert pt_ragged == torch.sum(seq_lens - prefix_lens)

    init_ragged(batch_size, prefix_lens, seq_lens)
    init_pagged(batch_size, seq_lens, prefix_lens)

    scaling = 128**-0.5
    logit_cap = 0

    o1, s1 = flashinfer_prefill_wrapper_ragged.forward_return_lse(
        q.contiguous().view(-1, num_qo_heads, head_dim),
        k.contiguous().view(-1, num_kv_heads, head_dim),
        v.contiguous().view(-1, num_kv_heads, head_dim),
        causal=True,
        sm_scale=scaling,
        logits_soft_cap=logit_cap,
    )

    o2, s2 = flashinfer_prefill_wrapper_paged2.forward_return_lse(
        q.contiguous().view(-1, num_qo_heads, head_dim),
        token_to_kv_pool.get_kv_buffer(0),
        causal=False,
        sm_scale=scaling,
        logits_soft_cap=logit_cap,
    )

    o3, _ = merge_state(o1, s1, o2, s2)

    o = flashinfer_prefill_wrapper_paged.forward(
        q.contiguous().view(-1, num_qo_heads, head_dim),
        token_to_kv_pool.get_kv_buffer(0),
        causal=True,
        sm_scale=scaling,
        logits_soft_cap=logit_cap,
    )

    o_triton = run_triton(
        prefix_lens, seq_lens, q, k, v, num_qo_heads, head_dim, scaling, logit_cap
    )

    print("=" * 20, "Diff With Triton", "=" * 20)

    print("Mean: ", torch.mean(torch.abs(o - o_triton)))
    print("Max: ", torch.max(torch.abs(o - o_triton)))

    print("=" * 20, "Diff With Ragged", "=" * 20)

    print("Mean: ", torch.mean(torch.abs(o - o3)))
    print("Max: ", torch.max(torch.abs(o - o3)))


if __name__ == "__main__":
    main()
