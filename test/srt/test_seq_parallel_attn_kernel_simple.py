import pytest
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.cascade import merge_state
from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

from sglang.srt.layers.extend_attention import extend_attention_fwd, redundant_attention
from sglang.srt.layers.token_attention import token_attention_fwd

flashinfer_prefill_wrapper_ragged = None
flashinfer_prefill_wrapper_paged = None
flashinfer_decode_wrapper = None


def get_next_partition_id(curr_partition_id, num_partitions):
    assert curr_partition_id < num_partitions
    return (curr_partition_id - 1) % num_partitions


def get_sp_prev_local_rank(rank, num_partitions):
    return (rank - 1) % num_partitions


def get_sp_next_local_rank(rank, num_partitions):
    return (rank + 1) % num_partitions


def append_merge_partition(partition_list, o, s):
    if len(partition_list) == 0:
        partition_list.append((o, s))
    else:
        o_prev, s_prev = partition_list[-1]
        o, s = merge_state(o_prev, s_prev, o, s)
        partition_list[-1] = (o, s)


def seq_parallel_attn(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    q,
    k,
    v,
    rank: int,
    sp_size: int,
):
    """Simulate a sequence parallel attention kernel. It takes full Q, K, and V
    with simulated communication. TODO: replace with actual communication.
    """
    num_partitions = sp_size
    num_iters = sp_size
    # NOTE: we assume sequence length is divisible by num_partitions
    qo_len_per_iter = qo_len // num_iters
    kv_len_per_partition = kv_len // num_partitions

    qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len_per_iter
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len_per_partition
    flashinfer_prefill_wrapper_ragged.end_forward()
    flashinfer_prefill_wrapper_ragged.begin_forward(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
    )

    kv_indices = torch.arange(0, batch_size * kv_len_per_partition).to(0).int()
    kv_last_page_len = torch.full((batch_size,), 1, dtype=torch.int32).to(0)
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

    local_k, local_v = (
        k[:, rank * kv_len_per_partition : (rank + 1) * kv_len_per_partition]
        .contiguous()
        .view(-1, num_kv_heads, head_dim),
        v[:, rank * kv_len_per_partition : (rank + 1) * kv_len_per_partition]
        .contiguous()
        .view(-1, num_kv_heads, head_dim),
    )
    k_partition, v_partition = local_k, local_v

    owned_pids = [rank]
    owned_partitions = [None for _ in range(num_partitions)]
    owned_partitions[rank] = (local_k, local_v)
    o_partitions = [[] for _ in range(num_partitions)]

    to_rank = rank  # which SP worker to send my sequence KV partition to.
    from_rank = rank  # which SP worker to receive the sequence KV partition from.

    pid = rank  # start from the worker's own partition
    for _ in range(num_iters):
        # TODO: send-recv communication here
        to_rank = get_sp_next_local_rank(to_rank, num_partitions)
        # send_to(to_rank, k, v)
        q_partition = q[:, pid * qo_len_per_iter : (pid + 1) * qo_len_per_iter]
        k_partition, v_partition = owned_partitions[pid]
        # Ragged attention computation for self attention within the partition
        o, s = flashinfer_prefill_wrapper_ragged.forward_return_lse(
            q_partition.contiguous().view(-1, num_qo_heads, head_dim),
            k_partition.contiguous().view(-1, num_kv_heads, head_dim),
            v_partition.contiguous().view(-1, num_kv_heads, head_dim),
        )
        append_merge_partition(o_partitions[pid], o, s)
        # Paged attention computation for cross partition attention
        # NOTE: below schedule is for load balancing
        for existing_pid in owned_pids:
            if existing_pid == pid:
                continue
            i, j = (existing_pid, pid) if existing_pid > pid else (pid, existing_pid)
            q_data = q[:, i * qo_len_per_iter : (i + 1) * qo_len_per_iter]
            kv_data = torch.stack(owned_partitions[j], dim=1)
            o, s = flashinfer_prefill_wrapper_paged.forward_return_lse(
                q_data.contiguous().view(-1, num_qo_heads, head_dim),
                kv_data,
                causal=False,
            )
            append_merge_partition(o_partitions[i], o, s)

        # TODO: send-recv communication here
        from_rank = get_sp_prev_local_rank(from_rank, num_partitions)
        # recv_from(from_rank, k, v)
        pid = from_rank
        kv_recved = (
            k[:, pid * kv_len_per_partition : (pid + 1) * kv_len_per_partition]
            .contiguous()
            .view(-1, num_kv_heads, head_dim),
            v[:, pid * kv_len_per_partition : (pid + 1) * kv_len_per_partition]
            .contiguous()
            .view(-1, num_kv_heads, head_dim),
        )
        owned_pids.append(pid)
        owned_partitions[pid] = kv_recved

    # Reshape all o tensors so that we can concatenate along the sequence dimension
    # we must have len(partition_list) == 1 here
    os = [
        o.view(batch_size, qo_len_per_iter, num_qo_heads, head_dim)
        for partition_list in o_partitions
        for o, _ in partition_list
    ]
    o = torch.cat(os, dim=1).view(
        -1, num_qo_heads, head_dim
    )  # restore the original shape
    return o


@pytest.mark.parametrize("batch_size", [12, 37, 67])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [37, 17])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [32, 4])
@pytest.mark.parametrize("head_dim", [128])
def test_seq_parallel_prefill(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    rank: int = 0,
    sp_size: int = 2,
):
    init_flashinfer(num_qo_heads, num_kv_heads)

    q = torch.randn(batch_size, qo_len, num_qo_heads, head_dim).to(0).half()
    k = torch.randn(batch_size, kv_len, num_kv_heads, head_dim).to(0).half()
    v = torch.randn(batch_size, kv_len, num_kv_heads, head_dim).to(0).half()

    def reference_impl_ragged():
        qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
        kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len

        flashinfer_prefill_wrapper_ragged.end_forward()
        flashinfer_prefill_wrapper_ragged.begin_forward(
            qo_indptr,
            kv_indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim,
        )
        o = flashinfer_prefill_wrapper_ragged.forward(
            q.contiguous().view(-1, num_qo_heads, head_dim),
            k.contiguous().view(-1, num_kv_heads, head_dim),
            v.contiguous().view(-1, num_kv_heads, head_dim),
        )
        flashinfer_prefill_wrapper_ragged.end_forward()
        return o

    def reference_impl_paged():
        qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
        total_tokens = kv_len * batch_size

        kv_data = torch.zeros(total_tokens, 2, num_kv_heads, head_dim).to(0).half()
        kv_data[:, 0] = k.contiguous().view(-1, num_kv_heads, head_dim)
        kv_data[:, 1] = v.contiguous().view(-1, num_kv_heads, head_dim)
        kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
        kv_indices = torch.arange(0, total_tokens).to(0).int()
        kv_last_page_len = torch.full((batch_size,), 1, dtype=torch.int32).to(0)

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
        o = flashinfer_prefill_wrapper_paged.forward(
            q.contiguous().view(-1, num_qo_heads, head_dim), kv_data
        )
        flashinfer_prefill_wrapper_paged.end_forward()
        return o

    o_sp = seq_parallel_attn(
        batch_size,
        kv_len,
        qo_len,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        q,
        k,
        v,
        rank=1,
        sp_size=4,
    )
    o_truth = reference_impl_paged()

    print("Mean: ", torch.mean(torch.abs(o_sp - o_truth)))
    print("Max: ", torch.max(torch.abs(o_sp - o_truth)))
    assert torch.allclose(o_sp, o_truth, rtol=1e-2, atol=1e-3)


def init_flashinfer(num_attention_heads, num_kv_heads):
    if not _grouped_size_compiled_for_decode_kernels(num_attention_heads, num_kv_heads):
        use_tensor_cores = True
    else:
        use_tensor_cores = False

    workspace_buffer = torch.empty(
        3, 128 * 1024 * 1024, dtype=torch.int8, device="cuda"
    )

    global flashinfer_prefill_wrapper_ragged, flashinfer_prefill_wrapper_paged, flashinfer_decode_wrapper

    flashinfer_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer[0], "NHD"
    )
    flashinfer_prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer[1], "NHD"
    )
    flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer[2], "NHD", use_tensor_cores=use_tensor_cores
    )


if __name__ == "__main__":
    test_seq_parallel_prefill(12, 128, 128, 8, 8, 128, rank=3, sp_size=4)
    test_seq_parallel_prefill(12, 4096, 4096, 8, 8, 128, rank=4, sp_size=8)
    test_seq_parallel_prefill(12, 1024, 1024, 32, 32, 128, rank=1, sp_size=2)
