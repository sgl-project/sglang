import multiprocessing
import random

import torch
from vllm.distributed import init_distributed_environment

from sglang.srt.layers.parallel_utils.parallel_state import initialize_model_parallel
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.controller.model_runner import InputMetadata

NUM_HEADS = 32
HEAD_DIM = 128
SCALING = 1
NUM_KV_HEADS = 8
LAYER_ID = 0
LOGIT_CAP = -1


BATCH_SIZE = 1
QO_LEN = 128
KV_LEN = 128


def gen_qkv(rank: int = 0, sp_size: int = 1):
    torch.manual_seed(42)
    random.seed(42)
    q = torch.randn(BATCH_SIZE * QO_LEN, NUM_HEADS, HEAD_DIM).cuda().half()
    k = torch.randn(BATCH_SIZE * KV_LEN, NUM_KV_HEADS, HEAD_DIM).cuda().half()
    v = torch.randn(BATCH_SIZE * KV_LEN, NUM_KV_HEADS, HEAD_DIM).cuda().half()

    # num_heads_per_partition = NUM_HEADS // sp_size
    # q = q[
    #     :, :, num_heads_per_partition * rank : num_heads_per_partition * (rank + 1)
    # ].contiguous()
    # kv_len_per_partition = KV_LEN // sp_size
    # k = k[
    #     :, kv_len_per_partition * rank : kv_len_per_partition * (rank + 1)
    # ].contiguous()
    # v = v[
    #     :, kv_len_per_partition * rank : kv_len_per_partition * (rank + 1)
    # ].contiguous()

    return q, k, v


def get_input_metadata(sp_size: int = 1, tp_size: int = 1):
    from flashinfer import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )

    input_metadata = InputMetadata(
        forward_mode=None,
        batch_size=BATCH_SIZE,
        total_num_tokens=BATCH_SIZE * QO_LEN,
        req_pool_indices=None,
        seq_lens=None,
        positions=None,
        req_to_token_pool=None,
        token_to_kv_pool=None,
        out_cache_loc=None,
        extend_seq_lens=None,
        extend_start_loc=None,
        extend_no_prefix=True,
        return_logprob=None,
        top_logprobs_nums=None,
        flashinfer_prefill_wrapper_ragged=None,
        flashinfer_prefill_wrapper_paged=None,
        flashinfer_decode_wrapper=None,
        sp_size=sp_size,
    )

    workspace_buffer = torch.empty(
        2, 128 * 1024 * 1024, dtype=torch.int8, device="cuda"
    )

    input_metadata.flashinfer_prefill_wrapper_ragged = (
        BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer[0], "NHD")
    )
    input_metadata.flashinfer_prefill_wrapper_paged = (
        BatchPrefillWithPagedKVCacheWrapper(workspace_buffer[1], "NHD")
    )

    num_qo_heads = NUM_HEADS // sp_size
    num_kv_heads = NUM_KV_HEADS
    qo_len_per_iter = QO_LEN // sp_size
    kv_len_per_partition = KV_LEN // sp_size

    qo_indptr = torch.arange(0, BATCH_SIZE + 1).cuda().int() * qo_len_per_iter
    kv_indptr = torch.arange(0, BATCH_SIZE + 1).cuda().int() * kv_len_per_partition
    input_metadata.flashinfer_prefill_wrapper_ragged.end_forward()
    input_metadata.flashinfer_prefill_wrapper_ragged.begin_forward(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
    )

    # cached part
    kv_indices = torch.arange(0, BATCH_SIZE * kv_len_per_partition).cuda().int()
    kv_last_page_len = torch.full((BATCH_SIZE,), 1, dtype=torch.int32).cuda()
    input_metadata.flashinfer_prefill_wrapper_paged.end_forward()
    input_metadata.flashinfer_prefill_wrapper_paged.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        1,
    )

    return input_metadata


def sp_worker(rank: int = 0, sp_size: int = 1, tp_size: int = 1):
    torch.manual_seed(42)
    random.seed(42)

    def init_comm():
        nccl_init_method = f"tcp://127.0.0.1:28888"
        init_distributed_environment(
            backend="nccl",
            world_size=tp_size,
            rank=rank,
            local_rank=rank,
            distributed_init_method=nccl_init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=tp_size, sequence_parallel_size=sp_size
        )
        torch.cuda.set_device(rank)

    init_comm()

    def init_attention():
        attention = RadixAttention(
            num_heads=NUM_HEADS // sp_size,
            head_dim=HEAD_DIM,
            scaling=SCALING,
            num_kv_heads=NUM_KV_HEADS,
            layer_id=LAYER_ID,
            logit_cap=LOGIT_CAP,
        )
        return attention

    attn = init_attention()
    print("SP worker", rank, "initialized on", torch.cuda.current_device())

    # Computation
    input_metadata = get_input_metadata(sp_size=sp_size, tp_size=tp_size)
    q, k, v = gen_qkv(rank, sp_size)
    qs, ks, vs = [], [], []
    q_head_idxes = _get_sequence_parallel_head_idxes(
        NUM_HEADS, NUM_KV_HEADS, rank, sp_size
    )
    print(rank, q_head_idxes)
    for i in range(sp_size):
        qs.append(
            q[(QO_LEN // sp_size) * i : (QO_LEN // sp_size) * (i + 1), q_head_idxes]
        )
        ks.append(k[(KV_LEN // sp_size) * i : (KV_LEN // sp_size) * (i + 1)])
        vs.append(v[(KV_LEN // sp_size) * i : (KV_LEN // sp_size) * (i + 1)])

    output = attn.seq_parallel_extend_forward_flashinfer(qs, ks, vs, input_metadata)

    o_truth = reference_attn()
    o_truth = (
        o_truth.contiguous()
        .view(-1, NUM_HEADS, HEAD_DIM)[:, q_head_idxes]
        .view(-1, NUM_HEADS // sp_size * HEAD_DIM)
    )

    print("SP worker", rank, "results:")
    print("Mean: ", torch.mean(torch.abs(output - o_truth)))
    print("Max: ", torch.max(torch.abs(output - o_truth)))
    assert torch.allclose(output, o_truth, rtol=1e-2, atol=1e-3)


def _get_sequence_parallel_head_idxes(total_num_heads, num_kv_heads, sp_rank, sp_size):
    group_num = num_kv_heads
    group_size = total_num_heads // num_kv_heads
    shard_num_heads = group_size // sp_size
    idxes = [
        group_size * i + sp_rank * shard_num_heads + j
        for i in range(group_num)
        for j in range(0, shard_num_heads)
    ]
    return idxes


def reference_attn():
    torch.manual_seed(42)
    random.seed(42)

    attn = RadixAttention(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        scaling=SCALING,
        num_kv_heads=NUM_KV_HEADS,
        layer_id=LAYER_ID,
        logit_cap=LOGIT_CAP,
    )

    input_metadata = get_input_metadata()
    q, k, v = gen_qkv()

    return attn.extend_forward_flashinfer(q, k, v, input_metadata)


def main():
    sp_size = 2
    tp_size = 2

    multiprocessing.set_start_method("spawn", force=True)
    sp_procs = []
    for rank in range(1, sp_size):
        sp_proc = multiprocessing.Process(
            target=sp_worker, args=(rank, sp_size, tp_size)
        )
        sp_proc.start()
        sp_procs.append(sp_proc)

    output = sp_worker(0, sp_size, tp_size)

    for sp_proc in sp_procs:
        sp_proc.join()


if __name__ == "__main__":
    main()
