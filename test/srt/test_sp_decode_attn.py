import multiprocessing
import random

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, merge_state
from vllm.distributed import init_distributed_environment

from sglang.srt.layers.parallel_utils import get_sp_group, initialize_model_parallel

NUM_HEADS = 32
HEAD_DIM = 128
SCALING = 1
NUM_KV_HEADS = 8
LAYER_ID = 0
LOGIT_CAP = -1


BATCH_SIZE = 3
SEQ_LENS = [16, 64, 128]


def gen_qkv(sp_rank: int = 0, sp_size: int = 1):
    torch.manual_seed(42)
    random.seed(42)

    q = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_DIM).cuda().half()
    total_num_context_tokens = sum(SEQ_LENS)
    kv_cache = (
        torch.randn(total_num_context_tokens, 2, NUM_KV_HEADS, HEAD_DIM).cuda().half()
    )

    if sp_size > 1:
        q_head_idxes = _get_sequence_parallel_head_idxes(
            NUM_HEADS, NUM_KV_HEADS, sp_rank, sp_size
        )
        q = q[:, q_head_idxes].contiguous()

        sp_kv_cache = (
            torch.empty(total_num_context_tokens // sp_size, 2, NUM_KV_HEADS, HEAD_DIM)
            .cuda()
            .half()
        )
        sp_stt, stt = 0, 0
        for i in range(BATCH_SIZE):
            seq_len = SEQ_LENS[i]
            sp_seq_len = seq_len // sp_size

            sp_end = sp_stt + sp_seq_len
            end = stt + seq_len

            sp_kv_cache[sp_stt:sp_end] = kv_cache[
                stt + sp_rank * sp_seq_len : stt + (sp_rank + 1) * sp_seq_len
            ]
            sp_stt = sp_end
            stt = end
        kv_cache = sp_kv_cache

    return q, kv_cache


def init_flashinfer(sp_size: int = 1, tp_size: int = 1):

    workspace_buffer = torch.empty(
        1, 128 * 1024 * 1024, dtype=torch.int8, device="cuda"
    )

    flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer[0], "NHD"
    )

    num_qo_heads = NUM_HEADS
    num_kv_heads = NUM_KV_HEADS

    seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int32, device="cuda")
    seq_lens = seq_lens // sp_size
    total_num_context_tokens = sum(SEQ_LENS) // sp_size

    kv_indptr = torch.zeros((BATCH_SIZE + 1,), dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(
        total_num_context_tokens, dtype=torch.int32, device="cuda"
    )
    kv_last_page_len = torch.ones((BATCH_SIZE,), dtype=torch.int32, device="cuda")

    flashinfer_decode_wrapper.end_forward()
    flashinfer_decode_wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        1,
    )

    return flashinfer_decode_wrapper


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

    print("SP worker", rank, "initialized on", torch.cuda.current_device())

    decode_wrapper = init_flashinfer(sp_size=sp_size, tp_size=tp_size)
    q, kv_cache = gen_qkv(rank, sp_size)

    gathered_q = get_sp_group().all_gather(q.view(1, *q.shape), dim=0)
    q = torch.empty_like(gathered_q).view(-1, NUM_HEADS, HEAD_DIM)

    for i in range(sp_size):
        idxes = _get_sequence_parallel_head_idxes(NUM_HEADS, NUM_KV_HEADS, i, sp_size)
        q[:, idxes] = gathered_q[i]

    # Computation
    o, s = decode_wrapper.forward_return_lse(q, kv_cache)

    os = get_sp_group().all_gather(o.view(1, *o.shape), dim=0)
    ss = get_sp_group().all_gather(s.view(1, *s.shape), dim=0)
    for i in range(sp_size):
        if i != rank:
            o, s = merge_state(os[i], ss[i], o, s)
    output = o

    o_truth = reference_attn()

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

    decode_wrapper = init_flashinfer()
    q, kv_cache = gen_qkv()

    return decode_wrapper.forward(q, kv_cache)


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

    sp_worker(0, sp_size, tp_size)

    for sp_proc in sp_procs:
        sp_proc.join()


if __name__ == "__main__":
    main()
