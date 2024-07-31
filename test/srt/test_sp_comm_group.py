import multiprocessing
import random

import torch
from vllm.distributed import init_distributed_environment

from sglang.srt.layers.parallel_utils import get_sp_group, initialize_model_parallel

NUM_TOKENS = 3
NUM_KV_HEADS = 2
HEAD_DIM = 4


def gen_kv(rank: int = 0, sp_size: int = 1):
    torch.manual_seed(42)
    random.seed(42)
    k = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM).cuda().half()
    v = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM).cuda().half()

    return k, v


def sp_worker(rank: int = 0, sp_size: int = 1, tp_size: int = 1):
    torch.manual_seed(42)
    random.seed(42)

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
    print("SP worker", rank, "initialized on", torch.cuda.current_device())

    k, v = gen_kv(rank, sp_size)

    ks = get_sp_group().all_gather(k.view(1, *k.shape), dim=0)
    vs = get_sp_group().all_gather(v.view(1, *v.shape), dim=0)

    print("SP worker", rank, "all-gathered ks", ks)
    print("SP worker", rank, "all-gathered vs", vs)


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
