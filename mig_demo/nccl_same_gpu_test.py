"""NCCL 2.31 probe: 2 ranks on a single physical GPU (no MIG).

Launches 2 processes, both bound to cuda:0 of the same H200, and runs an
all_reduce over NCCL. This is the no-MIG variant of the single-GPU
multi-rank emulation from stas00/ml-engineering (emulate-multi-node.md).

Requires NCCL>=2.31 preloaded (LD_PRELOAD) and NCCL_MULTI_RANK_GPU_ENABLE=1.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29617"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    x = torch.full((1 << 20,), float(rank + 1), device="cuda:0")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    dist.barrier()
    expected = sum(r + 1 for r in range(world_size))
    assert x.mean().item() == expected, f"rank {rank}: got {x.mean().item()}"
    print(
        f"rank {rank}/{world_size} OK on {torch.cuda.get_device_name(0)} "
        f"(all_reduce sum correct: {expected})",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("PASS: 2 NCCL ranks coexisted on one physical GPU")
