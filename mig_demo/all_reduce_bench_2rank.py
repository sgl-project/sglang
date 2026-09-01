"""all_reduce busbw between 2 emulated ranks on one physical H200.

Single-GPU variant of stas00/ml-engineering network/benchmarks/all_reduce_bench.py:
measures the interconnect the emulated TP=2 sglang ranks actually get
(NCCL 2.31 multi-rank on one device, host-SHM transport expected).
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PAYLOAD_BYTES = 512 * 1024**2  # 512 MiB, same as the doc's tables
WARMUP, ITERS = 5, 20


def worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29618"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    x = torch.empty(PAYLOAD_BYTES // 4, dtype=torch.float32, device="cuda:0")

    for _ in range(WARMUP):
        dist.all_reduce(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        dist.all_reduce(x)
    end.record()
    torch.cuda.synchronize()

    if rank == 0:
        sec = start.elapsed_time(end) / 1e3 / ITERS
        algbw = PAYLOAD_BYTES / sec / 1e9
        busbw = algbw * 2 * (world_size - 1) / world_size
        print(f"payload=512MiB iters={ITERS} algbw={algbw:.1f}GB/s busbw={busbw:.1f}GB/s")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
