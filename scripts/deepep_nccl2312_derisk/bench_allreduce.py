#!/usr/bin/env python3
"""Minimal cross-node all_reduce / all_to_all bench for R7 (NCCL 2.30.7 vs 2.31.2).
Run on both ranks: MASTER_ADDR=<rank0-addr> WORLD_SIZE=2 RANK=<r> python3 bench_allreduce.py
Spawns 4 local procs (one per GPU)."""

import ctypes
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [2**24, 2**27, 2**30]  # 16MB, 128MB, 1GB in bf16 elems*2


def bench_one(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def worker(local_rank, num_local):
    nccl_v = ctypes.c_int()
    ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(nccl_v))
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '8361')}",
        world_size=int(os.environ["WORLD_SIZE"]) * num_local,
        rank=int(os.environ["RANK"]) * num_local + local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    torch.cuda.set_device(local_rank)
    world = dist.get_world_size()
    if dist.get_rank() == 0:
        print(f"world={world} nccl_runtime={nccl_v.value}", flush=True)
    for n in SIZES:
        x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        t = bench_one(lambda: dist.all_reduce(x))
        if dist.get_rank() == 0:
            bw = n * 2 * 2 * (world - 1) / world / t / 1e9  # algbw GB/s (send+recv)
            print(
                f"all_reduce {n * 2 / 1e6:.0f}MB: {t * 1e3:.3f} ms, algbw {bw:.1f} GB/s",
                flush=True,
            )
        del x
        torch.cuda.empty_cache()
    # all_to_all single
    for n in [2**22, 2**25]:
        x = torch.randn(world, n, dtype=torch.bfloat16, device="cuda")
        y = torch.empty_like(x)
        t = bench_one(lambda: dist.all_to_all_single(y, x))
        if dist.get_rank() == 0:
            bw = n * 2 * (world - 1) / t / 1e9
            print(
                f"all_to_all {n * 2 / 1e6:.0f}MB/rank: {t * 1e3:.3f} ms, bw {bw:.1f} GB/s",
                flush=True,
            )
        del x, y
        torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(4,), nprocs=4)
