#!/usr/bin/env python3
"""Time 10 KiB fp16 custom-AR vs NCCL, eager and CUDA-graph.

Launch (NVLink pair 0–4, then NVLink quad 0–3):

  CUDA_VISIBLE_DEVICES=0,4 SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \\
    torchrun --standalone --nproc_per_node=2 scripts/dsv41_v100_custom_ar_microbench.py

  CUDA_VISIBLE_DEVICES=0,1,2,3 SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \\
    torchrun --standalone --nproc_per_node=4 scripts/dsv41_v100_custom_ar_microbench.py
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

N_ELEM = 5120  # 10 KiB fp16, DSV4.1 hidden
N_AR = 83
WARMUP = 10
ITERS = 20
MAX_BYTES = 2 * 1024 * 1024


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _us_per_ar(ms: float) -> float:
    return (ms * 1000.0) / N_AR


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    cpu_group = dist.new_group(backend="gloo")
    dist.barrier()

    from sglang.srt.distributed.device_communicators import custom_all_reduce_utils
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    # P2P cache path wants sglang's world group. These GPUs are NVLink.
    nvis = torch.cuda.device_count()
    custom_all_reduce_utils._gpu_p2p_access_cache = {
        f"{i}->{j}": True for i in range(nvis) for j in range(nvis)
    }

    ca = CustomAllreduce(group=cpu_group, device=device, max_size=MAX_BYTES)
    if getattr(ca, "disabled", True):
        raise SystemExit(f"rank {rank}: CustomAllreduce disabled (full_nvlink/P2P)")

    torch.manual_seed(0)
    x = torch.randn(N_ELEM, dtype=torch.float16, device=device)
    x_nccl = x.clone()
    dist.all_reduce(x_nccl)
    y = ca._all_reduce_impl(x.clone(), registered=False)
    if y is None:
        raise SystemExit(f"rank {rank}: custom AR returned None")
    max_abs = (y.float() - x_nccl.float()).abs().max().item()
    _log(rank, f"world={world} device={device} full_nvlink={ca.full_nvlink} max_abs={max_abs:.4g}")
    if max_abs > 0.05:
        raise SystemExit(f"rank {rank}: CA vs NCCL mismatch max_abs={max_abs}")

    def time_eager_nccl() -> float:
        buf = x.clone()
        for _ in range(WARMUP):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            for _ in range(N_AR):
                dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        return (time.perf_counter() - t0) / ITERS * 1000.0

    def time_eager_ca() -> float:
        buf = x.clone()
        for _ in range(WARMUP):
            out = ca._all_reduce_impl(buf, registered=False)
            buf.copy_(out)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            for _ in range(N_AR):
                out = ca._all_reduce_impl(buf, registered=False)
                buf.copy_(out)
        torch.cuda.synchronize()
        dist.barrier()
        return (time.perf_counter() - t0) / ITERS * 1000.0

    nccl_ms = time_eager_nccl()
    ca_ms = time_eager_ca()
    _log(
        rank,
        f"eager  {N_AR} AR: NCCL {nccl_ms:.1f} ms/tok ({_us_per_ar(nccl_ms):.0f} us/AR)  "
        f"CA {ca_ms:.1f} ms/tok ({_us_per_ar(ca_ms):.0f} us/AR)",
    )

    buf = x.clone()
    g = torch.cuda.CUDAGraph()
    dist.barrier()
    with ca.capture():
        for _ in range(3):
            out = ca._all_reduce_impl(buf, registered=False)
            buf.copy_(out)
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.graph(g):
            for _ in range(N_AR):
                out = ca._all_reduce_impl(buf, registered=False)
                buf.copy_(out)
        dist.barrier()
    dist.barrier()

    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    graph_ms = (time.perf_counter() - t0) / ITERS * 1000.0
    _log(
        rank,
        f"graph  {N_AR} AR: CA {graph_ms:.1f} ms/tok ({_us_per_ar(graph_ms):.0f} us/AR)",
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
