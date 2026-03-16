from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from flashinfer.norm import rmsnorm

from sglang.jit_kernel.all_reduce import (
    fused_parallel_qknorm,
    get_fused_parallel_qknorm_max_occupancy,
)
from sglang.jit_kernel.benchmark.utils import is_in_ci

Q_K_DIMS = [(6144, 1024)]
DTYPE = torch.bfloat16
EPS = 1e-6
BATCH_SIZES = [1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384]
if is_in_ci():
    BATCH_SIZES = [1, 64, 1024]
NUM_LAYERS = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def init_distributed():
    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = local_rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )

    cpu_group = coord.cpu_group
    nccl_group = coord.device_group
    assert nccl_group is not None

    max_pull_size = 16 * 1024 * 1024
    max_push_size = 16 * 1024 * 1024
    max_occupancy = get_fused_parallel_qknorm_max_occupancy(
        DTYPE, world_size, Q_K_DIMS[0][0], Q_K_DIMS[0][1]
    )
    if rank == 0:
        print(f"Max occupancy for fused_parallel_qknorm: {max_occupancy} blocks/SM")
    comm = CustomAllReduceV2(
        cpu_group,
        device,
        max_pull_size,
        max_push_size,
        max_push_blocks=148 * max_occupancy,
    )
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")

    return rank, world_size, device, cpu_group, nccl_group, comm


def _all_gather_cat(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(gathered, x, group=group)
    return torch.cat(gathered, dim=-1)


@torch.inference_mode()
def bench_one(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(0)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(NUM_LAYERS):
            fn(i)

    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    graph.replay()
    start.record()
    for i in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (iters * NUM_LAYERS)


def main():
    args = parse_args()
    rank, world_size, device, _, nccl_group, comm = init_distributed()
    torch.cuda.set_stream(torch.cuda.Stream())

    if rank == 0:
        print(
            f"{'q_dim':>8} {'k_dim':>8} {'batch':>8} {'fused_us':>12} {'baseline_us':>12}"
        )

    def all_gather(x: torch.Tensor) -> torch.Tensor:
        return _all_gather_cat(x, nccl_group)

    for q_dim, k_dim in Q_K_DIMS:
        local_q_dim = q_dim // world_size
        local_k_dim = k_dim // world_size
        for batch_size in BATCH_SIZES:
            q = torch.randn(
                NUM_LAYERS, batch_size, local_q_dim, device=device, dtype=DTYPE
            )
            k = torch.randn(
                NUM_LAYERS, batch_size, local_k_dim, device=device, dtype=DTYPE
            )
            q_weight = torch.randn(NUM_LAYERS, local_q_dim, device=device, dtype=DTYPE)
            k_weight = torch.randn(NUM_LAYERS, local_k_dim, device=device, dtype=DTYPE)
            q_full = all_gather(q)
            k_full = all_gather(k)
            q_weight_full = all_gather(q_weight)
            k_weight_full = all_gather(k_weight)

            def run_fused(i: int):
                fused_parallel_qknorm(
                    comm.obj,
                    q[i],
                    k[i],
                    q_weight[i],
                    k_weight[i],
                    EPS,
                )

            def run_baseline(i: int):
                rmsnorm(q_full[i], q_weight_full[i], out=q_full[i], eps=EPS)
                rmsnorm(k_full[i], k_weight_full[i], out=k_full[i], eps=EPS)

            fused_us = bench_one(run_fused, args.warmup, args.iters)
            baseline_us = bench_one(run_baseline, args.warmup, args.iters)

            if rank == 0:
                print(
                    f"{q_dim:8d} {k_dim:8d} {batch_size:8d} "
                    f"{fused_us:12.1f} {baseline_us:12.1f}"
                )

    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
