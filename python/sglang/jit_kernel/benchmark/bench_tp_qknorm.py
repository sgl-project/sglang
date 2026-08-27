from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.all_reduce import (
    fused_parallel_qknorm,
    get_fused_parallel_qknorm_max_occupancy,
)
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    suite="stage-b-kernel-benchmark-1-gpu-large",
    disabled="requires multi-GPU, self-skips in CI",
)

Q_K_DIMS = [(6144, 1024)]
DTYPE = torch.bfloat16
EPS = 1e-6
BATCH_SIZES = get_ci_test_range([2**i for i in range(15)], [1, 64, 1024])
NUM_LAYERS = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def init_distributed():
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
    max_occupancy = get_fused_parallel_qknorm_max_occupancy(
        DTYPE, world_size, Q_K_DIMS[0][0], Q_K_DIMS[0][1]
    )
    if rank == 0:
        print(f"Max occupancy for fused_parallel_qknorm: {max_occupancy} blocks/SM")

    props = torch.cuda.get_device_properties(device)
    comm = CustomAllReduceV2(
        cpu_group,
        device,
        max_pull_size=0,
        max_push_size=8 * max(BATCH_SIZES),
        max_push_blocks=props.multi_processor_count * max_occupancy,
    )
    comm_ = CustomAllReduceV2(cpu_group, device)
    if comm.disabled or comm_.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
    return rank, world_size, device, cpu_group, comm, comm_


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


def rmsnorm_baseline(
    comm_,
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    world_size: int,
) -> None:
    from sglang.srt.models.minimax_m2 import rms_apply_serial, rms_sumsq_serial

    sum_sq = rms_sumsq_serial(q, k)
    sum_sq = comm_.custom_all_reduce(sum_sq)
    rms_apply_serial(q, k, q_weight, k_weight, sum_sq, world_size, EPS)


def main():
    args = parse_args()
    rank, world_size, device, _, comm, comm_ = init_distributed()
    torch.cuda.set_stream(torch.cuda.Stream())

    if rank == 0:
        print(
            f"{'q_dim':>8} {'k_dim':>8} {'batch':>8} {'fused_us':>12} {'baseline_us':>12}"
        )

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
                rmsnorm_baseline(
                    comm_,
                    q[i],
                    k[i],
                    q_weight[i],
                    k_weight[i],
                    world_size,
                )

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
