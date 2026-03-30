"""
Benchmark symmetric-memory collectives: AllGather, ReduceScatter, All-to-All.

Compares torch.distributed eager vs PyNccl symmetric-memory CUDA graph for each
collective across a sweep of message sizes.

Usage:
    torchrun --nproc_per_node=8 benchmark_symm_mem.py
    torchrun --nproc_per_node=8 benchmark_symm_mem.py --warmup 10 --iters 200
    torchrun --nproc_per_node=4 benchmark_symm_mem.py --ops ag rs a2a
"""

import argparse
from typing import Dict, List

import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    SymmetricMemoryContext,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark symmetric-memory collectives across message sizes."
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Benchmark iterations per size."
    )
    parser.add_argument(
        "--graph-loop",
        type=int,
        default=100,
        help="Ops per CUDA graph replay (amortizes launch overhead).",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["ag", "rs", "a2a"],
        choices=["ag", "rs", "a2a"],
        help="Which collectives to benchmark.",
    )
    parser.add_argument(
        "--min-exp", type=int, default=10, help="Min message size as 2^N bytes."
    )
    parser.add_argument(
        "--max-exp", type=int, default=16, help="Max message size as 2^N bytes."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for tensors.",
    )
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def human_size(nbytes: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if nbytes < 1024.0 or unit == "GiB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} GiB"


def bench_eager(func, output, inp, warmup: int, iters: int) -> float:
    """Benchmark an eager collective. Returns average latency in microseconds."""
    eager_input = inp.clone()
    for _ in range(warmup):
        func(output, eager_input)
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        func(output, eager_input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # ms -> us


def bench_symm_graph(
    func, output, inp, group, warmup: int, iters: int, graph_loop: int
) -> float:
    """Benchmark a collective with symmetric memory + CUDA graph. Returns avg us."""
    with SymmetricMemoryContext(group):
        graph_input = inp.clone()
        graph_output = output.clone()

    with graph_capture() as ctx:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=ctx.stream):
            for _ in range(graph_loop):
                func(graph_output, graph_input)

    graph.replay()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    latencies: List[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))

    graph.reset()
    return sum(latencies) / len(latencies) / graph_loop * 1000  # ms -> us


def print_table(data: List[Dict[str, str]]):
    if not data:
        return
    try:
        from tabulate import tabulate

        print(tabulate(data, headers="keys", tablefmt="github", floatfmt=".2f"))
    except ImportError:
        headers = list(data[0].keys())
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in data:
            print("| " + " | ".join(str(row[h]) for h in headers) + " |")


def main():
    args = parse_args()
    dtype = get_dtype(args.dtype)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % 8)
    device = torch.cuda.current_device()

    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank % 8)
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    tp_group = get_tensor_model_parallel_group()
    nccl_group = tp_group.device_group
    pynccl_comm: PyNcclCommunicator = tp_group.pynccl_comm
    dist.barrier()

    results: List[Dict] = []

    for exp in range(args.min_exp, args.max_exp + 1):
        sz = 2**exp
        if sz * dtype.itemsize > 2**24:
            break

        inp = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
        row: Dict = {"msg_size": human_size(inp.nbytes)}

        # -- AllGather --
        if "ag" in args.ops:
            ag_out = torch.empty(sz * world_size, dtype=dtype, device=device)

            eager_us = bench_eager(
                lambda o, i: dist.all_gather_into_tensor(o, i, group=nccl_group),
                ag_out,
                inp,
                args.warmup,
                args.iters,
            )
            graph_us = bench_symm_graph(
                lambda o, i: pynccl_comm.all_gather(o, i),
                ag_out,
                inp,
                tp_group,
                args.warmup,
                args.iters,
                args.graph_loop,
            )

            row["AG eager (us)"] = f"{eager_us:.2f}"
            row["AG symm graph (us)"] = f"{graph_us:.2f}"

            if rank == 0:
                print(f"AG sz={sz}: eager={eager_us:.2f}us graph={graph_us:.2f}us")

        # -- ReduceScatter --
        if "rs" in args.ops:
            rs_out = torch.empty(sz // world_size, dtype=dtype, device=device)

            eager_us = bench_eager(
                lambda o, i: dist.reduce_scatter_tensor(o, i, group=nccl_group),
                rs_out,
                inp,
                args.warmup,
                args.iters,
            )
            graph_us = bench_symm_graph(
                lambda o, i: pynccl_comm.reduce_scatter(o, i),
                rs_out,
                inp,
                tp_group,
                args.warmup,
                args.iters,
                args.graph_loop,
            )

            row["RS eager (us)"] = f"{eager_us:.2f}"
            row["RS symm graph (us)"] = f"{graph_us:.2f}"

            if rank == 0:
                print(f"RS sz={sz}: eager={eager_us:.2f}us graph={graph_us:.2f}us")

        # -- All-to-All --
        if "a2a" in args.ops:
            a2a_out = torch.empty_like(inp)

            eager_us = bench_eager(
                lambda o, i: dist.all_to_all_single(o, i, group=nccl_group),
                a2a_out,
                inp,
                args.warmup,
                args.iters,
            )
            graph_us = bench_symm_graph(
                lambda o, i: pynccl_comm.all_to_all_single(o, i),
                a2a_out,
                inp,
                tp_group,
                args.warmup,
                args.iters,
                args.graph_loop,
            )

            row["A2A eager (us)"] = f"{eager_us:.2f}"
            row["A2A symm graph (us)"] = f"{graph_us:.2f}"

            if rank == 0:
                print(f"A2A sz={sz}: eager={eager_us:.2f}us graph={graph_us:.2f}us")

        results.append(row)

    if rank == 0:
        print("\n=== Results ===")
        print_table(results)

    dist.barrier()
    # Skip dist.destroy_process_group() -- symmetric memory + NCCL cleanup
    # order causes segfaults; torchrun handles process lifecycle.


if __name__ == "__main__":
    main()
