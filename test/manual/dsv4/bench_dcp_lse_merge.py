"""Benchmark and validate DSV4 DCP attention LSE merge collectives.

Run on a two-GPU H20 node:

    torchrun --standalone --nproc_per_node=2 \
        test/manual/dsv4/bench_dcp_lse_merge.py

To reproduce the production TP8/DCP2 communicator topology, run:

    torchrun --standalone --nproc_per_node=8 \
        test/manual/dsv4/bench_dcp_lse_merge.py \
        --use-sglang-group --dcp-size 2

The benchmark compares the reference all-gather plus all-reduce path with the
destination-head all-to-all candidate.
"""

from __future__ import annotations

import argparse
import gc
import os

import torch
import torch.distributed as dist

from sglang.kernels.ops.attention.utils import (
    cp_lse_a2a_out_rs,
    cp_lse_ag_out_rs,
)
from sglang.srt.distributed.parallel_state import (
    get_dcp_group,
    init_distributed_environment,
    initialize_model_parallel,
)


class DCPGroup:
    """Minimal GroupCoordinator adapter for a standalone DCP=2 benchmark."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank

    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        input_ = input_.contiguous()
        output = torch.empty(
            (self.world_size * input_.shape[0],) + input_.shape[1:],
            dtype=input_.dtype,
            device=input_.device,
        )
        dist.all_gather_into_tensor(output, input_)
        output = output.reshape((self.world_size,) + input_.shape)
        output = output.movedim(0, dim)
        return output.reshape(
            input_.shape[:dim]
            + (self.world_size * input_.shape[dim],)
            + input_.shape[dim + 1 :]
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_)
        return input_

    def all_to_all_single(
        self, output: torch.Tensor, input_: torch.Tensor
    ) -> None:
        dist.all_to_all_single(output, input_)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DSV4 DCP attention LSE merge collectives."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture each merge path and benchmark CUDA Graph replay.",
    )
    parser.add_argument(
        "--use-sglang-group",
        action="store_true",
        help="Use the production GroupCoordinator instead of the local adapter.",
    )
    return parser.parse_args()


def init_distributed(
    use_sglang_group: bool,
    dcp_size: int,
) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if dcp_size <= 0 or world_size % dcp_size != 0:
        raise RuntimeError(
            f"DCP size must divide world size, got {world_size=} and {dcp_size=}"
        )
    if not use_sglang_group and world_size != dcp_size:
        raise RuntimeError(
            "The local group adapter requires world size to equal DCP size; "
            "use --use-sglang-group for TP8/DCP2 topology"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if use_sglang_group:
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            decode_context_parallel_size=dcp_size,
        )
    else:
        dist.init_process_group("nccl")
    return world_size, rank, device


def time_cuda_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    dist.barrier()
    return start.elapsed_time(end) / iters


def reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def capture_cuda_graph(fn, warmup: int):
    for _ in range(max(warmup, 1)):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return graph


def main() -> None:
    args = parse_args()
    world_size, rank, device = init_distributed(
        args.use_sglang_group, args.dcp_size
    )
    if args.batch_size <= 0 or args.head_dim <= 0:
        raise ValueError("batch size and head dimension must be positive")
    group = (
        get_dcp_group()
        if args.use_sglang_group
        else DCPGroup(world_size, rank)
    )
    if args.num_heads % group.world_size != 0:
        raise ValueError("num heads must be divisible by DCP world size")

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    partial_out = torch.randn(
        args.batch_size,
        args.num_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    partial_lse = torch.randn(
        args.batch_size,
        args.num_heads,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    ref_out, ref_lse = cp_lse_ag_out_rs(
        partial_out, partial_lse, group, return_lse=True
    )
    a2a_out, a2a_lse = cp_lse_a2a_out_rs(
        partial_out, partial_lse, group, return_lse=True
    )
    torch.testing.assert_close(a2a_lse, ref_lse, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(a2a_out, ref_out, rtol=2e-3, atol=2e-3)

    def ref_fn():
        return cp_lse_ag_out_rs(partial_out, partial_lse, group)

    def a2a_fn():
        return cp_lse_a2a_out_rs(partial_out, partial_lse, group)

    captured_graphs = []
    if args.cuda_graph:
        ref_graph = capture_cuda_graph(ref_fn, args.warmup)
        a2a_graph = capture_cuda_graph(a2a_fn, args.warmup)
        captured_graphs = [ref_graph, a2a_graph]
        ref_fn = ref_graph.replay
        a2a_fn = a2a_graph.replay

    ref_ms = reduce_max(
        time_cuda_ms(
            ref_fn,
            args.warmup,
            args.iters,
        ),
        device,
    )
    a2a_ms = reduce_max(
        time_cuda_ms(
            a2a_fn,
            args.warmup,
            args.iters,
        ),
        device,
    )
    max_abs_out = (a2a_out - ref_out).abs().max()
    max_abs_lse = (a2a_lse - ref_lse).abs().max()
    dist.all_reduce(max_abs_out, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_abs_lse, op=dist.ReduceOp.MAX)

    if rank == 0:
        print("DeepSeek-V4 DCP attention LSE merge benchmark")
        print(f"world_size / DCP size      : {world_size}/{group.world_size}")
        print(f"execution mode             : {'cuda graph' if args.cuda_graph else 'eager'}")
        print(f"batch_size                 : {args.batch_size}")
        print(f"total heads/head_dim       : {args.num_heads}/{args.head_dim}")
        print(f"reference AG+AR ms         : {ref_ms:.3f}")
        print(f"all-to-all merge ms        : {a2a_ms:.3f}")
        print(f"all-to-all speedup         : {ref_ms / a2a_ms:.3f}x")
        print(f"max abs output difference  : {max_abs_out.item():.6g}")
        print(f"max abs LSE difference     : {max_abs_lse.item():.6g}")

    if captured_graphs:
        torch.cuda.synchronize()
        dist.barrier()
        for graph in captured_graphs:
            graph.reset()
        ref_fn = a2a_fn = ref_graph = a2a_graph = graph = None
        captured_graphs.clear()
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
