"""Benchmark and profile DSV4 DCP attention LSE merge collectives.

Run each path in a separate process for production-shape DCP8 profiling:

    torchrun --standalone --nproc_per_node=8 \
        test/manual/dsv4/bench_dcp_lse_merge.py \
        --use-sglang-group --dcp-size 8 --batch-size 128 \
        --path a2a --layers 43 --cuda-graph

The runtime reference called ``cp_lse_ag_out_rs`` is all-gather plus
all-reduce followed by a local head slice. This benchmark also retains a true
all-gather plus reduce-scatter candidate so the three communication patterns
can be compared without changing production behavior.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Callable

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
PATH_LABELS = {
    "ag-ar": "all-gather + all-reduce + slice",
    "ag-rs": "all-gather + reduce-scatter",
    "a2a": "destination-head all-to-all",
}


class DCPGroup:
    """Minimal GroupCoordinator adapter for a standalone DCP benchmark."""

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

    def reduce_scatter_along_dim(
        self, input_: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        input_ = input_.movedim(dim, 0).contiguous()
        output = torch.empty(
            (input_.shape[0] // self.world_size,) + input_.shape[1:],
            dtype=input_.dtype,
            device=input_.device,
        )
        dist.reduce_scatter_tensor(output, input_)
        return output.movedim(0, dim)

    def all_to_all_single(
        self, output: torch.Tensor, input_: torch.Tensor
    ) -> None:
        dist.all_to_all_single(output, input_)


def cp_lse_ag_out_reduce_scatter(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group,
    return_lse: bool = False,
):
    """Benchmark-only true reduce-scatter implementation."""
    if cp_group.world_size == 1:
        return (cp_attn_out, cp_attn_lse) if return_lse else cp_attn_out

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = cp_group.all_gather(cp_attn_lse, dim=0).view(
        (cp_group.world_size,) + cp_attn_lse.shape
    )
    global_lse = torch.logsumexp(lses, dim=0)
    scale = torch.exp(cp_attn_lse - global_lse).unsqueeze(-1)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    out = torch.nan_to_num(
        cp_attn_out, nan=0.0, posinf=0.0, neginf=0.0
    ) * scale
    out = cp_group.reduce_scatter_along_dim(out, dim=1).contiguous()
    if return_lse:
        local_heads = global_lse.shape[1] // cp_group.world_size
        head_start = local_heads * cp_group.rank_in_group
        head_end = head_start + local_heads
        return out, global_lse[:, head_start:head_end].contiguous()
    return out


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
        "--path",
        choices=("all", *PATH_LABELS),
        default="all",
        help="Run one isolated path for trustworthy graph-memory accounting.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of sequential attention merges captured in one graph.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture the merge path and benchmark CUDA Graph replay.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Export one GPU-only Chrome trace per rank.",
    )
    parser.add_argument("--trace-replays", type=int, default=4)
    parser.add_argument(
        "--memory-snapshot",
        type=Path,
        help="Dump a rank-0 PyTorch allocator snapshot around graph capture.",
    )
    parser.add_argument("--json-output", type=Path)
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
            "use --use-sglang-group for a production communicator topology"
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


def reduce_max_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def reduce_max_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def time_cuda_ms(
    fn: Callable[[], object], warmup: int, iters: int, device: torch.device
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = 1000 * (time.perf_counter() - start_wall) / iters
    event_ms = start.elapsed_time(end) / iters
    dist.barrier()
    return (
        reduce_max_float(event_ms, device),
        reduce_max_float(wall_ms, device),
    )


def get_memory_stats() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
    }


def memory_delta(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, int]:
    reserved_delta = after["reserved_bytes"] - before["reserved_bytes"]
    device_used_delta = before["free_bytes"] - after["free_bytes"]
    return {
        "allocated_delta_bytes": (
            after["allocated_bytes"] - before["allocated_bytes"]
        ),
        "reserved_delta_bytes": reserved_delta,
        "peak_allocated_over_baseline_bytes": (
            after["peak_allocated_bytes"] - before["allocated_bytes"]
        ),
        "peak_reserved_over_baseline_bytes": (
            after["peak_reserved_bytes"] - before["reserved_bytes"]
        ),
        "device_used_delta_bytes": device_used_delta,
        "non_allocator_delta_bytes": device_used_delta - reserved_delta,
    }


def make_layered_fn(
    fn: Callable[[], torch.Tensor], layers: int
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        result = None
        for _ in range(layers):
            result = fn()
        assert result is not None
        return result

    return run


def snapshot_path_for_rank(path: Path, rank: int) -> Path:
    suffix = path.suffix or ".pickle"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}.rank{rank}{suffix}")


def capture_cuda_graph(
    fn: Callable[[], object],
    warmup: int,
    rank: int,
    memory_snapshot: Path | None,
) -> tuple[torch.cuda.CUDAGraph, dict[str, int], dict[str, int], Path | None]:
    for _ in range(max(warmup, 1)):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()

    snapshot_path = None
    if memory_snapshot is not None and rank == 0:
        snapshot_path = snapshot_path_for_rank(memory_snapshot, rank)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=200000)

    torch.cuda.reset_peak_memory_stats()
    before = get_memory_stats()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        after = get_memory_stats()
        if snapshot_path is not None:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
    finally:
        if snapshot_path is not None:
            torch.cuda.memory._record_memory_history(enabled=None)
    return graph, before, after, snapshot_path


def export_trace(
    fn: Callable[[], object],
    trace_dir: Path,
    path_name: str,
    rank: int,
    replays: int,
) -> Path:
    if rank == 0:
        trace_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        for _ in range(replays):
            fn()
        torch.cuda.synchronize()
    trace_path = trace_dir / f"dcp_lse_{path_name}_rank{rank}.json"
    profile.export_chrome_trace(str(trace_path))
    dist.barrier()
    return trace_path


def max_correctness_error(
    actual: torch.Tensor, expected: torch.Tensor
) -> torch.Tensor:
    error = (actual - expected).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    return error


def bytes_to_gib(value: int) -> float:
    return value / 1024**3


def main() -> None:
    args = parse_args()
    world_size, rank, device = init_distributed(
        args.use_sglang_group, args.dcp_size
    )
    if any(
        value <= 0
        for value in (
            args.batch_size,
            args.head_dim,
            args.layers,
            args.warmup,
            args.iters,
            args.trace_replays,
        )
    ):
        raise ValueError(
            "shape, layer, warmup, iteration, and replay counts must be positive"
        )
    if args.memory_snapshot is not None and not args.cuda_graph:
        raise ValueError("--memory-snapshot requires --cuda-graph")
    if args.memory_snapshot is not None and args.path == "all":
        raise ValueError("--memory-snapshot requires one isolated --path")

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

    path_fns = {
        "ag-ar": lambda: cp_lse_ag_out_rs(partial_out, partial_lse, group),
        "ag-rs": lambda: cp_lse_ag_out_reduce_scatter(
            partial_out, partial_lse, group
        ),
        "a2a": lambda: cp_lse_a2a_out_rs(partial_out, partial_lse, group),
    }
    selected_paths = list(PATH_LABELS) if args.path == "all" else [args.path]

    ref_out, ref_lse = cp_lse_ag_out_rs(
        partial_out, partial_lse, group, return_lse=True
    )
    correctness = {}
    for path_name in selected_paths:
        if path_name == "ag-ar":
            out, lse = ref_out, ref_lse
        elif path_name == "ag-rs":
            out, lse = cp_lse_ag_out_reduce_scatter(
                partial_out, partial_lse, group, return_lse=True
            )
        else:
            out, lse = cp_lse_a2a_out_rs(
                partial_out, partial_lse, group, return_lse=True
            )
        torch.testing.assert_close(out, ref_out, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(lse, ref_lse, rtol=1e-6, atol=1e-6)
        correctness[path_name] = {
            "max_abs_output": float(max_correctness_error(out, ref_out).item()),
            "max_abs_lse": float(max_correctness_error(lse, ref_lse).item()),
        }
    del ref_out, ref_lse, out, lse

    results = {}
    for path_name in selected_paths:
        layered_fn = make_layered_fn(path_fns[path_name], args.layers)
        timed_fn = layered_fn
        graph = None
        graph_memory = None
        snapshot_path = None
        if args.cuda_graph:
            graph, before, after, snapshot_path = capture_cuda_graph(
                layered_fn,
                args.warmup,
                rank,
                args.memory_snapshot,
            )
            timed_fn = graph.replay
            local_delta = memory_delta(before, after)
            graph_memory = {
                "rank0_before": before if rank == 0 else None,
                "rank0_after": after if rank == 0 else None,
                "max_delta_across_ranks": {
                    name: reduce_max_int(value, device)
                    for name, value in local_delta.items()
                },
            }

        event_ms, wall_ms = time_cuda_ms(
            timed_fn, args.warmup, args.iters, device
        )
        timing_error = abs(event_ms - wall_ms) / wall_ms
        trace_path = None
        if args.trace_dir is not None:
            trace_path = export_trace(
                timed_fn,
                args.trace_dir,
                path_name,
                rank,
                args.trace_replays,
            )

        results[path_name] = {
            "label": PATH_LABELS[path_name],
            "cuda_event_graph_ms": event_ms,
            "wall_graph_ms": wall_ms,
            "cuda_event_per_merge_ms": event_ms / args.layers,
            "wall_per_merge_ms": wall_ms / args.layers,
            "event_wall_error_percent": 100 * timing_error,
            "timing_valid": (not args.cuda_graph) or timing_error <= 0.05,
            "graph_memory": graph_memory,
            "rank0_memory_snapshot": (
                str(snapshot_path) if snapshot_path is not None else None
            ),
            "rank0_trace": str(trace_path) if rank == 0 and trace_path else None,
        }

        if graph is not None:
            torch.cuda.synchronize()
            dist.barrier()
            graph.reset()
            graph = timed_fn = layered_fn = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            dist.barrier()

    record = {
        "world_size": world_size,
        "dcp_size": group.world_size,
        "execution_mode": "cuda_graph" if args.cuda_graph else "eager",
        "batch_size": args.batch_size,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "layers_per_graph": args.layers,
        "warmup": args.warmup,
        "iters": args.iters,
        "isolated_path": args.path if args.path != "all" else None,
        "correctness": correctness,
        "results": results,
    }

    if rank == 0:
        print("DeepSeek-V4 DCP attention LSE merge benchmark")
        print(f"world_size / DCP size      : {world_size}/{group.world_size}")
        print(f"execution mode             : {record['execution_mode']}")
        print(
            "batch/heads/head_dim       : "
            f"{args.batch_size}/{args.num_heads}/{args.head_dim}"
        )
        print(f"merges per graph           : {args.layers}")
        for path_name, result in results.items():
            print(
                f"{path_name:5s} event/wall per merge : "
                f"{result['cuda_event_per_merge_ms']:.4f}/"
                f"{result['wall_per_merge_ms']:.4f} ms"
            )
            print(
                f"{path_name:5s} event-wall error     : "
                f"{result['event_wall_error_percent']:.2f}% "
                f"(valid={result['timing_valid']})"
            )
            memory = result["graph_memory"]
            if memory is not None:
                delta = memory["max_delta_across_ranks"]
                print(
                    f"{path_name:5s} graph reserved delta  : "
                    f"{bytes_to_gib(delta['reserved_delta_bytes']):.3f} GiB"
                )
                print(
                    f"{path_name:5s} device-used delta     : "
                    f"{bytes_to_gib(delta['device_used_delta_bytes']):.3f} GiB"
                )
                print(
                    f"{path_name:5s} non-allocator delta   : "
                    f"{bytes_to_gib(delta['non_allocator_delta_bytes']):.3f} GiB"
                )

        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(record, indent=2, sort_keys=True))
            print(f"JSON result                : {args.json_output}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
