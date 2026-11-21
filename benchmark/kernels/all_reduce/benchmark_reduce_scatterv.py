"""Compare the DP-attention MoE combine collectives at decode sizes.

The decode combine under attention DP + EP MoE (moe_a2a_backend=none) reduces
a [world * rows_per_rank, hidden] buffer and hands each rank its own rows.
The FFN exit can run it as reduce_scatterv (a group of world_size ncclReduce
calls, one per root) or as the fixed-size reduce_scatter_tensor (one
ncclReduceScatter); with equal MAX_LEN rows both are valid. This measures the
two against an AllGather of the same rows, eagerly and inside a CUDA graph.

Inputs stay L2-hot on purpose: in serving the combine reads a buffer the
preceding moeFinalize kernel just wrote, and the cost is NVLink, not HBM.

Usage (one node, inside the serving image so the same NCCL library loads):
    torchrun --nproc_per_node 8 benchmark_reduce_scatterv.py --hidden 6144
"""

import argparse
import os
from typing import Callable, Dict, List

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

OPS = ("all_gather", "reduce_scatter_grouped", "reduce_scatter_single")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument(
        "--rows-per-rank",
        type=int,
        nargs="+",
        # Decode verify (~28 rows) through a 4096-token prefill chunk per rank.
        default=[4, 8, 16, 28, 32, 64, 128, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--graph-calls", type=int, default=20)
    return parser.parse_args()


def _make_op(
    name: str, comm: PyNcclCommunicator, world: int, rows: int, hidden: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    device = comm.device
    if name == "all_gather":
        gathered = torch.empty(world * rows, hidden, dtype=torch.bfloat16, device=device)

        def op(local: torch.Tensor) -> torch.Tensor:
            comm.all_gather(gathered, local)
            return gathered

        return op

    out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
    sizes = [rows] * world if name == "reduce_scatter_grouped" else None

    def op(full: torch.Tensor) -> torch.Tensor:
        comm.reduce_scatter(out, full, sizes=sizes)
        return out

    return op


def _time_eager_us(op, inp: torch.Tensor, iters: int) -> float:
    for _ in range(10):
        op(inp)
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        op(inp)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def _time_graph_us(op, inp: torch.Tensor, calls: int, replays: int) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(inp)  # warm the NCCL plan outside capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            op(inp)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    graph.reset()
    return start.elapsed_time(end) * 1000 / (replays * calls)


def _max_abs_error_vs_fp32(
    comm: PyNcclCommunicator, full: torch.Tensor, rows: int, rank: int
) -> Dict[str, float]:
    reference = full.float().clone()
    comm.all_reduce(reference)
    expected = reference.narrow(0, rank * rows, rows)
    errors = {}
    for name in ("reduce_scatter_grouped", "reduce_scatter_single"):
        out = _make_op(name, comm, comm.world_size, rows, full.shape[1])(full)
        errors[name] = (out.float() - expected).abs().max().item()
    torch.cuda.synchronize()
    # A sum of world_size N(0,1) bf16 values is off by ~1e-2 after rounding; a
    # large or non-finite error means the collective did not run.
    for name, err in errors.items():
        if not err < 0.5:
            raise RuntimeError(
                f"{name} rows={rows}: max |err| {err}, collective did not run"
            )
    return errors


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    comm = PyNcclCommunicator(group=dist.group.WORLD, device=local_rank)
    # Some sglang versions build the communicator disabled and enable it only
    # inside change_state(); a disabled comm turns every call into a no-op.
    assert comm.available, "NCCL library not loaded"
    comm.disabled = False

    results: List[List[str]] = []
    for rows in args.rows_per_rank:
        generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
        local = torch.randn(
            rows, args.hidden, dtype=torch.bfloat16, device="cuda", generator=generator
        )
        full = torch.randn(
            world * rows,
            args.hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        inputs = {"all_gather": local}
        gathered = _make_op("all_gather", comm, world, rows, args.hidden)(local)
        torch.cuda.synchronize()
        if not torch.equal(gathered[rank * rows : (rank + 1) * rows], local):
            raise RuntimeError(f"all_gather rows={rows}: own rows not returned")
        timings = {}
        for name in OPS:
            op = _make_op(name, comm, world, rows, args.hidden)
            inp = inputs.get(name, full)
            timings[name] = (
                _time_eager_us(op, inp, args.iters),
                _time_graph_us(op, inp, args.graph_calls, replays=20),
            )
        errors = _max_abs_error_vs_fp32(comm, full, rows, rank)
        payload_kib = world * rows * args.hidden * 2 / 1024
        grouped_graph = timings["reduce_scatter_grouped"][1]
        single_graph = timings["reduce_scatter_single"][1]
        results.append(
            [
                str(rows),
                f"{payload_kib:.0f}",
                *(f"{timings[n][0]:.1f} / {timings[n][1]:.1f}" for n in OPS),
                f"{grouped_graph / single_graph:.2f}x",
                f"{errors['reduce_scatter_grouped']:.3g} / "
                f"{errors['reduce_scatter_single']:.3g}",
            ]
        )

    if rank == 0:
        header = [
            "rows/rank",
            "buffer KiB",
            "AllGather us (eager / graph)",
            "RS grouped ncclReduce us",
            "RS single ncclReduceScatter us",
            "grouped / single (graph)",
            "max |err| vs fp32 (grouped / single)",
        ]
        print(f"world={world} hidden={args.hidden} dtype=bf16")
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in results:
            print("| " + " | ".join(row) + " |")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
