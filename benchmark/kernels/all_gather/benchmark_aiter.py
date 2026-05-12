"""
Benchmark SGLang logical TP all-gather against Aiter custom all-gather.

This benchmark is intended for captured logits all-gather shapes such as
``1,32320;2,32320;4,32320``. It compares the current RCCL
``dist.all_gather_into_tensor`` route with Aiter's custom all-gather, validates
candidate correctness before timing, and reports per-rank average latency.

Usage:
    torchrun --nproc_per_node=4 benchmark/kernels/all_gather/benchmark_aiter.py \
      --dtype bfloat16 --shapes "1,32320;2,32320;4,32320"
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.distributed as dist

Shape = tuple[int, ...]


def parse_shape_list(value: str) -> list[Shape]:
    shapes: list[Shape] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        shape = tuple(int(dim.strip()) for dim in item.split(",") if dim.strip())
        if not shape or any(dim <= 0 for dim in shape):
            raise argparse.ArgumentTypeError(f"invalid shape: {item!r}")
        shapes.append(shape)
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RCCL vs Aiter custom all-gather for explicit shapes."
    )
    parser.add_argument(
        "--backend",
        default="cpu:gloo,cuda:nccl",
        help="Process group backend for torch.distributed.",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shape_list,
        default=parse_shape_list("1,32320;2,32320;4,32320"),
        help='Semicolon-separated input shapes, e.g. "1,32320;2,32320;4,32320".',
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Input dtype.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--max-size-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Aiter CustomAllreduce IPC pool size.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-rank diagnostic details.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def logical_output_shape(input_shape: Shape, world_size: int, dim: int) -> Shape:
    if dim < 0:
        dim += len(input_shape)
    return input_shape[:dim] + (input_shape[dim] * world_size,) + input_shape[dim + 1 :]


def raw_allgather_shape(input_shape: Shape, world_size: int) -> Shape:
    return (input_shape[0] * world_size,) + input_shape[1:]


def reshape_logical(raw: torch.Tensor, input_shape: Shape, world_size: int, dim: int):
    if dim < 0:
        dim += len(input_shape)
    return (
        raw.reshape((world_size,) + input_shape)
        .movedim(0, dim)
        .reshape(logical_output_shape(input_shape, world_size, dim))
    )


def make_input(shape: Shape, dtype: torch.dtype, device: torch.device, rank: int):
    # Distinct per-rank values make rank-order errors visible in correctness.
    x = torch.arange(
        rank * 1000,
        rank * 1000 + int(torch.tensor(shape).prod().item()),
        device=device,
        dtype=torch.float32,
    )
    return x.reshape(shape).to(dtype)


@torch.inference_mode()
def rccl_logical_all_gather(
    inp: torch.Tensor,
    raw_out: torch.Tensor,
    pg: dist.ProcessGroup,
    dim: int = -1,
):
    dist.all_gather_into_tensor(raw_out, inp, group=pg)
    return reshape_logical(raw_out, tuple(inp.shape), dist.get_world_size(pg), dim)


@torch.inference_mode()
def aiter_logical_all_gather(
    comm,
    inp: torch.Tensor,
    raw_out: torch.Tensor,
    dim: int = -1,
):
    # SGLang's patched path writes Aiter output into the same preallocated raw
    # buffer used by all_gather_into_tensor, then applies the standard reshape.
    comm.all_gather_unreg(inp, out=raw_out, dim=0)
    return reshape_logical(raw_out, tuple(inp.shape), comm.world_size, dim)


def sync_avg(value: float, device: torch.device, pg: dist.ProcessGroup) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=pg)
    return float(tensor.item())


def sync_max(value: float, device: torch.device, pg: dist.ProcessGroup) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=pg)
    return float(tensor.item())


def fmt_optional_us(value: object) -> str:
    if value is None:
        return "None"
    return f"{float(value):.2f}"


def time_us(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times), statistics.mean(times)


def check_correctness(
    comm,
    inp: torch.Tensor,
    rccl_raw: torch.Tensor,
    aiter_raw: torch.Tensor,
    pg: dist.ProcessGroup,
):
    ref = rccl_logical_all_gather(inp, rccl_raw, pg)
    out = aiter_logical_all_gather(comm, inp, aiter_raw)
    if ref.shape != out.shape:
        raise AssertionError(
            f"shape mismatch: ref={tuple(ref.shape)} out={tuple(out.shape)}"
        )
    if not torch.equal(ref, out):
        max_abs = (ref.float() - out.float()).abs().max().item()
        raise AssertionError(f"Aiter output mismatch, max_abs={max_abs}")


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend=args.backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(torch.cuda.device_count() > 0 and rank % torch.cuda.device_count())
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = dtype_from_name(args.dtype)
    pg = dist.group.WORLD

    from aiter.dist.device_communicators.custom_all_reduce import (
        CustomAllreduce as AiterCustomAllreduce,
    )

    gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    max_input_bytes = max(
        int(torch.tensor(shape).prod().item())
        * torch.tensor([], dtype=dtype).element_size()
        for shape in args.shapes
    )
    pool_size = max(args.max_size_bytes, max_input_bytes * world_size * 2)
    comm = AiterCustomAllreduce(group=gloo_group, device=device, max_size=pool_size)

    rows: list[dict[str, object]] = []
    for shape in args.shapes:
        inp = make_input(shape, dtype, device, rank).contiguous()
        input_bytes = inp.numel() * inp.element_size()
        raw_shape = raw_allgather_shape(shape, world_size)
        rccl_raw = torch.empty(raw_shape, dtype=dtype, device=device)
        aiter_raw = torch.empty_like(rccl_raw)

        can_aiter = bool(comm.should_custom_ag(inp))
        if not can_aiter:
            if rank == 0:
                print(f"SKIP shape={shape}: Aiter should_custom_ag=False")
            rows.append(
                {
                    "shape": shape,
                    "input_bytes": input_bytes,
                    "correct": False,
                    "rccl_us": None,
                    "aiter_us": None,
                    "speedup": None,
                }
            )
            continue

        check_correctness(comm, inp, rccl_raw, aiter_raw, pg)
        correct_flag = sync_max(0.0, device, pg) == 0.0

        dist.barrier(group=pg)
        rccl_median_us, rccl_mean_us = time_us(
            lambda: rccl_logical_all_gather(inp, rccl_raw, pg),
            args.warmup,
            args.iters,
        )
        dist.barrier(group=pg)
        aiter_median_us, aiter_mean_us = time_us(
            lambda: aiter_logical_all_gather(comm, inp, aiter_raw),
            args.warmup,
            args.iters,
        )
        dist.barrier(group=pg)

        rccl_median_us = sync_avg(rccl_median_us, device, pg)
        rccl_mean_us = sync_avg(rccl_mean_us, device, pg)
        aiter_median_us = sync_avg(aiter_median_us, device, pg)
        aiter_mean_us = sync_avg(aiter_mean_us, device, pg)
        speedup = rccl_median_us / aiter_median_us if aiter_median_us > 0 else 0.0

        rows.append(
            {
                "shape": shape,
                "input_bytes": input_bytes,
                "correct": correct_flag,
                "rccl_us": rccl_median_us,
                "aiter_us": aiter_median_us,
                "rccl_mean_us": rccl_mean_us,
                "aiter_mean_us": aiter_mean_us,
                "speedup": speedup,
            }
        )
        if args.verbose:
            print(
                f"[rank {rank}] shape={shape} rccl_median_us={rccl_median_us:.2f} "
                f"aiter_median_us={aiter_median_us:.2f}"
            )

    if hasattr(comm, "close"):
        comm.close()

    if rank == 0:
        print("\nResults (logical dim=-1 all-gather, avg median us across ranks)")
        header = (
            f"{'Shape':>14}  {'Input Bytes':>12}  {'Correct':>7}  "
            f"{'RCCL us':>10}  {'Aiter us':>10}  {'Speedup':>8}"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            rccl = row["rccl_us"]
            aiter = row["aiter_us"]
            speedup = row["speedup"]
            print(
                f"{str(row['shape']):>14}  "
                f"{row['input_bytes']:>12}  "
                f"{str(row['correct']):>7}  "
                f"{fmt_optional_us(rccl):>10}  "
                f"{fmt_optional_us(aiter):>10}  "
                f"{speedup if speedup is not None else 0.0:>7.2f}x"
            )

    dist.barrier(group=pg)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
