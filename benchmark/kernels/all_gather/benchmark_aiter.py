"""
Benchmark SGLang logical TP all-gather against Aiter custom all-gather.

This benchmark is intended for captured logits all-gather shapes such as
``1,32320;2,32320;4,32320`` and for correctness coverage across metadata
integer dtypes. It compares the current RCCL ``dist.all_gather_into_tensor``
route with Aiter's custom all-gather when RCCL supports the dtype, validates
candidate correctness against deterministic expected outputs, and reports
per-rank average latency.

Usage:
    torchrun --nproc_per_node=4 benchmark/kernels/all_gather/benchmark_aiter.py \
      --dtype bfloat16 --shapes "1,32320;2,32320;4,32320"
"""

from __future__ import annotations

import argparse
import os
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


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "uint64_t": torch.uint64,
    "u64": torch.uint64,
    "int64_t": torch.int64,
    "i64": torch.int64,
    "uint32_t": torch.uint32,
    "u32": torch.uint32,
    "int32_t": torch.int32,
    "i32": torch.int32,
    "int16_t": torch.int16,
    "i16": torch.int16,
    "uint8_t": torch.uint8,
    "u8": torch.uint8,
    "int8_t": torch.int8,
    "i8": torch.int8,
}


def parse_dtype_list(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one dtype is required")
    unknown = [name for name in names if name not in DTYPE_MAP]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown dtype(s): {unknown}; choices={sorted(DTYPE_MAP)}"
        )
    return names


def parse_dim_list(value: str) -> list[int]:
    dims = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not dims:
        raise argparse.ArgumentTypeError("at least one dim is required")
    return dims


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
        type=parse_dtype_list,
        default=parse_dtype_list("bfloat16"),
        help="Input dtype or comma-separated dtypes.",
    )
    parser.add_argument(
        "--dims",
        type=parse_dim_list,
        default=parse_dim_list("-1"),
        help='Comma-separated logical gather dims, e.g. "-1" or "0,-1".',
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run correctness checks without latency timing.",
    )
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
    return DTYPE_MAP[name]


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
    numel = int(torch.tensor(shape).prod().item())
    x = torch.arange(rank * (numel + 17), rank * (numel + 17) + numel, device=device)
    if dtype == torch.uint64:
        return x.reshape(shape).to(torch.uint64)
    if dtype == torch.uint32:
        return (x % (2**31)).reshape(shape).to(torch.uint32)
    if dtype == torch.int16:
        return (x % (2**14)).reshape(shape).to(torch.int16)
    if dtype == torch.uint8:
        return (x % (2**8)).reshape(shape).to(torch.uint8)
    if dtype == torch.int8:
        return (x % (2**7)).reshape(shape).to(torch.int8)
    return x.reshape(shape).to(dtype)


def expected_logical_all_gather(
    input_shape: Shape,
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
    dim: int,
) -> torch.Tensor:
    parts = [make_input(input_shape, dtype, device, rank) for rank in range(world_size)]
    if dim < 0:
        dim += len(input_shape)
    return torch.cat(parts, dim=dim)


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


def install_aiter_aot_import_shims() -> None:
    """Let this standalone test import only the AITER pieces it needs.

    Some dev environments have optional top-level AITER deps (for example
    FlyDSL) that are unrelated to custom all-gather. `AITER_AOT_IMPORT=1`
    avoids importing those modules; these shims provide the attributes that
    AITER's distributed helpers expect from the top-level package.
    """
    if os.getenv("AITER_AOT_IMPORT") != "1":
        return

    import aiter
    from aiter.jit.utils.torch_guard import torch_compile_guard
    from aiter.ops import custom_all_reduce
    from aiter.ops.quant import get_hip_quant

    aiter.torch_compile_guard = torch_compile_guard
    aiter.get_hip_quant = get_hip_quant
    for name in dir(custom_all_reduce):
        if not name.startswith("_"):
            setattr(aiter, name, getattr(custom_all_reduce, name))


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
    expected: torch.Tensor,
    pg: dist.ProcessGroup | None,
    dim: int = -1,
):
    ref = expected
    out = aiter_logical_all_gather(comm, inp, aiter_raw, dim=dim)
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
    pg = dist.group.WORLD

    install_aiter_aot_import_shims()
    from aiter.dist.device_communicators.custom_all_reduce import (
        CustomAllreduce as AiterCustomAllreduce,
    )

    gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    dtypes = [dtype_from_name(name) for name in args.dtype]
    max_input_bytes = max(
        int(torch.tensor(shape).prod().item())
        * torch.tensor([], dtype=dtype).element_size()
        for shape in args.shapes
        for dtype in dtypes
    )
    pool_size = max(args.max_size_bytes, max_input_bytes * world_size * 2)
    comm = AiterCustomAllreduce(group=gloo_group, device=device, max_size=pool_size)

    rows: list[dict[str, object]] = []
    for dtype_name, dtype in zip(args.dtype, dtypes):
        for shape in args.shapes:
            for dim in args.dims:
                inp = make_input(shape, dtype, device, rank).contiguous()
                input_bytes = inp.numel() * inp.element_size()
                raw_shape = raw_allgather_shape(shape, world_size)
                rccl_raw = torch.empty(raw_shape, dtype=dtype, device=device)
                aiter_raw = torch.empty_like(rccl_raw)
                expected = expected_logical_all_gather(
                    shape, dtype, device, world_size, dim
                )

                can_aiter = bool(comm.should_custom_ag(inp))
                if not can_aiter:
                    if rank == 0:
                        print(
                            f"SKIP dtype={dtype_name} shape={shape} dim={dim}: "
                            "Aiter should_custom_ag=False"
                        )
                    rows.append(
                        {
                            "dtype": dtype_name,
                            "shape": shape,
                            "dim": dim,
                            "input_bytes": input_bytes,
                            "correct": False,
                            "rccl_us": None,
                            "aiter_us": None,
                            "speedup": None,
                        }
                    )
                    continue

                check_correctness(comm, inp, rccl_raw, aiter_raw, expected, pg, dim=dim)
                correct_flag = sync_max(0.0, device, pg) == 0.0

                rccl_median_us = rccl_mean_us = None
                aiter_median_us = aiter_mean_us = None
                speedup = None
                if not args.correctness_only:
                    dist.barrier(group=pg)
                    try:
                        rccl_median_us, rccl_mean_us = time_us(
                            lambda: rccl_logical_all_gather(inp, rccl_raw, pg, dim=dim),
                            args.warmup,
                            args.iters,
                        )
                    except Exception:
                        rccl_median_us = rccl_mean_us = None
                    dist.barrier(group=pg)
                    aiter_median_us, aiter_mean_us = time_us(
                        lambda: aiter_logical_all_gather(comm, inp, aiter_raw, dim=dim),
                        args.warmup,
                        args.iters,
                    )
                    dist.barrier(group=pg)

                    if rccl_median_us is not None:
                        rccl_median_us = sync_avg(rccl_median_us, device, pg)
                        rccl_mean_us = sync_avg(rccl_mean_us, device, pg)
                    aiter_median_us = sync_avg(aiter_median_us, device, pg)
                    aiter_mean_us = sync_avg(aiter_mean_us, device, pg)
                    speedup = (
                        rccl_median_us / aiter_median_us
                        if rccl_median_us is not None and aiter_median_us > 0
                        else None
                    )

                rows.append(
                    {
                        "dtype": dtype_name,
                        "shape": shape,
                        "dim": dim,
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
                        f"[rank {rank}] dtype={dtype_name} shape={shape} dim={dim} "
                        f"rccl_median_us={fmt_optional_us(rccl_median_us)} "
                        f"aiter_median_us={fmt_optional_us(aiter_median_us)}"
                    )

    if hasattr(comm, "close"):
        comm.close()

    if rank == 0:
        print("\nResults (logical all-gather, avg median us across ranks)")
        header = (
            f"{'DType':>10}  {'Shape':>14}  {'Dim':>4}  "
            f"{'Input Bytes':>12}  {'Correct':>7}  "
            f"{'RCCL us':>10}  {'Aiter us':>10}  {'Speedup':>8}"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            rccl = row["rccl_us"]
            aiter = row["aiter_us"]
            speedup = row["speedup"]
            print(
                f"{str(row['dtype']):>10}  "
                f"{str(row['shape']):>14}  "
                f"{row['dim']:>4}  "
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
