"""
Benchmark SGLang vs Aiter custom all-reduce across message sizes.
Usage:
    torchrun --nproc_per_node=2 benchmark_aiter.py
    torchrun --nproc_per_node=4 benchmark_aiter.py
    torchrun --nproc_per_node=8 benchmark_aiter.py
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang vs Aiter custom all-reduce across message sizes."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="Process group backend for the custom-AR control path (must NOT be nccl).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per size per implementation.",
    )
    parser.add_argument(
        "--iters-small",
        type=int,
        default=50,
        help="Benchmark iterations for sizes <= 1MB.",
    )
    parser.add_argument(
        "--iters-large",
        type=int,
        default=20,
        help="Benchmark iterations for sizes > 1MB.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-iteration timings on rank 0 for debugging.",
    )
    return parser.parse_args()


def get_env_rank_world() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def init_dist(backend: str):
    rank, world_size, _ = get_env_rank_world()
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )


def get_device(local_rank: int) -> torch.device:
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def human_size(num_bytes: int) -> str:
    units = [("B", 1), ("K", 1024), ("M", 1024 * 1024), ("G", 1024 * 1024 * 1024)]
    for suf, base in reversed(units):
        if num_bytes % base == 0 and num_bytes >= base:
            val = num_bytes // base
            return f"{val}{suf}"
    return f"{num_bytes}B"


def get_message_sizes() -> List[int]:
    return [
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1 * 1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024,
    ]


@torch.inference_mode()
def run_once(comm, inp: torch.Tensor) -> Optional[torch.Tensor]:
    if hasattr(comm, "all_reduce_unreg"):
        return comm.all_reduce_unreg(inp)
    if hasattr(comm, "custom_all_reduce"):
        return comm.custom_all_reduce(inp)
    raise RuntimeError("No known all-reduce method found on the communicator.")


@torch.inference_mode()
def bench_impl(
    name: str,
    comm,
    sizes: List[int],
    device: torch.device,
    warmup: int,
    iters_small: int,
    iters_large: int,
    verbose: bool,
    pg: Optional[dist.ProcessGroup] = None,
) -> List[Tuple[int, Optional[float]]]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    results: List[Tuple[int, Optional[float]]] = []

    for size_bytes in sizes:
        elems = size_bytes // 2  # float16: 2 bytes per element
        inp = torch.empty(elems, dtype=torch.float16, device=device)
        inp.uniform_(0, 1)

        disabled = False
        dist.barrier(group=pg)
        for _ in range(warmup):
            torch.cuda.synchronize()
            out = run_once(comm, inp)
            torch.cuda.synchronize()
            if out is None:
                disabled = True
                break
        dist.barrier(group=pg)

        if disabled:
            if rank == 0:
                print(
                    f"[{name}] {human_size(size_bytes)}: custom AR disabled (skipped)"
                )
            results.append((size_bytes, None))
            continue

        num_iters = iters_small if size_bytes <= (1 * 1024 * 1024) else iters_large

        times_ms: List[float] = []
        for it in range(num_iters):
            dist.barrier(group=pg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = run_once(comm, inp)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dist.barrier(group=pg)

            if out is None:
                disabled = True
                break

            dt_ms = (t1 - t0) * 1000.0
            times_ms.append(dt_ms)

            if verbose and rank == 0:
                print(
                    f"[{name}] size={human_size(size_bytes)} iter={it} time={dt_ms:.3f} ms"
                )

        if disabled or not times_ms:
            if rank == 0:
                print(
                    f"[{name}] {human_size(size_bytes)}: custom AR disabled (no timings)"
                )
            results.append((size_bytes, None))
            continue

        avg_ms_local = sum(times_ms) / len(times_ms)
        avg_tensor = torch.tensor([avg_ms_local], dtype=torch.float64, device=device)
        gather_list = [torch.zeros_like(avg_tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, avg_tensor, group=pg)
        if rank == 0:
            avg_ms = float(torch.stack(gather_list).mean().item())
            print(
                f"[{name}] {human_size(size_bytes)}: {avg_ms:.3f} ms (avg across ranks)"
            )
            results.append((size_bytes, avg_ms))
        else:
            results.append((size_bytes, None))

    return results


def main():
    args = parse_args()
    rank, world_size, local_rank = get_env_rank_world()

    if world_size not in (2, 4, 6, 8):
        print(
            f"[rank {rank}] WARNING: world_size={world_size} not in supported set (2,4,6,8). "
            "Custom AR may disable itself.",
            file=sys.stderr,
        )

    init_dist(args.backend)
    device = get_device(local_rank)

    # Import after dist init; some libs query torch dist state on import
    sgl_comm = None
    aiter_comm = None
    HAVE_SGLANG = False
    HAVE_AITER = False

    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce as SGLCustomAllreduce,
        )

        HAVE_SGLANG = True
    except Exception as e:
        if rank == 0:
            print(f"SGLang CustomAllreduce import failed: {e}", file=sys.stderr)

    try:
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as AiterCustomAllreduce,
        )

        HAVE_AITER = True
    except Exception as e:
        if rank == 0:
            print(f"Aiter CustomAllreduce import failed: {e}", file=sys.stderr)

    if rank == 0:
        print(f"Initialized PG backend={args.backend} world_size={world_size}")
        print(f"Device: {device.type}:{device.index}")
        print(f"SGLang available: {HAVE_SGLANG}, Aiter available: {HAVE_AITER}")

    pg = dist.group.WORLD
    sizes = get_message_sizes()
    max_size = max(sizes) if sizes else (64 * 1024 * 1024)

    if HAVE_SGLANG:
        try:
            sgl_comm = SGLCustomAllreduce(group=pg, device=device, max_size=max_size)
        except Exception as e:
            if rank == 0:
                print(
                    f"Failed to construct SGLang CustomAllreduce: {e}", file=sys.stderr
                )
            sgl_comm = None

    if HAVE_AITER:
        try:
            aiter_comm = AiterCustomAllreduce(
                group=pg, device=device, max_size=max_size
            )
        except Exception as e:
            if rank == 0:
                print(
                    f"Failed to construct Aiter CustomAllreduce: {e}", file=sys.stderr
                )
            aiter_comm = None

    sgl_results: List[Tuple[int, Optional[float]]] = []
    aiter_results: List[Tuple[int, Optional[float]]] = []

    if sgl_comm is not None:
        sgl_results = bench_impl(
            name="SGLang",
            comm=sgl_comm,
            sizes=sizes,
            device=device,
            warmup=args.warmup,
            iters_small=args.iters_small,
            iters_large=args.iters_large,
            verbose=args.verbose,
            pg=pg,
        )

    if aiter_comm is not None:
        aiter_results = bench_impl(
            name="Aiter",
            comm=aiter_comm,
            sizes=sizes,
            device=device,
            warmup=args.warmup,
            iters_small=args.iters_small,
            iters_large=args.iters_large,
            verbose=args.verbose,
            pg=pg,
        )

    for comm in (sgl_comm, aiter_comm):
        if comm is not None and hasattr(comm, "close"):
            try:
                comm.close()
            except Exception:
                pass

    if dist.get_rank() == 0:
        print("\nResults (avg ms across ranks; None = disabled/unavailable):")
        header = f"{'Size':>8}  {'SGLang(ms)':>12}  {'Aiter(ms)':>11}"
        print(header)
        print("-" * len(header))

        sgl_map = {s: v for s, v in sgl_results if v is not None}
        aiter_map = {s: v for s, v in aiter_results if v is not None}

        for s in sizes:
            sgl_ms = sgl_map.get(s, None)
            aiter_ms = aiter_map.get(s, None)
            print(
                f"{human_size(s):>8}  {('%.3f' % sgl_ms) if sgl_ms is not None else 'None':>12}  "
                f"{('%.3f' % aiter_ms) if aiter_ms is not None else 'None':>11}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
