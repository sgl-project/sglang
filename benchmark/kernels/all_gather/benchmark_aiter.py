"""
Benchmark baseline NCCL all-gather vs Aiter custom all-gather across message sizes.

Aiter exposes ``CustomAllreduce.custom_all_gather(inp, dim=0)`` (see
``aiter/aiter/dist/device_communicators/custom_all_reduce.py``), which dispatches
to ``all_gather_reg`` / ``all_gather_unreg`` based on whether the current stream
is capturing. This script measures per-rank time vs. the default
``torch.distributed.all_gather_into_tensor`` (RCCL) path for apples-to-apples
comparison, mirroring the structure of the all-reduce benchmark.

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
        description="Benchmark NCCL vs Aiter custom all-gather across message sizes."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu:gloo,cuda:nccl",
        help=(
            "Process group backend. Default routes CPU collectives through gloo "
            "and CUDA collectives through NCCL (RCCL on ROCm), giving a real "
            "RCCL baseline for all-gather while still letting Aiter's "
            "CustomAllreduce use the gloo sub-PG it requires."
        ),
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
        help="Benchmark iterations for sizes <= 1MB (per-rank input bytes).",
    )
    parser.add_argument(
        "--iters-large",
        type=int,
        default=20,
        help="Benchmark iterations for sizes > 1MB (per-rank input bytes).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Element dtype (2 bytes).",
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
    """Per-rank input sizes in bytes.

    For all-gather, total on-the-wire traffic per step is roughly
    ``world_size * size`` bytes (each rank receives world_size-1 shards).
    """
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
def nccl_all_gather(
    inp: torch.Tensor, out: torch.Tensor, group: Optional[dist.ProcessGroup]
) -> torch.Tensor:
    dist.all_gather_into_tensor(out, inp, group=group)
    return out


@torch.inference_mode()
def aiter_all_gather(comm, inp: torch.Tensor) -> Optional[torch.Tensor]:
    """Call Aiter's custom all-gather along dim=0 (out is allocated internally)."""
    if hasattr(comm, "custom_all_gather"):
        return comm.custom_all_gather(inp, dim=0)
    raise RuntimeError("Aiter communicator does not expose custom_all_gather.")


def _should_custom_ag(comm, inp: torch.Tensor) -> bool:
    """Replicate Aiter's ``should_custom_ag`` gate without importing the
    private attribute if missing; fall back to ``True`` and let the kernel
    raise (we still skip if the comm is disabled)."""
    if comm is None or getattr(comm, "disabled", True):
        return False
    fn = getattr(comm, "should_custom_ag", None)
    if fn is None:
        return True
    return bool(fn(inp))


@torch.inference_mode()
def bench_nccl(
    sizes: List[int],
    device: torch.device,
    dtype: torch.dtype,
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
        elems = size_bytes // torch.tensor([], dtype=dtype).element_size()
        inp = torch.empty(elems, dtype=dtype, device=device)
        inp.uniform_(0, 1) if dtype != torch.bfloat16 else inp.fill_(1.0)
        out = torch.empty(elems * world_size, dtype=dtype, device=device)

        dist.barrier(group=pg)
        for _ in range(warmup):
            nccl_all_gather(inp, out, pg)
        torch.cuda.synchronize()
        dist.barrier(group=pg)

        num_iters = iters_small if size_bytes <= (1 * 1024 * 1024) else iters_large
        times_ms: List[float] = []
        for it in range(num_iters):
            dist.barrier(group=pg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nccl_all_gather(inp, out, pg)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dist.barrier(group=pg)
            dt_ms = (t1 - t0) * 1000.0
            times_ms.append(dt_ms)
            if verbose and rank == 0:
                print(
                    f"[NCCL] size={human_size(size_bytes)} iter={it} time={dt_ms:.3f} ms"
                )

        avg_ms_local = sum(times_ms) / len(times_ms)
        avg_tensor = torch.tensor([avg_ms_local], dtype=torch.float64, device=device)
        gather_list = [torch.zeros_like(avg_tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, avg_tensor, group=pg)
        if rank == 0:
            avg_ms = float(torch.stack(gather_list).mean().item())
            print(
                f"[NCCL] {human_size(size_bytes)}: {avg_ms:.3f} ms (avg across ranks)"
            )
            results.append((size_bytes, avg_ms))
        else:
            results.append((size_bytes, None))

    return results


@torch.inference_mode()
def bench_aiter(
    comm,
    sizes: List[int],
    device: torch.device,
    dtype: torch.dtype,
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
        elems = size_bytes // torch.tensor([], dtype=dtype).element_size()
        inp = torch.empty(elems, dtype=dtype, device=device)
        inp.uniform_(0, 1) if dtype != torch.bfloat16 else inp.fill_(1.0)

        # Aiter's custom all-gather requires 16B alignment of the per-rank
        # byte size AND the per-rank input to fit in max_size / (world_size*2).
        if not _should_custom_ag(comm, inp):
            if rank == 0:
                print(
                    f"[Aiter] {human_size(size_bytes)}: skipped (should_custom_ag=False)"
                )
            results.append((size_bytes, None))
            continue

        disabled = False
        dist.barrier(group=pg)
        for _ in range(warmup):
            torch.cuda.synchronize()
            out = aiter_all_gather(comm, inp)
            torch.cuda.synchronize()
            if out is None:
                disabled = True
                break
        dist.barrier(group=pg)

        if disabled:
            if rank == 0:
                print(f"[Aiter] {human_size(size_bytes)}: custom AG disabled (skipped)")
            results.append((size_bytes, None))
            continue

        # One-shot correctness check vs. NCCL reference on this size.
        ref = torch.empty(elems * world_size, dtype=dtype, device=device)
        dist.all_gather_into_tensor(ref, inp, group=pg)
        aiter_out = aiter_all_gather(comm, inp)
        if aiter_out is None or aiter_out.shape != ref.shape:
            if rank == 0:
                print(
                    f"[Aiter] {human_size(size_bytes)}: correctness skipped "
                    f"(out={None if aiter_out is None else tuple(aiter_out.shape)} "
                    f"ref={tuple(ref.shape)})"
                )
        else:
            mismatch = not torch.equal(aiter_out, ref)
            if mismatch and rank == 0:
                print(
                    f"[Aiter] {human_size(size_bytes)}: WARNING output differs from NCCL reference"
                )

        num_iters = iters_small if size_bytes <= (1 * 1024 * 1024) else iters_large
        times_ms: List[float] = []
        for it in range(num_iters):
            dist.barrier(group=pg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = aiter_all_gather(comm, inp)
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
                    f"[Aiter] size={human_size(size_bytes)} iter={it} time={dt_ms:.3f} ms"
                )

        if disabled or not times_ms:
            if rank == 0:
                print(
                    f"[Aiter] {human_size(size_bytes)}: custom AG disabled (no timings)"
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
                f"[Aiter] {human_size(size_bytes)}: {avg_ms:.3f} ms (avg across ranks)"
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
            "Aiter custom AG may disable itself.",
            file=sys.stderr,
        )

    init_dist(args.backend)
    device = get_device(local_rank)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    aiter_comm = None
    HAVE_AITER = False
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
        print(f"Device: {device.type}:{device.index} dtype={dtype}")
        print(f"Aiter available: {HAVE_AITER}")

    pg = dist.group.WORLD
    sizes = get_message_sizes()
    max_size = max(sizes) if sizes else (64 * 1024 * 1024)

    # Aiter's CustomAllreduce refuses an NCCL-backed PG (it needs a pure TCP
    # store for IPC handle exchange). SGLang creates a separate gloo group for
    # the same reason; replicate that here.
    ranks = list(range(world_size))
    gloo_group = dist.new_group(ranks=ranks, backend="gloo")

    if HAVE_AITER:
        try:
            aiter_comm = AiterCustomAllreduce(
                group=gloo_group, device=device, max_size=max_size
            )
        except Exception as e:
            if rank == 0:
                print(
                    f"Failed to construct Aiter CustomAllreduce: {e}", file=sys.stderr
                )
            aiter_comm = None

    nccl_results = bench_nccl(
        sizes=sizes,
        device=device,
        dtype=dtype,
        warmup=args.warmup,
        iters_small=args.iters_small,
        iters_large=args.iters_large,
        verbose=args.verbose,
        pg=pg,
    )

    aiter_results: List[Tuple[int, Optional[float]]] = []
    if aiter_comm is not None:
        aiter_results = bench_aiter(
            comm=aiter_comm,
            sizes=sizes,
            device=device,
            dtype=dtype,
            warmup=args.warmup,
            iters_small=args.iters_small,
            iters_large=args.iters_large,
            verbose=args.verbose,
            pg=pg,
        )

    if aiter_comm is not None and hasattr(aiter_comm, "close"):
        try:
            aiter_comm.close()
        except Exception:
            pass

    if dist.get_rank() == 0:
        print("\nResults (avg ms across ranks; None = disabled/unavailable):")
        header = f"{'Size':>8}  {'NCCL(ms)':>11}  {'Aiter(ms)':>11}  {'Speedup':>8}"
        print(header)
        print("-" * len(header))

        nccl_map = {s: v for s, v in nccl_results if v is not None}
        aiter_map = {s: v for s, v in aiter_results if v is not None}

        for s in sizes:
            nccl_ms = nccl_map.get(s, None)
            aiter_ms = aiter_map.get(s, None)
            speedup = (
                f"{nccl_ms / aiter_ms:.2f}x"
                if (nccl_ms is not None and aiter_ms is not None and aiter_ms > 0)
                else "-"
            )
            print(
                f"{human_size(s):>8}  "
                f"{('%.3f' % nccl_ms) if nccl_ms is not None else 'None':>11}  "
                f"{('%.3f' % aiter_ms) if aiter_ms is not None else 'None':>11}  "
                f"{speedup:>8}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
