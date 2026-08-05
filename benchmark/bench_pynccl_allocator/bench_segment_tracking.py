"""
Benchmark for comparing CPU overhead of segment tracking methods:
1. nccl_allocator_register_segments_with_comm() - C++ registration with index tracking
2. torch.cuda.memory.memory_snapshot() - PyTorch memory snapshot

Usage:
    python benchmark/bench_pynccl_allocator/bench_segment_tracking.py --num-segments 50 --num-iters 1000
"""

import argparse
import time
import warnings
from typing import List

import torch

warnings.filterwarnings("ignore")


def setup_segments(num_segments: int, segment_size: int = 1024 * 1024):
    """
    Allocate a specified number of segments using the NCCL allocator.
    """
    import os

    import torch.distributed as dist

    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        get_nccl_mem_pool,
    )

    # Initialize distributed if not already done
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="nccl",
            rank=0,
            world_size=1,
            device_id=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )

    mem_pool = get_nccl_mem_pool()

    # Allocate segments in the pool
    tensors: List[torch.Tensor] = []
    with torch.cuda.use_mem_pool(mem_pool):
        for _ in range(num_segments):
            t = torch.empty(segment_size, dtype=torch.uint8, device="cuda")
            tensors.append(t)

    # Keep tensors alive by returning them (caller should hold reference)
    return tensors, mem_pool


def bench_register_segments_with_comm(
    nccl_lib, comm_ptr: int, num_iters: int = 10000
) -> float:
    """
    Benchmark nccl_allocator_register_segments_with_comm() function.

    Args:
        nccl_lib: The loaded NCCL allocator library
        comm_ptr: The communicator pointer value
        num_iters: Number of iterations

    Returns:
        Average time per call in microseconds.
    """
    import ctypes

    # Setup the C function signature
    register_func = nccl_lib.nccl_allocator_register_segments_with_comm
    register_func.restype = ctypes.c_int
    register_func.argtypes = [ctypes.c_uint64]

    # Warmup
    for _ in range(100):
        register_func(comm_ptr)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        register_func(comm_ptr)
    end = time.perf_counter()

    avg_us = (end - start) / num_iters * 1e6
    return avg_us


def bench_mempool_snapshot(
    mem_pool: torch.cuda.MemPool, num_iters: int = 10000
) -> float:
    """
    Benchmark torch.cuda.MemPool.snapshot() function.

    Returns:
        Average time per call in microseconds.
    """
    # Warmup
    for _ in range(100):
        mem_pool.snapshot()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        mem_pool.snapshot()
    end = time.perf_counter()

    avg_us = (end - start) / num_iters * 1e6
    return avg_us


def bench_with_various_segment_counts(
    segment_counts: List[int],
    num_iters: int = 10000,
    segment_size: int = 1024 * 1024,  # 1MB per segment
):
    """
    Run benchmarks with various numbers of tracked segments.
    """
    print("=" * 80)
    print("Benchmark: Segment Registration CPU Overhead")
    print("=" * 80)
    print(f"Segment size: {segment_size / 1024 / 1024:.2f} MB")
    print(f"Iterations per measurement: {num_iters}")
    print()
    print(
        f"{'Segments':<12} {'register_segments (µs)':<30} {'snapshot (µs)':<20} {'Speedup':<10}"
    )
    print("-" * 80)

    all_tensors = []  # Keep all tensors alive
    comm_ptr = 0  # Use dummy comm_ptr for benchmarking (no actual NCCL registration)

    for num_segments in segment_counts:
        # Clean up previous segments
        all_tensors = []

        # Allocate segments (this initializes _nccl_allocator_lib via get_nccl_mem_pool)
        tensors, mem_pool = setup_segments(num_segments, segment_size)
        all_tensors.extend(tensors)

        # Sync to ensure allocations are complete
        torch.cuda.synchronize()

        # Import _nccl_allocator_lib after setup_segments (ensures library is loaded)
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            _nccl_allocator_lib,
        )

        # Run benchmarks
        time_register = bench_register_segments_with_comm(
            _nccl_allocator_lib, comm_ptr, num_iters
        )
        time_snapshot = bench_mempool_snapshot(mem_pool, num_iters)

        speedup = time_snapshot / time_register if time_register > 0 else float("inf")

        print(
            f"{num_segments:<12} {time_register:<30.3f} {time_snapshot:<20.3f} {speedup:<10.2f}x"
        )

    print("-" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark segment tracking methods in pynccl_allocator"
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        nargs="+",
        default=[10, 50, 100, 200, 500, 1000],
        help="Number of segments to track (can specify multiple values)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10000,
        help="Number of iterations for each measurement",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=1024 * 1024,  # 1MB
        help="Size of each segment in bytes",
    )
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a GPU.")
        return

    # Initialize CUDA context by creating a small tensor
    _ = torch.zeros(1, device="cuda")

    # Run benchmarks
    bench_with_various_segment_counts(
        segment_counts=args.num_segments,
        num_iters=args.num_iters,
        segment_size=args.segment_size,
    )


if __name__ == "__main__":
    main()
