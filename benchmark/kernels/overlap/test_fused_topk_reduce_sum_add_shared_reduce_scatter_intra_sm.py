"""
Test & Benchmark for Fused TopK Reduce-Sum + Add Shared + Reduce-Scatter Overlap Kernel
========================================================================================

Four test modes:
  1. correctness  — verify overlap kernel output matches reference (PyTorch + NCCL)
  2. performance  — benchmark compute-only, comm-only, non-overlap, and overlap
  3. multi_size   — correctness sweep across problem shapes and dtypes
  4. stability    — repeated correctness checks to catch intermittent hang/failures

Usage:
    # Correctness test (2 GPUs)
    torchrun --nproc_per_node=2 test_fused_topk_reduce_sum_add_shared_reduce_scatter_intra_sm.py --case correctness

    # Performance benchmark
    torchrun --nproc_per_node=2 test_fused_topk_reduce_sum_add_shared_reduce_scatter_intra_sm.py --case performance

    # Multi-size sweep
    torchrun --nproc_per_node=2 test_fused_topk_reduce_sum_add_shared_reduce_scatter_intra_sm.py --case multi_size

    # Stability (hang detection)
    torchrun --nproc_per_node=2 test_fused_topk_reduce_sum_add_shared_reduce_scatter_intra_sm.py --case stability
"""

import argparse
import os
import time
import torch
import torch.distributed as dist

from fused_topk_reduce_sum_add_shared_reduce_scatter_intra_sm import (
    TopkRsSOverlapContext,
    create_topk_rss_overlap_context,
    fused_topk_reduce_sum_add_shared_reduce_scatter,
)


# ---------------------------------------------------------------------------
# Distributed Initialization
# ---------------------------------------------------------------------------

def initialize_distributed():
    """Initialize distributed environment."""
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        device_id=torch.device(f"cuda:{LOCAL_RANK}"),
    )
    assert dist.is_initialized()

    tp_group = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    dist.barrier(tp_group)

    return RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, tp_group


# ---------------------------------------------------------------------------
# Precision Check
# ---------------------------------------------------------------------------

def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Cosine-similarity based diff metric. Threshold: < 0.001."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


# ---------------------------------------------------------------------------
# Reference Implementation
# ---------------------------------------------------------------------------

def reference_topk_rss(
    expert_outputs: torch.Tensor,
    shared_expert_output: torch.Tensor,
    pg,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    """
    Reference implementation using standard PyTorch ops + NCCL reduce-scatter.

    Args:
        expert_outputs: [M, TOPK, N] routed expert outputs
        shared_expert_output: [M, N] shared expert output
        pg: process group for reduce-scatter
        routed_scaling_factor: scalar applied after topk reduce-sum (default 1.0)

    Returns:
        [M_per_rank, N] reduced and scattered output
    """
    M, topk, N = expert_outputs.shape

    # topk reduce-sum, then apply routed_scaling_factor
    result = expert_outputs.sum(dim=1) * routed_scaling_factor

    # add shared expert output
    result = result + shared_expert_output

    # reduce-scatter
    world_size = dist.get_world_size(pg)
    M_per_rank = M // world_size
    output = torch.empty(
        (M_per_rank, N), dtype=result.dtype, device=result.device,
    )
    dist.reduce_scatter_tensor(output, result, op=dist.ReduceOp.SUM, group=pg)

    return output


def compute_only(
    expert_outputs: torch.Tensor,
    shared_expert_output: torch.Tensor,
) -> torch.Tensor:
    """Standalone compute: topk_reduce_sum + add shared expert output (no comm)."""
    result = expert_outputs.sum(dim=1)
    result = result * 2.5
    return result + shared_expert_output


def comm_only(tensor: torch.Tensor, output: torch.Tensor, pg) -> torch.Tensor:
    """Standalone communication: NCCL reduce-scatter only."""
    dist.reduce_scatter_tensor(output, tensor, op=dist.ReduceOp.SUM, group=pg)
    return output


def non_overlap(
    expert_outputs: torch.Tensor,
    shared_expert_output: torch.Tensor,
    output: torch.Tensor,
    pg,
) -> torch.Tensor:
    """Non-overlap: compute + comm sequentially (no overlap)."""
    result = compute_only(expert_outputs, shared_expert_output)
    dist.reduce_scatter_tensor(output, result, op=dist.ReduceOp.SUM, group=pg)
    return output


# ---------------------------------------------------------------------------
# Performance Measurement
# ---------------------------------------------------------------------------

def perf_func(func, warmup_iters=10, iters=100, *args, **kwargs):
    """Performance measurement: warmup then timed iterations."""
    # Warmup
    for _ in range(warmup_iters):
        output = func(*args, **kwargs)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        output = func(*args, **kwargs)
    end_event.record()

    torch.cuda.synchronize()
    duration_ms = start_event.elapsed_time(end_event) / iters

    return output, duration_ms


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_correctness(args):
    """
    Verify overlap kernel output matches reference implementation.
    Tests: overlap kernel vs NCCL reference.
    """
    rank, local_rank, world_size, local_world_size, pg = initialize_distributed()
    torch.manual_seed(42 + rank)

    M = args.M
    N = args.N
    topk = args.topk
    dtype = getattr(torch, args.dtype)

    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"

    # Create context
    ctx = create_topk_rss_overlap_context(
        max_M=M, N=N, topk=topk, dtype=dtype,
    )

    # Generate random inputs
    expert_outputs = torch.randn(M, topk, N, dtype=dtype, device=f"cuda:{local_rank}")
    shared_expert_output = torch.randn(M, N, dtype=dtype, device=f"cuda:{local_rank}")

    # Run overlap kernel
    overlap_out = fused_topk_reduce_sum_add_shared_reduce_scatter(
        ctx, expert_outputs, shared_expert_output,
        routed_scaling_factor=1.0,
    )

    # Run reference
    ref_out = reference_topk_rss(
        expert_outputs, shared_expert_output, pg,
        routed_scaling_factor=1.0,
    )

    # Compare
    torch.cuda.synchronize()
    diff = calc_diff(overlap_out, ref_out)
    max_diff = (overlap_out - ref_out).abs().max().item()
    mean_diff = (overlap_out - ref_out).abs().mean().item()

    if rank == 0:
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        passed = diff < 0.001
        status = "PASSED" if passed else "FAILED"
        print(f"[Correctness] {status}")
        print(f"  Config: M={M}, N={N}, topk={topk}, dtype={args.dtype}, "
              f"world_size={world_size}")
        print(f"  cosine_diff={diff:.6f}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        if not passed:
            print(f"  Expected (first row): {ref_out[0, :8]}")
            print(f"  Got      (first row): {overlap_out[0, :8]}")

    ctx.finalize()
    dist.destroy_process_group()


def test_performance(args):
    """
    Benchmark four implementations:
    1. Compute-only: topk_reduce_sum + add shared (no comm)
    2. Comm-only: NCCL reduce-scatter
    3. Non-overlap: compute + comm sequentially
    4. Overlap kernel: fused compute + comm

    Reports latency (us) and speedup.
    """
    rank, local_rank, world_size, local_world_size, pg = initialize_distributed()
    torch.manual_seed(42 + rank)

    M = args.M
    N = args.N
    topk = args.topk
    dtype = getattr(torch, args.dtype)
    warmup = args.warmup
    iters = args.iters

    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"
    M_per_rank = M // world_size

    # Create context
    ctx = create_topk_rss_overlap_context(
        max_M=M, N=N, topk=topk, dtype=dtype,
    )

    # Generate inputs
    expert_outputs = torch.randn(M, topk, N, dtype=dtype, device=f"cuda:{local_rank}")
    shared_expert_output = torch.randn(M, N, dtype=dtype, device=f"cuda:{local_rank}")

    # Pre-allocate comm-only tensors
    compute_result = torch.randn(M, N, dtype=dtype, device=f"cuda:{local_rank}")
    comm_output = torch.randn(M_per_rank, N, dtype=dtype, device=f"cuda:{local_rank}")
    non_overlap_output = torch.randn(M_per_rank, N, dtype=dtype, device=f"cuda:{local_rank}")

    # Data volume for bandwidth calculation
    elem_bytes = expert_outputs.element_size()
    input_bytes = M * topk * N * elem_bytes + M * N * elem_bytes  # expert_outputs + shared
    output_bytes = M_per_rank * N * elem_bytes
    comm_bytes = M * N * elem_bytes  # reduce-scatter input size

    # 1. Compute-only
    _, compute_ms = perf_func(
        compute_only, warmup, iters,
        expert_outputs, shared_expert_output,
    )

    # 2. Comm-only (NCCL reduce-scatter)
    # Need a fresh compute_result for comm input
    fresh_result = compute_only(expert_outputs, shared_expert_output)
    _, comm_ms = perf_func(
        comm_only, warmup, iters,
        fresh_result, comm_output, pg,
    )

    # 3. Non-overlap (compute + comm sequential)
    _, non_overlap_ms = perf_func(
        non_overlap, warmup, iters,
        expert_outputs, shared_expert_output,
        non_overlap_output, pg,
    )

    # 4. Overlap kernel
    _, overlap_ms = perf_func(
        fused_topk_reduce_sum_add_shared_reduce_scatter, warmup, iters,
        ctx, expert_outputs, shared_expert_output, 1.0,
    )

    from moe_reduce_rs import (
        MoEReduceRSSymmMemContext,
        create_moe_rs_symm_mem_context,
        reduce_topk_reduce_scatter_a2a_intra_node_with_shared_expert_symm_mem,
    )

    ctx = create_moe_rs_symm_mem_context(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        max_token_num=M,
        hidden_dim=N,
        num_experts=0,
        topk=topk,
        input_dtype=dtype,
        group=pg,
    )
    output = torch.empty(
        (M_per_rank, N), dtype=expert_outputs.dtype, device=expert_outputs.device,
    )
    _, overlap_ms = perf_func(
        reduce_topk_reduce_scatter_a2a_intra_node_with_shared_expert_symm_mem, warmup, iters,
        expert_outputs, shared_expert_output, ctx, M, 7, output, 1.0,
    )

    if rank == 0:
        compute_us = compute_ms * 1000
        comm_us = comm_ms * 1000
        non_overlap_us = non_overlap_ms * 1000
        overlap_us = overlap_ms * 1000

        compute_bw = input_bytes / (compute_ms * 1e-3) / 1e9  # GB/s
        comm_bw = comm_bytes / (comm_ms * 1e-3) / 1e9
        non_overlap_bw = (input_bytes + output_bytes) / (non_overlap_ms * 1e-3) / 1e9
        overlap_bw = (input_bytes + output_bytes) / (overlap_ms * 1e-3) / 1e9

        speedup_vs_seq = non_overlap_us / overlap_us
        overlap_efficiency = max(compute_us, comm_us) / overlap_us

        print(f"\n{'Implementation':<25} {'Latency (us)':>14} {'BW (GB/s)':>12} {'Speedup':>10}")
        print("-" * 65)
        print(f"{'Compute-only':<25} {compute_us:>14.2f} {compute_bw:>12.2f} {'-':>10}")
        print(f"{'Comm-only (NCCL)':<25} {comm_us:>14.2f} {comm_bw:>12.2f} {'-':>10}")
        print(f"{'Non-overlap':<25} {non_overlap_us:>14.2f} {non_overlap_bw:>12.2f} {1.0:>10.3f}")
        print(f"{'Overlap kernel':<25} {overlap_us:>14.2f} {overlap_bw:>12.2f} {speedup_vs_seq:>10.3f}")
        print("-" * 65)
        print(f"  Overlap efficiency = max(compute, comm) / overlap = "
              f"{overlap_efficiency:.3f}")
        print(f"  Config: M={M}, N={N}, topk={topk}, dtype={args.dtype}, "
              f"world_size={world_size}")

    ctx.finalize()
    dist.destroy_process_group()


def test_multi_size(args):
    """Correctness across multiple problem shapes and dtypes."""
    rank, local_rank, world_size, local_world_size, pg = initialize_distributed()

    configs = [
        # (M, N, topk, dtype)
        (1024, 4096, 8, "bfloat16"),
        (2048, 7168, 8, "bfloat16"),
        (4096, 7168, 6, "bfloat16"),
        (1024, 4096, 4, "float16"),
        (2048, 3584, 8, "float16"),
        (512,  2048, 4, "bfloat16"),
    ]

    all_passed = True
    for M, N, topk, dtype_name in configs:
        if M % world_size != 0:
            if rank == 0:
                print(f"  SKIP M={M}, N={N}, topk={topk}, dtype={dtype_name} "
                      f"(M not divisible by world_size={world_size})")
            continue

        dtype = getattr(torch, dtype_name)
        torch.manual_seed(42 + rank)

        ctx = create_topk_rss_overlap_context(
            max_M=M, N=N, topk=topk, dtype=dtype,
        )

        expert_outputs = torch.randn(M, topk, N, dtype=dtype, device=f"cuda:{local_rank}")
        shared_expert_output = torch.randn(M, N, dtype=dtype, device=f"cuda:{local_rank}")

        overlap_out = fused_topk_reduce_sum_add_shared_reduce_scatter(
            ctx, expert_outputs, shared_expert_output,
            routed_scaling_factor=1.0,
        )
        ref_out = reference_topk_rss(expert_outputs, shared_expert_output, pg)

        torch.cuda.synchronize()
        diff = calc_diff(overlap_out, ref_out)
        passed = diff < 0.001

        if rank == 0:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] M={M}, N={N}, topk={topk}, dtype={dtype_name}, "
                  f"diff={diff:.6f}")

        if not passed:
            all_passed = False

        ctx.finalize()

    if rank == 0:
        print(f"\n[Multi-size] Overall: {'PASSED' if all_passed else 'FAILED'}")

    dist.destroy_process_group()


def test_stability(args, n_iters=50):
    """Repeated correctness checks to catch non-deterministic hangs/failures."""
    rank, local_rank, world_size, local_world_size, pg = initialize_distributed()
    torch.manual_seed(42 + rank)

    M = args.M
    N = args.N
    topk = args.topk
    dtype = getattr(torch, args.dtype)

    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"

    ctx = create_topk_rss_overlap_context(
        max_M=M, N=N, topk=topk, dtype=dtype,
    )

    n_fail = 0
    max_diff_overall = 0.0

    for i in range(n_iters):
        torch.manual_seed(100 + i + rank)
        expert_outputs = torch.randn(M, topk, N, dtype=dtype, device=f"cuda:{local_rank}")
        shared_expert_output = torch.randn(M, N, dtype=dtype, device=f"cuda:{local_rank}")

        overlap_out = fused_topk_reduce_sum_add_shared_reduce_scatter(
            ctx, expert_outputs, shared_expert_output,
            routed_scaling_factor=1.0,
        )
        ref_out = reference_topk_rss(expert_outputs, shared_expert_output, pg)

        torch.cuda.synchronize()
        diff = calc_diff(overlap_out, ref_out)
        max_diff_overall = max(max_diff_overall, diff)

        if diff >= 0.001:
            n_fail += 1
            if rank == 0 and n_fail <= 3:
                print(f"  [iter {i}] FAIL diff={diff:.6f}")

    if rank == 0:
        status = "PASSED" if n_fail == 0 else "FAILED"
        print(f"[Stability] {status}: {n_iters - n_fail}/{n_iters} passed, "
              f"max_diff={max_diff_overall:.6f}")

    ctx.finalize()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Fused TopK Reduce-Sum + Add Shared + Reduce-Scatter Overlap Kernel",
    )
    parser.add_argument(
        "--case", type=str, default="correctness",
        choices=["correctness", "performance", "multi_size", "stability"],
        help="Test mode",
    )
    # Shape parameters
    parser.add_argument("--M", type=int, default=8192, help="Total token count (must be divisible by world_size)")
    parser.add_argument("--N", type=int, default=7168, help="Hidden dimension")
    parser.add_argument("--topk", type=int, default=8, help="Number of selected experts")
    # Benchmark parameters
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    # Misc
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"], help="Data type")
    parser.add_argument("--profile", action="store_true",
                        help="Export torch profiler trace")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.case == "correctness":
        test_correctness(args)
    elif args.case == "performance":
        test_performance(args)
    elif args.case == "multi_size":
        test_multi_size(args)
    elif args.case == "stability":
        test_stability(args)


if __name__ == "__main__":
    main()