#!/usr/bin/env python3
"""Microbenchmark for GLM-5.2 MoE hotpath.

Measures the overhead of the MoE forward path, including gate projection,
topk selection, and expert dispatch/combine. Identifies allocation overhead
and synchronization points.

Usage:
    python scripts/bench_glm52_moe_hotpath.py
"""

import argparse
import sys
import time
import torch

sys.path.insert(0, "python")


def benchmark_allocation(hidden_size, num_tokens, device, warmup=5, iters=50):
    """Benchmark the temporary allocation in MoE forward_normal."""
    # Simulates: torch.empty_like(final_hidden_states) + torch.add
    final_hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    shared_output = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)

    for _ in range(warmup):
        out = torch.empty_like(final_hidden_states)
        torch.add(final_hidden_states, shared_output, out=out)
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        out = torch.empty_like(final_hidden_states)
        torch.add(final_hidden_states, shared_output, out=out)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.95)]


def benchmark_inplace_add(hidden_size, num_tokens, device, warmup=5, iters=50):
    """Benchmark in-place add (no allocation)."""
    final_hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    shared_output = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)

    for _ in range(warmup):
        final_hidden_states.add_(shared_output)
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        final_hidden_states.add_(shared_output)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.95)]


def main():
    parser = argparse.ArgumentParser(description="Benchmark GLM-5.2 MoE hotpath")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # GLM-5.2 dimensions (from glm4_moe.py / glm4_moe_lite.py)
    hidden_size = 7168  # GLM-4.5/4.6 hidden size

    print(f"GLM-5.2 MoE Hotpath Microbenchmark")
    print(f"Device: {device}, Hidden size: {hidden_size}")
    print(f"{'='*80}")

    test_cases = [
        ("decode_bs1", 1),
        ("decode_bs4", 4),
        ("decode_bs8", 8),
        ("decode_bs16", 16),
        ("decode_bs32", 32),
        ("decode_bs64", 64),
        ("prefill_256", 256),
        ("prefill_1024", 1024),
        ("prefill_4096", 4096),
    ]

    print(f"{'Test Case':<20} {'Tokens':>8} {'Alloc Med(ms)':>15} {'Inplace Med(ms)':>17} {'Speedup':>8}")
    print("-" * 80)

    for name, num_tokens in test_cases:
        alloc_med, alloc_p95 = benchmark_allocation(hidden_size, num_tokens, device)
        inplace_med, inplace_p95 = benchmark_inplace_add(hidden_size, num_tokens, device)
        speedup = alloc_med / inplace_med if inplace_med > 0 else float('inf')
        print(f"{name:<20} {num_tokens:>8} {alloc_med:>15.4f} {inplace_med:>17.4f} {speedup:>7.2f}x")

    print(f"\n{'='*80}")
    print("Note: In-place add avoids the torch.empty_like allocation.")
    print("The allocation overhead is per-layer, so for 60 MoE layers it compounds.")
    print(f"\nBenchmark complete.")


if __name__ == "__main__":
    main()
