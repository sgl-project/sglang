#!/usr/bin/env python3
"""Benchmark dLLM post-processing Triton kernel vs PyTorch reference."""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.kernels.post_process import (
    dllm_post_process_pytorch as pytorch_reference,
)


def test_correctness(block_size: int, vocab_size: int, mask_ratio: float = 0.5):
    """Test correctness of Triton kernel against PyTorch reference."""
    from sglang.srt.dllm.kernels.post_process import dllm_post_process_fused

    print(f"\n{'='*60}")
    print(f"Testing correctness: block_size={block_size}, vocab_size={vocab_size}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    mask_id = vocab_size - 1  # Use last token as mask
    threshold = 0.95

    # Generate test data
    torch.manual_seed(42)
    logits = torch.randn(block_size, vocab_size, dtype=torch.float16, device=device)

    # Create input_ids with some masked positions
    input_ids_base = torch.randint(
        0, vocab_size - 1, (block_size,), dtype=torch.int64, device=device
    )
    mask_positions = torch.rand(block_size, device=device) < mask_ratio
    input_ids_base[mask_positions] = mask_id

    # Test with different scenarios
    scenarios = [
        (
            "All masked",
            torch.full((block_size,), mask_id, dtype=torch.int64, device=device),
        ),
        (
            "No masked",
            torch.randint(
                0, vocab_size - 1, (block_size,), dtype=torch.int64, device=device
            ),
        ),
        ("Mixed", input_ids_base.clone()),
    ]

    all_passed = True

    for scenario_name, input_ids in scenarios:
        print(f"\nScenario: {scenario_name}")

        # PyTorch reference
        input_ids_pt = input_ids.clone()
        out_pt, transfer_pt, conf_pt, num_pt = pytorch_reference(
            logits, input_ids_pt, mask_id, threshold
        )

        # Triton kernel
        input_ids_tr = input_ids.clone()
        dllm_post_process_fused(logits, input_ids_tr, mask_id, threshold)

        # Compare results - Triton kernel modifies input_ids in-place
        ids_match = torch.equal(out_pt, input_ids_tr)

        print(f"  Output IDs match: {ids_match}")
        print(f"  Num transfers (PyTorch): {num_pt}")

        if not ids_match:
            diff_positions = (out_pt != input_ids_tr).nonzero(as_tuple=True)[0]
            print(f"  Differences at positions: {diff_positions[:10].tolist()}...")
            all_passed = False

    return all_passed


def benchmark(
    block_size: int,
    vocab_size: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
    ncu_mode: bool = False,
):
    """Benchmark Triton kernel vs PyTorch implementation."""
    from sglang.srt.dllm.kernels.post_process import dllm_post_process_fused

    print(f"\n{'='*60}")
    print(f"Benchmarking: block_size={block_size}, vocab_size={vocab_size}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    mask_id = vocab_size - 1
    threshold = 0.95

    # Generate test data
    torch.manual_seed(42)
    logits = torch.randn(block_size, vocab_size, dtype=torch.float16, device=device)
    input_ids_template = torch.full(
        (block_size,), mask_id, dtype=torch.int64, device=device
    )
    # Make some positions non-masked
    input_ids_template[::2] = torch.randint(
        0, vocab_size - 1, (block_size // 2 + 1,), device=device
    )[: block_size // 2 + (block_size % 2)]

    if ncu_mode:
        # NCU mode: run a few iterations without timing
        print("NCU profiling mode - running kernels for profiling...")
        for _ in range(3):
            input_ids = input_ids_template.clone()
            dllm_post_process_fused(logits, input_ids, mask_id, threshold)
        torch.cuda.synchronize()
        print("Done. Check NCU output for kernel analysis.")
        return

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        input_ids = input_ids_template.clone()
        dllm_post_process_fused(logits, input_ids, mask_id, threshold)
    torch.cuda.synchronize()

    # Benchmark Triton kernel
    print(f"Benchmarking Triton kernel ({num_iterations} iterations)...")
    triton_times = []
    for _ in range(num_iterations):
        input_ids = input_ids_template.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        dllm_post_process_fused(logits, input_ids, mask_id, threshold)
        torch.cuda.synchronize()
        triton_times.append((time.perf_counter() - start) * 1000)

    # Benchmark PyTorch reference
    print(f"Benchmarking PyTorch reference ({num_iterations} iterations)...")
    pytorch_times = []
    for _ in range(num_iterations):
        input_ids = input_ids_template.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        pytorch_reference(logits, input_ids, mask_id, threshold)
        torch.cuda.synchronize()
        pytorch_times.append((time.perf_counter() - start) * 1000)

    # Report results
    triton_mean = np.mean(triton_times)
    triton_std = np.std(triton_times)
    pytorch_mean = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)

    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Triton kernel:     {triton_mean:.3f} ± {triton_std:.3f} ms")
    print(f"PyTorch reference: {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")
    print(f"Speedup:           {pytorch_mean / triton_mean:.2f}x")
    print()

    # Per-iteration breakdown for dLLM context
    print("Impact on dLLM decoding (block_size iterations):")
    print(f"  Total Triton:  {triton_mean * block_size:.1f} ms (worst case)")
    print(f"  Total PyTorch: {pytorch_mean * block_size:.1f} ms (worst case)")
    print(f"  Savings:       {(pytorch_mean - triton_mean) * block_size:.1f} ms")


def benchmark_individual_ops(
    block_size: int, vocab_size: int, num_iterations: int = 100
):
    """Benchmark individual PyTorch operations for comparison."""
    print(f"\n{'='*60}")
    print(
        f"Individual operation breakdown: block_size={block_size}, vocab_size={vocab_size}"
    )
    print(f"{'='*60}")

    device = torch.device("cuda")
    mask_id = vocab_size - 1
    threshold = 0.95

    logits = torch.randn(block_size, vocab_size, dtype=torch.float16, device=device)
    input_ids = torch.full((block_size,), mask_id, dtype=torch.int64, device=device)

    ops = {
        "argmax": lambda: torch.argmax(logits, dim=-1),
        "softmax": lambda: F.softmax(logits.float(), dim=-1),
        "gather": lambda: torch.gather(
            F.softmax(logits.float(), dim=-1),
            dim=-1,
            index=torch.argmax(logits, dim=-1).unsqueeze(-1),
        ),
        "where (mask)": lambda: torch.where(
            input_ids == mask_id, torch.argmax(logits, dim=-1), input_ids
        ),
        "threshold": lambda: (
            F.softmax(logits.float(), dim=-1).max(dim=-1).values > threshold
        ),
    }

    print(f"\nOperation breakdown (avg of {num_iterations} iterations):")
    print("-" * 40)

    total_time = 0
    for op_name, op_fn in ops.items():
        # Warmup
        for _ in range(10):
            op_fn()
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            op_fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        total_time += avg_time
        print(f"{op_name:20s}: {avg_time:.3f} ms")

    print("-" * 40)
    print(f"{'Total':20s}: {total_time:.3f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark dLLM post-processing Triton kernel"
    )
    parser.add_argument(
        "--block-size", type=int, default=32, help="Block size (default: 32)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=128000,
        help="Vocabulary size (default: 128000)",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--ncu", action="store_true", help="NCU profiling mode (minimal iterations)"
    )
    parser.add_argument(
        "--skip-correctness", action="store_true", help="Skip correctness tests"
    )
    parser.add_argument(
        "--breakdown", action="store_true", help="Show individual operation breakdown"
    )
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Configuration: block_size={args.block_size}, vocab_size={args.vocab_size}")

    if not args.skip_correctness and not args.ncu:
        # Test correctness
        passed = test_correctness(args.block_size, args.vocab_size)
        if not passed:
            print("\n⚠️  Correctness test failed! Fix bugs before benchmarking.")
            return
        print("\n✓ Correctness tests passed!")

    # Benchmark
    benchmark(
        args.block_size,
        args.vocab_size,
        num_iterations=args.num_iterations,
        ncu_mode=args.ncu,
    )

    if args.breakdown and not args.ncu:
        benchmark_individual_ops(args.block_size, args.vocab_size, args.num_iterations)


if __name__ == "__main__":
    main()
