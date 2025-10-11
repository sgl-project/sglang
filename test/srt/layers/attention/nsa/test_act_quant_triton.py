"""
Unit tests comparing TileLang and Triton implementations of activation quantization.
Tests both accuracy and performance.
"""

import time
from typing import Tuple

import pytest
import torch

from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant as act_quant_triton


def benchmark_kernel(
    fn,
    x: torch.Tensor,
    block_size: int,
    scale_fmt,
    warmup: int = 10,
    repeat: int = 100,
    use_cuda_graph: bool = True,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Benchmark a kernel function.

    Args:
        fn: Function to benchmark
        x: Input tensor
        block_size: Block size for quantization
        scale_fmt: Scale format
        warmup: Number of warmup iterations
        repeat: Number of repeat iterations
        use_cuda_graph: Whether to use CUDA graphs for more accurate timing

    Returns:
        Tuple of (avg_time_ms, quantized_output, scales)
    """
    # Warmup
    for _ in range(warmup):
        y, s = fn(x, block_size=block_size, scale_fmt=scale_fmt)

    if not x.is_cuda or not use_cuda_graph:
        # Fallback to regular timing
        if x.is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeat):
            y, s = fn(x, block_size=block_size, scale_fmt=scale_fmt)

        if x.is_cuda:
            torch.cuda.synchronize()

        end = time.perf_counter()
        avg_time_ms = (end - start) / repeat * 1000

        return avg_time_ms, y, s

    # Use CUDA graph for more accurate timing
    torch.cuda.synchronize()

    # Allocate output buffers
    N = x.size(-1)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)

    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_cap, s_cap = fn(x, block_size=block_size, scale_fmt=scale_fmt)

    # Warmup with graph
    for _ in range(warmup):
        graph.replay()

    torch.cuda.synchronize()

    # Timing with CUDA graph
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        graph.replay()
    end_event.record()

    torch.cuda.synchronize()

    avg_time_ms = start_event.elapsed_time(end_event) / repeat

    return avg_time_ms, y_cap, s_cap


def check_accuracy(
    y_ref: torch.Tensor,
    s_ref: torch.Tensor,
    y_test: torch.Tensor,
    s_test: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Tuple[bool, dict]:
    """
    Check accuracy between reference and test outputs.

    Args:
        y_ref: Reference quantized output
        s_ref: Reference scales
        y_test: Test quantized output
        s_test: Test scales
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (passed, metrics_dict)
    """
    # Convert FP8 to float for comparison
    y_ref_float = y_ref.float()
    y_test_float = y_test.float()

    # Compute differences
    y_diff = torch.abs(y_ref_float - y_test_float)
    s_diff = torch.abs(s_ref - s_test)

    # Compute metrics
    y_max_diff = y_diff.max().item()
    y_mean_diff = y_diff.mean().item()
    s_max_diff = s_diff.max().item()
    s_mean_diff = s_diff.mean().item()

    # Check relative and absolute tolerance
    y_close = torch.allclose(y_ref_float, y_test_float, rtol=rtol, atol=atol)
    s_close = torch.allclose(s_ref, s_test, rtol=rtol, atol=atol)

    # Compute percentage of matching elements
    y_match_pct = (y_ref_float == y_test_float).float().mean().item() * 100

    metrics = {
        "y_max_diff": y_max_diff,
        "y_mean_diff": y_mean_diff,
        "y_match_pct": y_match_pct,
        "s_max_diff": s_max_diff,
        "s_mean_diff": s_mean_diff,
        "y_close": y_close,
        "s_close": s_close,
    }

    passed = y_close and s_close

    return passed, metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_quant_comprehensive_benchmark(scale_fmt=None):
    """Comprehensive benchmark across multiple sizes with CUDA graphs."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size = 128

    shapes = [
        (128, 512),
        (256, 1024),
        (512, 2048),
        (1024, 4096),
        (2048, 8192),
        (4096, 16384),
    ]

    print("\n" + "=" * 100)
    print("Comprehensive Performance Benchmark with CUDA Graphs")
    print("=" * 100)
    print(
        f"{'Shape':<20} {'TileLang (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10} {'Status'}"
    )
    print("-" * 100)

    for shape in shapes:
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=dtype, device=device)

        try:
            # Benchmark both with CUDA graphs
            time_tilelang, y_ref, s_ref = benchmark_kernel(
                act_quant,
                x,
                block_size,
                scale_fmt,
                warmup=5,
                repeat=50,
                use_cuda_graph=True,
            )
            time_triton, y_triton, s_triton = benchmark_kernel(
                act_quant_triton,
                x,
                block_size,
                scale_fmt,
                warmup=5,
                repeat=50,
                use_cuda_graph=True,
            )

            # Check accuracy
            passed, _ = check_accuracy(y_ref, s_ref, y_triton, s_triton)

            speedup = time_tilelang / time_triton if time_triton > 0 else 0
            status = "✓ PASS" if passed else "✗ FAIL"

            print(
                f"{str(shape):<20} {time_tilelang:<15.4f} {time_triton:<15.4f} "
                f"{speedup:<10.2f} {status}"
            )
        except Exception as e:
            print(f"{str(shape):<20} ERROR: {str(e)}")

    print("=" * 100)

    # Also run without CUDA graphs for comparison
    print("\n" + "=" * 100)
    print("Performance Benchmark WITHOUT CUDA Graphs (for comparison)")
    print("=" * 100)
    print(
        f"{'Shape':<20} {'TileLang (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10} {'Status'}"
    )
    print("-" * 100)

    for shape in shapes:
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=dtype, device=device)

        try:
            # Benchmark both without CUDA graphs
            time_tilelang, y_ref, s_ref = benchmark_kernel(
                act_quant,
                x,
                block_size,
                scale_fmt,
                warmup=5,
                repeat=50,
                use_cuda_graph=False,
            )
            time_triton, y_triton, s_triton = benchmark_kernel(
                act_quant_triton,
                x,
                block_size,
                scale_fmt,
                warmup=5,
                repeat=50,
                use_cuda_graph=False,
            )

            # Check accuracy
            passed, _ = check_accuracy(y_ref, s_ref, y_triton, s_triton)

            speedup = time_tilelang / time_triton if time_triton > 0 else 0
            status = "✓ PASS" if passed else "✗ FAIL"

            print(
                f"{str(shape):<20} {time_tilelang:<15.4f} {time_triton:<15.4f} "
                f"{speedup:<10.2f} {status}"
            )
        except Exception as e:
            print(f"{str(shape):<20} ERROR: {str(e)}")

    print("=" * 100)


if __name__ == "__main__":
    # Run comprehensive benchmark
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("Running Comprehensive Benchmark with scale_fmt=None")
        print("=" * 80)
        test_act_quant_comprehensive_benchmark(scale_fmt=None)

        print("\n" + "=" * 80)
        print("Running Comprehensive Benchmark with scale_fmt!=None")
        print("=" * 80)
        test_act_quant_comprehensive_benchmark(scale_fmt="any")
    else:
        print("CUDA not available. Skipping tests.")
