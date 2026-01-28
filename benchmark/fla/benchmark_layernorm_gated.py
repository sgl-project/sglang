from typing import Optional

import numpy as np
import torch

# Import the function to benchmark
from sglang.srt.layers.attention.fla.layernorm_gated import (
    _layer_norm_fwd as layer_norm_fwd,
)
from sglang.srt.layers.attention.fla.layernorm_gated import rms_norm_ref


def benchmark_layer_norm_fwd(
    M: int = 65536,
    N: int = 128,
    eps: float = 1e-6,
    has_z: bool = True,
    has_bias: bool = False,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = True,
    dtype: torch.dtype = torch.float16,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    device: str = "cuda",
    verbose: bool = True,
):
    """
    Benchmark layer_norm_fwd with specified parameters.

    Args:
        M: Number of rows (batch size)
        N: Number of columns (hidden dimension)
        eps: Epsilon for numerical stability
        has_z: Whether to use gating tensor z
        has_bias: Whether to use bias
        group_size: Group size for group normalization (None = full dimension)
        norm_before_gate: Whether to normalize before gating
        is_rms_norm: Whether to use RMS normalization (vs LayerNorm)
        dtype: Data type for tensors
        warmup_iters: Number of warmup iterations
        benchmark_iters: Number of benchmark iterations
        device: Device to run on
    """
    if verbose:
        print("=" * 80)
        print("LayerNorm Forward Pass Benchmark")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  x.shape: torch.Size([{M}, {N}])")
        print(f"  weight.shape: torch.Size([{N}])")
        print(f"  bias: {'torch.Size([{}])'.format(N) if has_bias else None}")
        print(f"  eps: {eps}")
        print(f"  z: {'torch.Size([{}, {}])'.format(M, N) if has_z else None}")
        print(f"  out: None")
        print(f"  group_size: {group_size}")
        print(f"  norm_before_gate: {norm_before_gate}")
        print(f"  is_rms_norm: {is_rms_norm}")
        print(f"  dtype: {dtype}")
        print(f"  device: {device}")
        print()

    # Create input tensors
    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device) if has_bias else None
    z = torch.randn(M, N, dtype=dtype, device=device) if has_z else None

    # Ensure contiguous memory layout
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if z is not None:
        z = z.contiguous()

    if verbose:
        print("Warming up...")
    # Warmup
    for _ in range(warmup_iters):
        out, mean, rstd = layer_norm_fwd(
            x=x,
            weight=weight,
            bias=bias,
            eps=eps,
            z=z,
            out=None,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        torch.cuda.synchronize()

    if verbose:
        print(f"Capturing CUDA graph...")

    # Capture the kernel execution in a CUDA graph
    runs_per_measurement = 100

    # Create output tensor for graph capture
    out_graph = torch.empty_like(x)
    mean_graph = (
        torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd_graph = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)

    # Capture the graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(runs_per_measurement):
            out, mean, rstd = layer_norm_fwd(
                x=x,
                weight=weight,
                bias=bias,
                eps=eps,
                z=z,
                out=out_graph,
                group_size=group_size,
                norm_before_gate=norm_before_gate,
                is_rms_norm=is_rms_norm,
            )

    if verbose:
        print(
            f"Running benchmark with {benchmark_iters} iterations using CUDA graph..."
        )

    # Benchmark by replaying the graph
    times = []
    for i in range(benchmark_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()

        # elapsed_time_ms returns milliseconds, divide by runs_per_measurement
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(
            elapsed_ms / 1000.0 / runs_per_measurement
        )  # Convert to seconds per run

    # Compute statistics
    times = np.array(times) * 1_000_000  # Convert to microseconds
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)

    # Calculate throughput
    num_elements = M * N
    throughput_gelements_per_sec = (num_elements / mean_time) * 1_000_000 / 1e9

    # Calculate memory bandwidth
    # Read: x, weight, z (if has_z)
    # Write: out, rstd, mean (if not rms_norm)
    bytes_per_element = 2 if dtype == torch.float16 else 4  # fp16 or fp32
    read_bytes = (M * N + N) * bytes_per_element  # x + weight
    if has_z:
        read_bytes += M * N * bytes_per_element  # z
    write_bytes = M * N * bytes_per_element  # out
    write_bytes += M * 4  # rstd (float32)
    if not is_rms_norm:
        write_bytes += M * 4  # mean (float32)

    total_bytes = read_bytes + write_bytes
    bandwidth_gb_per_sec = (total_bytes / mean_time) * 1_000_000 / 1e9

    if verbose:
        print("\n" + "=" * 80)
        print("Benchmark Results")
        print("=" * 80)
        print(f"\nTiming Statistics (microseconds):")
        print(f"  Mean:     {mean_time:.2f} us")
        print(f"  Std Dev:  {std_time:.2f} us")
        print(f"  Min:      {min_time:.2f} us")
        print(f"  Max:      {max_time:.2f} us")
        print(f"  Median:   {median_time:.2f} us")
        print(f"  P95:      {p95_time:.2f} us")
        print(f"  P99:      {p99_time:.2f} us")

        print(f"\nThroughput:")
        print(f"  {throughput_gelements_per_sec:.2f} GElements/sec")
        print(f"  {bandwidth_gb_per_sec:.2f} GB/sec")

        print(f"\nMemory Usage:")
        print(f"  Input size: {read_bytes / 1e6:.2f} MB")
        print(f"  Output size: {write_bytes / 1e6:.2f} MB")
        print(f"  Total: {total_bytes / 1e6:.2f} MB")

    # Verify correctness against reference implementation
    if verbose:
        print("\nVerifying correctness...")
    out_triton, mean_triton, rstd_triton = layer_norm_fwd(
        x=x,
        weight=weight,
        bias=bias,
        eps=eps,
        z=z,
        out=None,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )

    # Compute reference output
    out_ref = rms_norm_ref(
        x=x,
        weight=weight,
        bias=bias,
        z=z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        upcast=True,
    )

    # Compare outputs
    max_diff = torch.max(torch.abs(out_triton - out_ref)).item()
    mean_diff = torch.mean(torch.abs(out_triton - out_ref)).item()
    rel_diff = torch.mean(
        torch.abs(out_triton - out_ref) / (torch.abs(out_ref) + 1e-5)
    ).item()

    if verbose:
        print(f"\nCorrectness Check (vs Reference Implementation):")
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  Mean relative difference: {rel_diff:.6e}")

        if max_diff < 1e-2:
            print("  ✓ PASS: Results match reference implementation")
        else:
            print("  ✗ FAIL: Results do not match reference implementation")

        print("\n" + "=" * 80)

    return {
        "mean_time_us": mean_time,
        "std_time_us": std_time,
        "min_time_us": min_time,
        "max_time_us": max_time,
        "median_time_us": median_time,
        "p95_time_us": p95_time,
        "p99_time_us": p99_time,
        "throughput_gelements_per_sec": throughput_gelements_per_sec,
        "bandwidth_gb_per_sec": bandwidth_gb_per_sec,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_diff": rel_diff,
    }


def main():
    """Run the benchmark with the specified configuration."""
    # Configuration from user
    config = {
        "M": 65536,
        "N": 128,
        "eps": 1e-6,
        "has_z": True,
        "has_bias": False,
        "group_size": None,
        "norm_before_gate": True,
        "is_rms_norm": True,
        "dtype": torch.float16,
        "warmup_iters": 10,
        "benchmark_iters": 100,
        "device": "cuda",
    }

    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
        return

    results = benchmark_layer_norm_fwd(**config)

    # Collect all results
    all_results = []
    # Test with different batch sizes
    print("\nRunning benchmarks for varying batch sizes...")
    for M in [256, 512, 1024, 4096, 16384, 65536, 2**17, 2**18]:
        config_var = config.copy()
        config_var["M"] = M
        config_var["warmup_iters"] = 5
        config_var["benchmark_iters"] = 50
        config_var["verbose"] = False
        result = benchmark_layer_norm_fwd(**config_var)
        all_results.append({"M": M, "N": config_var["N"], **result})
        print(f"  M={M:>5}: {result['mean_time_us']:>7.2f} us")

    # Print summary table
    print("\n\n")
    print("=" * 30)
    print("SUMMARY TABLE - Varying Batch Size (M) with N=128")
    print("=" * 30)
    print(f"{'M':>8} | {'Median (us)':>12}")
    print("-" * 30)
    for r in all_results:
        print(f"{r['M']:>8} | {r['median_time_us']:>12.2f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
