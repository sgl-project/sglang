"""
Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN).

This test compares the CuTe DSL implementation against the Triton reference
implementation for both precision and performance.

Batch sizes tested:
- B=16: Uses small batch kernel (B < 32 threshold)
- B=128: Uses big batch kernel (B >= 32 threshold)

Usage:
    # Run pytest tests
    pytest test_cutedsl_gdn.py -v -k performance

    # Run standalone benchmark with CUDA Graph + profiler (recommended)
    python test_cutedsl_gdn.py --bench --B 16 128 --profile --iters 3 --run-iters 100

    # Run without CUDA Graph (for NCU profiling)
    python test_cutedsl_gdn.py --bench --B 16 128 --no-cudagraph --iters 3 --run-iters 100
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

# Check if cutlass/cute is available
try:
    import cutlass  # noqa: F401

    CUTEDSL_AVAILABLE = True
except ImportError:
    CUTEDSL_AVAILABLE = False

# Check if sglang Triton kernel is available
try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    TRITON_KERNEL_AVAILABLE = True
except ImportError:
    TRITON_KERNEL_AVAILABLE = False

# Directly load cutedsl_gdn.py without going through sgl_kernel package
# This avoids the sgl_kernel/__init__.py which requires compiled C++ extensions
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CUTEDSL_GDN_PATH = os.path.join(
    _SCRIPT_DIR, "..", "python", "sgl_kernel", "cutedsl_gdn.py"
)

cutedsl_gdn_module = None
if os.path.exists(_CUTEDSL_GDN_PATH) and CUTEDSL_AVAILABLE:
    try:
        spec = importlib.util.spec_from_file_location("cutedsl_gdn", _CUTEDSL_GDN_PATH)
        if spec and spec.loader:
            cutedsl_gdn_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cutedsl_gdn_module)
    except Exception as e:
        print(f"Warning: Failed to load cutedsl_gdn module: {e}")
        cutedsl_gdn_module = None


def is_hopper():
    """Check if we're running on Hopper (SM90)."""
    return torch.cuda.get_device_properties(0).major >= 9


def skip_if_cutedsl_unavailable():
    """Skip test if CuTe DSL is not available."""
    if not CUTEDSL_AVAILABLE:
        pytest.skip("CuTe DSL (cutlass/cute) not available")


def skip_if_triton_unavailable():
    """Skip test if Triton kernel is not available."""
    if not TRITON_KERNEL_AVAILABLE:
        pytest.skip("Triton GDN kernel (sglang) not available")


def print_precision_table(metrics: dict, B: int, kernel_type: str):
    """Print a formatted precision comparison table."""
    print("\n" + "=" * 70)
    print(f"  PRECISION COMPARISON: B={B} ({kernel_type} Kernel)")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Value':>20}")
    print("-" * 70)
    print(f"  {'Total Elements':<30} {metrics['total_elements']:>20,}")
    print(f"  {'Max Absolute Diff':<30} {metrics['max_abs_diff']:>20.6e}")
    print(f"  {'Mean Absolute Diff':<30} {metrics['mean_abs_diff']:>20.6e}")
    print("-" * 70)
    print(f"  {'Pass Rate (diff ≤ 1e-3)':<30} {metrics['pass_rate_1e3']:>19.2f}%")
    print(f"  {'Pass Rate (diff ≤ 1e-2)':<30} {metrics['pass_rate_1e2']:>19.2f}%")
    print(f"  {'Pass Rate (diff ≤ 1e-1)':<30} {metrics['pass_rate_1e1']:>19.2f}%")
    print(f"  {'Fail Rate (diff > 1e-1)':<30} {metrics['fail_rate_1e1']:>19.2f}%")
    print("-" * 70)
    print(f"  {'Has NaN/Inf':<30} {'Yes' if metrics['has_nan_inf'] else 'No':>20}")
    status = "✓ PASSED" if metrics['passed'] else "✗ FAILED"
    print(f"  {'Result':<30} {status:>20}")
    print("=" * 70 + "\n")


def print_performance_table(
    B: int,
    kernel_type: str,
    triton_mean: float,
    triton_std: float,
    cutedsl_mean: float,
    cutedsl_std: float,
):
    """Print a formatted performance comparison table with mean ± std."""
    speedup = triton_mean / cutedsl_mean
    faster = "CuTe DSL" if speedup > 1.0 else "Triton"
    ratio = speedup if speedup > 1.0 else 1.0 / speedup

    print("\n" + "=" * 70)
    print(f"  PERFORMANCE COMPARISON: B={B} ({kernel_type} Kernel)")
    print("=" * 70)
    print(f"  {'Kernel':<15} {'Mean (μs)':>12} {'± Std':>10} {'Speedup':>12}")
    print("-" * 70)
    print(f"  {'Triton':<15} {triton_mean:>12.3f} {triton_std:>10.3f} {'1.00x':>12}")
    print(f"  {'CuTe DSL':<15} {cutedsl_mean:>12.3f} {cutedsl_std:>10.3f} {triton_mean/cutedsl_mean:>11.2f}x")
    print("-" * 70)
    print(f"  Winner: {faster} is {ratio:.2f}x faster")
    print("=" * 70 + "\n")


def evict_caches(buf: torch.Tensor):
    """Evict GPU caches by reading a large buffer."""
    # Simple cache eviction: force GPU to touch different memory
    _ = buf.sum()
    torch.cuda.synchronize()


def parse_trace_file(trace_file: str) -> dict:
    """
    Parse a torch profiler trace file and extract kernel timings.

    Args:
        trace_file: Path to the trace JSON file

    Returns:
        dict with 'triton_times' and 'cutedsl_times' lists (in microseconds)
    """
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)

    # Triton kernel pattern (from sglang)
    # Actual name: fused_sigmoid_gating_delta_rule_update_kernel
    triton_patterns = ['fused_sigmoid_gating_delta_rule_update_kernel']

    # CuTe DSL kernel patterns (from cutedsl_gdn.py)
    # Actual names:
    #   - cpasync_swizzle_kernel_small_batch (B < 32)
    #   - cpasync_swizzle_kernel_big_batch (B >= 32)
    cutedsl_patterns = [
        'cpasync_swizzle_kernel_small_batch',
        'cpasync_swizzle_kernel_big_batch',
    ]

    triton_durations = []
    cutedsl_durations = []

    for event in trace_data.get('traceEvents', []):
        if event.get('cat') != 'kernel':
            continue

        name = event.get('name', '')
        dur = event.get('dur', 0)  # duration in microseconds

        # Check if it's a Triton kernel
        if any(pattern in name for pattern in triton_patterns):
            triton_durations.append(dur)
        # Check if it's a CuTe DSL kernel
        elif any(pattern in name for pattern in cutedsl_patterns):
            cutedsl_durations.append(dur)

    return {
        'triton_times': triton_durations,
        'cutedsl_times': cutedsl_durations,
    }


def print_trace_summary(trace_results: dict, B: int, kernel_type: str):
    """Print summary of trace-based kernel timings."""
    triton_times = np.array(trace_results['triton_times'])
    cutedsl_times = np.array(trace_results['cutedsl_times'])

    if len(triton_times) == 0 or len(cutedsl_times) == 0:
        print(f"\n⚠ Warning: Could not find kernel timings in trace file")
        print(f"  Triton kernels found: {len(triton_times)}")
        print(f"  CuTe DSL kernels found: {len(cutedsl_times)}")
        return None

    triton_mean = triton_times.mean()
    triton_std = triton_times.std()
    cutedsl_mean = cutedsl_times.mean()
    cutedsl_std = cutedsl_times.std()
    speedup = triton_mean / cutedsl_mean

    print("\n" + "=" * 70)
    print(f"  TRACE-BASED TIMING: B={B} ({kernel_type} Kernel)")
    print("=" * 70)
    print(f"  {'Kernel':<15} {'Mean (μs)':>12} {'± Std':>10} {'Count':>8} {'Speedup':>12}")
    print("-" * 70)
    print(f"  {'Triton':<15} {triton_mean:>12.3f} {triton_std:>10.3f} {len(triton_times):>8} {'1.00x':>12}")
    print(f"  {'CuTe DSL':<15} {cutedsl_mean:>12.3f} {cutedsl_std:>10.3f} {len(cutedsl_times):>8} {speedup:>11.2f}x")
    print("-" * 70)
    faster = "CuTe DSL" if speedup > 1.0 else "Triton"
    ratio = speedup if speedup > 1.0 else 1.0 / speedup
    print(f"  Winner: {faster} is {ratio:.2f}x faster")
    print("=" * 70 + "\n")

    return {
        'triton_mean': triton_mean,
        'triton_std': triton_std,
        'cutedsl_mean': cutedsl_mean,
        'cutedsl_std': cutedsl_std,
        'speedup': speedup,
    }


def comprehensive_precision_check(
    out_ref: torch.Tensor,
    out_test: torch.Tensor,
    name: str = "output",
    rtol: float = 1e-5,
    atol: float = 1e-4,
):
    """
    Comprehensive precision comparison between reference and test outputs.

    Pass criteria: diff > 1e-1 must be < 1% of total elements

    Args:
        out_ref: Reference output tensor (from Triton)
        out_test: Test output tensor (from CuTe DSL)
        name: Name of the comparison for display
        rtol: Relative tolerance threshold (for info)
        atol: Absolute tolerance threshold (for info)

    Returns:
        dict: Dictionary containing precision metrics
    """
    ref = out_ref.float().flatten()
    test = out_test.float().flatten()

    total_elements = ref.numel()

    # Absolute Error
    abs_diff = (ref - test).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    # Pass Rates
    pass_rate_1e3 = (abs_diff <= 1e-3).float().mean().item() * 100
    pass_rate_1e2 = (abs_diff <= 1e-2).float().mean().item() * 100
    pass_rate_1e1 = (abs_diff <= 1e-1).float().mean().item() * 100
    fail_rate_1e1 = 100.0 - pass_rate_1e1

    # NaN/Inf Check
    nan_ref = torch.isnan(ref).sum().item()
    nan_test = torch.isnan(test).sum().item()
    inf_ref = torch.isinf(ref).sum().item()
    inf_test = torch.isinf(test).sum().item()
    has_nan_inf = (nan_ref + nan_test + inf_ref + inf_test) > 0

    # Pass/Fail Criteria
    passed = (fail_rate_1e1 < 1.0) and not has_nan_inf

    return {
        "total_elements": total_elements,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "pass_rate_1e3": pass_rate_1e3,
        "pass_rate_1e2": pass_rate_1e2,
        "pass_rate_1e1": pass_rate_1e1,
        "fail_rate_1e1": fail_rate_1e1,
        "has_nan_inf": has_nan_inf,
        "passed": passed,
    }


def run_triton_kernel(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """Run the Triton reference kernel."""
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
    )


# ==============================================================================
# Precision Tests
# ==============================================================================


@pytest.mark.skipif(
    not CUTEDSL_AVAILABLE, reason="CuTe DSL (cutlass/cute) not available"
)
@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton GDN kernel (sglang) not available"
)
@pytest.mark.skipif(cutedsl_gdn_module is None, reason="cutedsl_gdn module not loaded")
@pytest.mark.parametrize("B", [16, 128])
def test_cutedsl_gdn_precision(B: int):
    """Test precision of CuTe DSL GDN kernel against Triton reference."""
    torch.manual_seed(2025)

    # GDN configuration (same as qwen3_next model)
    T, H, K, V = 1, 16, 128, 128
    HV = 32

    scale = K**-0.5
    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Create input tensors
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    initial_state_indices = torch.arange(B, dtype=torch.int32, device="cuda")

    # Create independent initial states for each kernel
    # Shape: [B, HV, K, V] - directly indexed by initial_state_indices
    initial_state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    initial_state_triton = initial_state_cutedsl.clone().reshape(-1).contiguous()

    # Pre-compile CuTe DSL kernel (JIT compilation happens on first call)
    _ = cutedsl_gdn_module.cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_cutedsl.clone(),
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )
    torch.cuda.synchronize()

    # Reset initial states after pre-compilation
    initial_state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    initial_state_triton = initial_state_cutedsl.clone().reshape(-1).contiguous()

    # Run CuTe DSL kernel using high-level API
    out_cutedsl = cutedsl_gdn_module.cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_cutedsl,
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )
    torch.cuda.synchronize()

    # Run Triton kernel
    out_triton = run_triton_kernel(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state=initial_state_triton,
        initial_state_indices=initial_state_indices,  # Same as CuTe DSL: arange(B)
        scale=scale,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )

    # Precision check
    metrics = comprehensive_precision_check(
        out_triton, out_cutedsl, name=f"GDN Triton vs CuTe DSL (B={B})"
    )

    # Print formatted comparison table
    kernel_type = "SmallBatch" if B < 32 else "BigBatch"
    print_precision_table(metrics, B, kernel_type)

    # Assert pass criteria
    assert metrics["passed"], (
        f"Precision check failed for B={B}:\n"
        f"  fail_rate_1e1={metrics['fail_rate_1e1']:.2f}% (should be < 1%)\n"
        f"  has_nan_inf={metrics['has_nan_inf']}"
    )


# ==============================================================================
# Performance Tests
# ==============================================================================


@pytest.mark.skipif(
    not CUTEDSL_AVAILABLE, reason="CuTe DSL (cutlass/cute) not available"
)
@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton GDN kernel (sglang) not available"
)
@pytest.mark.skipif(cutedsl_gdn_module is None, reason="cutedsl_gdn module not loaded")
@pytest.mark.parametrize("B", [16, 128])
def test_cutedsl_gdn_performance(B: int):
    """Benchmark CuTe DSL GDN kernel against Triton reference.

    Uses per-iteration CUDA event timing with cache eviction for fair comparison.
    Methodology matches the reference test in flash-linear-attention.
    """
    torch.manual_seed(2025)

    # GDN configuration
    T, H, K, V = 1, 16, 128, 128
    HV = 32

    scale = K**-0.5
    softplus_beta = 1.0
    softplus_threshold = 20.0

    warmup_iters = 10
    bench_iters = 100

    # Create input tensors
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    # Each batch must have its own state (arange) to avoid race condition
    initial_state_indices = torch.arange(B, dtype=torch.int32, device="cuda")

    initial_state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    initial_state_triton = initial_state_cutedsl.clone().reshape(-1).contiguous()

    # Cache eviction buffer (8MB to flush L2 cache)
    evict_buf = torch.randn(2 * 1024 * 1024, dtype=torch.float32, device="cuda")

    # Define kernel runner functions for cleaner code
    def run_cutedsl():
        return cutedsl_gdn_module.cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=initial_state_cutedsl,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        )

    def run_triton():
        return run_triton_kernel(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            initial_state_triton,
            initial_state_indices,  # Same as CuTe DSL: arange(B)
            scale,
        )

    # ========== Pre-compile kernels ==========
    print(f"\nPre-compiling kernels...")
    # Run once to trigger JIT compilation (not timed)
    _ = run_cutedsl()
    torch.cuda.synchronize()
    _ = run_triton()
    torch.cuda.synchronize()
    print("  CuTe DSL and Triton kernels compiled.")

    # ========== Warmup (alternating) ==========
    print(f"\nWarmup {warmup_iters} iterations (alternating)...")
    for _ in range(warmup_iters):
        _ = run_cutedsl()
        torch.cuda.synchronize()
        evict_caches(evict_buf)

        _ = run_triton()
        torch.cuda.synchronize()
        evict_caches(evict_buf)

    torch.cuda.synchronize()

    # ========== Benchmark with per-iteration timing ==========
    print(f"Benchmarking {bench_iters} iterations...")
    triton_times = []
    cutedsl_times = []

    for i in range(bench_iters):
        # Time Triton
        evict_caches(evict_buf)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = run_triton()
        end.record()
        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

        # Time CuTe DSL
        evict_caches(evict_buf)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = run_cutedsl()
        end.record()
        torch.cuda.synchronize()
        cutedsl_times.append(start.elapsed_time(end))

        # Print progress every 20 iterations
        if (i + 1) % 20 == 0:
            print(
                f"  Iter {i+1}/{bench_iters}: "
                f"Triton={triton_times[-1]*1000:.3f}μs, "
                f"CuTe DSL={cutedsl_times[-1]*1000:.3f}μs"
            )

    # Convert to numpy for statistics
    triton_times = np.array(triton_times) * 1000  # Convert ms to μs
    cutedsl_times = np.array(cutedsl_times) * 1000

    triton_mean = triton_times.mean()
    triton_std = triton_times.std()
    cutedsl_mean = cutedsl_times.mean()
    cutedsl_std = cutedsl_times.std()

    # Print formatted comparison table
    kernel_type = "SmallBatch" if B < 32 else "BigBatch"
    print_performance_table(
        B, kernel_type, triton_mean, triton_std, cutedsl_mean, cutedsl_std
    )

    # No strict speedup assertion - just verify it runs correctly
    # Performance can vary based on hardware and compilation


# ==============================================================================
# Integration Test - Using the high-level API
# ==============================================================================


@pytest.mark.skipif(
    not CUTEDSL_AVAILABLE, reason="CuTe DSL (cutlass/cute) not available"
)
@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton GDN kernel (sglang) not available"
)
@pytest.mark.skipif(cutedsl_gdn_module is None, reason="cutedsl_gdn module not loaded")
@pytest.mark.parametrize("B", [16, 128])
def test_cutedsl_gdn_high_level_api(B: int):
    """Test the high-level API of CuTe DSL GDN kernel."""
    # Use directly loaded module
    cutedsl_fused_sigmoid_gating_delta_rule_update = (
        cutedsl_gdn_module.cutedsl_fused_sigmoid_gating_delta_rule_update
    )
    is_cutedsl_gdn_available = cutedsl_gdn_module.is_cutedsl_gdn_available

    assert is_cutedsl_gdn_available(), "CuTe DSL GDN should be available"

    torch.manual_seed(2025)

    T, H, K, V = 1, 16, 128, 128
    HV = 32

    # Create input tensors
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    # Each batch must have its own state (arange) to avoid race condition
    initial_state_indices = torch.arange(B, dtype=torch.int32, device="cuda")

    # Create initial state tensors - B states, one for each batch
    initial_state_base = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    initial_state_cutedsl = initial_state_base.clone()
    initial_state_triton = initial_state_base.clone().reshape(-1).contiguous()

    # Pre-compile CuTe DSL kernel (JIT compilation happens on first call)
    _ = cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_cutedsl.clone(),
        initial_state_indices=initial_state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    # Reset initial states after pre-compilation
    initial_state_cutedsl = initial_state_base.clone()

    # Run high-level CuTe DSL API
    out_cutedsl = cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_cutedsl,
        initial_state_indices=initial_state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    # Run Triton reference
    out_triton = run_triton_kernel(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state=initial_state_triton,
        initial_state_indices=initial_state_indices,  # Same as CuTe DSL: arange(B)
        scale=K**-0.5,
    )

    # Precision check
    metrics = comprehensive_precision_check(out_triton, out_cutedsl)

    # Print formatted comparison table
    kernel_type = "SmallBatch" if B < 32 else "BigBatch"
    print_precision_table(metrics, B, f"{kernel_type} - High-Level API")

    assert metrics["passed"], f"High-level API precision check failed for B={B}"


# ==============================================================================
# Standalone Benchmark with Profiler Support + CUDA Graph
# ==============================================================================


# Null context manager for when profiler is disabled
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def run_benchmark(
    B: int,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    run_iters: int = 10,
    profile: bool = False,
    trace_dir: str = "traces",
    evict_mb: int = 128,
    use_cudagraph: bool = True,
):
    """
    Run standalone benchmark with optional torch profiler and CUDA graph.

    Uses low-level API to avoid Python wrapper overhead (similar to test_decode_fused_gdn.py).

    Args:
        B: Batch size
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations (outer loop)
        run_iters: Number of kernel runs per benchmark iteration (amortizes launch overhead)
        profile: Whether to enable torch profiler
        trace_dir: Directory to save trace files
        evict_mb: Cache eviction buffer size in MB (0 to disable)
        use_cudagraph: Whether to use CUDA graph for kernel replay

    Returns:
        dict: Benchmark results
    """
    if cutedsl_gdn_module is None:
        raise RuntimeError("CuTe DSL module not available")
    if not TRITON_KERNEL_AVAILABLE:
        raise RuntimeError("Triton kernel not available")

    # Import CuTe dependencies from module
    try:
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        import cuda.bindings.driver as cuda_driver
    except ImportError as e:
        raise RuntimeError(f"CuTe DSL dependencies not available: {e}")

    torch.manual_seed(2025)

    # GDN configuration
    T, H, K, V = 1, 16, 128, 128
    HV = 32

    scale = K**-0.5
    softplus_beta = 1.0
    softplus_threshold = 20.0
    use_small_batch = B < 32

    # ========== Create tensors (done once, no Python overhead in benchmark loop) ==========
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    # Initial state setup:
    # - Use randn for initial state (cannot be all zeros - affects computation)
    # - Each batch must have its own state (arange) to avoid race condition
    #   because kernel writes final state back to h0_source[idx]
    initial_state_indices = torch.arange(B, dtype=torch.int32, device="cuda")
    initial_state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    initial_state_triton = initial_state_cutedsl.reshape(-1).contiguous()
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device="cuda")

    # h0_source for CuTe: [B*HV, K, V] without transpose
    h0_source = initial_state_cutedsl.reshape(B * HV, K, V).contiguous()

    # Output tensor for CuTe
    o_cutedsl = torch.zeros(B, T, HV, V, dtype=torch.bfloat16, device="cuda")

    # Create multiple input tensor sets to avoid L2 cache hits during run_iters
    q_list, k_list, v_list, a_list, b_list = [], [], [], [], []
    q_tensor_list, k_tensor_list, v_tensor_list, a_tensor_list, b_tensor_list = [], [], [], [], []
    for ri in range(run_iters):
        torch.manual_seed(2025 + ri)
        q_i = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
        k_i = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
        v_i = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
        a_i = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
        b_i = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
        q_list.append(q_i)
        k_list.append(k_i)
        v_list.append(v_i)
        a_list.append(a_i)
        b_list.append(b_i)
        # Pre-create CuTe tensors (avoid from_dlpack overhead in benchmark loop)
        q_tensor_list.append(from_dlpack(q_i, assumed_align=16))
        k_tensor_list.append(from_dlpack(k_i, assumed_align=16))
        v_tensor_list.append(from_dlpack(v_i, assumed_align=16))
        a_tensor_list.append(from_dlpack(a_i, assumed_align=16))
        b_tensor_list.append(from_dlpack(b_i, assumed_align=16))

    # Pre-create CuTe tensors for static inputs
    A_log_tensor = from_dlpack(A_log, assumed_align=16)
    dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
    h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
    h0_indices_tensor = from_dlpack(initial_state_indices, assumed_align=16)
    o_tensor = from_dlpack(o_cutedsl, assumed_align=16)
    cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

    # Create dedicated PyTorch stream for CuTe kernel (for CUDA Graph capture)
    torch_stream_cutedsl = torch.cuda.Stream()
    stream = cuda_driver.CUstream(torch_stream_cutedsl.cuda_stream)

    # Cache eviction buffer
    evict_buf = None
    if evict_mb > 0:
        evict_buf = torch.randn(evict_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

    kernel_type = "SmallBatch" if use_small_batch else "BigBatch"
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK: B={B} ({kernel_type} Kernel)")
    print(f"  Warmup: {warmup_iters}, Bench: {bench_iters}, RunIters: {run_iters}")
    print(f"  Profile: {profile}, CUDA Graph: {use_cudagraph}")
    print(f"{'=' * 70}")

    # ========== Pre-compile CuTe kernel ==========
    print(f"\nPre-compiling kernels...")
    
    # Initialize cutlass module internals (required before calling _get_compiled_kernel)
    cutedsl_gdn_module._check_cutlass_available()
    
    # Get the compiled kernel (this triggers compilation if not cached)
    compiled_kernel = cutedsl_gdn_module._get_compiled_kernel(B, T, H, HV, K, V, use_small_batch)
    torch.cuda.synchronize()
    print("  ✓ CuTe DSL kernel compiled.")
    
    # Warmup Triton
    for ri in range(run_iters):
        _ = run_triton_kernel(
            A_log, dt_bias, q_list[ri], k_list[ri], v_list[ri], a_list[ri], b_list[ri],
            initial_state_triton, initial_state_indices, scale,
        )
    torch.cuda.synchronize()
    print("  ✓ Triton kernel compiled.")

    # ========== Define low-level kernel runner functions ==========
    # These only call compiled kernels, no Python overhead
    def run_cutedsl():
        for ri in range(run_iters):
            compiled_kernel(
                cu_seqlens_tensor,
                q_tensor_list[ri],
                k_tensor_list[ri],
                v_tensor_list[ri],
                a_tensor_list[ri],
                b_tensor_list[ri],
                A_log_tensor,
                dt_bias_tensor,
                h0_source_tensor,
                h0_indices_tensor,
                o_tensor,
                stream,
            )

    def run_triton():
        for ri in range(run_iters):
            _ = run_triton_kernel(
                A_log, dt_bias, q_list[ri], k_list[ri], v_list[ri], a_list[ri], b_list[ri],
                initial_state_triton, initial_state_indices, scale,
            )

    # ========== Pre-warmup kernels ==========
    print(f"\nPre-warming kernels...")
    with torch.cuda.stream(torch_stream_cutedsl):
        run_cutedsl()
    torch.cuda.synchronize()
    run_triton()
    torch.cuda.synchronize()
    print("  ✓ Pre-warming complete.")

    # ========== Capture CUDA Graphs ==========
    graph_triton = None
    graph_cutedsl = None

    if use_cudagraph:
        print(f"\nCapturing CUDA graphs...")

        # Capture Triton graph
        try:
            graph_triton = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_triton):
                run_triton()
            print("  ✓ Triton graph captured")
        except Exception as e:
            print(f"  ✗ Triton graph failed: {e}")
            graph_triton = None

        # Capture CuTe DSL graph (must use the same stream as the kernel)
        try:
            graph_cutedsl = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_cutedsl, stream=torch_stream_cutedsl):
                run_cutedsl()
            torch.cuda.synchronize()
            print("  ✓ CuTe DSL graph captured")
        except Exception as e:
            print(f"  ✗ CuTe DSL graph failed: {e}")
            graph_cutedsl = None

        # Enforce fairness: if one fails, disable both
        if graph_triton is not None and graph_cutedsl is not None:
            print("✓ Both kernels will use CUDA graph replay")
        elif graph_triton is not None or graph_cutedsl is not None:
            print("⚠ Mixed mode detected - enforcing direct call for fair comparison")
            graph_triton = graph_cutedsl = None
        else:
            print("✗ CUDA graph capture failed for both kernels, using direct calls")

    torch.cuda.synchronize()

    # ========== Warmup (alternating) ==========
    print(f"\nWarmup {warmup_iters} iterations (alternating)...")
    for _ in range(warmup_iters):
        if graph_cutedsl:
            graph_cutedsl.replay()
        else:
            with torch.cuda.stream(torch_stream_cutedsl):
                run_cutedsl()
        torch.cuda.synchronize()
        if evict_buf is not None:
            evict_caches(evict_buf)

        if graph_triton:
            graph_triton.replay()
        else:
            run_triton()
        torch.cuda.synchronize()
        if evict_buf is not None:
            evict_caches(evict_buf)

    torch.cuda.synchronize()

    # ========== Benchmark with per-iteration timing ==========
    print(f"\nBenchmarking {bench_iters} iterations...")
    triton_times = []
    cutedsl_times = []

    # Setup profiler if enabled
    profiler = None
    trace_file = None
    if profile:
        os.makedirs(trace_dir, exist_ok=True)
        trace_file = os.path.join(trace_dir, f"trace_gdn_B{B}.json")
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        profiler.__enter__()

    for i in range(bench_iters):
        # Time Triton
        if evict_buf is not None:
            evict_caches(evict_buf)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.profiler.record_function(f"Triton_iter{i}") if profile else nullcontext():
            if graph_triton:
                graph_triton.replay()
            else:
                run_triton()
        end.record()
        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

        # Time CuTe DSL (must use the same stream as the kernel)
        if evict_buf is not None:
            evict_caches(evict_buf)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(torch_stream_cutedsl):
            start.record()
            with torch.profiler.record_function(f"CuTeDSL_iter{i}") if profile else nullcontext():
                if graph_cutedsl:
                    graph_cutedsl.replay()
                else:
                    run_cutedsl()
            end.record()
        torch.cuda.synchronize()
        cutedsl_times.append(start.elapsed_time(end))

        # Print progress every iteration for visibility
        if (i + 1) % max(1, bench_iters // 5) == 0 or i == 0:
            print(
                f"  Iter {i+1}/{bench_iters}: "
                f"Triton={triton_times[-1]/run_iters*1000:.3f}μs, "
                f"CuTe DSL={cutedsl_times[-1]/run_iters*1000:.3f}μs"
            )

    # Export profiler trace if enabled
    if profiler is not None:
        profiler.__exit__(None, None, None)
        profiler.export_chrome_trace(trace_file)
        print(f"\n  ✓ Profiler trace saved to: {trace_file}")

    # Convert to numpy for statistics (ms -> μs, divide by run_iters for per-kernel time)
    triton_times = np.array(triton_times) / run_iters * 1000
    cutedsl_times = np.array(cutedsl_times) / run_iters * 1000

    triton_mean = triton_times.mean()
    triton_std = triton_times.std()
    cutedsl_mean = cutedsl_times.mean()
    cutedsl_std = cutedsl_times.std()

    # Print CUDA event timing results
    print_performance_table(B, kernel_type, triton_mean, triton_std, cutedsl_mean, cutedsl_std)

    result = {
        'B': B,
        'kernel_type': kernel_type,
        'triton_mean': triton_mean,
        'triton_std': triton_std,
        'cutedsl_mean': cutedsl_mean,
        'cutedsl_std': cutedsl_std,
        'speedup': triton_mean / cutedsl_mean,
        'trace_file': trace_file,
        'use_cudagraph': graph_triton is not None,
    }

    # Parse trace file and print trace-based timing
    if trace_file is not None and os.path.exists(trace_file):
        trace_results = parse_trace_file(trace_file)
        trace_summary = print_trace_summary(trace_results, B, kernel_type)
        if trace_summary is not None:
            result['trace_triton_mean'] = trace_summary['triton_mean']
            result['trace_triton_std'] = trace_summary['triton_std']
            result['trace_cutedsl_mean'] = trace_summary['cutedsl_mean']
            result['trace_cutedsl_std'] = trace_summary['cutedsl_std']
            result['trace_speedup'] = trace_summary['speedup']

    return result


def print_final_summary(all_results: list):
    """Print final summary table for all benchmark results."""
    if not all_results:
        return

    has_trace = any('trace_triton_mean' in r for r in all_results)
    has_cudagraph = any(r.get('use_cudagraph', False) for r in all_results)

    print("\n\n" + "=" * 120)
    print("  FINAL BENCHMARK SUMMARY")
    print("=" * 120)

    # Header
    header = f"  {'B':<6} {'Type':<12}{'Graph':<6}"
    header += f"{'Triton (μs)':<18}{'CuTe DSL (μs)':<18}{'Speedup':<10}"
    if has_trace:
        header += f"{'Trace Triton':<15}{'Trace CuTe':<15}{'Trace Spd':<10}"
    print(header)
    print("-" * 120)

    for r in all_results:
        cudagraph_str = "Yes" if r.get('use_cudagraph', False) else "No"
        row = f"  {r['B']:<6} {r['kernel_type']:<12}{cudagraph_str:<6}"
        row += f"{r['triton_mean']:.3f}±{r['triton_std']:.3f}".ljust(18)
        row += f"{r['cutedsl_mean']:.3f}±{r['cutedsl_std']:.3f}".ljust(18)
        row += f"{r['speedup']:.2f}x".ljust(10)

        if has_trace and 'trace_triton_mean' in r:
            row += f"{r['trace_triton_mean']:.3f}±{r['trace_triton_std']:.3f}".ljust(15)
            row += f"{r['trace_cutedsl_mean']:.3f}±{r['trace_cutedsl_std']:.3f}".ljust(15)
            row += f"{r['trace_speedup']:.2f}x".ljust(10)

        print(row)

    print("=" * 120)

    # Overall statistics
    print("\nOverall Statistics (CUDA Event Timing):")
    speedups = [r['speedup'] for r in all_results]
    print(f"  Speedup: min={min(speedups):.2f}x, avg={np.mean(speedups):.2f}x, max={max(speedups):.2f}x")

    if has_trace:
        trace_speedups = [r['trace_speedup'] for r in all_results if 'trace_speedup' in r]
        if trace_speedups:
            print("\nOverall Statistics (Trace-Based Timing):")
            print(f"  Speedup: min={min(trace_speedups):.2f}x, avg={np.mean(trace_speedups):.2f}x, max={max(trace_speedups):.2f}x")

    print("=" * 120 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test CuTe DSL GDN kernel for precision and performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pytest tests
  pytest test_cutedsl_gdn.py -v -k performance

  # Run standalone benchmark with CUDA Graph (recommended)
  python test_cutedsl_gdn.py --bench --B 16 128 --iters 3 --run-iters 100

  # Run benchmark with profiler + CUDA Graph
  python test_cutedsl_gdn.py --bench --B 16 128 --profile --iters 3 --run-iters 100

  # Run without CUDA Graph (for NCU/nsight profiling)
  python test_cutedsl_gdn.py --bench --B 16 128 --no-cudagraph --iters 3 --run-iters 100

  # Run precision tests only
  python test_cutedsl_gdn.py --precision --B 16 128
""",
    )
    parser.add_argument(
        "--bench", action="store_true",
        help="Run standalone benchmark (not pytest)"
    )
    parser.add_argument(
        "--precision", action="store_true",
        help="Run precision check only"
    )
    parser.add_argument(
        "--B", type=int, nargs="+", default=[16, 128],
        help="Batch sizes to test (default: 16 128)"
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable torch profiler and save trace files"
    )
    parser.add_argument(
        "--trace-dir", type=str, default="traces",
        help="Directory to save trace files (default: traces)"
    )
    parser.add_argument(
        "--evict-mb", type=int, default=128,
        help="L2 cache eviction buffer size in MB (0 to disable, default: 128)"
    )
    parser.add_argument(
        "--run-iters", type=int, default=10,
        help="Kernel runs per benchmark iteration to amortize launch overhead (default: 10)"
    )
    parser.add_argument(
        "--no-cudagraph", action="store_true",
        help="Disable CUDA graph for kernel replay (required for NCU/profilers)"
    )

    args = parser.parse_args()

    if args.bench:
        # Run standalone benchmark
        print(f"\nCuTe DSL GDN Benchmark")
        print(f"Batch sizes: {args.B}")
        print(f"Iterations: warmup={args.warmup}, bench={args.iters}, run_iters={args.run_iters}")
        print(f"CUDA Graph: {'disabled' if args.no_cudagraph else 'enabled'}")
        print(f"Profiler: {'enabled' if args.profile else 'disabled'}")
        if args.profile:
            print(f"Trace directory: {args.trace_dir}")

        all_results = []
        for B in args.B:
            result = run_benchmark(
                B=B,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
                run_iters=args.run_iters,
                profile=args.profile,
                trace_dir=args.trace_dir,
                evict_mb=args.evict_mb,
                use_cudagraph=not args.no_cudagraph,
            )
            all_results.append(result)

        print_final_summary(all_results)

    elif args.precision:
        # Run precision check
        print(f"\nCuTe DSL GDN Precision Check")
        print(f"Batch sizes: {args.B}")

        for B in args.B:
            # Manually call precision test logic
            torch.manual_seed(2025)
            T, H, K, V = 1, 16, 128, 128
            HV = 32
            scale = K**-0.5
            softplus_beta = 1.0
            softplus_threshold = 20.0

            A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
            dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
            a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
            b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
            q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
            initial_state_indices = torch.arange(B, dtype=torch.int32, device="cuda")

            initial_state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
            initial_state_triton = initial_state_cutedsl.clone().reshape(-1).contiguous()

            # Pre-compile
            _ = cutedsl_gdn_module.cutedsl_fused_sigmoid_gating_delta_rule_update(
                A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
                initial_state_source=initial_state_cutedsl.clone(),
                initial_state_indices=initial_state_indices,
                scale=scale, use_qk_l2norm_in_kernel=True,
                softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
            )
            torch.cuda.synchronize()

            # Reset and run
            initial_state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
            initial_state_triton = initial_state_cutedsl.clone().reshape(-1).contiguous()

            out_cutedsl = cutedsl_gdn_module.cutedsl_fused_sigmoid_gating_delta_rule_update(
                A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
                initial_state_source=initial_state_cutedsl,
                initial_state_indices=initial_state_indices,
                scale=scale, use_qk_l2norm_in_kernel=True,
                softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
            )
            torch.cuda.synchronize()

            out_triton = run_triton_kernel(
                A_log, dt_bias, q, k, v, a, b,
                initial_state_triton,
                initial_state_indices,  # Same as CuTe DSL: arange(B)
                scale, softplus_beta, softplus_threshold,
            )

            metrics = comprehensive_precision_check(out_triton, out_cutedsl)
            kernel_type = "SmallBatch" if B < 32 else "BigBatch"
            print_precision_table(metrics, B, kernel_type)

    else:
        # Run pytest by default
        pytest.main([__file__, "-v"])
