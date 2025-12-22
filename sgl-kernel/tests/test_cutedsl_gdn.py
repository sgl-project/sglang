"""
Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN).

This test compares the CuTe DSL implementation against the Triton reference
implementation for both precision and performance.

Batch sizes tested:
- B=16: Uses small batch kernel (B < 32 threshold)
- B=128: Uses big batch kernel (B >= 32 threshold)
"""

import importlib.util
import os
import sys

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
        initial_state_indices=torch.arange(B + 1, dtype=torch.int32, device="cuda"),
        scale=scale,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )

    # Precision check
    metrics = comprehensive_precision_check(
        out_triton, out_cutedsl, name=f"GDN Triton vs CuTe DSL (B={B})"
    )

    # Print metrics for debugging
    print(f"\nPrecision Check: B={B}")
    print(f"  Max abs diff: {metrics['max_abs_diff']:.6e}")
    print(f"  Mean abs diff: {metrics['mean_abs_diff']:.6e}")
    print(f"  Pass rate (≤1e-3): {metrics['pass_rate_1e3']:.2f}%")
    print(f"  Pass rate (≤1e-2): {metrics['pass_rate_1e2']:.2f}%")
    print(f"  Pass rate (≤1e-1): {metrics['pass_rate_1e1']:.2f}%")
    print(f"  Fail rate (>1e-1): {metrics['fail_rate_1e1']:.2f}%")

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
    """Benchmark CuTe DSL GDN kernel against Triton reference."""
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
    initial_state_indices = torch.arange(B, dtype=torch.int32, device="cuda")
    initial_state_indices_triton = torch.arange(B + 1, dtype=torch.int32, device="cuda")

    initial_state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    initial_state_triton = initial_state_cutedsl.clone().reshape(-1).contiguous()

    # Warmup both kernels
    for _ in range(warmup_iters):
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
        _ = run_triton_kernel(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            initial_state_triton.clone(),
            initial_state_indices_triton,
            scale,
        )
    torch.cuda.synchronize()

    # Benchmark CuTe DSL
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(bench_iters):
        _ = cutedsl_gdn_module.cutedsl_fused_sigmoid_gating_delta_rule_update(
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
    end_event.record()
    torch.cuda.synchronize()
    cutedsl_time_ms = start_event.elapsed_time(end_event) / bench_iters

    # Benchmark Triton
    start_event.record()
    for _ in range(bench_iters):
        _ = run_triton_kernel(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            initial_state_triton,
            initial_state_indices_triton,
            scale,
        )
    end_event.record()
    torch.cuda.synchronize()
    triton_time_ms = start_event.elapsed_time(end_event) / bench_iters

    # Convert to microseconds
    cutedsl_time_us = cutedsl_time_ms * 1000
    triton_time_us = triton_time_ms * 1000
    speedup = triton_time_us / cutedsl_time_us

    kernel_type = "SmallBatch" if B < 32 else "BigBatch"
    print(f"\nPerformance: B={B} ({kernel_type})")
    print(f"  Triton:    {triton_time_us:.3f} us")
    print(f"  CuTe DSL:  {cutedsl_time_us:.3f} us")
    print(f"  Speedup:   {speedup:.2f}x")

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
    initial_state_indices = torch.arange(B + 1, dtype=torch.int32, device="cuda")

    # Create initial state tensors
    initial_state_base = torch.randn(
        B + 1, HV, K, V, dtype=torch.float32, device="cuda"
    )
    initial_state_cutedsl = initial_state_base.clone()
    initial_state_triton = initial_state_base.clone()[:B].reshape(-1).contiguous()

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
        initial_state_indices=initial_state_indices[:B],
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
        initial_state_indices=initial_state_indices,
        scale=K**-0.5,
    )

    # Precision check
    metrics = comprehensive_precision_check(out_triton, out_cutedsl)

    print(f"\nHigh-Level API Test: B={B}")
    print(f"  Max abs diff: {metrics['max_abs_diff']:.6e}")
    print(f"  Pass rate (≤1e-2): {metrics['pass_rate_1e2']:.2f}%")

    assert metrics["passed"], f"High-level API precision check failed for B={B}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
