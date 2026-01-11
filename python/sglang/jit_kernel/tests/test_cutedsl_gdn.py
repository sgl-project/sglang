"""Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN)."""

import argparse

import numpy as np
import pytest
import torch

try:
    import cutlass  # noqa: F401

    CUTEDSL_AVAILABLE = True
except ImportError:
    CUTEDSL_AVAILABLE = False

try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    TRITON_KERNEL_AVAILABLE = True
except ImportError:
    TRITON_KERNEL_AVAILABLE = False

# Import cutedsl_gdn module from parent directory
cutedsl_gdn_module = None
if CUTEDSL_AVAILABLE:
    try:
        from sglang.jit_kernel import cutedsl_gdn as cutedsl_gdn_module
    except ImportError as e:
        print(f"Warning: Failed to import cutedsl_gdn module: {e}")
        cutedsl_gdn_module = None


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
    status = "✓ PASSED" if metrics["passed"] else "✗ FAILED"
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
    print(
        f"  {'CuTe DSL':<15} {cutedsl_mean:>12.3f} {cutedsl_std:>10.3f} {triton_mean/cutedsl_mean:>11.2f}x"
    )
    print("-" * 70)
    print(f"  {faster} is {ratio:.2f}x faster")
    print("=" * 70 + "\n")


def evict_caches(buf: torch.Tensor):
    """Evict GPU caches by reading a large buffer."""
    _ = buf.sum()
    torch.cuda.synchronize()


def comprehensive_precision_check(out_ref: torch.Tensor, out_test: torch.Tensor):
    """Precision comparison. Pass: diff > 1e-1 must be < 1% of elements."""
    ref = out_ref.float().flatten()
    test = out_test.float().flatten()
    total_elements = ref.numel()

    abs_diff = (ref - test).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    pass_rate_1e3 = (abs_diff <= 1e-3).float().mean().item() * 100
    pass_rate_1e2 = (abs_diff <= 1e-2).float().mean().item() * 100
    pass_rate_1e1 = (abs_diff <= 1e-1).float().mean().item() * 100
    fail_rate_1e1 = 100.0 - pass_rate_1e1

    nan_ref = torch.isnan(ref).sum().item()
    nan_test = torch.isnan(test).sum().item()
    inf_ref = torch.isinf(ref).sum().item()
    inf_test = torch.isinf(test).sum().item()
    has_nan_inf = (nan_ref + nan_test + inf_ref + inf_test) > 0

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

    initial_state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    initial_state_triton = initial_state_cutedsl.clone().reshape(-1).contiguous()

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
        scale=scale,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )

    metrics = comprehensive_precision_check(out_triton, out_cutedsl)
    kernel_type = "SmallBatch" if B < 32 else "LargeBatch"
    print_precision_table(metrics, B, kernel_type)

    assert metrics["passed"], (
        f"Precision check failed for B={B}:\n"
        f"  fail_rate_1e1={metrics['fail_rate_1e1']:.2f}% (should be < 1%)\n"
        f"  has_nan_inf={metrics['has_nan_inf']}"
    )


@pytest.mark.skipif(
    not CUTEDSL_AVAILABLE, reason="CuTe DSL (cutlass/cute) not available"
)
@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton GDN kernel (sglang) not available"
)
@pytest.mark.skipif(cutedsl_gdn_module is None, reason="cutedsl_gdn module not loaded")
@pytest.mark.parametrize("B", [1, 128])
def test_cutedsl_gdn_performance(B: int):
    """Benchmark CuTe DSL GDN kernel against Triton reference."""
    try:
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import from_dlpack
    except ImportError as e:
        pytest.skip(f"CuTe DSL dependencies not available: {e}")

    torch.manual_seed(2025)
    T, H, K, V = 1, 16, 128, 128
    HV = 32
    N = B

    scale = K**-0.5
    use_small_batch = N < 32
    is_varlen_decode = True  # Test varlen version, consistent with end-to-end model
    warmup_iters = 10
    bench_iters = 100
    run_iters = 10

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    initial_state_indices = torch.arange(N, dtype=torch.int32, device="cuda")
    pool_size = N
    initial_state_cutedsl = torch.randn(
        pool_size, HV, K, V, dtype=torch.float32, device="cuda"
    )
    initial_state_triton = initial_state_cutedsl.reshape(-1).contiguous()
    cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device="cuda")
    h0_source = initial_state_cutedsl
    if is_varlen_decode:
        o_cutedsl = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
    else:
        o_cutedsl = torch.zeros(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")

    q_list, k_list, v_list, a_list, b_list = [], [], [], [], []
    q_tensor_list, k_tensor_list, v_tensor_list, a_tensor_list, b_tensor_list = (
        [],
        [],
        [],
        [],
        [],
    )
    for ri in range(run_iters):
        torch.manual_seed(2025 + ri)
        if is_varlen_decode:
            # Varlen decode format: (1, N, H, K), a/b are 2D (N, HV)
            q_i = torch.randn(1, N, H, K, dtype=torch.bfloat16, device="cuda")
            k_i = torch.randn(1, N, H, K, dtype=torch.bfloat16, device="cuda")
            v_i = torch.randn(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
            a_i = torch.randn(N, HV, dtype=torch.bfloat16, device="cuda")
            b_i = torch.randn(N, HV, dtype=torch.bfloat16, device="cuda")
        else:
            q_i = torch.randn(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
            k_i = torch.randn(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
            v_i = torch.randn(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")
            a_i = torch.randn(N, 1, HV, dtype=torch.bfloat16, device="cuda")
            b_i = torch.randn(N, 1, HV, dtype=torch.bfloat16, device="cuda")
        q_list.append(q_i)
        k_list.append(k_i)
        v_list.append(v_i)
        a_list.append(a_i)
        b_list.append(b_i)
        q_tensor_list.append(from_dlpack(q_i, assumed_align=16))
        k_tensor_list.append(from_dlpack(k_i, assumed_align=16))
        v_tensor_list.append(from_dlpack(v_i, assumed_align=16))
        a_tensor_list.append(from_dlpack(a_i, assumed_align=16))
        b_tensor_list.append(from_dlpack(b_i, assumed_align=16))

    A_log_tensor = from_dlpack(A_log, assumed_align=16)
    dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
    h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
    h0_indices_tensor = from_dlpack(initial_state_indices, assumed_align=16)
    o_tensor = from_dlpack(o_cutedsl, assumed_align=16)
    cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

    torch_stream_cutedsl = torch.cuda.Stream()
    stream = cuda_driver.CUstream(torch_stream_cutedsl.cuda_stream)
    evict_buf = torch.randn(32 * 1024 * 1024, dtype=torch.float32, device="cuda")

    print(f"\nPre-compiling kernels...")
    cutedsl_gdn_module._check_cutlass_available()
    compiled_kernel = cutedsl_gdn_module._get_compiled_kernel(
        N, H, HV, K, V, pool_size, use_small_batch, is_varlen_decode
    )
    torch.cuda.synchronize()
    print("  ✓ CuTe DSL kernel compiled.")

    for ri in range(run_iters):
        _ = run_triton_kernel(
            A_log,
            dt_bias,
            q_list[ri],
            k_list[ri],
            v_list[ri],
            a_list[ri],
            b_list[ri],
            initial_state_triton,
            initial_state_indices,
            scale,
        )
    torch.cuda.synchronize()
    print("  ✓ Triton kernel compiled.")

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
                A_log,
                dt_bias,
                q_list[ri],
                k_list[ri],
                v_list[ri],
                a_list[ri],
                b_list[ri],
                initial_state_triton,
                initial_state_indices,
                scale,
            )

    print(f"Pre-warming kernels...")
    with torch.cuda.stream(torch_stream_cutedsl):
        run_cutedsl()
    torch.cuda.synchronize()
    run_triton()
    torch.cuda.synchronize()
    print("  ✓ Pre-warming complete.")

    print(f"Capturing CUDA graphs...")
    graph_triton = None
    graph_cutedsl = None

    try:
        graph_triton = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_triton):
            run_triton()
        print("  ✓ Triton graph captured")
    except Exception as e:
        print(f"  ✗ Triton graph failed: {e}")
        graph_triton = None

    try:
        graph_cutedsl = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_cutedsl, stream=torch_stream_cutedsl):
            run_cutedsl()
        torch.cuda.synchronize()
        print("  ✓ CuTe DSL graph captured")
    except Exception as e:
        print(f"  ✗ CuTe DSL graph failed: {e}")
        graph_cutedsl = None

    use_cudagraph = graph_triton is not None and graph_cutedsl is not None
    if use_cudagraph:
        print("✓ Both kernels will use CUDA graph replay")
    else:
        print("⚠ CUDA graph not available, using direct calls")
        graph_triton = graph_cutedsl = None

    torch.cuda.synchronize()

    print(f"\nWarmup {warmup_iters} iterations (alternating)...")
    for _ in range(warmup_iters):
        if graph_cutedsl:
            graph_cutedsl.replay()
        else:
            with torch.cuda.stream(torch_stream_cutedsl):
                run_cutedsl()
        torch.cuda.synchronize()
        evict_caches(evict_buf)

        if graph_triton:
            graph_triton.replay()
        else:
            run_triton()
        torch.cuda.synchronize()
        evict_caches(evict_buf)

    torch.cuda.synchronize()

    print(f"Benchmarking {bench_iters} iterations...")
    triton_times = []
    cutedsl_times = []

    for i in range(bench_iters):
        evict_caches(evict_buf)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph_triton:
            graph_triton.replay()
        else:
            run_triton()
        end.record()
        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

        evict_caches(evict_buf)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(torch_stream_cutedsl):
            start.record()
            if graph_cutedsl:
                graph_cutedsl.replay()
            else:
                run_cutedsl()
            end.record()
        torch.cuda.synchronize()
        cutedsl_times.append(start.elapsed_time(end))

        if (i + 1) % 20 == 0:
            print(
                f"  Iter {i+1}/{bench_iters}: "
                f"Triton={triton_times[-1]/run_iters*1000:.3f}μs, "
                f"CuTe DSL={cutedsl_times[-1]/run_iters*1000:.3f}μs"
            )

    triton_times = np.array(triton_times) / run_iters * 1000
    cutedsl_times = np.array(cutedsl_times) / run_iters * 1000

    triton_mean = triton_times.mean()
    triton_std = triton_times.std()
    cutedsl_mean = cutedsl_times.mean()
    cutedsl_std = cutedsl_times.std()

    kernel_type = "SmallBatch" if B < 32 else "LargeBatch"
    print_performance_table(
        B, kernel_type, triton_mean, triton_std, cutedsl_mean, cutedsl_std
    )

    speedup = triton_mean / cutedsl_mean
    min_speedup = 1.0 if B < 32 else 1.15
    assert (
        speedup >= min_speedup
    ), f"CuTe DSL speedup {speedup:.2f}x < {min_speedup}x for B={B} ({kernel_type})"


def run_benchmark(
    B: int,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    run_iters: int = 10,
    evict_mb: int = 128,
    use_cudagraph: bool = True,
):
    """Run standalone benchmark."""
    if cutedsl_gdn_module is None:
        raise RuntimeError("CuTe DSL module not available")
    if not TRITON_KERNEL_AVAILABLE:
        raise RuntimeError("Triton kernel not available")

    try:
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import from_dlpack
    except ImportError as e:
        raise RuntimeError(f"CuTe DSL dependencies not available: {e}")

    torch.manual_seed(2025)
    T, H, K, V = 1, 16, 128, 128
    HV = 32
    N = B

    scale = K**-0.5
    use_small_batch = N < 32
    is_varlen_decode = True  # Test varlen version, consistent with end-to-end model

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    initial_state_indices = torch.arange(N, dtype=torch.int32, device="cuda")
    pool_size = N
    initial_state_cutedsl = torch.randn(
        pool_size, HV, K, V, dtype=torch.float32, device="cuda"
    )
    initial_state_triton = initial_state_cutedsl.reshape(-1).contiguous()
    cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device="cuda")
    h0_source = initial_state_cutedsl  # (pool_size, HV, K, V)
    if is_varlen_decode:
        o_cutedsl = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
    else:
        o_cutedsl = torch.zeros(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")

    q_list, k_list, v_list, a_list, b_list = [], [], [], [], []
    q_tensor_list, k_tensor_list, v_tensor_list, a_tensor_list, b_tensor_list = (
        [],
        [],
        [],
        [],
        [],
    )
    for ri in range(run_iters):
        torch.manual_seed(2025 + ri)
        if is_varlen_decode:
            # Varlen decode format: (1, N, H, K), a/b are 2D (N, HV)
            q_i = torch.randn(1, N, H, K, dtype=torch.bfloat16, device="cuda")
            k_i = torch.randn(1, N, H, K, dtype=torch.bfloat16, device="cuda")
            v_i = torch.randn(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
            a_i = torch.randn(N, HV, dtype=torch.bfloat16, device="cuda")
            b_i = torch.randn(N, HV, dtype=torch.bfloat16, device="cuda")
        else:
            q_i = torch.randn(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
            k_i = torch.randn(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
            v_i = torch.randn(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")
            a_i = torch.randn(N, 1, HV, dtype=torch.bfloat16, device="cuda")
            b_i = torch.randn(N, 1, HV, dtype=torch.bfloat16, device="cuda")
        q_list.append(q_i)
        k_list.append(k_i)
        v_list.append(v_i)
        a_list.append(a_i)
        b_list.append(b_i)
        q_tensor_list.append(from_dlpack(q_i, assumed_align=16))
        k_tensor_list.append(from_dlpack(k_i, assumed_align=16))
        v_tensor_list.append(from_dlpack(v_i, assumed_align=16))
        a_tensor_list.append(from_dlpack(a_i, assumed_align=16))
        b_tensor_list.append(from_dlpack(b_i, assumed_align=16))

    A_log_tensor = from_dlpack(A_log, assumed_align=16)
    dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
    h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
    h0_indices_tensor = from_dlpack(initial_state_indices, assumed_align=16)
    o_tensor = from_dlpack(o_cutedsl, assumed_align=16)
    cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

    torch_stream_cutedsl = torch.cuda.Stream()
    stream = cuda_driver.CUstream(torch_stream_cutedsl.cuda_stream)
    evict_buf = (
        torch.randn(evict_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        if evict_mb > 0
        else None
    )

    kernel_type = "SmallBatch" if use_small_batch else "LargeBatch"
    print(
        f"\nBENCHMARK: B={B} ({kernel_type}), warmup={warmup_iters}, samples={bench_iters}, runs={run_iters}"
    )

    print(f"\nPre-compiling kernels...")
    cutedsl_gdn_module._check_cutlass_available()
    compiled_kernel = cutedsl_gdn_module._get_compiled_kernel(
        N, H, HV, K, V, pool_size, use_small_batch, is_varlen_decode
    )
    torch.cuda.synchronize()
    print("  ✓ CuTe DSL kernel compiled.")

    for ri in range(run_iters):
        _ = run_triton_kernel(
            A_log,
            dt_bias,
            q_list[ri],
            k_list[ri],
            v_list[ri],
            a_list[ri],
            b_list[ri],
            initial_state_triton,
            initial_state_indices,
            scale,
        )
    torch.cuda.synchronize()
    print("  ✓ Triton kernel compiled.")

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
                A_log,
                dt_bias,
                q_list[ri],
                k_list[ri],
                v_list[ri],
                a_list[ri],
                b_list[ri],
                initial_state_triton,
                initial_state_indices,
                scale,
            )

    print(f"\nPre-warming kernels...")
    with torch.cuda.stream(torch_stream_cutedsl):
        run_cutedsl()
    torch.cuda.synchronize()
    run_triton()
    torch.cuda.synchronize()
    print("  ✓ Pre-warming complete.")

    graph_triton = None
    graph_cutedsl = None

    if use_cudagraph:
        print(f"\nCapturing CUDA graphs...")
        try:
            graph_triton = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_triton):
                run_triton()
            print("  ✓ Triton graph captured")
        except Exception as e:
            print(f"  ✗ Triton graph failed: {e}")
            graph_triton = None

        try:
            graph_cutedsl = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_cutedsl, stream=torch_stream_cutedsl):
                run_cutedsl()
            torch.cuda.synchronize()
            print("  ✓ CuTe DSL graph captured")
        except Exception as e:
            print(f"  ✗ CuTe DSL graph failed: {e}")
            graph_cutedsl = None

        if graph_triton is not None and graph_cutedsl is not None:
            print("✓ Both kernels will use CUDA graph replay")
        elif graph_triton is not None or graph_cutedsl is not None:
            print("⚠ Mixed mode detected - enforcing direct call for fair comparison")
            graph_triton = graph_cutedsl = None
        else:
            print("✗ CUDA graph capture failed for both kernels, using direct calls")

    torch.cuda.synchronize()

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

    print(f"\nBenchmarking {bench_iters} iterations...")
    triton_times = []
    cutedsl_times = []

    for i in range(bench_iters):
        if evict_buf is not None:
            evict_caches(evict_buf)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph_triton:
            graph_triton.replay()
        else:
            run_triton()
        end.record()
        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

        if evict_buf is not None:
            evict_caches(evict_buf)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(torch_stream_cutedsl):
            start.record()
            if graph_cutedsl:
                graph_cutedsl.replay()
            else:
                run_cutedsl()
            end.record()
        torch.cuda.synchronize()
        cutedsl_times.append(start.elapsed_time(end))

        if (i + 1) % max(1, bench_iters // 5) == 0 or i == 0:
            print(
                f"  Iter {i+1}/{bench_iters}: "
                f"Triton={triton_times[-1]/run_iters*1000:.3f}μs, "
                f"CuTe DSL={cutedsl_times[-1]/run_iters*1000:.3f}μs"
            )

    triton_times = np.array(triton_times) / run_iters * 1000
    cutedsl_times = np.array(cutedsl_times) / run_iters * 1000

    triton_mean = triton_times.mean()
    triton_std = triton_times.std()
    cutedsl_mean = cutedsl_times.mean()
    cutedsl_std = cutedsl_times.std()

    print_performance_table(
        B, kernel_type, triton_mean, triton_std, cutedsl_mean, cutedsl_std
    )

    return {
        "B": B,
        "kernel_type": kernel_type,
        "triton_mean": triton_mean,
        "triton_std": triton_std,
        "cutedsl_mean": cutedsl_mean,
        "cutedsl_std": cutedsl_std,
        "speedup": triton_mean / cutedsl_mean,
    }


def print_final_summary(all_results: list):
    """Print final summary table for all benchmark results."""
    if not all_results:
        return

    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"  {'B':<6} {'Type':<12}{'Triton (μs)':<18}{'CuTe DSL (μs)':<18}{'Speedup':<10}"
    )
    print("-" * 70)

    for r in all_results:
        triton_str = f"{r['triton_mean']:.3f}±{r['triton_std']:.3f}"
        cutedsl_str = f"{r['cutedsl_mean']:.3f}±{r['cutedsl_std']:.3f}"
        print(
            f"  {r['B']:<6} {r['kernel_type']:<12}{triton_str:<18}{cutedsl_str:<18}{r['speedup']:.2f}x"
        )

    print("=" * 70)
    speedups = [r["speedup"] for r in all_results]
    print(
        f"  Avg speedup: {np.mean(speedups):.2f}x (min={min(speedups):.2f}x, max={max(speedups):.2f}x)"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test CuTe DSL GDN kernel for precision and performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all pytest tests
  python test_cutedsl_gdn.py

  # Run precision tests only
  python test_cutedsl_gdn.py --precision

  # Run standalone benchmark with CUDA Graph (recommended)
  python test_cutedsl_gdn.py --bench --B 1 128 --samples 100 --runs 10

  # Run without CUDA Graph (for NCU/nsight profiling)
  python test_cutedsl_gdn.py --bench --B 1 128 --no-cudagraph --samples 3 --runs 100
""",
    )
    parser.add_argument(
        "--bench", action="store_true", help="Run standalone benchmark (not pytest)"
    )
    parser.add_argument(
        "--precision", action="store_true", help="Run precision tests via pytest"
    )
    parser.add_argument(
        "--B",
        type=int,
        nargs="+",
        default=[1, 128],
        help="Batch sizes for --bench mode (default: 1 128)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of timing samples for statistics (default: 100)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--evict-mb",
        type=int,
        default=0,
        help="L2 cache eviction buffer size in MB (0 to disable, default: 128)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Kernel runs per sample to amortize launch overhead (default: 10)",
    )
    parser.add_argument(
        "--no-cudagraph",
        action="store_true",
        help="Disable CUDA graph for kernel replay (required for NCU/profilers)",
    )

    args = parser.parse_args()

    if args.bench:
        # Run standalone benchmark
        print(f"\nCuTe DSL GDN Benchmark")
        print(f"Batch sizes: {args.B}")
        print(
            f"Iterations: warmup={args.warmup}, samples={args.samples}, runs={args.runs}"
        )
        print(f"CUDA Graph: {'disabled' if args.no_cudagraph else 'enabled'}")

        all_results = []
        for B in args.B:
            result = run_benchmark(
                B=B,
                warmup_iters=args.warmup,
                bench_iters=args.samples,
                run_iters=args.runs,
                evict_mb=args.evict_mb,
                use_cudagraph=not args.no_cudagraph,
            )
            all_results.append(result)

        print_final_summary(all_results)

    elif args.precision:
        # Run precision tests via pytest
        pytest.main([__file__, "-v", "-k", "precision"])

    else:
        # Run pytest by default
        pytest.main([__file__, "-v"])
