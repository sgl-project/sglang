"""Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN)."""

import atexit
import numpy as np
import pytest
import torch

import logging
logging.basicConfig(level=logging.INFO)

# Global result collector for summary table
_results = {
    "precision": [],
    "performance": [],
    "mtp_precision": [],
    "mtp_performance": [],
}

def print_summary_tables():
    """Print summary table at the end of test session."""
    if not any(_results.values()):
        return
    
    print("\n" + "="*80)
    print("SUMMARY TABLES")
    print("="*80)
    
    # Precision results
    if _results["precision"]:
        print("\nPrecision Test Results:")
        print(" B(BS)| kernel                    | dtype | max_diff | mean_diff | fail_rate | check")
        print("-" * 85)
        for B, kernel_name, dtype_str, max_diff, mean_diff, fail_rate, has_nan in sorted(_results["precision"]):
            check_str = "ok" if not has_nan and fail_rate < 1.0 else "fail"
            print(f" {B:>4} | {kernel_name:24} | {dtype_str:5} | {max_diff:8.2e} | {mean_diff:9.2e} | {fail_rate:8.2f}% | {check_str}")
    
    # Performance results
    if _results["performance"]:
        print("\nPerformance Test Results (decode mode):")
        print(" B(BS)| Triton (us) | CuTeDSL (us) | speedup | check")
        print("-" * 70)
        for B, dtype_str, triton_mean, triton_std, cutedsl_mean, cutedsl_std, speedup, cutedsl_fused_mean, cutedsl_fused_std, speedup_fused in sorted(_results["performance"]):
            if cutedsl_mean is not None:
                print(f" {B:>4} | {triton_mean:6.2f}±{triton_std:5.2f} | {cutedsl_mean:7.2f}±{cutedsl_std:5.2f} | {speedup:6.2f}x | ok [cutedsl_gdn, {dtype_str}]")
            print(f" {B:>4} | {triton_mean:6.2f}±{triton_std:5.2f} | {cutedsl_fused_mean:7.2f}±{cutedsl_fused_std:5.2f} | {speedu_fused:6.2f}x | ok [cutedsl_gdn_transpose, {dtype_str}]")
    
    # MTP Precision results
    if _results["mtp_precision"]:
        print("\nMTP Precision Test Results:")
        print(" T(BS)| max_diff | mean_diff | fail_rate | check")
        print("-" * 60)
        for T, max_diff, mean_diff, fail_rate, has_nan in sorted(_results["mtp_precision"]):
            check_str = "ok" if not has_nan and fail_rate < 1.0 else "fail"
            print(f" {T:>4} | {max_diff:8.2e} | {mean_diff:9.2e} | {fail_rate:8.2f}% | {check_str}")
    
    # MTP Performance results
    if _results["mtp_performance"]:
        print("\nMTP Performance Test Results:")
        print(" T(BS)| Triton (us) | CuTeDSL (us) | speedup | check")
        print("-" * 70)
        for T, dtype_str, triton_mean, triton_std, cutedsl_mean, cutedsl_std, speedup in sorted(_results["mtp_performance"]):
            print(f" {T:>4} | {triton_mean:6.2f}±{triton_std:5.2f} | {cutedsl_mean:7.2f}±{cutedsl_std:5.2f} | {speedup:6.2f}x | ok [cutedsl_gdn_transpose, {dtype_str}]")
    
    print("="*80 + "\n")

# Register the summary printer to run at exit
atexit.register(print_summary_tables)

try:
    import cuda.bindings.driver as cuda_driver
    import cutlass  # noqa: F401
    from cutlass.cute.runtime import from_dlpack

    from sglang.jit_kernel import cutedsl_gdn
    from sglang.jit_kernel import cutedsl_gdn_transpose

    CUTEDSL_AVAILABLE = True
except ImportError:
    CUTEDSL_AVAILABLE = False
    cutedsl_gdn = None

try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update,
    )

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def run_triton_kernel(A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale, cu_seqlens=None):
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )

def run_triton_mtp_kernel(q, k, v, g, b, initial_state, indices, intermediate_state, intermediate_indices, scale, cu_seqlens=None, NUM_DRAFT_TOKENS=3):
    return fused_recurrent_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        disable_state_update=True,
        intermediate_states_buffer=intermediate_state,
        intermediate_state_indices=intermediate_indices,
        cache_steps=NUM_DRAFT_TOKENS,
        retrieve_parent_token=None,
    )

def run_fused_recurrent_kernel(A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale, stream=None, cu_seqlens=None):
    return cutedsl_gdn_transpose.cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        stream=stream
    )

def run_fused_recurrent_mtp_kernel(q, k, v, g, b, initial_state, indices, intermediate_state, intermediate_indices, scale, stream=None, cu_seqlens=None, NUM_DRAFT_TOKENS=3):
    return cutedsl_gdn_transpose.cutedsl_fused_recurrent_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        disable_state_update=True,
        intermediate_states_buffer=intermediate_state,
        intermediate_state_indices=intermediate_indices,
        cache_steps=NUM_DRAFT_TOKENS,
        retrieve_parent_token=None,
        stream=stream
    )


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("B", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
def test_cutedsl_gdn_precision(B: int, state_dtype: torch.dtype):
    """Test precision of CuTe DSL GDN kernel against Triton reference."""
    torch.manual_seed(2025)
    T, H, K, V, HV = 1, 16, 128, 128, 32
    scale = K**-0.5

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(B, dtype=torch.int32, device="cuda")
    state_cutedsl = torch.randn(B, HV, K, V, dtype=state_dtype, device="cuda")
    state_triton = state_cutedsl.clone().reshape(-1).contiguous()
    state_cutedsl_fused_recurrent = state_cutedsl.clone().transpose(-2, -1).contiguous().transpose(-2, -1)

    # Warmup compilation
    if state_dtype == torch.float32:
        _ = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log, dt_bias, q, k, v, a, b, state_cutedsl.clone(), indices, scale=scale
        )
        torch.cuda.synchronize()

    _ = run_fused_recurrent_kernel(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl_fused_recurrent.clone(), indices, scale=scale
    )
    torch.cuda.synchronize()

    # Fresh state for actual test
    state_cutedsl = torch.randn(B, HV, K, V, dtype=state_dtype, device="cuda")
    state_triton = state_cutedsl.clone().reshape(-1).contiguous()
    state_cutedsl_fused_recurrent = state_cutedsl.clone().transpose(-2, -1).contiguous().transpose(-2, -1)

    if state_dtype == torch.float32:
        out_cutedsl = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log, dt_bias, q, k, v, a, b, state_cutedsl, indices, scale=scale
        )
    else:
        out_cutedsl = None
    
    out_triton = run_triton_kernel(
        A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale
    )
    out_cutedsl_fused_recurrent = run_fused_recurrent_kernel(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl_fused_recurrent, indices, scale=scale
    )

    # Check precision: diff > 0.1 must be < 1% of elements
    if state_dtype == torch.float32:
        abs_diff = (out_triton.float() - out_cutedsl.float()).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        fail_rate = (abs_diff > 0.1).float().mean().item() * 100
        has_nan = torch.isnan(out_cutedsl).any() or torch.isinf(out_cutedsl).any()

        kernel_type = "SmallBatch" if B < 32 else "LargeBatch"
        dtype_str = "f32" if state_dtype == torch.float32 else "bf16"
        print(
            f"\n  B={B} ({kernel_type}): max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, fail_rate={fail_rate:.2f}%"
        )

        # Collect results for summary table (cutedsl_gdn)
        _results["precision"].append((B, "cutedsl_gdn", dtype_str, max_diff, mean_diff, fail_rate, has_nan))

        assert not has_nan, "Output contains NaN/Inf"
        assert fail_rate < 1.0, f"Fail rate {fail_rate:.2f}% >= 1%"

    # Check precision for fused_recurrent kernel: diff > 0.1 must be < 1% of elements
    abs_diff_fused = (out_triton.float() - out_cutedsl_fused_recurrent.float()).abs()
    max_diff_fused = abs_diff_fused.max().item()
    mean_diff_fused = abs_diff_fused.mean().item()
    fail_rate_fused = (abs_diff_fused > 0.1).float().mean().item() * 100
    has_nan_fused = torch.isnan(out_cutedsl_fused_recurrent).any() or torch.isinf(out_cutedsl_fused_recurrent).any()

    kernel_type = "SmallBatch" if B < 32 else "LargeBatch"
    dtype_str = "f32" if state_dtype == torch.float32 else "bf16"

    # Collect results for summary table (cutedsl_fused_recurrent)
    _results["precision"].append((B, "cutedsl_gdn_transpose", dtype_str, max_diff_fused, mean_diff_fused, fail_rate_fused, has_nan_fused))

    assert not has_nan_fused, "Fused recurrent output contains NaN/Inf"
    assert fail_rate_fused < 1.0, f"Fused recurrent fail rate {fail_rate_fused:.2f}% >= 1%"


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("B", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
def test_cutedsl_gdn_performance(B: int, state_dtype: torch.dtype):
    """Benchmark CuTe DSL GDN kernel against Triton reference."""
    torch.manual_seed(2025)
    T, H, K, V, HV = 1, 32, 128, 128, 32
    N = B
    scale = K**-0.5
    is_varlen = True
    warmup, bench_iters, run_iters = 10, 100, 100

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(N, dtype=torch.int32, device="cuda")
    state_cutedsl = torch.randn(N, HV, K, V, dtype=state_dtype, device="cuda")
    state_triton = state_cutedsl.reshape(-1).contiguous()
    state_cutedsl_fused_recurrent = state_cutedsl.clone().transpose(-2, -1).contiguous().transpose(-2, -1)
    cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device="cuda")
    o_cutedsl = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")

    # Prepare tensors for multiple runs
    q_list, k_list, v_list, a_list, b_list = [], [], [], [], []
    q_tensor_list, k_tensor_list, v_tensor_list, a_tensor_list, b_tensor_list = (
        [],
        [],
        [],
        [],
        [],
    )
    q_triton, k_triton, v_triton, a_triton, b_triton = [], [], [], [], []

    for ri in range(run_iters):
        torch.manual_seed(2025 + ri)
        q_i = torch.randn(1, N, H, K, dtype=torch.bfloat16, device="cuda")
        k_i = torch.randn(1, N, H, K, dtype=torch.bfloat16, device="cuda")
        v_i = torch.randn(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
        a_i = torch.randn(N, HV, dtype=torch.bfloat16, device="cuda")
        b_i = torch.randn(N, HV, dtype=torch.bfloat16, device="cuda")

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
        q_triton.append(q_i.transpose(0, 1).contiguous())
        k_triton.append(k_i.transpose(0, 1).contiguous())
        v_triton.append(v_i.transpose(0, 1).contiguous())
        a_triton.append(a_i.unsqueeze(1).contiguous())
        b_triton.append(b_i.unsqueeze(1).contiguous())

    A_log_t = from_dlpack(A_log, assumed_align=16)
    dt_bias_t = from_dlpack(dt_bias, assumed_align=16)
    h0_t = from_dlpack(state_cutedsl, assumed_align=16)
    idx_t = from_dlpack(indices, assumed_align=16)
    o_t = from_dlpack(o_cutedsl, assumed_align=16)
    cu_t = from_dlpack(cu_seqlens, assumed_align=16)

    torch_stream = torch.cuda.Stream()
    stream = cuda_driver.CUstream(torch_stream.cuda_stream)
    torch_stream_fused_recurrent = torch.cuda.Stream()
    stream_fused_recurrent = cuda_driver.CUstream(torch_stream_fused_recurrent.cuda_stream)

    # Compile kernels
    compiled = None
    if state_dtype == torch.float32:
        compiled = cutedsl_gdn._get_compiled_kernel(N, H, HV, K, V, N, N < 32, is_varlen)
        torch.cuda.synchronize()

    for ri in range(run_iters):
        _ = run_triton_kernel(
            A_log,
            dt_bias,
            q_triton[ri],
            k_triton[ri],
            v_triton[ri],
            a_triton[ri],
            b_triton[ri],
            state_triton,
            indices,
            scale,
        )
    torch.cuda.synchronize()

    def run_cutedsl():
        for ri in range(run_iters):
            compiled(
                cu_t,
                q_tensor_list[ri],
                k_tensor_list[ri],
                v_tensor_list[ri],
                a_tensor_list[ri],
                b_tensor_list[ri],
                A_log_t,
                dt_bias_t,
                h0_t,
                idx_t,
                o_t,
                stream,
            )

    def run_triton():
        for ri in range(run_iters):
            _ = run_triton_kernel(
                A_log,
                dt_bias,
                q_triton[ri],
                k_triton[ri],
                v_triton[ri],
                a_triton[ri],
                b_triton[ri],
                state_triton,
                indices,
                scale,
            )

    def run_fused_recurrent():
        for ri in range(run_iters):
            _ = run_fused_recurrent_kernel(
                A_log, 
                dt_bias, 
                q_triton[ri], 
                k_triton[ri], 
                v_triton[ri], 
                a_triton[ri], 
                b_triton[ri], 
                state_cutedsl_fused_recurrent, 
                indices, 
                scale=scale,
                stream=stream_fused_recurrent
            )

    # Warmup
    if state_dtype == torch.float32:
        with torch.cuda.stream(torch_stream):
            run_cutedsl()
        torch.cuda.synchronize()
    run_triton()
    torch.cuda.synchronize()
    with torch.cuda.stream(torch_stream_fused_recurrent):
        run_fused_recurrent()
    torch.cuda.synchronize()

    # Capture CUDA graphs
    graph_triton = torch.cuda.CUDAGraph()
    graph_cutedsl = None if state_dtype in [torch.bfloat16, torch.float16] else torch.cuda.CUDAGraph()
    graph_cutedsl_fused_recurrent = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph_triton):
            run_triton()
        if state_dtype == torch.float32:
            with torch.cuda.graph(graph_cutedsl, stream=torch_stream):
                run_cutedsl()
        with torch.cuda.graph(graph_cutedsl_fused_recurrent, stream=torch_stream_fused_recurrent):
            run_fused_recurrent()
        torch.cuda.synchronize()
    except Exception:
        graph_triton = graph_cutedsl = graph_cutedsl_fused_recurrent = None

    # Warmup with graphs
    for _ in range(warmup):
        if state_dtype == torch.float32:
            if graph_cutedsl:
                graph_cutedsl.replay()
            else:
                with torch.cuda.stream(torch_stream):
                    run_cutedsl()
            torch.cuda.synchronize()

        if graph_triton:
            graph_triton.replay()
        else:
            run_triton()
        torch.cuda.synchronize()

        if graph_cutedsl_fused_recurrent:
            graph_cutedsl_fused_recurrent.replay()
        else:
            with torch.cuda.stream(torch_stream_fused_recurrent):
                run_fused_recurrent()
        torch.cuda.synchronize()

    # Benchmark
    triton_times, cutedsl_times, cutedsl_fused_recurrent_times = [], [], []
    for _ in range(bench_iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        if graph_triton:
            graph_triton.replay()
        else:
            run_triton()
        end.record()
        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

        if state_dtype == torch.float32:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            with torch.cuda.stream(torch_stream):
                start.record()
                if graph_cutedsl:
                    graph_cutedsl.replay()
                else:
                    run_cutedsl()
                end.record()
            torch.cuda.synchronize()
            cutedsl_times.append(start.elapsed_time(end))

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        with torch.cuda.stream(torch_stream_fused_recurrent):
            start.record()
            if graph_cutedsl_fused_recurrent:
                graph_cutedsl_fused_recurrent.replay()
            else:
                run_fused_recurrent()
            end.record()
        torch.cuda.synchronize()
        cutedsl_fused_recurrent_times.append(start.elapsed_time(end))

    triton_mean = np.mean(triton_times) / run_iters * 1000
    triton_std = np.std(triton_times) / run_iters * 1000
    cutedsl_fused_recurrent_mean = np.mean(cutedsl_fused_recurrent_times) / run_iters * 1000
    cutedsl_fused_recurrent_std = np.std(cutedsl_fused_recurrent_times) / run_iters * 1000
    speedup_fused_recurrent = triton_mean / cutedsl_fused_recurrent_mean

    if state_dtype == torch.float32:
        cutedsl_mean = np.mean(cutedsl_times) / run_iters * 1000
        cutedsl_std = np.std(cutedsl_times) / run_iters * 1000
        speedup = triton_mean / cutedsl_mean

    kernel_type = "SmallBatch" if B < 32 else "LargeBatch"
    dtype_str = "f32" if state_dtype == torch.float32 else "bf16"

    # Collect results for summary table
    cutedsl_mean_val = cutedsl_mean if state_dtype == torch.float32 else None
    cutedsl_std_val = cutedsl_std if state_dtype == torch.float32 else None
    speedup_val = speedup if state_dtype == torch.float32 else None
    _results["performance"].append((
        B, dtype_str, triton_mean, triton_std,
        cutedsl_mean_val, cutedsl_std_val, speedup_val,
        cutedsl_fused_recurrent_mean, cutedsl_fused_recurrent_std, speedup_fused_recurrent
    ))

    if state_dtype == torch.float32:
        min_speedup = 1.0 if B < 32 else 1.15
        assert speedup >= min_speedup, f"Speedup {speedup:.2f}x < {min_speedup}x for B={B}"


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("T", [4, 8, 16, 32, 48])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
def test_cutedsl_gdn_mtp_precision(T: int, state_dtype: torch.dtype):
    """Test precision of CuTe DSL GDN kernel against Triton reference."""
    torch.manual_seed(2025)
    NUM_DRAFT_TOKENS = 3
    B, H, K, V, HV = 1, 16, 128, 128, 32
    scale = K**-0.5
    original_T = T  # Save original T value for summary table
    T = T * NUM_DRAFT_TOKENS

    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(T // NUM_DRAFT_TOKENS, dtype=torch.int32, device="cuda")
    state_triton = torch.randn(T // NUM_DRAFT_TOKENS, HV, K, V, dtype=state_dtype, device="cuda")
    state_cutedsl_fused_recurrent = state_triton.clone().transpose(-2, -1).contiguous().transpose(-2, -1)

    cu_seqlens = torch.ones(T // NUM_DRAFT_TOKENS, dtype=torch.int32).cuda()
    cu_seqlens = torch.cumsum(cu_seqlens, dim=0)
    zero_tensor = torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    cu_seqlens = torch.cat([zero_tensor, cu_seqlens]).to(torch.int32)
    cu_seqlens *= NUM_DRAFT_TOKENS

    intermediate_state = torch.zeros(T // NUM_DRAFT_TOKENS + 1, NUM_DRAFT_TOKENS, HV, K, V, device="cuda", dtype=state_dtype)
    intermediate_state_indices = torch.arange(T // NUM_DRAFT_TOKENS, device="cuda", dtype=torch.int32)
    intermediate_state_fused_recurrent = intermediate_state.clone().transpose(-2, -1).contiguous().transpose(-2, -1)

    _ = run_fused_recurrent_mtp_kernel(
        q, k, v, g, b, state_cutedsl_fused_recurrent.clone(), indices, intermediate_state_fused_recurrent, intermediate_state_indices, scale=scale, cu_seqlens=cu_seqlens, NUM_DRAFT_TOKENS=NUM_DRAFT_TOKENS
    )
    torch.cuda.synchronize()

    # Fresh state for actual test
    state_triton = torch.randn(T // NUM_DRAFT_TOKENS, HV, K, V, dtype=state_dtype, device="cuda")
    state_cutedsl_fused_recurrent = state_triton.clone().transpose(-2, -1).contiguous().transpose(-2, -1)

    intermediate_state = torch.zeros(T // NUM_DRAFT_TOKENS + 1, NUM_DRAFT_TOKENS, HV, K, V, device="cuda", dtype=state_dtype)
    intermediate_state_indices = torch.arange(T // NUM_DRAFT_TOKENS, device="cuda", dtype=torch.int32)
    intermediate_state_fused_recurrent = intermediate_state.clone().transpose(-2, -1).contiguous().transpose(-2, -1)
    
    out_triton = run_triton_mtp_kernel(
        q, k, v, g, b, state_triton, indices, intermediate_state, intermediate_state_indices, scale, cu_seqlens=cu_seqlens, NUM_DRAFT_TOKENS=NUM_DRAFT_TOKENS
    )
    out_cutedsl_fused_recurrent = run_fused_recurrent_mtp_kernel(
        q, k, v, g, b, state_cutedsl_fused_recurrent, indices, intermediate_state_fused_recurrent, intermediate_state_indices, scale=scale, cu_seqlens=cu_seqlens, NUM_DRAFT_TOKENS=NUM_DRAFT_TOKENS
    )

    # Check precision for fused_recurrent kernel: diff > 0.1 must be < 1% of elements
    abs_diff_fused = (out_triton.float() - out_cutedsl_fused_recurrent.float()).abs()
    max_diff_fused = abs_diff_fused.max().item()
    mean_diff_fused = abs_diff_fused.mean().item()
    fail_rate_fused = (abs_diff_fused > 0.1).float().mean().item() * 100
    has_nan_fused = torch.isnan(out_cutedsl_fused_recurrent).any() or torch.isinf(out_cutedsl_fused_recurrent).any()

    kernel_type = "SmallBatch" if B < 32 else "LargeBatch"
    dtype_str = "f32" if state_dtype == torch.float32 else "bf16"

    # Collect results for summary table
    _results["mtp_precision"].append((original_T, max_diff_fused, mean_diff_fused, fail_rate_fused, has_nan_fused))

    assert not has_nan_fused, "Fused recurrent output contains NaN/Inf"
    assert fail_rate_fused < 1.0, f"Fused recurrent fail rate {fail_rate_fused:.2f}% >= 1%"


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("T", [4, 8, 16, 32, 48])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
def test_cutedsl_gdn_mtp_performance(T: int, state_dtype: torch.dtype):
    """Benchmark CuTe DSL fused recurrent kernel against Triton reference for MTP (Multi-Token Prediction) scenario."""
    torch.manual_seed(2025)
    NUM_DRAFT_TOKENS = 3
    B, H, K, V, HV = 1, 16, 128, 128, 16
    scale = K**-0.5
    num_sequences = T  # Number of sequences
    total_tokens = T * NUM_DRAFT_TOKENS
    warmup, bench_iters, run_iters = 10, 100, 10

    # Construct cu_seqlens for MTP scenario: [0, 3, 6, 9, ...] for T sequences
    cu_seqlens = torch.arange(0, (num_sequences + 1) * NUM_DRAFT_TOKENS, NUM_DRAFT_TOKENS, dtype=torch.int32, device="cuda")
    
    indices = torch.arange(num_sequences, dtype=torch.int32, device="cuda")
    intermediate_state_indices = torch.arange(num_sequences, dtype=torch.int32, device="cuda")

    # Prepare tensors for multiple runs
    q_list, k_list, v_list, g_list, b_list = [], [], [], [], []
    state_triton_list, state_cutedsl_fused_recurrent_list = [], []
    intermediate_state_list, intermediate_state_fused_recurrent_list = [], []

    for ri in range(run_iters):
        torch.manual_seed(2025 + ri)
        q_i = torch.randn(B, total_tokens, H, K, dtype=torch.bfloat16, device="cuda")
        k_i = torch.randn(B, total_tokens, H, K, dtype=torch.bfloat16, device="cuda")
        v_i = torch.randn(B, total_tokens, HV, V, dtype=torch.bfloat16, device="cuda")
        g_i = torch.randn(B, total_tokens, HV, dtype=torch.bfloat16, device="cuda")
        b_i = torch.randn(B, total_tokens, HV, dtype=torch.bfloat16, device="cuda")
        
        state_triton_i = torch.randn(num_sequences, HV, K, V, dtype=state_dtype, device="cuda")
        state_cutedsl_fused_recurrent_i = state_triton_i.clone().transpose(-2, -1).contiguous().transpose(-2, -1)
        
        intermediate_state_i = torch.zeros(num_sequences + 1, NUM_DRAFT_TOKENS, HV, K, V, device="cuda", dtype=state_dtype)
        intermediate_state_fused_recurrent_i = intermediate_state_i.clone().transpose(-2, -1).contiguous().transpose(-2, -1)

        q_list.append(q_i)
        k_list.append(k_i)
        v_list.append(v_i)
        g_list.append(g_i)
        b_list.append(b_i)
        state_triton_list.append(state_triton_i)
        state_cutedsl_fused_recurrent_list.append(state_cutedsl_fused_recurrent_i)
        intermediate_state_list.append(intermediate_state_i)
        intermediate_state_fused_recurrent_list.append(intermediate_state_fused_recurrent_i)

    torch_stream_fused_recurrent = torch.cuda.Stream()
    stream_fused_recurrent = cuda_driver.CUstream(torch_stream_fused_recurrent.cuda_stream)

    # Warmup compilation
    _ = run_triton_mtp_kernel(
        q_list[0], k_list[0], v_list[0], g_list[0], b_list[0],
        state_triton_list[0], indices, intermediate_state_list[0], intermediate_state_indices,
        scale, cu_seqlens=cu_seqlens, NUM_DRAFT_TOKENS=NUM_DRAFT_TOKENS
    )
    torch.cuda.synchronize()

    _ = run_fused_recurrent_mtp_kernel(
        q_list[0], k_list[0], v_list[0], g_list[0], b_list[0],
        state_cutedsl_fused_recurrent_list[0], indices,
        intermediate_state_fused_recurrent_list[0], intermediate_state_indices,
        scale=scale, cu_seqlens=cu_seqlens, NUM_DRAFT_TOKENS=NUM_DRAFT_TOKENS,
        stream=stream_fused_recurrent
    )
    torch.cuda.synchronize()

    def run_triton():
        for ri in range(run_iters):
            _ = run_triton_mtp_kernel(
                q_list[ri], k_list[ri], v_list[ri], g_list[ri], b_list[ri],
                state_triton_list[ri], indices,
                intermediate_state_list[ri], intermediate_state_indices,
                scale, cu_seqlens=cu_seqlens, NUM_DRAFT_TOKENS=NUM_DRAFT_TOKENS
            )

    def run_fused_recurrent():
        for ri in range(run_iters):
            _ = run_fused_recurrent_mtp_kernel(
                q_list[ri], k_list[ri], v_list[ri], g_list[ri], b_list[ri],
                state_cutedsl_fused_recurrent_list[ri], indices,
                intermediate_state_fused_recurrent_list[ri], intermediate_state_indices,
                scale=scale, cu_seqlens=cu_seqlens, NUM_DRAFT_TOKENS=NUM_DRAFT_TOKENS,
                stream=stream_fused_recurrent
            )

    # Warmup
    run_triton()
    torch.cuda.synchronize()
    with torch.cuda.stream(torch_stream_fused_recurrent):
        run_fused_recurrent()
    torch.cuda.synchronize()

    # Capture CUDA graphs
    graph_triton = torch.cuda.CUDAGraph()
    graph_cutedsl_fused_recurrent = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph_triton):
            run_triton()
        with torch.cuda.graph(graph_cutedsl_fused_recurrent, stream=torch_stream_fused_recurrent):
            run_fused_recurrent()
        torch.cuda.synchronize()
    except Exception:
        graph_triton = graph_cutedsl_fused_recurrent = None

    # Warmup with graphs
    for _ in range(warmup):
        if graph_triton:
            graph_triton.replay()
        else:
            run_triton()
        torch.cuda.synchronize()

        if graph_cutedsl_fused_recurrent:
            graph_cutedsl_fused_recurrent.replay()
        else:
            with torch.cuda.stream(torch_stream_fused_recurrent):
                run_fused_recurrent()
        torch.cuda.synchronize()

    # Benchmark
    triton_times, cutedsl_fused_recurrent_times = [], []
    for _ in range(bench_iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if graph_triton:
            graph_triton.replay()
        else:
            run_triton()
        end.record()
        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(torch_stream_fused_recurrent):
            start.record()
            if graph_cutedsl_fused_recurrent:
                graph_cutedsl_fused_recurrent.replay()
            else:
                run_fused_recurrent()
            end.record()
        torch.cuda.synchronize()
        cutedsl_fused_recurrent_times.append(start.elapsed_time(end))

    triton_mean = np.mean(triton_times) / run_iters * 1000
    triton_std = np.std(triton_times) / run_iters * 1000
    cutedsl_fused_recurrent_mean = np.mean(cutedsl_fused_recurrent_times) / run_iters * 1000
    cutedsl_fused_recurrent_std = np.std(cutedsl_fused_recurrent_times) / run_iters * 1000
    speedup_fused_recurrent = triton_mean / cutedsl_fused_recurrent_mean

    dtype_str = "f32" if state_dtype == torch.float32 else "bf16"
    # Collect results for summary table
    _results["mtp_performance"].append((
        T, dtype_str, triton_mean, triton_std,
        cutedsl_fused_recurrent_mean, cutedsl_fused_recurrent_std, speedup_fused_recurrent
    ))

    min_speedup = 1.0
    assert speedup_fused_recurrent >= min_speedup, f"Speedup {speedup_fused_recurrent:.2f}x < {min_speedup}x for T={T}"

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print summary table at the end of test session."""
    print_summary_tables()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
