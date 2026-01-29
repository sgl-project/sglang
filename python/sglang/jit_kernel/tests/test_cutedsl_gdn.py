"""Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN)."""

import numpy as np
import pytest
import torch

try:
    import cuda.bindings.driver as cuda_driver
    import cutlass  # noqa: F401
    from cutlass.cute.runtime import from_dlpack

    from sglang.jit_kernel import cutedsl_gdn

    CUTEDSL_AVAILABLE = True
except ImportError:
    CUTEDSL_AVAILABLE = False
    cutedsl_gdn = None

try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def run_triton_kernel(A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale):
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
        cu_seqlens=None,
    )


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("B", [16, 128])
def test_cutedsl_gdn_precision(B: int):
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
    state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.clone().reshape(-1).contiguous()

    # Warmup compilation
    _ = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl.clone(), indices, scale=scale
    )
    torch.cuda.synchronize()

    # Fresh state for actual test
    state_cutedsl = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.clone().reshape(-1).contiguous()

    out_cutedsl = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl, indices, scale=scale
    )
    out_triton = run_triton_kernel(
        A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale
    )

    # Check precision: diff > 0.1 must be < 1% of elements
    abs_diff = (out_triton.float() - out_cutedsl.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    fail_rate = (abs_diff > 0.1).float().mean().item() * 100
    has_nan = torch.isnan(out_cutedsl).any() or torch.isinf(out_cutedsl).any()

    kernel_type = "SmallBatch" if B < 32 else "LargeBatch"
    print(
        f"\n  B={B} ({kernel_type}): max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, fail_rate={fail_rate:.2f}%"
    )

    assert not has_nan, "Output contains NaN/Inf"
    assert fail_rate < 1.0, f"Fail rate {fail_rate:.2f}% >= 1%"


@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton kernel not available")
@pytest.mark.parametrize("B", [1, 128])
def test_cutedsl_gdn_performance(B: int):
    """Benchmark CuTe DSL GDN kernel against Triton reference."""
    torch.manual_seed(2025)
    T, H, K, V, HV = 1, 16, 128, 128, 32
    N = B
    scale = K**-0.5
    is_varlen = True
    warmup, bench_iters, run_iters = 10, 100, 10

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(N, dtype=torch.int32, device="cuda")
    state_cutedsl = torch.randn(N, HV, K, V, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.reshape(-1).contiguous()
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

    # Compile kernels
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

    # Warmup
    with torch.cuda.stream(torch_stream):
        run_cutedsl()
    torch.cuda.synchronize()
    run_triton()
    torch.cuda.synchronize()

    # Capture CUDA graphs
    graph_triton = torch.cuda.CUDAGraph()
    graph_cutedsl = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph_triton):
            run_triton()
        with torch.cuda.graph(graph_cutedsl, stream=torch_stream):
            run_cutedsl()
        torch.cuda.synchronize()
    except Exception:
        graph_triton = graph_cutedsl = None

    # Warmup with graphs
    for _ in range(warmup):
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

    # Benchmark
    triton_times, cutedsl_times = [], []
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

    triton_mean = np.mean(triton_times) / run_iters * 1000
    triton_std = np.std(triton_times) / run_iters * 1000
    cutedsl_mean = np.mean(cutedsl_times) / run_iters * 1000
    cutedsl_std = np.std(cutedsl_times) / run_iters * 1000
    speedup = triton_mean / cutedsl_mean

    kernel_type = "SmallBatch" if B < 32 else "LargeBatch"
    print(
        f"\n  B={B} ({kernel_type}): Triton={triton_mean:.2f}±{triton_std:.2f}μs, CuTeDSL={cutedsl_mean:.2f}±{cutedsl_std:.2f}μs, speedup={speedup:.2f}x"
    )

    min_speedup = 1.0 if B < 32 else 1.15
    assert speedup >= min_speedup, f"Speedup {speedup:.2f}x < {min_speedup}x for B={B}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
