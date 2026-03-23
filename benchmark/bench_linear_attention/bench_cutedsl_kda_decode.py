"""
Benchmark & Correctness: CuTe DSL KDA Decode vs Triton KDA Decode.

Benchmark modes:
  - Default (wall-clock): Measures Python dispatch + GPU execution.
    CuTe DSL has ~120us Python overhead from from_dlpack/mark_layout_dynamic
    that Triton doesn't have. This does NOT reflect production performance.
  - --cuda-graph: Uses CUDA graph capture/replay, eliminating ALL Python dispatch
    overhead for both backends. This reflects actual production performance
    where SGLang uses CUDA graphs for decode.

Usage:
    python bench_cutedsl_kda_decode.py --mode bench --cuda-graph   # recommended
    python bench_cutedsl_kda_decode.py --mode bench                # wall-clock
    python bench_cutedsl_kda_decode.py --mode correctness
    python bench_cutedsl_kda_decode.py                             # all
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import triton

from sglang.jit_kernel.cutedsl_kda import (
    cutedsl_fused_sigmoid_gating_kda_update,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)


def make_inputs(B, H, HV, K, V, pool_size, device, dtype, layout, seed=42):
    torch.manual_seed(seed)
    assert K == 128
    assert V % 16 == 0 and V % 32 == 0

    if layout == "varlen":
        q = torch.randn(1, B, H, K, device=device, dtype=dtype)
        k = torch.randn(1, B, H, K, device=device, dtype=dtype)
        v = torch.randn(1, B, HV, V, device=device, dtype=dtype)
        a = torch.randn(B, HV, K, device=device, dtype=dtype)
        b = torch.randn(B, HV, device=device, dtype=dtype)
        cu_seqlens = torch.arange(B + 1, device=device, dtype=torch.int32)
    else:
        q = torch.randn(B, 1, H, K, device=device, dtype=dtype)
        k = torch.randn(B, 1, H, K, device=device, dtype=dtype)
        v = torch.randn(B, 1, HV, V, device=device, dtype=dtype)
        a = torch.randn(B, 1, HV, K, device=device, dtype=dtype)
        b = torch.randn(B, 1, HV, device=device, dtype=dtype)
        cu_seqlens = torch.arange(B + 1, device=device, dtype=torch.int32)

    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, K, device=device, dtype=dtype)
    ssm_states_cute = (
        torch.randn(pool_size, HV, K, V, device=device, dtype=torch.float32) * 0.1
    )
    ssm_states_triton = ssm_states_cute.transpose(-1, -2).contiguous()
    cache_indices = torch.arange(B, device=device, dtype=torch.int32)

    return dict(
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        pool_size=pool_size,
        layout=layout,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states_cute=ssm_states_cute,
        ssm_states_triton=ssm_states_triton,
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
    )


def run_baseline(inp):
    state = inp["ssm_states_triton"].clone()
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        a=inp["a"],
        b=inp["b"],
        initial_state_source=state,
        initial_state_indices=inp["cache_indices"],
        cu_seqlens=inp["cu_seqlens"] if inp["layout"] == "varlen" else None,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
    )
    return o, state


def run_cutedsl(inp):
    state = inp["ssm_states_cute"].clone()
    o = cutedsl_fused_sigmoid_gating_kda_update(
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        a=inp["a"],
        b=inp["b"],
        initial_state_source=state,
        initial_state_indices=inp["cache_indices"],
        cu_seqlens=inp["cu_seqlens"],
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )
    return o, state


def check_correctness(B, H, HV, K, V, pool_size, device, dtype, layout, seed=42):
    tag = f"layout={layout:<6} B={B:>2} H={H:>2} HV={HV:>2} K={K:>3} V={V:>3} pool={pool_size:>4}"
    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype, layout, seed=seed)
    o_ref, st_ref = run_baseline(inp)
    o_cute, st_cute = run_cutedsl(inp)

    atol = 3e-2 if dtype != torch.float32 else 1e-4
    rtol = 2e-2 if dtype != torch.float32 else 1e-4
    output_ok = state_ok = True
    out_diff = st_diff = 0.0

    try:
        torch.testing.assert_close(o_cute.float(), o_ref.float(), atol=atol, rtol=rtol)
    except AssertionError:
        output_ok = False
        out_diff = (o_cute.float() - o_ref.float()).abs().max().item()

    st_cute_vk = st_cute.transpose(-1, -2).contiguous()
    valid_slots = inp["cache_indices"][inp["cache_indices"] >= 0].unique()
    try:
        torch.testing.assert_close(
            st_cute_vk[valid_slots].float(),
            st_ref[valid_slots].float(),
            atol=atol,
            rtol=rtol,
        )
    except AssertionError:
        state_ok = False
        st_diff = (
            (st_cute_vk[valid_slots].float() - st_ref[valid_slots].float())
            .abs()
            .max()
            .item()
        )

    passed = output_ok and state_ok
    if passed:
        print(f"  [PASS] {tag}")
    else:
        details = []
        if not output_ok:
            details.append(f"output max_diff={out_diff:.6f}")
        if not state_ok:
            details.append(f"state max_diff={st_diff:.6f}")
        print(f"  [FAIL] {tag}  ({', '.join(details)})")
    return passed


def bench_cuda_graph(fn, warmup=50, rep=200):
    """Capture fn into CUDA graph, then benchmark replay with CUDA events."""
    # Warmup + trigger JIT
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()

    # Warmup replay
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    # Timed replay
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        g.replay()
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    return times_ms[len(times_ms) // 2]


def bench_shape(B, H, HV, K, V, pool_size, device, dtype, layout, use_cuda_graph=False):
    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype, layout)
    state_ref = inp["ssm_states_triton"].clone()
    state_cute = inp["ssm_states_cute"].clone()

    def fn_baseline():
        fused_sigmoid_gating_delta_rule_update(
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            a=inp["a"],
            b=inp["b"],
            initial_state_source=state_ref,
            initial_state_indices=inp["cache_indices"],
            cu_seqlens=inp["cu_seqlens"] if inp["layout"] == "varlen" else None,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
        )

    def fn_cutedsl():
        cutedsl_fused_sigmoid_gating_kda_update(
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            a=inp["a"],
            b=inp["b"],
            initial_state_source=state_cute,
            initial_state_indices=inp["cache_indices"],
            cu_seqlens=inp["cu_seqlens"],
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    # Warmup
    for _ in range(10):
        fn_baseline()
        fn_cutedsl()
    torch.cuda.synchronize()

    if use_cuda_graph:
        ms_baseline = bench_cuda_graph(fn_baseline)
        ms_cutedsl = bench_cuda_graph(fn_cutedsl)
    else:
        quantiles = [0.5, 0.2, 0.8]
        try:
            ms_baseline, _, _ = triton.testing.do_bench(
                fn_baseline, quantiles=quantiles, warmup=50, rep=200
            )
            ms_cutedsl, _, _ = triton.testing.do_bench(
                fn_cutedsl, quantiles=quantiles, warmup=50, rep=200
            )
        except Exception:
            torch.cuda.synchronize()
            N = 200
            start = time.perf_counter()
            for _ in range(N):
                fn_baseline()
            torch.cuda.synchronize()
            ms_baseline = (time.perf_counter() - start) / N * 1000
            start = time.perf_counter()
            for _ in range(N):
                fn_cutedsl()
            torch.cuda.synchronize()
            ms_cutedsl = (time.perf_counter() - start) / N * 1000

    speedup = ms_baseline / ms_cutedsl if ms_cutedsl > 0 else float("inf")
    delta_us = (ms_baseline - ms_cutedsl) * 1000
    print(
        f"  {layout:>6}  {B:>5}  {H:>3}  {HV:>3}  {K:>3}  {V:>3} | "
        f"{ms_baseline * 1000:>10.1f} | {ms_cutedsl * 1000:>10.1f} | "
        f"{speedup:>7.2f}x | {delta_us:>+9.1f}"
    )


def run_correctness(device, dtype):
    print("=" * 78)
    print("Correctness: Triton KDA Decode vs CuTe DSL KDA Decode")
    print("=" * 78)
    shapes = [
        ("dense", 1, 8, 16, 128, 128, 32),
        ("dense", 4, 8, 16, 128, 128, 32),
        ("dense", 32, 8, 16, 128, 128, 128),
        ("dense", 64, 8, 16, 128, 128, 128),
        ("varlen", 4, 8, 16, 128, 128, 32),
        ("varlen", 16, 8, 16, 128, 128, 64),
        ("varlen", 32, 8, 16, 128, 128, 128),
        ("varlen", 64, 8, 16, 128, 128, 128),
        ("varlen", 1, 16, 32, 128, 128, 32),
        ("varlen", 32, 16, 32, 128, 128, 128),
        ("varlen", 64, 16, 16, 128, 128, 128),
    ]
    all_pass = True
    for layout, B, H, HV, K, V, pool_size in shapes:
        if not check_correctness(B, H, HV, K, V, pool_size, device, dtype, layout):
            all_pass = False

    print("\n  PAD_SLOT_ID test (indices with -1):")
    inp = make_inputs(32, 8, 16, 128, 128, 128, device, dtype, layout="varlen")
    inp["cache_indices"][::5] = -1
    st_ref_before = inp["ssm_states_triton"].clone()
    st_cute_before = inp["ssm_states_cute"].clone()
    o_ref, st_ref = run_baseline(inp)
    o_cute, st_cute = run_cutedsl(inp)
    st_cute_vk = st_cute.transpose(-1, -2).contiguous()
    st_cute_before_vk = st_cute_before.transpose(-1, -2).contiguous()
    valid_token_pos = (inp["cache_indices"] >= 0).nonzero(as_tuple=False).squeeze(-1)
    valid_slots = inp["cache_indices"][inp["cache_indices"] >= 0].unique()
    atol = 3e-2 if dtype != torch.float32 else 1e-4
    rtol = 2e-2 if dtype != torch.float32 else 1e-4
    pad_ok = True
    details = []
    try:
        torch.testing.assert_close(
            o_cute[:, valid_token_pos].float(),
            o_ref[:, valid_token_pos].float(),
            atol=atol,
            rtol=rtol,
        )
    except AssertionError:
        pad_ok = False
        details.append(
            f"valid_output max_diff={(o_cute[:, valid_token_pos].float() - o_ref[:, valid_token_pos].float()).abs().max().item():.6f}"
        )
    try:
        torch.testing.assert_close(
            st_cute_vk[valid_slots].float(),
            st_ref[valid_slots].float(),
            atol=atol,
            rtol=rtol,
        )
    except AssertionError:
        pad_ok = False
        details.append(
            f"valid_state max_diff={(st_cute_vk[valid_slots].float() - st_ref[valid_slots].float()).abs().max().item():.6f}"
        )
    untouched = torch.ones(inp["pool_size"], device=device, dtype=torch.bool)
    untouched[valid_slots] = False
    try:
        torch.testing.assert_close(
            st_ref[untouched].float(),
            st_ref_before[untouched].float(),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            st_cute_vk[untouched].float(),
            st_cute_before_vk[untouched].float(),
            atol=0.0,
            rtol=0.0,
        )
    except AssertionError:
        pad_ok = False
        details.append("untouched_slots modified")
    print(
        f"  [{'PASS' if pad_ok else 'FAIL'}] PAD_SLOT_ID=-1 handling"
        + (f" ({', '.join(details)})" if details else " (valid outputs/states only)")
    )
    if not pad_ok:
        all_pass = False
    print()
    print("ALL PASSED." if all_pass else "SOME FAILED.")
    return all_pass


def run_benchmark(device, dtype, use_cuda_graph=False):
    print()
    print("=" * 92)
    if use_cuda_graph:
        print(
            "Benchmark: Triton vs CuTe DSL KDA Decode  [CUDA Graph replay — production mode]"
        )
    else:
        print(
            "Benchmark: Triton vs CuTe DSL KDA Decode  [Wall-clock — includes Python dispatch]"
        )
        print(
            "  NOTE: CuTe DSL shows ~120us Python overhead here that vanishes with CUDA graphs."
        )
    print("=" * 92)

    configs = [
        ("dense", 1, 8, 16),
        ("dense", 4, 8, 16),
        ("dense", 32, 8, 16),
        ("dense", 64, 8, 16),
        ("varlen", 1, 8, 16),
        ("varlen", 4, 8, 16),
        ("varlen", 8, 8, 16),
        ("varlen", 16, 8, 16),
        ("varlen", 32, 8, 16),
        ("varlen", 64, 8, 16),
        ("varlen", 128, 8, 16),
        ("varlen", 32, 16, 32),
        ("varlen", 64, 16, 16),
    ]
    K, V, pool_size = 128, 128, 512
    print(f"  Config: K={K}, V={V}, pool_size={pool_size}, dtype={dtype}")
    print(
        f"  {'layout':>6}  {'B':>5}  {'H':>3}  {'HV':>3}  {'K':>3}  {'V':>3} | {'triton (us)':>12} | {'cutedsl (us)':>13} | {'speedup':>8} | {'delta (us)':>11}"
    )
    print("  " + "-" * 82)
    for layout, B, H, HV in configs:
        bench_shape(
            B,
            H,
            HV,
            K,
            V,
            max(pool_size, B + 16),
            device,
            dtype,
            layout,
            use_cuda_graph=use_cuda_graph,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["all", "correctness", "bench"], default="all"
    )
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        default=False,
        help="Use CUDA graph capture/replay (recommended, matches production)",
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  (SM {cap[0]}{cap[1]})")

    if args.mode in ("all", "correctness"):
        all_pass = run_correctness(device, dtype)
        if not all_pass and args.mode == "all":
            print("\nSkipping benchmark due to correctness failures.")
            return 1

    if args.mode in ("all", "bench"):
        if args.cuda_graph:
            run_benchmark(device, dtype, use_cuda_graph=True)
        else:
            run_benchmark(device, dtype, use_cuda_graph=False)
            run_benchmark(device, dtype, use_cuda_graph=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
