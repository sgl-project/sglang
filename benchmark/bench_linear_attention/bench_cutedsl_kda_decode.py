"""Benchmark & Correctness: CuTe DSL KDA Decode vs Triton KDA Decode.

This benchmark assumes the production / Triton canonical state layout:
    ssm_states.shape == (pool_size, HV, V, K)

Both the Triton baseline and the CuTe DSL candidate operate directly on that VK
layout. No transpose is performed anywhere in the benchmark.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch
import triton

from sglang.jit_kernel.cutedsl_kda import cutedsl_fused_sigmoid_gating_kda_update
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.fla.kda import chunk_kda


def make_inputs(
    B: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    device: str,
    dtype: torch.dtype,
    layout: str,
    seed: int = 42,
):
    torch.manual_seed(seed)

    assert K == 128
    assert V % 16 == 0 and V % 32 == 0

    if layout == "varlen":
        q = torch.randn(1, B, H, K, device=device, dtype=dtype)
        k = torch.randn(1, B, H, K, device=device, dtype=dtype)
        v = torch.randn(1, B, HV, V, device=device, dtype=dtype)

        # decode params
        a = torch.randn(B, HV, K, device=device, dtype=dtype)
        b = torch.randn(B, HV, device=device, dtype=dtype)

        # prefill params for chunk_kda must keep batch dim = 1
        # chunk_kda requires g, beta, v to have the same head count as k (H),
        # matching the real KimiLinear model where num_heads == num_kv_heads.
        prefill_v = torch.randn(1, B, H, V, device=device, dtype=dtype)
        prefill_g = torch.randn(1, B, H, K, device=device, dtype=dtype)
        prefill_beta = torch.sigmoid(torch.randn(1, B, H, device=device, dtype=dtype))

        cu_seqlens = torch.arange(B + 1, device=device, dtype=torch.int32)

    elif layout == "dense":
        q = torch.randn(B, 1, H, K, device=device, dtype=dtype)
        k = torch.randn(B, 1, H, K, device=device, dtype=dtype)
        v = torch.randn(B, 1, HV, V, device=device, dtype=dtype)

        # decode params
        a = torch.randn(B, 1, HV, K, device=device, dtype=dtype)
        b = torch.randn(B, 1, HV, device=device, dtype=dtype)

        # prefill params for chunk_kda dense path
        # chunk_kda requires g, beta, v to have the same head count as k (H),
        # matching the real KimiLinear model where num_heads == num_kv_heads.
        prefill_v = torch.randn(B, 1, H, V, device=device, dtype=dtype)
        prefill_g = torch.randn(B, 1, H, K, device=device, dtype=dtype)
        prefill_beta = torch.sigmoid(torch.randn(B, 1, H, device=device, dtype=dtype))

        cu_seqlens = torch.arange(B + 1, device=device, dtype=torch.int32)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, K, device=device, dtype=dtype)

    ssm_states = (
        torch.randn(pool_size, HV, V, K, device=device, dtype=torch.float32) * 0.1
    )
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
        prefill_v=prefill_v,
        prefill_g=prefill_g,
        prefill_beta=prefill_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
    )


def run_baseline(inp):
    state = inp["ssm_states"].clone()
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
    state = inp["ssm_states"].clone()
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
        cu_seqlens=inp["cu_seqlens"] if inp["layout"] == "varlen" else None,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )
    return o, state


def run_prefill_then_decode_baseline(inp):
    ssm_states = inp["ssm_states"].clone()
    prefill_v_clone = inp["prefill_v"].clone()
    v_clone = inp["v"].clone()

    _ = chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=prefill_v_clone,
        g=inp["prefill_g"],
        beta=inp["prefill_beta"],
        initial_state=ssm_states,
        initial_state_indices=inp["cache_indices"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu_seqlens"] if inp["layout"] == "varlen" else None,
    )

    o = fused_sigmoid_gating_delta_rule_update(
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        q=inp["q"],
        k=inp["k"],
        v=v_clone,
        a=inp["a"],
        b=inp["b"],
        initial_state_source=ssm_states,
        initial_state_indices=inp["cache_indices"],
        cu_seqlens=inp["cu_seqlens"] if inp["layout"] == "varlen" else None,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
    )
    return o, ssm_states


def run_prefill_then_decode_cutedsl(inp):
    ssm_states = inp["ssm_states"].clone()
    prefill_v_clone = inp["prefill_v"].clone()
    v_clone = inp["v"].clone()

    _ = chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=prefill_v_clone,
        g=inp["prefill_g"],
        beta=inp["prefill_beta"],
        initial_state=ssm_states,
        initial_state_indices=inp["cache_indices"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu_seqlens"] if inp["layout"] == "varlen" else None,
    )

    o = cutedsl_fused_sigmoid_gating_kda_update(
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        q=inp["q"],
        k=inp["k"],
        v=v_clone,
        a=inp["a"],
        b=inp["b"],
        initial_state_source=ssm_states,
        initial_state_indices=inp["cache_indices"],
        cu_seqlens=inp["cu_seqlens"] if inp["layout"] == "varlen" else None,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )
    return o, ssm_states


def _assert_close(name, x, y, atol=3e-2, rtol=2e-2):
    try:
        torch.testing.assert_close(x.float(), y.float(), atol=atol, rtol=rtol)
        return True, 0.0
    except AssertionError:
        max_diff = (x - y).abs().max().item()
        return False, max_diff


def check_correctness(B, H, HV, K, V, pool_size, device, dtype, layout):
    tag = (
        f"layout={layout:<6} B={B:>4} H={H:>2} HV={HV:>2} "
        f"K={K:>3} V={V:>3} pool={pool_size:>4}"
    )
    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype, layout)

    o_ref, st_ref = run_baseline(inp)
    o_cute, st_cute = run_cutedsl(inp)

    ok_o, diff_o = _assert_close("output", o_cute, o_ref)
    valid_mask = inp["cache_indices"] >= 0
    valid_idx = inp["cache_indices"][valid_mask]
    ok_s, diff_s = _assert_close("state", st_cute[valid_idx], st_ref[valid_idx])

    if ok_o and ok_s:
        print(f"  [PASS] {tag}")
        return True

    details = []
    if not ok_o:
        details.append(f"output max_diff={diff_o:.6f}")
    if not ok_s:
        details.append(f"state max_diff={diff_s:.6f}")
    print(f"  [FAIL] {tag} ({', '.join(details)})")
    return False


def check_prefill_chain(B, H, HV, K, V, pool_size, device, dtype, layout):
    tag = (
        f"[prefill->decode] layout={layout:<6} B={B:>4} H={H:>2} HV={HV:>2} "
        f"K={K:>3} V={V:>3} pool={pool_size:>4}"
    )
    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype, layout)

    o_ref, st_ref = run_prefill_then_decode_baseline(inp)
    o_cute, st_cute = run_prefill_then_decode_cutedsl(inp)

    ok_o, diff_o = _assert_close("output", o_cute, o_ref)
    valid_mask = inp["cache_indices"] >= 0
    valid_idx = inp["cache_indices"][valid_mask]
    ok_s, diff_s = _assert_close("state", st_cute[valid_idx], st_ref[valid_idx])

    if ok_o and ok_s:
        print(f"  [PASS] {tag}")
        return True

    details = []
    if not ok_o:
        details.append(f"output max_diff={diff_o:.6f}")
    if not ok_s:
        details.append(f"state max_diff={diff_s:.6f}")
    print(f"  [FAIL] {tag} ({', '.join(details)})")
    return False


def bench_shape(B, H, HV, K, V, pool_size, device, dtype, layout):
    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype, layout)

    def fn_triton():
        fused_sigmoid_gating_delta_rule_update(
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            a=inp["a"],
            b=inp["b"],
            initial_state_source=inp["ssm_states"],
            initial_state_indices=inp["cache_indices"],
            cu_seqlens=inp["cu_seqlens"] if inp["layout"] == "varlen" else None,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
        )

    def fn_cute():
        cutedsl_fused_sigmoid_gating_kda_update(
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            a=inp["a"],
            b=inp["b"],
            initial_state_source=inp["ssm_states"],
            initial_state_indices=inp["cache_indices"],
            cu_seqlens=inp["cu_seqlens"] if inp["layout"] == "varlen" else None,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    for _ in range(10):
        fn_triton()
        fn_cute()
    torch.cuda.synchronize()

    try:
        ms_triton, _, _ = triton.testing.do_bench(
            fn_triton, quantiles=[0.5, 0.2, 0.8], warmup=50, rep=200
        )
        ms_cute, _, _ = triton.testing.do_bench(
            fn_cute, quantiles=[0.5, 0.2, 0.8], warmup=50, rep=200
        )
    except Exception:
        rep = 100
        st = time.perf_counter()
        for _ in range(rep):
            fn_triton()
        torch.cuda.synchronize()
        ms_triton = (time.perf_counter() - st) / rep * 1000

        st = time.perf_counter()
        for _ in range(rep):
            fn_cute()
        torch.cuda.synchronize()
        ms_cute = (time.perf_counter() - st) / rep * 1000

    speedup = ms_triton / ms_cute if ms_cute > 0 else float("inf")
    delta = (ms_cute - ms_triton) * 1000
    print(
        f"  {layout:>6}  {B:>5}  {H:>3}  {HV:>3}  {K:>3}  {V:>3} | "
        f"{ms_triton * 1000:>12.1f} | "
        f"{ms_cute * 1000:>13.1f} | "
        f"{speedup:>8.2f} | "
        f"{delta:>11.1f}"
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

    print()
    print("=" * 78)
    print("Correctness: Triton prefill/extend -> CuTe decode chain")
    print("=" * 78)
    for layout, B, H, HV, K, V, pool_size in shapes[:8]:
        if not check_prefill_chain(B, H, HV, K, V, pool_size, device, dtype, layout):
            all_pass = False

    print()
    print("ALL PASSED." if all_pass else "SOME FAILED.")
    return all_pass


def run_benchmark(device, dtype):
    print()
    print("=" * 92)
    print("Benchmark: Triton KDA Decode vs CuTe DSL KDA Decode")
    print("=" * 92)

    bench_configs = [
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

    K = 128
    V = 128
    pool_size = 512

    print(f"  Config: K={K}, V={V}, pool_size={pool_size}, dtype={dtype}")
    print(
        f"  {'layout':>6}  {'B':>5}  {'H':>3}  {'HV':>3}  {'K':>3}  {'V':>3} | "
        f"{'triton (μs)':>12} | "
        f"{'cutedsl (μs)':>13} | "
        f"{'speedup':>8} | "
        f"{'delta (μs)':>11}"
    )
    print("  " + "-" * 82)

    for layout, B, H, HV in bench_configs:
        actual_pool = max(pool_size, B + 16)
        bench_shape(B, H, HV, K, V, actual_pool, device, dtype, layout)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark & Correctness: Triton KDA Decode vs CuTe DSL KDA Decode"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "correctness", "bench"],
        default="all",
        help="Run mode (default: all)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)

    cap = torch.cuda.get_device_capability()
    dev_name = torch.cuda.get_device_name()
    print(f"Device: {dev_name}  (SM {cap[0]}{cap[1]})")

    if args.mode in ("all", "correctness"):
        all_pass = run_correctness(device, dtype)
        if not all_pass and args.mode == "all":
            print("\nSkipping benchmark due to correctness failures.")
            return 1

    if args.mode in ("all", "bench"):
        run_benchmark(device, dtype)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
