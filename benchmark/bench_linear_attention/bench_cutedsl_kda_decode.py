"""
Benchmark & Correctness: CuTe DSL KDA Decode vs Triton KDA Decode.

Compares:
  - Baseline: Triton fused_sigmoid_gating_delta_rule_update(..., is_kda=True)
  - CuTe DSL: cutedsl_fused_sigmoid_gating_delta_rule_update_kda

Notes:
  - Triton KDA state layout is [pool, HV, V, K]
  - CuTe DSL KDA state layout here is [pool, HV, K, V]
  - We initialize both from the same canonical values and transpose when comparing states.

Usage:
    python bench_cutedsl_kda_decode.py
    python bench_cutedsl_kda_decode.py --mode correctness
    python bench_cutedsl_kda_decode.py --mode bench
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

    assert K == 128, "Current CuTe DSL KDA kernel assumes K=128"
    assert (
        V % 16 == 0 and V % 32 == 0
    ), "Current CuTe DSL KDA kernel expects tile-friendly V"

    if layout == "varlen":
        q = torch.randn(1, B, H, K, device=device, dtype=dtype)
        k = torch.randn(1, B, H, K, device=device, dtype=dtype)
        v = torch.randn(1, B, HV, V, device=device, dtype=dtype)
        a = torch.randn(B, HV, K, device=device, dtype=dtype)
        b = torch.randn(B, HV, device=device, dtype=dtype)
        cu_seqlens = torch.arange(B + 1, device=device, dtype=torch.int32)
    elif layout == "dense":
        q = torch.randn(B, 1, H, K, device=device, dtype=dtype)
        k = torch.randn(B, 1, H, K, device=device, dtype=dtype)
        v = torch.randn(B, 1, HV, V, device=device, dtype=dtype)
        a = torch.randn(B, 1, HV, K, device=device, dtype=dtype)
        b = torch.randn(B, 1, HV, device=device, dtype=dtype)
        cu_seqlens = torch.arange(B + 1, device=device, dtype=torch.int32)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, K, device=device, dtype=dtype)

    # Canonical logical state in [pool, HV, K, V] for CuTe.
    ssm_states_cute = (
        torch.randn(pool_size, HV, K, V, device=device, dtype=torch.float32) * 0.1
    )
    # Triton expects [pool, HV, V, K].
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
    tag = (
        f"layout={layout:<6} "
        f"B={B:>2} H={H:>2} HV={HV:>2} K={K:>3} V={V:>3} pool={pool_size:>4}"
    )

    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype, layout, seed=seed)

    o_ref, st_ref = run_baseline(inp)
    o_cute, st_cute = run_cutedsl(inp)

    # Compare outputs directly; both return same logical layout as input q/v path.
    atol = 3e-2 if dtype != torch.float32 else 1e-4
    rtol = 2e-2 if dtype != torch.float32 else 1e-4

    output_ok = True
    state_ok = True
    out_diff = 0.0
    st_diff = 0.0

    try:
        torch.testing.assert_close(o_cute.float(), o_ref.float(), atol=atol, rtol=rtol)
    except AssertionError:
        output_ok = False
        out_diff = (o_cute.float() - o_ref.float()).abs().max().item()

    # Triton: [pool, HV, V, K], CuTe: [pool, HV, K, V]
    st_cute_vk = st_cute.transpose(-1, -2).contiguous()
    idx = inp["cache_indices"]
    valid_slots = idx[idx >= 0].unique()

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


def bench_shape(B, H, HV, K, V, pool_size, device, dtype, layout):
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

    for _ in range(10):
        fn_baseline()
        fn_cutedsl()
    torch.cuda.synchronize()

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
        f"{ms_baseline * 1000:>10.1f} | "
        f"{ms_cutedsl * 1000:>10.1f} | "
        f"{speedup:>7.2f}x | "
        f"{delta_us:>+9.1f}"
    )


def run_correctness(device, dtype):
    print("=" * 78)
    print("Correctness: Triton KDA Decode vs CuTe DSL KDA Decode")
    print("=" * 78)

    shapes = [
        # layout, B, H, HV, K, V, pool
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

    # Save pristine states to verify untouched pool slots remain unchanged.
    st_ref_before = inp["ssm_states_triton"].clone()
    st_cute_before = inp["ssm_states_cute"].clone()

    o_ref, st_ref = run_baseline(inp)
    o_cute, st_cute = run_cutedsl(inp)

    # Triton: [pool, HV, V, K], CuTe: [pool, HV, K, V] -> [pool, HV, V, K]
    st_cute_vk = st_cute.transpose(-1, -2).contiguous()
    st_cute_before_vk = st_cute_before.transpose(-1, -2).contiguous()

    # token positions in batch dimension
    valid_token_pos = (inp["cache_indices"] >= 0).nonzero(as_tuple=False).squeeze(-1)
    # pool slots actually referenced by valid tokens
    valid_slots = inp["cache_indices"][inp["cache_indices"] >= 0].unique()

    atol = 3e-2 if dtype != torch.float32 else 1e-4
    rtol = 2e-2 if dtype != torch.float32 else 1e-4

    pad_ok = True
    details = []

    # 1) Compare outputs only for valid tokens.
    try:
        torch.testing.assert_close(
            o_cute[:, valid_token_pos].float(),
            o_ref[:, valid_token_pos].float(),
            atol=atol,
            rtol=rtol,
        )
    except AssertionError:
        pad_ok = False
        out_diff = (
            (o_cute[:, valid_token_pos].float() - o_ref[:, valid_token_pos].float())
            .abs()
            .max()
            .item()
        )
        details.append(f"valid_output max_diff={out_diff:.6f}")

    # 2) Compare states only for valid pool slots.
    try:
        torch.testing.assert_close(
            st_cute_vk[valid_slots].float(),
            st_ref[valid_slots].float(),
            atol=atol,
            rtol=rtol,
        )
    except AssertionError:
        pad_ok = False
        st_diff = (
            (st_cute_vk[valid_slots].float() - st_ref[valid_slots].float())
            .abs()
            .max()
            .item()
        )
        details.append(f"valid_state max_diff={st_diff:.6f}")

    # 3) Untouched pool slots must remain unchanged.
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

    if pad_ok:
        print("  [PASS] PAD_SLOT_ID=-1 handling (valid outputs/states only)")
    else:
        print(f"  [FAIL] PAD_SLOT_ID=-1 handling ({', '.join(details)})")
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
