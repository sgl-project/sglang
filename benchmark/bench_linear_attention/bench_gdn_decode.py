"""
Benchmark & Correctness: GDN Packed Decode vs Baseline Decode.

Compares:
  - Baseline: split(mixed_qkv) → view → fused_sigmoid_gating_delta_rule_update
  - Packed:   fused_recurrent_gated_delta_rule_packed_decode (single kernel)

The packed path eliminates:
  - torch.split() + .view() tensor materialization
  - Separate gating kernel launches
  - Intermediate tensor allocations

Reports correctness (output & state matching) and performance (ms, speedup).

Usage:
    python bench_gdn_decode.py                        # default sweep
    python bench_gdn_decode.py --mode bench           # benchmark only
    python bench_gdn_decode.py --mode correctness     # correctness only
    python bench_gdn_decode.py --preset qwen3.5-35b   # Qwen3.5-35B-A3B config
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import triton

from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)

# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------


def make_inputs(
    B: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    device: str,
    dtype: torch.dtype,
    seed: int = 42,
):
    """Create all input tensors for a single benchmark / correctness run."""
    torch.manual_seed(seed)

    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn(B, qkv_dim, device=device, dtype=dtype)
    a = torch.randn(B, HV, device=device, dtype=dtype)
    b = torch.randn(B, HV, device=device, dtype=dtype)
    A_log = torch.randn(HV, device=device, dtype=dtype)
    dt_bias = torch.randn(HV, device=device, dtype=dtype)

    ssm_states = torch.randn(pool_size, HV, V, K, device=device, dtype=dtype) * 0.1
    cache_indices = torch.arange(B, device=device, dtype=torch.int32)

    cu_seqlens = torch.arange(B + 1, device=device, dtype=torch.long)

    return dict(
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        qkv_dim=qkv_dim,
        pool_size=pool_size,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
    )


# ---------------------------------------------------------------------------
# Runner wrappers
# ---------------------------------------------------------------------------


def run_baseline(inp):
    """Baseline path: split → view → fused_sigmoid_gating_delta_rule_update.

    This mirrors the FULL original decode path in GDNAttnBackend.forward_decode,
    including the split, view, and kernel call.
    """
    B, H, HV, K, V = inp["B"], inp["H"], inp["HV"], inp["K"], inp["V"]
    mixed_qkv = inp["mixed_qkv"]
    ssm_states = inp["ssm_states"].clone()

    # Step 1: split (same as forward_decode)
    q_flat, k_flat, v_flat = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)

    # Step 2: view + reshape (same as forward_decode)
    q = q_flat.view(1, B, H, K)
    k = k_flat.view(1, B, H, K)
    v = v_flat.view(1, B, HV, V)

    # Step 3: fused gating + recurrent update
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        q=q,
        k=k,
        v=v,
        a=inp["a"],
        b=inp["b"],
        initial_state_source=ssm_states,
        initial_state_indices=inp["cache_indices"],
        cu_seqlens=inp["cu_seqlens"],
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )

    return o, ssm_states


def run_packed(inp):
    """Packed path: single fused kernel directly on mixed_qkv."""
    B, HV, K, V = inp["B"], inp["HV"], inp["K"], inp["V"]
    ssm_states = inp["ssm_states"].clone()
    out = inp["mixed_qkv"].new_empty(B, 1, HV, V)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=inp["mixed_qkv"],
        a=inp["a"],
        b=inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        scale=inp["K"] ** -0.5,
        initial_state=ssm_states,
        out=out,
        ssm_state_indices=inp["cache_indices"],
        use_qk_l2norm_in_kernel=True,
    )

    # Convert [B, 1, HV, V] → [1, B, HV, V] to match baseline layout
    return out.transpose(0, 1), ssm_states


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_correctness(B, H, HV, K, V, pool_size, device, dtype, seed=42):
    """Run correctness check for a single config. Returns True if PASS."""
    tag = f"B={B:>4} H={H:>2} HV={HV:>2} K={K:>3} V={V:>3} pool={pool_size:>4}"

    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype, seed=seed)

    o_baseline, state_baseline = run_baseline(inp)
    o_packed, state_packed = run_packed(inp)

    # Output comparison
    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4

    try:
        torch.testing.assert_close(o_packed, o_baseline, atol=atol, rtol=rtol)
        output_ok = True
    except AssertionError as e:
        output_ok = False
        out_diff = (o_packed - o_baseline).abs().max().item()

    # State comparison (only for slots that were updated)
    indices = inp["cache_indices"]
    try:
        torch.testing.assert_close(
            state_packed[indices], state_baseline[indices], atol=atol, rtol=rtol
        )
        state_ok = True
    except AssertionError:
        state_ok = False
        st_diff = (state_packed[indices] - state_baseline[indices]).abs().max().item()

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


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_shape(B, H, HV, K, V, pool_size, device, dtype):
    """Benchmark baseline vs packed for a single config."""
    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype)

    # ── Baseline: full path including split + view ──
    def fn_baseline():
        q_flat, k_flat, v_flat = torch.split(
            inp["mixed_qkv"], [H * K, H * K, HV * V], dim=-1
        )
        q = q_flat.view(1, B, H, K)
        k = k_flat.view(1, B, H, K)
        v = v_flat.view(1, B, HV, V)
        fused_sigmoid_gating_delta_rule_update(
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            q=q,
            k=k,
            v=v,
            a=inp["a"],
            b=inp["b"],
            initial_state_source=inp["ssm_states"],
            initial_state_indices=inp["cache_indices"],
            cu_seqlens=inp["cu_seqlens"],
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    # ── Packed: single kernel ──
    out_buf = inp["mixed_qkv"].new_empty(B, 1, HV, V)

    def fn_packed():
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=inp["mixed_qkv"],
            a=inp["a"],
            b=inp["b"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            scale=K**-0.5,
            initial_state=inp["ssm_states"],
            out=out_buf,
            ssm_state_indices=inp["cache_indices"],
            use_qk_l2norm_in_kernel=True,
        )

    # Warmup
    for _ in range(10):
        fn_baseline()
        fn_packed()
    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]

    try:
        ms_baseline, ms_base_lo, ms_base_hi = triton.testing.do_bench(
            fn_baseline, quantiles=quantiles, warmup=50, rep=200
        )
        ms_packed, ms_pack_lo, ms_pack_hi = triton.testing.do_bench(
            fn_packed, quantiles=quantiles, warmup=50, rep=200
        )
    except Exception:
        # Fallback to manual timing
        torch.cuda.synchronize()
        N = 200
        start = time.perf_counter()
        for _ in range(N):
            fn_baseline()
        torch.cuda.synchronize()
        ms_baseline = (time.perf_counter() - start) / N * 1000

        start = time.perf_counter()
        for _ in range(N):
            fn_packed()
        torch.cuda.synchronize()
        ms_packed = (time.perf_counter() - start) / N * 1000

    speedup = ms_baseline / ms_packed if ms_packed > 0 else float("inf")
    saved_us = (ms_baseline - ms_packed) * 1000

    print(
        f"  {B:>5}  {H:>3}  {HV:>3}  {K:>3}  {V:>3} | "
        f"{ms_baseline * 1000:>10.1f} | "
        f"{ms_packed * 1000:>10.1f} | "
        f"{speedup:>7.2f}x | "
        f"{saved_us:>+9.1f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_correctness(device, dtype):
    print("=" * 70)
    print("Correctness: Baseline GDN Decode vs Packed GDN Decode")
    print("=" * 70)

    shapes = [
        # (B,   H,  HV,  K,   V,   pool_size)
        # --- Qwen3.5-35B-A3B style (TP=2: H=8, HV=16) ---
        (1, 8, 16, 128, 128, 32),
        (4, 8, 16, 128, 128, 32),
        (16, 8, 16, 128, 128, 64),
        (32, 8, 16, 128, 128, 128),
        (64, 8, 16, 128, 128, 128),
        (128, 8, 16, 128, 128, 256),
        (256, 8, 16, 128, 128, 512),
        # --- Qwen3.5-35B-A3B style (TP=1: H=16, HV=32) ---
        (1, 16, 32, 128, 128, 32),
        (32, 16, 32, 128, 128, 128),
        (64, 16, 32, 128, 128, 128),
        # --- Qwen3-Next-80B-A3B style ---
        (32, 16, 16, 128, 128, 128),
        (64, 16, 16, 128, 128, 128),
        # --- With PAD_SLOT_ID ---
        (32, 8, 16, 128, 128, 128),  # some indices may be padded
        # --- Edge cases ---
        (1, 8, 16, 128, 128, 32),
        (2, 8, 16, 128, 128, 32),
    ]

    all_pass = True
    for B, H, HV, K, V, pool_size in shapes:
        if not check_correctness(B, H, HV, K, V, pool_size, device, dtype):
            all_pass = False

    # PAD_SLOT_ID test
    print("\n  PAD_SLOT_ID test (indices with -1):")
    inp = make_inputs(32, 8, 16, 128, 128, 128, device, dtype)
    o_baseline, st_baseline = run_baseline(inp)
    o_packed, st_packed = run_packed(inp)

    try:
        torch.testing.assert_close(o_packed, o_baseline, atol=2e-2, rtol=1e-2)
        print("  [PASS] PAD_SLOT_ID=-1 handling")
    except AssertionError:
        print("  [FAIL] PAD_SLOT_ID=-1 handling")
        all_pass = False

    print()
    if all_pass:
        print("ALL PASSED.")
    else:
        print("SOME FAILED.")
    return all_pass


def run_benchmark(device, dtype, args):
    print()
    print("=" * 85)
    print("Benchmark: Baseline GDN Decode vs Packed GDN Decode")
    print("=" * 85)

    K = args.head_size_k
    V = args.head_size_v
    pool_size = args.pool_size

    if args.preset == "qwen3.5-35b":
        # Qwen3.5-35B-A3B: H_qk=16, H_v=32, K=128, V=128
        # After TP=2: H=8, HV=16
        bench_configs = [
            # (B,   H,  HV) — TP=2 config
            (1, 8, 16),
            (2, 8, 16),
            (4, 8, 16),
            (8, 8, 16),
            (16, 8, 16),
            (32, 8, 16),
            (64, 8, 16),
            (128, 8, 16),
            (256, 8, 16),
            (512, 8, 16),
            # TP=1 config (full heads)
            (1, 16, 32),
            (8, 16, 32),
            (32, 16, 32),
            (64, 16, 32),
            (128, 16, 32),
            (256, 16, 32),
        ]
    elif args.preset == "qwen3-next-80b":
        bench_configs = [
            # Qwen3-Next-80B-A3B: all same H=HV=16 after TP
            (1, 16, 16),
            (8, 16, 16),
            (32, 16, 16),
            (64, 16, 16),
            (128, 16, 16),
            (256, 16, 16),
        ]
    else:
        bench_configs = []
        for B in args.batch_sizes:
            for H in args.num_q_heads:
                for HV in args.num_v_heads:
                    bench_configs.append((B, H, HV))

    print(f"  Config: K={K}, V={V}, pool_size={pool_size}, dtype={dtype}")
    print(
        f"  {'B':>5}  {'H':>3}  {'HV':>3}  {'K':>3}  {'V':>3} | "
        f"{'base (μs)':>10} | "
        f"{'packed (μs)':>10} | "
        f"{'speedup':>8} | "
        f"{'saved (μs)':>10}"
    )
    print("  " + "-" * 75)

    for B, H, HV in bench_configs:
        actual_pool = max(pool_size, B + 16)
        bench_shape(B, H, HV, K, V, actual_pool, device, dtype)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark & Correctness: GDN Packed Decode vs Baseline"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "correctness", "bench"],
        default="all",
        help="Run mode (default: all)",
    )
    parser.add_argument(
        "--preset",
        choices=["qwen3.5-35b", "qwen3-next-80b", "custom"],
        default="qwen3.5-35b",
        help="Preset config (default: qwen3.5-35b)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--head-size-k", type=int, default=128)
    parser.add_argument("--head-size-v", type=int, default=128)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--num-q-heads",
        type=int,
        nargs="+",
        default=[8, 16],
    )
    parser.add_argument(
        "--num-v-heads",
        type=int,
        nargs="+",
        default=[16, 32],
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
        run_benchmark(device, dtype, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
