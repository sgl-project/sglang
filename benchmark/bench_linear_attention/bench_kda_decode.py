"""
Benchmark & Correctness: KDA Packed Decode vs Baseline Decode.

Compares:
  - Baseline: split(mixed_qkv) -> view -> fused_sigmoid_gating_delta_rule_update(is_kda=True)
  - Packed:   fused_recurrent_kda_packed_decode (single fused kernel)

Differences from the GDN packed decode benchmark:
  - KDA gate ``a`` is per-K with shape ``[B, HV * K]`` (instead of ``[B, HV]``).
  - KDA ``dt_bias`` is per-K with shape ``[HV * K]`` (instead of ``[HV]``).
  - State decay in the kernel is a per-K vector ``exp(g)`` (instead of a scalar).

Reports correctness (output & state matching) and performance (us, speedup).

Usage:
    python bench_kda_decode.py                        # default sweep
    python bench_kda_decode.py --mode bench           # benchmark only
    python bench_kda_decode.py --mode correctness     # correctness only
"""

import argparse

import torch

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
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
    seed: int = 42,
):
    """Create all input tensors for a single benchmark / correctness run."""
    torch.manual_seed(seed)

    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn(B, qkv_dim, device=device, dtype=dtype) * 0.1
    # KDA per-K gate: a is [B, HV*K], dt_bias is [HV*K].
    a = torch.randn(B, HV * K, device=device, dtype=dtype) * 0.5 - 1.0
    b = torch.randn(B, HV, device=device, dtype=dtype) * 0.5
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.2
    dt_bias = torch.randn(HV * K, device=device, dtype=torch.float32) * 0.1

    ssm_states = torch.randn(pool_size, HV, V, K, device=device, dtype=dtype) * 0.01
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
        mixed_qkv=mixed_qkv.contiguous(),
        a=a.contiguous(),
        b=b.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=ssm_states.contiguous(),
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
    )


def run_baseline(inp):
    """Baseline path: split -> view -> fused_sigmoid_gating_delta_rule_update.

    Mirrors the existing decode path in ``KDAAttnBackend.forward_decode``
    (post conv1d, pre-packed-optimization).
    """
    B, H, HV, K, V = inp["B"], inp["H"], inp["HV"], inp["K"], inp["V"]
    mixed_qkv = inp["mixed_qkv"]
    ssm_states = inp["ssm_states"].clone()

    q_flat, k_flat, v_flat = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q_flat.view(1, B, H, K)
    k = k_flat.view(1, B, H, K)
    v = v_flat.view(1, B, HV, V)

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
        is_kda=True,
    )
    return o, ssm_states


def run_packed(inp):
    """Packed path: single fused kernel directly on mixed_qkv."""
    B, HV, K, V = inp["B"], inp["HV"], inp["K"], inp["V"]
    ssm_states = inp["ssm_states"].clone()
    out = inp["mixed_qkv"].new_empty(B, 1, HV, V)

    fused_recurrent_kda_packed_decode(
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
    return out.transpose(0, 1), ssm_states


def check_correctness(B, H, HV, K, V, pool_size, device, dtype, seed=42):
    """Run correctness check for a single config. Returns True if PASS."""
    tag = f"B={B:>4} H={H:>2} HV={HV:>2} K={K:>3} V={V:>3} pool={pool_size:>4}"

    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype, seed=seed)

    o_baseline, state_baseline = run_baseline(inp)
    o_packed, state_packed = run_packed(inp)

    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4

    out_diff = (o_packed.float() - o_baseline.float()).abs().max().item()
    output_ok = out_diff <= max(atol, rtol * o_baseline.float().abs().max().item())

    indices = inp["cache_indices"]
    st_diff = (
        (state_packed[indices].float() - state_baseline[indices].float())
        .abs()
        .max()
        .item()
    )
    state_ok = st_diff <= max(
        atol, rtol * state_baseline[indices].float().abs().max().item()
    )

    passed = output_ok and state_ok
    if passed:
        print(
            f"  [PASS] {tag}  (out max_diff={out_diff:.2e}, state max_diff={st_diff:.2e})"
        )
    else:
        print(
            f"  [FAIL] {tag}  out max_diff={out_diff:.6f}, state max_diff={st_diff:.6f}"
        )
    return passed


def bench_shape(B, H, HV, K, V, pool_size, device, dtype):
    """Benchmark baseline vs packed for a single config."""
    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype)

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
            is_kda=True,
        )

    out_buf = inp["mixed_qkv"].new_empty(B, 1, HV, V)

    def fn_packed():
        fused_recurrent_kda_packed_decode(
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

    # Intentionally wall-clock CUDA-event timing, not the shared do_bench /
    # do_bench_cudagraph util: ~2/3 of the packed win is eager CPU dispatch
    # (split + 3x unflatten + extra launch), which graph capture / L2-flush
    # harnesses amortize away. Decode runs these ops eagerly every step, so
    # wall-clock is the production-relevant metric (~1.7x vs ~1.3x kernel-only).
    warmup, iters = 50, 200
    for _ in range(warmup):
        fn_baseline()
        fn_packed()
    torch.cuda.synchronize()

    def _time(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters  # ms

    ms_baseline = _time(fn_baseline)
    ms_packed = _time(fn_packed)

    speedup = ms_baseline / ms_packed if ms_packed > 0 else float("inf")
    saved_us = (ms_baseline - ms_packed) * 1000

    print(
        f"  {B:>5}  {H:>3}  {HV:>3}  {K:>3}  {V:>3} | "
        f"{ms_baseline * 1000:>10.1f} | "
        f"{ms_packed * 1000:>10.1f} | "
        f"{speedup:>7.2f}x | "
        f"{saved_us:>+9.1f}"
    )


def run_correctness(device, dtype):
    print("=" * 80)
    print("Correctness: Baseline KDA Decode vs Packed KDA Decode")
    print("=" * 80)

    shapes = [
        # (B, H, HV, K, V, pool_size)
        (1, 16, 16, 128, 128, 32),
        (4, 16, 16, 128, 128, 32),
        (16, 16, 16, 128, 128, 64),
        (32, 16, 16, 128, 128, 128),
        (64, 16, 16, 128, 128, 128),
        (128, 16, 16, 128, 128, 256),
        (256, 16, 16, 128, 128, 512),
        # Asymmetric H vs HV
        (1, 32, 32, 128, 128, 32),
        (32, 32, 32, 128, 128, 128),
        (64, 32, 32, 128, 128, 128),
        # Edge case
        (1, 16, 16, 128, 128, 32),
        (2, 16, 16, 128, 128, 32),
    ]

    all_pass = True
    for B, H, HV, K, V, pool_size in shapes:
        if not check_correctness(B, H, HV, K, V, pool_size, device, dtype):
            all_pass = False

    # PAD_SLOT_ID test: some indices < 0 should output zeros and skip state update.
    print("\n  PAD_SLOT_ID test (indices with -1):")
    inp = make_inputs(32, 16, 16, 128, 128, 128, device, dtype)
    pad_mask = torch.zeros(32, device=device, dtype=torch.bool)
    pad_mask[::4] = True
    inp["cache_indices"] = torch.where(
        pad_mask,
        torch.tensor(-1, device=device, dtype=torch.int32),
        inp["cache_indices"],
    )
    o_baseline, _ = run_baseline(inp)
    o_packed, _ = run_packed(inp)
    try:
        torch.testing.assert_close(o_packed, o_baseline, atol=2e-2, rtol=1e-2)
        print("  [PASS] PAD_SLOT_ID=-1 handling")
    except AssertionError as e:
        print(f"  [FAIL] PAD_SLOT_ID=-1 handling: {e}")
        all_pass = False

    print()
    print("ALL PASSED." if all_pass else "SOME FAILED.")
    return all_pass


def run_benchmark(device, dtype, args):
    print()
    print("=" * 85)
    print("Benchmark: Baseline KDA Decode vs Packed KDA Decode")
    print("=" * 85)

    K = args.head_size_k
    V = args.head_size_v
    pool_size = args.pool_size

    bench_configs = []
    for B in args.batch_sizes:
        for H in args.num_q_heads:
            for HV in args.num_v_heads:
                bench_configs.append((B, H, HV))

    print(f"  Config: K={K}, V={V}, pool_size={pool_size}, dtype={dtype}")
    print(
        f"  {'B':>5}  {'H':>3}  {'HV':>3}  {'K':>3}  {'V':>3} | "
        f"{'base (us)':>10} | "
        f"{'packed (us)':>10} | "
        f"{'speedup':>8} | "
        f"{'saved (us)':>10}"
    )
    print("  " + "-" * 80)

    for B, H, HV in bench_configs:
        # Packed kernel requires HV % H == 0 (GVA / grouped query layout).
        if HV % H != 0:
            continue
        actual_pool = max(pool_size, B + 16)
        bench_shape(B, H, HV, K, V, actual_pool, device, dtype)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark & Correctness: KDA Packed Decode vs Baseline"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "correctness", "bench"],
        default="all",
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
        default=[1, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--num-q-heads",
        type=int,
        nargs="+",
        default=[16, 32],
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
    raise SystemExit(main())
