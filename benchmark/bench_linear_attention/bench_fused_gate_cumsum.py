"""
Benchmark: Fused Gate+Cumsum vs Separate Gate + Cumsum.

Compares two paths:
  - Separate: torch gate activation -> chunk_local_cumsum (2 steps)
  - Fused:    kda_gate_chunk_cumsum (single kernel)

Both produce the same output: cumsum of gate-activated g.

Usage:
    python bench_fused_gate_cumsum.py
    python bench_fused_gate_cumsum.py --batch-sizes 4 16 64 128
    python bench_fused_gate_cumsum.py --seq-lens 64 128 256 512 1024
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch
import triton

from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.kda import kda_gate_chunk_cumsum

CHUNK_SIZE = 64


def make_inputs(
    B: int,
    T_per_seq: int,
    H: int,
    K: int,
    device: str,
    dtype: torch.dtype,
    seed: int = 42,
):
    T = B * T_per_seq
    torch.manual_seed(seed)

    # Raw gate: [1, T_total, H, K] (varlen format, before activation)
    raw_g = torch.randn(1, T, H, K, dtype=dtype, device=device)

    # A_log: [H] (per-head log-scale parameter)
    A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.5

    # dt_bias: [H*K] (per-head bias, flat)
    dt_bias = torch.randn(H * K, dtype=torch.float32, device=device) * 0.1

    # cu_seqlens for varlen mode
    cu_seqlens = torch.arange(
        0, (B + 1) * T_per_seq, T_per_seq, dtype=torch.long, device=device
    )

    return dict(
        raw_g=raw_g,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        B=B,
        T=T,
        T_per_seq=T_per_seq,
        H=H,
        K=K,
    )


def run_ref(inp):
    """Separate path: torch gate activation -> chunk_local_cumsum."""
    raw_g = inp["raw_g"]  # [1, T, H, K]
    A_log = inp["A_log"]  # [H]
    dt_bias = inp["dt_bias"]  # [H*K]
    cu_seqlens = inp["cu_seqlens"]
    H, K = inp["H"], inp["K"]

    # Step 1: gate activation using torch ops
    g_float = raw_g.float()
    if dt_bias is not None:
        g_float = g_float + dt_bias.float().view(1, 1, H, K)
    g_activated = -torch.exp(
        A_log.float().view(1, 1, H, 1)
    ) * torch.nn.functional.softplus(g_float)

    # Step 2: chunk-local cumsum
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    g_cumsum = chunk_local_cumsum(
        g_activated,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return g_cumsum


def run_fused(inp):
    """Fused path: kda_gate_chunk_cumsum (single kernel)."""
    raw_g = inp["raw_g"]
    A_log = inp["A_log"]
    dt_bias = inp["dt_bias"]
    cu_seqlens = inp["cu_seqlens"]

    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    g_cumsum = kda_gate_chunk_cumsum(
        raw_g,
        A_log=A_log,
        chunk_size=CHUNK_SIZE,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return g_cumsum


def verify_correctness(inp):
    """Verify fused and separate paths produce the same output."""
    out_separate = run_ref(inp)
    out_fused = run_fused(inp)

    max_diff = (out_separate - out_fused).abs().max().item()
    rel_diff = max_diff / (out_separate.abs().mean().item() + 1e-8)
    return max_diff, rel_diff


def bench_shape(B, H, T_per_seq, K, device, dtype):
    T = B * T_per_seq
    inp = make_inputs(B, T_per_seq, H, K, device, dtype)

    # Warmup (includes triton compilation)
    for _ in range(5):
        run_ref(inp)
        run_fused(inp)
    torch.cuda.synchronize()

    ms_sep, ms_sep_lo, ms_sep_hi = triton.testing.do_bench(
        lambda: run_ref(inp), quantiles=[0.5, 0.2, 0.8], warmup=50, rep=200
    )
    ms_fused, ms_fused_lo, ms_fused_hi = triton.testing.do_bench(
        lambda: run_fused(inp), quantiles=[0.5, 0.2, 0.8], warmup=50, rep=200
    )

    speedup = ms_sep / ms_fused if ms_fused > 0 else 0
    saved_us = (ms_sep - ms_fused) * 1000  # microseconds

    print(
        f"  {B:>5}  {H:>3}  {T_per_seq:>6}  {T:>7} | "
        f"{ms_sep:>8.3f}  {ms_fused:>8.3f} | "
        f"{speedup:>6.2f}x  {saved_us:>+8.1f}us"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: Fused vs Separate Gate+Cumsum"
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--head-size-k", type=int, default=128)
    parser.add_argument("--num-heads", type=int, nargs="+", default=[16])
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[64, 128, 256, 512, 1024]
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    K = args.head_size_k

    cap = torch.cuda.get_device_capability()
    dev_name = torch.cuda.get_device_name()
    print(f"Device: {dev_name}  (SM {cap[0]}{cap[1]})")
    print()

    # Correctness check
    print("=" * 80)
    print("Correctness verification")
    print("=" * 80)
    for H in args.num_heads:
        inp = make_inputs(16, 256, H, K, device, dtype)
        max_diff, rel_diff = verify_correctness(inp)
        print(
            f"  H={H:>3}, B=16, T/seq=256: "
            f"max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}  "
            f"{'PASS' if max_diff < 1e-3 else 'FAIL'}"
        )
    print()

    # Performance benchmark
    print("=" * 80)
    print("Performance: Separate (gate+cumsum) vs Fused (single kernel)")
    print("=" * 80)
    print(f"  Config: K={K}, chunk_size={CHUNK_SIZE}, dtype={dtype}")
    print(
        f"  {'B':>5}  {'H':>3}  {'T/seq':>6}  {'T_tot':>7} | "
        f"{'sep(ms)':>8}  {'fuse(ms)':>8} | "
        f"{'speedup':>6}  {'saved':>9}"
    )
    print("  " + "-" * 73)

    for H in args.num_heads:
        for B in args.batch_sizes:
            for T_per_seq in args.seq_lens:
                bench_shape(B, H, T_per_seq, K, device, dtype)
        if len(args.num_heads) > 1:
            print()


if __name__ == "__main__":
    sys.exit(main())
