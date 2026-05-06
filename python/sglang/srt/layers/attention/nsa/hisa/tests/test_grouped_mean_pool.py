"""Compare tilelang grouped block_mean_pooling vs triton grouped variant.

Tilelang's vanilla ``fp8_native_block_mean_pooling`` over-reads ``block_N - K``
rows past each pool block end and is unusable at K < 64. The new
``fp8_native_block_mean_pooling_grouped`` flips parallelism (one CTA per
``block_N=64`` tokens, producing G = 64/K pool blocks) and is bounds-safe.

This script:
  1. Cross-checks tilelang grouped vs triton grouped at K ∈ {8, 16, 32}
     (byte-equal expected modulo fp8 round-half-to-even, since both kernels
     do per-block fp32 mean → fp8 max-abs quantize with the same scale).
  2. Walltime A/B at production-ish shapes (long ragged K).
"""
from __future__ import annotations

import time

import torch

from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
    fp8_native_block_mean_pooling_grouped_interface,
    fp8_native_block_mean_pooling_interface,
)
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    block_mean_pooling_triton,
)


DEVICE = torch.device("cuda")
D = 128


def make_k(seq_kv, seed=0):
    torch.manual_seed(seed)
    k = torch.randn(seq_kv, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_scale = 0.05 + 0.02 * torch.rand(seq_kv, device=DEVICE, dtype=torch.float32)
    return k, k_scale


def diag(a, b):
    """Stats + verdict at fp8-strict tolerance.

    fp8e4m3fn has ~3-bit mantissa; per-element ULPs at magnitude M go up to
    M / 4. Cross-implementation reduce-order divergence can flip one fp8
    bucket near a boundary → max|Δ| up to M/4 on rare elements. Verdict =
    OK if (max|Δ| / max|val|) <= 1/16 (one fp8 ULP near saturation).
    """
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    val_max = max(a32.abs().max().item(), b32.abs().max().item(), 1e-9)
    rel = max_abs / val_max
    verdict = "OK" if rel <= 1.0 / 16 else "FAIL"
    return verdict, (
        f"max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  "
        f"max|Δ|/max|val|={rel:.3e}"
    )


def correctness(seq_kv, K):
    """Compare BlockedK + BlockedKScale across tilelang-grouped vs triton."""
    k, k_scale = make_k(seq_kv)
    bk_tl, bks_tl = fp8_native_block_mean_pooling_grouped_interface(k, k_scale, K)
    bk_t, bks_t = block_mean_pooling_triton(k_fp8=k, k_scale=k_scale, k_block_size=K)
    assert bk_tl.shape == bk_t.shape, f"{bk_tl.shape} vs {bk_t.shape}"
    assert bks_tl.shape == bks_t.shape, f"{bks_tl.shape} vs {bks_t.shape}"

    v_bk, d_bk = diag(bk_tl, bk_t)
    v_bks, d_bks = diag(bks_tl, bks_t)
    return v_bk, d_bk, v_bks, d_bks


def bench_one(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def main():
    print("=" * 105)
    print("CORRECTNESS — tilelang grouped vs triton grouped at fp8-strict tolerance")
    print("=" * 105)
    print(f"{'seq_kv':>8} | {'K':>3} | bk verdict | bk diag                                               | scale verdict")
    print("-" * 105)
    for seq_kv in (256, 1024, 4096, 8195):  # last one not multiple of block_N=64
        for K in (8, 16, 32):
            try:
                v_bk, d_bk, v_bks, _ = correctness(seq_kv, K)
                print(f"{seq_kv:>8} | {K:>3} | {v_bk:>10} | {d_bk:<53} | {v_bks}")
            except Exception as e:
                print(f"{seq_kv:>8} | {K:>3} | ERROR: {type(e).__name__}: {str(e)[:120]}")

    print()
    print("=" * 80)
    print("SPEED — wall-time per call (μs), avg over 300 iters after 50 warmup")
    print("=" * 80)
    print(f"{'seq_kv':>8} | {'K':>3} | tilelang grouped | triton grouped | tl / triton")
    print("-" * 80)
    from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
        fp8_native_block_mean_pooling_interface,
    )
    for seq_kv in (4096, 16384, 65536):
        for K in (8, 16, 32, 64, 128):
            k, k_scale = make_k(seq_kv)
            # Use grouped tilelang for K<64 (vanilla has OOB), vanilla for K>=64.
            if K < 64:
                tl_fn = lambda K=K: fp8_native_block_mean_pooling_grouped_interface(k, k_scale, K)
            else:
                tl_fn = lambda K=K: fp8_native_block_mean_pooling_interface(k, k_scale, K)
            triton_fn = lambda K=K: block_mean_pooling_triton(
                k_fp8=k, k_scale=k_scale, k_block_size=K,
            )
            try:
                tl_us = bench_one(tl_fn)
                tr_us = bench_one(triton_fn)
                ratio = tl_us / tr_us
                winner = "tilelang" if ratio < 1.0 else "triton"
                print(
                    f"{seq_kv:>8} | {K:>3} | {tl_us:>13.2f}    | "
                    f"{tr_us:>11.2f}    | {ratio:>5.2f}x  ({winner} faster)"
                )
            except Exception as e:
                print(f"{seq_kv:>8} | {K:>3} | ERROR: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
