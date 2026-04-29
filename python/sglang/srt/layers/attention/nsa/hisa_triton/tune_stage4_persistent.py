"""Sweep persistent kernel K_CHUNKS × num_stages for prefill stage 4.

Compares against the non-persistent grouped kernel (current baseline).
Shapes: sq=8192, K∈{16, 32}, skv∈{32K, 128K} (mid-late prefill chunks),
topk=2048.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    _block_sparse_mqa_grouped_kernel,
    _block_sparse_mqa_persistent_kernel,
)


DEVICE = torch.device("cuda")
H, D = 64, 128


def make_inputs(K, sq, skv, topk):
    torch.manual_seed(0)
    q = torch.randn(sq, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(skv, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_scale = 0.05 + 0.02 * torch.rand(skv, device=DEVICE, dtype=torch.float32)
    weights = torch.randn(sq, H, device=DEVICE, dtype=torch.float32)
    cu_ks = torch.zeros(sq, device=DEVICE, dtype=torch.int32)
    cu_ke = torch.linspace(skv // 4, skv, sq, device=DEVICE).to(torch.int32)
    num_blocks = skv // K
    topk_idx = torch.randint(
        0, num_blocks, (sq, topk), device=DEVICE, dtype=torch.int64,
    )
    return q, k_fp8, k_scale, weights, cu_ks, cu_ke, topk_idx


def bench_grouped_baseline(K, sq, skv, topk, iters=80, warmup=20):
    GEMM_TILE = 256
    GROUP_SIZE = GEMM_TILE // K
    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE

    q, k_fp8, k_scale, weights, cu_ks, cu_ke, topk_idx = make_inputs(K, sq, skv, topk)
    logits = torch.empty((sq, topk * K), device=DEVICE, dtype=torch.float32)
    grid = (sq, num_chunks)

    def fn():
        _block_sparse_mqa_grouped_kernel[grid](
            q, k_fp8, k_scale, topk_idx, logits, weights, cu_ks, cu_ke,
            q.stride(0), q.stride(1), q.stride(2),
            k_fp8.stride(0), k_fp8.stride(1),
            k_scale.stride(0),
            topk_idx.stride(0), topk_idx.stride(1),
            logits.stride(0), logits.stride(1),
            weights.stride(0), weights.stride(1),
            skv, topk,
            HEADS=H, DIM=D, KV_BLOCK_SIZE=K, GROUP_SIZE=GROUP_SIZE,
        )
    return _bench(fn, warmup, iters), logits.clone()


def bench_persistent(K, sq, skv, topk, K_CHUNKS, num_stages, num_warps,
                     iters=80, warmup=20):
    GEMM_TILE = 256
    GROUP_SIZE = GEMM_TILE // K
    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE
    outer = (num_chunks + K_CHUNKS - 1) // K_CHUNKS

    q, k_fp8, k_scale, weights, cu_ks, cu_ke, topk_idx = make_inputs(K, sq, skv, topk)
    logits = torch.empty((sq, topk * K), device=DEVICE, dtype=torch.float32)
    grid = (sq, outer)

    def fn():
        _block_sparse_mqa_persistent_kernel[grid](
            q, k_fp8, k_scale, topk_idx, logits, weights, cu_ks, cu_ke,
            q.stride(0), q.stride(1), q.stride(2),
            k_fp8.stride(0), k_fp8.stride(1),
            k_scale.stride(0),
            topk_idx.stride(0), topk_idx.stride(1),
            logits.stride(0), logits.stride(1),
            weights.stride(0), weights.stride(1),
            skv, topk,
            HEADS=H, DIM=D, KV_BLOCK_SIZE=K, GROUP_SIZE=GROUP_SIZE,
            K_CHUNKS=K_CHUNKS,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    try:
        t = _bench(fn, warmup, iters)
        return t, logits.clone()
    except Exception as ex:
        return f"ERR:{type(ex).__name__}", None


def _bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters  # μs


def correctness_check(out_baseline, out_persistent, label):
    """Both outputs should match bytewise (modulo NaN from random fp8 scale)."""
    nan_a = torch.isnan(out_baseline)
    nan_b = torch.isnan(out_persistent)
    if not torch.equal(nan_a, nan_b):
        return f"{label} NaN positions DIFFER"
    finite = ~nan_a
    if not torch.equal(out_baseline[finite], out_persistent[finite]):
        diff = (out_baseline[finite] - out_persistent[finite]).abs()
        return f"{label} finite mismatch max|Δ|={diff.max().item():.3e}"
    return f"{label} byte_eq"


CONFIGS = [
    # (K_CHUNKS, num_stages, num_warps)
    (2, 3, 4),
    (4, 2, 4),
    (4, 3, 4),
    (4, 4, 4),
    (8, 2, 4),
    (8, 3, 4),
    (8, 4, 4),
    (16, 2, 4),
    (16, 3, 4),
    (32, 2, 4),
    (32, 3, 4),
    # 8-warp variants for high register pressure cases
    (4, 2, 8),
    (8, 2, 8),
    (16, 2, 8),
]


def main():
    cases = [
        # Mid- and late-prefill chunks for the production 128K context
        ( 8, 8192,  65536, 2048),
        ( 8, 8192, 131072, 2048),
        (16, 8192,  65536, 2048),
        (16, 8192, 131072, 2048),
        (32, 8192,  65536, 2048),
        (32, 8192, 131072, 2048),
        (64, 8192,  65536, 2048),
        (64, 8192, 131072, 2048),
    ]

    width = 30 + 12 + len(CONFIGS) * 9 + 30
    print("=" * width)
    print("PERSISTENT KERNEL sweep (μs/call)  vs  baseline non-persistent grouped")
    print("=" * width)
    print(f"{'K':>3} {'sq':>5} {'skv':>7} | {'baseline':>9} | "
          + " ".join(f"K{kc}s{ns}w{w}".rjust(8) for kc, ns, w in CONFIGS) + " | best  speedup")
    print("-" * width)

    correctness_lines = []

    for K, sq, skv, topk in cases:
        base_us, base_out = bench_grouped_baseline(K, sq, skv, topk)
        results = []
        for K_CHUNKS, num_stages, num_warps in CONFIGS:
            t, pers_out = bench_persistent(
                K, sq, skv, topk, K_CHUNKS, num_stages, num_warps,
            )
            results.append(t)
            # Correctness on first config combo only (kernel structure is the same).
            if isinstance(t, float) and pers_out is not None and (K_CHUNKS, num_stages) == (CONFIGS[0][0], CONFIGS[0][1]):
                msg = correctness_check(base_out, pers_out, f"K={K} skv={skv} K{K_CHUNKS}s{num_stages}")
                correctness_lines.append(msg)
            del pers_out
        del base_out
        torch.cuda.empty_cache()

        floats = [t for t in results if isinstance(t, float)]
        best = min(floats) if floats else None
        speedup = f"{base_us / best:.2f}x" if best else "n/a"
        winner_idx = next((i for i, t in enumerate(results) if t == best), None)
        winner = (
            f"K{CONFIGS[winner_idx][0]}s{CONFIGS[winner_idx][1]}w{CONFIGS[winner_idx][2]}"
            if winner_idx is not None else "n/a"
        )

        def fmt(t):
            return f"{t:>8.0f}" if isinstance(t, float) else f"{t:>8}"

        line = (f"{K:>3} {sq:>5} {skv:>7} | {base_us:>9.0f} | "
                + " ".join(fmt(t) for t in results)
                + f" | {winner} {best:>5.0f}  {speedup}")
        print(line)

    print()
    print("Correctness checks (first config combo per case):")
    for line in correctness_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
