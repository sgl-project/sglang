"""Sweep GEMM_TILE for prefill stage 4 grouped kernel.

Current: GEMM_TILE=256 for all K<128 (SK7+SK11). Re-sweep specifically
for K=16/32 to check if smaller/larger tile wins on production sq=8192.

GEMM_TILE = K * GROUP_SIZE (kernel constraint). So:
  K=16 candidates: GROUP=4 (TILE=64), 8 (128), 16 (256, current), 32 (512)
  K=32 candidates: GROUP=2 (TILE=64), 4 (128), 8 (256, current), 16 (512)

For each (K, skv, GEMM_TILE): bench μs/call. Pick lowest per (K, skv).
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    _block_sparse_mqa_grouped_kernel,
)


DEVICE = torch.device("cuda")
H, D = 64, 128


def bench(K, sq, skv, topk, GEMM_TILE, warmup=10, iters=50):
    if GEMM_TILE % K != 0:
        return f"skip:tile%K"
    GROUP_SIZE = GEMM_TILE // K

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

    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE
    logits = torch.empty((sq, topk * K), device=DEVICE, dtype=torch.float32)
    grid = (sq, num_chunks)

    def fn():
        _block_sparse_mqa_grouped_kernel[grid](
            q, k_fp8, k_scale, topk_idx, logits, weights,
            cu_ks, cu_ke,
            q.stride(0), q.stride(1), q.stride(2),
            k_fp8.stride(0), k_fp8.stride(1),
            k_scale.stride(0),
            topk_idx.stride(0), topk_idx.stride(1),
            logits.stride(0), logits.stride(1),
            weights.stride(0), weights.stride(1),
            skv,
            topk,
            HEADS=H, DIM=D,
            KV_BLOCK_SIZE=K,
            GROUP_SIZE=GROUP_SIZE,
        )
    try:
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
        del q, k_fp8, k_scale, weights, cu_ks, cu_ke, topk_idx, logits
        torch.cuda.empty_cache()
        return s.elapsed_time(e) * 1e3 / iters
    except Exception as ex:
        del q, k_fp8, k_scale, weights, cu_ks, cu_ke, topk_idx, logits
        torch.cuda.empty_cache()
        return f"ERR:{type(ex).__name__}"


TILES = [64, 128, 256, 512]


def fmt(t):
    return f"{t:>8.0f}" if isinstance(t, float) else f"{t:>8}"


def main():
    print("=" * 100)
    print("GEMM_TILE sweep — prefill stage 4 grouped (μs/call)")
    print("=" * 100)
    print(f"{'K':>3} {'sq':>5} {'skv':>7} {'topk':>5} | "
          + " ".join(f"T{t}".rjust(9) for t in TILES) + " |   best   speedup_vs_T256")
    print("-" * 100)

    cases = [
        # K=16 across all 16 prefill chunks, picking representative skv
        (16,  8192,   8192, 2048),
        (16,  8192,  16384, 2048),
        (16,  8192,  32768, 2048),
        (16,  8192,  65536, 2048),
        (16,  8192, 131072, 2048),
        # K=32 same
        (32,  8192,   8192, 2048),
        (32,  8192,  16384, 2048),
        (32,  8192,  32768, 2048),
        (32,  8192,  65536, 2048),
        (32,  8192, 131072, 2048),
    ]
    for K, sq, skv, topk in cases:
        results = [bench(K, sq, skv, topk, t) for t in TILES]
        floats = [t for t in results if isinstance(t, float)]
        best = min(floats) if floats else None
        baseline = next((t for tile, t in zip(TILES, results)
                         if tile == 256 and isinstance(t, float)), None)
        speedup = f"{baseline / best:.2f}x" if best and baseline else "n/a"
        # Find which tile wins
        winner = None
        for tile, t in zip(TILES, results):
            if isinstance(t, float) and t == best:
                winner = tile
                break
        line = (f"{K:>3} {sq:>5} {skv:>7} {topk:>5} | "
                + " ".join(fmt(t) for t in results)
                + f" |  T{winner:>3} {best:>5.0f}  {speedup}")
        print(line)


if __name__ == "__main__":
    main()
