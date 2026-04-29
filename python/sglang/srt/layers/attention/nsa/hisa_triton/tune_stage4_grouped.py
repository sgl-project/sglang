"""Sweep (num_warps, num_stages) for stage 4 grouped kernels.

Two kernels under test:
  - _block_sparse_mqa_grouped_kernel  (prefill, K<128, GEMM_TILE=256)
  - _sparse_paged_mqa_grouped_kernel  (decode,  K<64,  GEMM_TILE=64)

Production-realistic shapes:
  Prefill: sq=8192, skv ∈ {32K, 128K} (mid + late chunk), K ∈ {16, 32}
  Decode:  B=1, ctx=128K, K ∈ {16, 32}

Reports μs/call per (kw, ns) combo. Pick the lowest per (K, shape).
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    _block_sparse_mqa_grouped_kernel,
    _sparse_paged_mqa_grouped_kernel,
)


DEVICE = torch.device("cuda")
H, D = 64, 128
PAGED = 64


# ---------------------------------------------------------------------------
# Prefill grouped: bench at fixed shape with explicit num_warps / num_stages
# ---------------------------------------------------------------------------

def bench_prefill_grouped(K, sq, skv, topk, num_warps, num_stages,
                          warmup=10, iters=50):
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

    GEMM_TILE = 256
    GROUP_SIZE = GEMM_TILE // K
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
            num_warps=num_warps,
            num_stages=num_stages,
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
        return s.elapsed_time(e) * 1e3 / iters  # μs
    except Exception as ex:
        del q, k_fp8, k_scale, weights, cu_ks, cu_ke, topk_idx, logits
        torch.cuda.empty_cache()
        return f"ERR:{type(ex).__name__}"


# ---------------------------------------------------------------------------
# Decode grouped: same idea, paged inputs
# ---------------------------------------------------------------------------

def bench_decode_grouped(K, B, ctx, topk, num_warps, num_stages,
                         warmup=10, iters=50):
    torch.manual_seed(1)
    num_kv_blocks_per_req = (ctx + PAGED - 1) // PAGED
    num_phys = B * num_kv_blocks_per_req + 16

    q = torch.randn(B, 1, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    kv_cache = torch.randint(
        0, 256, (num_phys, PAGED, 1, D + 4), device=DEVICE, dtype=torch.uint8,
    )
    weights = torch.randn(B, 1, H, device=DEVICE, dtype=torch.float32)
    context_lens = torch.full((B,), ctx, device=DEVICE, dtype=torch.int32)
    block_tables = torch.stack([
        torch.arange(b * num_kv_blocks_per_req, (b + 1) * num_kv_blocks_per_req,
                     device=DEVICE, dtype=torch.int32) for b in range(B)
    ])
    num_kv_blocks = ctx // K
    topk_idx = torch.randint(
        0, num_kv_blocks, (B, 1, topk), device=DEVICE, dtype=torch.int64,
    )

    kv_cache_flat = kv_cache.view(num_phys, -1)
    kv_fp8_view = kv_cache_flat.view(torch.float8_e4m3fn)
    kv_f32_view = kv_cache_flat.view(torch.float32)

    GROUP_SIZE = PAGED // K
    GEMM_TILE = PAGED
    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE
    logits = torch.empty((B, 1, topk * K), device=DEVICE, dtype=torch.float32)
    grid = (B, 1, num_chunks)

    def fn():
        _sparse_paged_mqa_grouped_kernel[grid](
            q, kv_fp8_view, kv_f32_view, topk_idx,
            logits, weights, context_lens, block_tables,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            kv_fp8_view.stride(0), kv_fp8_view.stride(1),
            kv_f32_view.stride(0), kv_f32_view.stride(1),
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            logits.stride(0), logits.stride(1), logits.stride(2),
            weights.stride(0), weights.stride(1), weights.stride(2),
            block_tables.stride(0), block_tables.stride(1),
            num_kv_blocks_per_req,
            num_phys,
            topk,
            PAGED_BLOCK_SIZE=PAGED,
            KV_BLOCK_SIZE=K,
            HEADS=H,
            DIM=D,
            GROUP_SIZE=GROUP_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
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
        del q, kv_cache, weights, context_lens, block_tables, topk_idx, logits
        torch.cuda.empty_cache()
        return s.elapsed_time(e) * 1e3 / iters
    except Exception as ex:
        del q, kv_cache, weights, context_lens, block_tables, topk_idx, logits
        torch.cuda.empty_cache()
        return f"ERR:{type(ex).__name__}"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

CONFIGS = [
    (4, 2),  # default
    (4, 3),
    (4, 4),
    (8, 2),
    (8, 3),
    (8, 4),
]


def fmt(t):
    return f"{t:>7.0f}" if isinstance(t, float) else f"{t:>7}"


def main():
    print("=" * 100)
    print("PREFILL grouped kernel — μs/call across (num_warps, num_stages)")
    print("=" * 100)
    print(f"{'K':>3} {'sq':>5} {'skv':>7} {'topk':>5} | "
          + " ".join(f"w{w}s{s}".rjust(8) for w, s in CONFIGS) + " |   best (us)  speedup")
    print("-" * 100)

    prefill_cases = [
        (16,  8192,  32768, 2048),
        (16,  8192, 131072, 2048),
        (32,  8192,  65536, 2048),
        (32,  8192, 131072, 2048),
    ]
    for K, sq, skv, topk in prefill_cases:
        results = [bench_prefill_grouped(K, sq, skv, topk, w, s) for w, s in CONFIGS]
        floats = [t for t in results if isinstance(t, float)]
        best = min(floats) if floats else None
        baseline = results[0] if isinstance(results[0], float) else None
        speedup = f"{baseline / best:.2f}x" if best and baseline else "n/a"
        line = (f"{K:>3} {sq:>5} {skv:>7} {topk:>5} | "
                + " ".join(fmt(t) for t in results)
                + f" |   {best:>7.0f}  {speedup}")
        print(line)

    print()
    print("=" * 100)
    print("DECODE grouped kernel — μs/call across (num_warps, num_stages)")
    print("=" * 100)
    print(f"{'K':>3} {'B':>3} {'ctx':>7} {'topk':>5} | "
          + " ".join(f"w{w}s{s}".rjust(8) for w, s in CONFIGS) + " |   best (us)  speedup")
    print("-" * 100)
    decode_cases = [
        (16, 1, 131072, 2048),
        (16, 4, 131072, 2048),
        (32, 1, 131072, 2048),
        (32, 4, 131072, 2048),
    ]
    for K, B, ctx, topk in decode_cases:
        results = [bench_decode_grouped(K, B, ctx, topk, w, s) for w, s in CONFIGS]
        floats = [t for t in results if isinstance(t, float)]
        best = min(floats) if floats else None
        baseline = results[0] if isinstance(results[0], float) else None
        speedup = f"{baseline / best:.2f}x" if best and baseline else "n/a"
        line = (f"{K:>3} {B:>3} {ctx:>7} {topk:>5} | "
                + " ".join(fmt(t) for t in results)
                + f" |   {best:>7.0f}  {speedup}")
        print(line)


if __name__ == "__main__":
    main()
