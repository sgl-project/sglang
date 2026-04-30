"""E2E-simulated kernel-level benchmark for hisa.

Workload (per request):
  - 128K context built via 16 chunked-prefills × 8K each
  - 256 decode steps at ctx=128K
  - 61 layers (DeepSeek-V3)

Measures per-call wall-time of the prefill / decode orchestrators (which call
all 4 stages internally), then aggregates:

  prefill_total = B × Σ(per-chunk × LAYERS)
  decode_total  = 256 × per-step(B, ctx=128K) × LAYERS
  e2e           = prefill_total + decode_total

Sweeps K ∈ {16, 32}, B ∈ {1, 4}.

Usage:
    python -m sglang.srt.layers.attention.nsa.hisa_triton.test_e2e_simulated

This is the canonical "did my optimization help" tool. Ref the percentage
breakdown (prefill / decode share) when picking what to optimize next.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa_triton.orchestrator import (
    fp8_native_hierarchy_mqa_logits,
    fp8_native_hierarchy_paged_mqa_logits,
)


DEVICE = torch.device("cuda")
H, D = 64, 128
PAGED = 64
POOL_PAGE = 64
LAYERS = 61
PREFILL_CHUNK = 8192
N_PREFILL_CHUNKS = 16
N_DECODE_STEPS = 256
CTX = N_PREFILL_CHUNKS * PREFILL_CHUNK  # 128K
BLOCK_TOPK_FORMULA = 8192  # Production: block_topk = BLOCK_TOPK_FORMULA // K
                            # → K=16:512, K=32:256, K=64:128, K=128:64


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------

def make_prefill_inputs(sq, skv):
    """Single ragged sequence: cu_seqlen_ke spans [skv//4, skv] linearly."""
    torch.manual_seed(0)
    q = torch.randn(sq, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(skv, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_scale = 0.05 + 0.02 * torch.rand(skv, device=DEVICE, dtype=torch.float32)
    weights = torch.randn(sq, H, device=DEVICE, dtype=torch.float32)
    cu_ks = torch.zeros(sq, device=DEVICE, dtype=torch.int32)
    cu_ke = torch.linspace(skv // 4, skv, sq, device=DEVICE).to(torch.int32)
    return q, (k_fp8, k_scale), weights, cu_ks, cu_ke


def make_decode_inputs(K, B, ctx):
    """B requests each at ctx tokens; pages are non-overlapping per request."""
    torch.manual_seed(1)
    num_kv_blocks_per_req = (ctx + PAGED - 1) // PAGED            # 2048 for ctx=128K
    num_phys = B * num_kv_blocks_per_req + 16                      # small margin
    num_pool_blocks_per_req = (ctx + K - 1) // K                   # 8K for K=16
    num_pool_pages_per_req = (
        num_pool_blocks_per_req + POOL_PAGE - 1
    ) // POOL_PAGE                                                  # 128 for K=16
    num_pool_phys = B * num_pool_pages_per_req + 8

    q = torch.randn(B, 1, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    kv_cache = torch.randint(
        0, 256, (num_phys, PAGED, 1, D + 4), device=DEVICE, dtype=torch.uint8,
    )
    pool_k_pages = torch.randint(
        0, 256, (num_pool_phys, POOL_PAGE * (D + 4)),
        device=DEVICE, dtype=torch.uint8,
    )
    weights = torch.randn(B, 1, H, device=DEVICE, dtype=torch.float32)
    context_lens = torch.full((B,), ctx, device=DEVICE, dtype=torch.int32)

    # Per-req non-overlapping page tables.
    block_tables = torch.stack([
        torch.arange(
            b * num_kv_blocks_per_req, (b + 1) * num_kv_blocks_per_req,
            device=DEVICE, dtype=torch.int32,
        )
        for b in range(B)
    ])
    pool_page_tables = torch.stack([
        torch.arange(
            b * num_pool_pages_per_req, (b + 1) * num_pool_pages_per_req,
            device=DEVICE, dtype=torch.int32,
        )
        for b in range(B)
    ])

    return (q, kv_cache, pool_k_pages, pool_page_tables, weights,
            context_lens, block_tables)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def cuda_bench(fn, warmup=10, iters=50):
    """CUDA-event-based timing. Returns μs/call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    # elapsed_time is ms.
    return start.elapsed_time(end) * 1e3 / iters


# ---------------------------------------------------------------------------
# Per-config benchmark
# ---------------------------------------------------------------------------

def bench_prefill(K):
    """16 prefill chunks; per chunk skv = (i+1) * 8K. Returns list of μs.

    Frees per-chunk inputs before next chunk to keep peak memory ~1.5GB
    (one chunk's intermediates).
    """
    block_topk = BLOCK_TOPK_FORMULA // K
    times = []
    for chunk_i in range(N_PREFILL_CHUNKS):
        skv = (chunk_i + 1) * PREFILL_CHUNK
        sq = PREFILL_CHUNK
        q, kv, w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)
        t = cuda_bench(lambda: fp8_native_hierarchy_mqa_logits(
            q, kv, w, cu_ks, cu_ke, K, block_topk,
        ))
        times.append(t)
        del q, kv, w, cu_ks, cu_ke
        torch.cuda.empty_cache()
    return times


def bench_decode_step(K, B, ctx):
    block_topk = BLOCK_TOPK_FORMULA // K
    inputs = make_decode_inputs(K, B, ctx)
    q, kv, pp, ppt, w, ctxl, bt = inputs
    t = cuda_bench(lambda: fp8_native_hierarchy_paged_mqa_logits(
        q_fp8=q, kv_cache_fp8=kv, pool_k_pages=pp, pool_page_tables=ppt,
        weights=w, context_lens=ctxl, block_tables=bt,
        k_block_size=K, pool_page_size=POOL_PAGE, block_topk=block_topk,
    ))
    del q, kv, pp, ppt, w, ctxl, bt
    torch.cuda.empty_cache()
    return t


def fmt_us_list(times):
    return " ".join(f"{t:>5.0f}" for t in times)


def main():
    print("=" * 100)
    print(f"hisa kernel-level e2e bench   "
          f"ctx={CTX//1024}K  prefill_chunk={PREFILL_CHUNK}  "
          f"decode_steps={N_DECODE_STEPS}  layers={LAYERS}")
    print("=" * 100)

    rows = []  # for summary

    for K in (16, 32):
        prefill_times = bench_prefill(K)  # B-independent (single ragged seq)
        prefill_per_req_us = sum(prefill_times) * LAYERS  # 16 chunks × 61 layers
        for B in (1, 4):
            decode_step_us = bench_decode_step(K, B, ctx=CTX)
            decode_total_us = decode_step_us * N_DECODE_STEPS * LAYERS
            prefill_total_us = prefill_per_req_us * B
            e2e_ms = (prefill_total_us + decode_total_us) / 1e3

            print()
            print(f"--- K={K}  B={B} ---")
            print(f"prefill chunks (μs/call, skv=8K..128K):")
            print(f"  {fmt_us_list(prefill_times)}")
            print(f"prefill: per-chunk-sum={sum(prefill_times):>7.0f}μs × "
                  f"{LAYERS}L × B={B} = {prefill_total_us/1e3:>7.1f} ms "
                  f"({prefill_total_us / e2e_ms / 10:>5.1f}%)")
            print(f"decode:  {decode_step_us:>7.0f}μs × {N_DECODE_STEPS}step × "
                  f"{LAYERS}L = {decode_total_us/1e3:>7.1f} ms "
                  f"({decode_total_us / e2e_ms / 10:>5.1f}%)")
            print(f"e2e: {e2e_ms:.1f} ms")
            rows.append((K, B, prefill_total_us/1e3, decode_total_us/1e3, e2e_ms))

    print()
    print("=" * 100)
    print(f"{'K':>4} {'B':>3} | {'prefill ms':>11} {'decode ms':>10} {'e2e ms':>10}")
    print("-" * 100)
    for K, B, p, d, e in rows:
        print(f"{K:>4} {B:>3} | {p:>11.1f} {d:>10.1f} {e:>10.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
