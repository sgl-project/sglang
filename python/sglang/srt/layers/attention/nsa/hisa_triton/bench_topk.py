"""Microbench torch.topk at production hisa stage 3 shapes.

Production formula: block_topk = 8192 // k_block_size
  K=16  → block_topk=512, num_blocks ≤ ctx/16  = 8192 (at 128K)
  K=32  → block_topk=256, num_blocks ≤ ctx/32  = 4096
  K=64  → block_topk=128, num_blocks ≤ ctx/64  = 2048
  K=128 → block_topk=64,  num_blocks ≤ ctx/128 = 1024

Stage 3 shapes:
  Prefill ragged: score [seq, num_blocks_max] f32 (or bf16). seq=8192.
  Decode paged:   score [B, max_pool_pages] f32. B ∈ {1, 4, 32}.

Measures: torch.topk(f32) vs torch.topk(bf16) μs/call across (rows, cols, k).
"""
from __future__ import annotations

import torch


DEVICE = torch.device("cuda")


def make_score(rows, cols, dtype=torch.float32):
    torch.manual_seed(0)
    return torch.randn(rows, cols, device=DEVICE, dtype=dtype)


def bench(fn, warmup=20, iters=100):
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


def main():
    print("=" * 100)
    print("torch.topk @ production hisa stage 3 shapes (μs/call)")
    print("=" * 100)
    print(f"{'mode':>10} {'rows':>5} {'cols':>5} {'k':>4} | {'f32':>8} {'bf16':>8} | {'best':>8}")
    print("-" * 100)

    # Prefill ragged: rows = seq_q = 8192. cols = num_pool_blocks_max
    # (K=16: 128K/16=8192; K=32: 4096; etc.)
    prefill_cases = [
        # (label, rows, cols, k)
        ("prefill",  8192, 8192, 512),  # K=16
        ("prefill",  8192, 4096, 256),  # K=32
        ("prefill",  8192, 2048, 128),  # K=64
        ("prefill",  8192, 1024,  64),  # K=128
    ]
    # Decode paged: rows = B. cols = max_pool_pages.
    decode_cases = [
        ("decode",   1, 8192, 512),  # B=1, K=16
        ("decode",   1, 4096, 256),  # B=1, K=32
        ("decode",   1, 2048, 128),  # B=1, K=64
        ("decode",   1, 1024,  64),  # B=1, K=128
        ("decode",   4, 8192, 512),
        ("decode",   4, 4096, 256),
        ("decode",   4, 2048, 128),
        ("decode",   4, 1024,  64),
        ("decode",  32, 8192, 512),
        ("decode",  32, 4096, 256),
        ("decode",  32, 2048, 128),
        ("decode",  32, 1024,  64),
    ]

    total_prefill = {"f32": 0.0, "bf16": 0.0}
    total_decode = {"f32": 0.0, "bf16": 0.0}

    for label, rows, cols, k in prefill_cases + decode_cases:
        s_f32 = make_score(rows, cols, torch.float32)
        s_bf16 = s_f32.to(torch.bfloat16)

        t_f32 = bench(lambda: torch.topk(s_f32, k=k, dim=-1, sorted=False))
        t_bf = bench(lambda: torch.topk(s_bf16, k=k, dim=-1, sorted=False))
        best = min(t_f32, t_bf)
        print(f"{label:>10} {rows:>5} {cols:>5} {k:>4} | "
              f"{t_f32:>8.1f} {t_bf:>8.1f} | {best:>8.1f}")
        if label == "prefill":
            total_prefill["f32"] += t_f32
            total_prefill["bf16"] += t_bf
        del s_f32, s_bf16

    print("-" * 100)
    print(f"prefill totals (one stage 3 call per K-config × 16 chunks "
          f"× 61 layers × B):")
    print(f"  f32 sum: {total_prefill['f32']:.1f}μs/chunk")
    print(f"  bf16 sum: {total_prefill['bf16']:.1f}μs/chunk")


if __name__ == "__main__":
    main()
