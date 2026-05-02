"""Final A/B: unified persistent block_sparse_mqa_triton vs tilelang
across all K∈{16,32,64,128} at production shapes.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    block_sparse_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa_triton.test_e2e_simulated import (
    BLOCK_TOPK_FORMULA, PREFILL_CHUNK, make_prefill_inputs,
)


def bench_eager(fn, w=20, n=50):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / n


def random_topk(K, sq, cu_ke, block_topk, seed=2):
    torch.manual_seed(seed)
    ke_blocks = ((cu_ke + K - 1) // K).long()
    n_blocks_max = int(ke_blocks.max().item())
    idx = torch.randint(
        0, n_blocks_max, (sq, block_topk),
        device=cu_ke.device, dtype=torch.int64,
    )
    idx = torch.minimum(idx, (ke_blocks - 1).clamp_min(0).unsqueeze(1))
    return idx


def main():
    sq = PREFILL_CHUNK
    print("=" * 100)
    print(f"Prefill stage 4 unified A/B (sq={sq})")
    print("=" * 100)
    print(f"{'K':>4} {'skv':>5} {'topk':>5} | "
          f"{'tile (μs)':>10} {'tri (μs)':>10} {'speedup':>8} {'Δ (μs)':>9}")
    print("-" * 100)

    for K in (16, 32, 64, 128):
        block_topk = BLOCK_TOPK_FORMULA // K
        for skv_k in (16, 32, 65, 128):
            skv = skv_k * 1024
            q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)
            topk_idx = random_topk(K, sq, cu_ke, block_topk)

            def call_tile():
                return fp8_native_block_sparse_mqa_attn_return_logits_interface(
                    q=q, k=k_fp8, k_scale=k_scale,
                    topk_block_index=topk_idx,
                    kv_block_size=K, weights=w,
                    cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
                )
            def call_tri():
                return block_sparse_mqa_triton(
                    q_fp8=q, k_fp8=k_fp8, k_scale=k_scale,
                    topk_block_index=topk_idx, kv_block_size=K, weights=w,
                    cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
                )

            t_tile = bench_eager(call_tile)
            t_tri = bench_eager(call_tri)
            sp = t_tile / t_tri
            print(f"{K:>4} {skv_k:>3}K {block_topk:>5} | "
                  f"{t_tile:>10.1f} {t_tri:>10.1f} "
                  f"{sp:>7.2f}x {t_tile-t_tri:>+9.1f}")

            del q, k_fp8, k_scale, w, cu_ks, cu_ke, topk_idx
            torch.cuda.empty_cache()
        print("-" * 100)


if __name__ == "__main__":
    main()
