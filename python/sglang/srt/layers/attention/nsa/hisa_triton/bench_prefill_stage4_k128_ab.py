"""Prefill stage 4 K=128 A/B: tilelang
``fp8_native_block_sparse_mqa_attn_return_logits_interface`` vs triton
``block_sparse_mqa_triton`` (legacy BLOCK_N=128 path).

Reuses the e2e prefill input generator and replicates the topk-index
distribution from running stages 1+2+3 on the same inputs (so the K
gather pattern is realistic, not random).
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    block_sparse_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa_triton.orchestrator import (
    fp8_native_hierarchy_mqa_logits,
)
from sglang.srt.layers.attention.nsa.hisa_triton.test_e2e_simulated import (
    BLOCK_TOPK_FORMULA, D, H, PREFILL_CHUNK, make_prefill_inputs,
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


def get_random_topk(K, sq, cu_ke, block_topk):
    """Random valid topk indices: per-row uniform in [0, ke[row] // K).
    The kernel time is dominated by the gather + GEMM, not by index
    distribution, so random ≈ realistic for timing."""
    torch.manual_seed(2)
    ke_blocks = ((cu_ke + K - 1) // K).long()  # [sq]
    n_blocks_max = int(ke_blocks.max().item())
    idx = torch.randint(
        0, n_blocks_max, (sq, block_topk),
        device=cu_ke.device, dtype=torch.int64,
    )
    idx = torch.minimum(idx, (ke_blocks - 1).clamp_min(0).unsqueeze(1))
    return idx


def main():
    K = 128
    block_topk = BLOCK_TOPK_FORMULA // K  # 64
    sq = PREFILL_CHUNK
    print("=" * 100)
    print(f"Prefill stage 4 K=128 A/B (sq={sq}, block_topk={block_topk})")
    print("=" * 100)
    print(f"{'skv':>5} | {'tile (μs)':>10} {'tri (μs)':>10} {'tri/tile':>9} "
          f"{'Δ (μs)':>8}")
    print("-" * 100)

    for skv_k in (16, 32, 65, 128):
        skv = skv_k * 1024
        q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)

        topk_idx = get_random_topk(K, sq, cu_ke, block_topk)
        # Stage 4 takes raw cu_seqlen (in tokens), not blocks. Confirm by
        # reading the kernel signatures.

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
                topk_block_index=topk_idx,
                kv_block_size=K, weights=w,
                cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
            )

        # Sanity: shapes match.
        out_tile = call_tile()
        out_tri = call_tri()
        assert out_tile.shape == out_tri.shape, (
            f"shape mismatch: tile {out_tile.shape} vs tri {out_tri.shape}"
        )

        t_tile = bench_eager(call_tile)
        t_tri = bench_eager(call_tri)
        ratio = t_tri / t_tile
        delta = t_tri - t_tile
        print(f"{skv//1024:>3}K | {t_tile:>10.1f} {t_tri:>10.1f} "
              f"{ratio:>8.2f}x {delta:>+8.1f}")

        del q, k_fp8, k_scale, w, cu_ks, cu_ke, topk_idx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
