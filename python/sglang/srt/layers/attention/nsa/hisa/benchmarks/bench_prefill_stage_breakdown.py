"""Per-stage prefill profile with stage-2 A/B (tilelang vs DeepGEMM).

Aligned with ``test_e2e_simulated``: same input generator (16 chunks at
sq=8192, skv ∈ {8K..128K}), same H/D, same K sweep. Times each of the 4
orchestrator stages in isolation, plus the orchestrator total. Stage 2
shows both the legacy tilelang path (A) and the current DG-based path (B).

Output per K:
  - Per-stage μs at the longest chunk (skv=128K) — the worst case
  - 16-chunk SUM (= per-req contribution at B=1, single-layer; multiply
    by 61 layers × B for full e2e bookkeeping)
"""
from __future__ import annotations

import deep_gemm
import torch

from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
    fp8_native_block_mean_pooling_grouped_interface,
    fp8_native_block_mean_pooling_interface,
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
    pool_mqa_attn_return_logits_fp8_interface,
)
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    block_sparse_mqa_triton,
    force_maintain_logits_triton,
)
from sglang.srt.layers.attention.nsa.hisa.orchestrator import (
    _stage3_topk_prefill,
    fp8_native_hierarchy_mqa_logits,
)
from sglang.srt.layers.attention.nsa.hisa.tests.test_e2e_simulated import (
    H, D, PREFILL_CHUNK, N_PREFILL_CHUNKS, BLOCK_TOPK_FORMULA,
    make_prefill_inputs,
)


DEVICE = torch.device("cuda")


def stage1(k_fp8, k_scale, K):
    if K < 64:
        return fp8_native_block_mean_pooling_grouped_interface(k_fp8, k_scale, K)
    return fp8_native_block_mean_pooling_interface(k_fp8, k_scale, K)


def stage2_tilelang(q, bk, bks, K, w, cu_ks_blk, cu_ke_blk):
    return pool_mqa_attn_return_logits_fp8_interface(
        q_fp8=q, blocked_kv_fp8=bk, blocked_kv_scale=bks,
        kv_block_size=K, weights_f32=w,
        cu_seqlen_blocked_ks=cu_ks_blk,
        cu_seqlen_blocked_ke=cu_ke_blk,
        clean_logits=True, force_maintain=True,
    )


def stage2_dg(q, bk, bks, w, cu_ks_blk, cu_ke_blk, n_blocks):
    out = deep_gemm.fp8_mqa_logits(
        q, (bk, bks), w, cu_ks_blk, cu_ke_blk, clean_logits=True,
    )[:, :n_blocks]
    force_maintain_logits_triton(out, cu_ks_blk, cu_ke_blk)
    return out


def stage3(score, K, block_topk):
    return _stage3_topk_prefill(score, block_topk, K)


def stage4(q, k_fp8, k_scale, topk_idx, K, w, cu_ks, cu_ke):
    if K == 128:
        return fp8_native_block_sparse_mqa_attn_return_logits_interface(
            q=q, k=k_fp8, k_scale=k_scale, topk_block_index=topk_idx,
            kv_block_size=K, weights=w, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        )
    return block_sparse_mqa_triton(
        q_fp8=q, k_fp8=k_fp8, k_scale=k_scale, topk_block_index=topk_idx,
        kv_block_size=K, weights=w, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )


def cuda_bench(fn, warmup=10, iters=50):
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
    return s.elapsed_time(e) * 1e3 / iters


def profile_chunk(K, sq, skv, block_topk):
    """Return dict of per-stage μs at one (K, skv)."""
    q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)

    # Set up shared inputs once.
    bk, bks = stage1(k_fp8, k_scale, K)
    n_blocks = (skv + K - 1) // K
    bk = bk[:n_blocks].contiguous()
    bks = bks[:n_blocks].contiguous()
    cu_ks_blk = cu_ks // K
    cu_ke_blk = (cu_ke + K - 1) // K
    score_a = stage2_tilelang(q, bk, bks, K, w, cu_ks_blk, cu_ke_blk)
    topk_idx = stage3(score_a, K, block_topk)

    t = {}
    t["stage1"] = cuda_bench(lambda: stage1(k_fp8, k_scale, K))
    t["stage2_tile"] = cuda_bench(
        lambda: stage2_tilelang(q, bk, bks, K, w, cu_ks_blk, cu_ke_blk)
    )
    t["stage2_dg"] = cuda_bench(
        lambda: stage2_dg(q, bk, bks, w, cu_ks_blk, cu_ke_blk, n_blocks)
    )
    t["stage3"] = cuda_bench(lambda: stage3(score_a, K, block_topk))
    t["stage4"] = cuda_bench(
        lambda: stage4(q, k_fp8, k_scale, topk_idx, K, w, cu_ks, cu_ke)
    )
    t["orch_total"] = cuda_bench(lambda: fp8_native_hierarchy_mqa_logits(
        q, (k_fp8, k_scale), w, cu_ks, cu_ke, K, block_topk,
    ))
    return t


def main():
    sq = PREFILL_CHUNK
    skv_list = [(i + 1) * PREFILL_CHUNK for i in range(N_PREFILL_CHUNKS)]
    skv_max = skv_list[-1]
    print("=" * 110)
    print(f"prefill stage breakdown   sq={sq}  H={H}  D={D}  "
          f"chunks={N_PREFILL_CHUNKS} (skv 8K..128K)")
    print("=" * 110)

    for K in (16, 32, 64, 128):
        block_topk = BLOCK_TOPK_FORMULA // K
        # Per-skv breakdown.
        per_chunk = []
        for skv in skv_list:
            per_chunk.append(profile_chunk(K, sq, skv, block_topk))
            torch.cuda.empty_cache()

        # SUM across 16 chunks per stage.
        sums = {k: sum(d[k] for d in per_chunk) for k in per_chunk[0]}
        worst = per_chunk[-1]  # skv=128K row

        # Implied orchestrator with each stage 2.
        # orch already times the production path (DG). For the legacy path
        # we estimate as orch - stage2_dg + stage2_tile (stages 1/3/4 are
        # invariant). Sanity-check: we also have orch_total directly.
        sums["orch_with_tile"] = (
            sums["orch_total"] - sums["stage2_dg"] + sums["stage2_tile"]
        )
        worst["orch_with_tile"] = (
            worst["orch_total"] - worst["stage2_dg"] + worst["stage2_tile"]
        )

        print()
        print(f"--- K={K}   block_topk={block_topk} ---")
        print(f"  {'stage':<28} {'skv=128K (μs)':>15} "
              f"{'16-chunk SUM (μs)':>20} {'% of orch_dg':>14}")
        print("  " + "-" * 80)
        order = [
            ("stage1 (mean-pool)", "stage1"),
            ("stage2A (tilelang)", "stage2_tile"),
            ("stage2B (DG+ftm) [prod]", "stage2_dg"),
            ("stage3 (topk)", "stage3"),
            ("stage4 (sparse-mqa)", "stage4"),
            ("orchestrator (DG)", "orch_total"),
            ("orchestrator (tile, est)", "orch_with_tile"),
        ]
        for label, key in order:
            pct = sums[key] / sums["orch_total"] * 100
            print(f"  {label:<28} {worst[key]:>15.1f} "
                  f"{sums[key]:>20.0f} {pct:>13.1f}%")
        delta = sums["stage2_tile"] - sums["stage2_dg"]
        speedup_stage2 = sums["stage2_tile"] / sums["stage2_dg"]
        speedup_orch = sums["orch_with_tile"] / sums["orch_total"]
        print(f"  stage2 saved per req     {worst['stage2_tile']-worst['stage2_dg']:>+15.1f} "
              f"{delta:>+20.0f}   stage2 {speedup_stage2:.2f}× / orch {speedup_orch:.2f}×")


if __name__ == "__main__":
    main()
