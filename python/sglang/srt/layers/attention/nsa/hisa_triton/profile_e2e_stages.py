"""Per-stage profile for e2e-simulated workload.

Mirrors the orchestrator logic from ``hisa_triton/orchestrator.py`` but
times each stage individually with CUDA events. Outputs a breakdown of
prefill / decode wall-time per stage at K ∈ {16, 32}, B ∈ {1, 4} on the
production-shape workload (ctx=128K, prefill_chunk=8K, decode_steps=256,
layers=61).

Read-only — does NOT modify any production kernels.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_mean_pooling_grouped_interface,
    fp8_native_block_mean_pooling_interface,
    pool_mqa_attn_return_logits_fp8_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    batch_decode_pool_mqa_triton,
    block_sparse_mqa_triton,
    sparse_paged_mqa_triton,
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
BLOCK_TOPK = 2048


# ---------------------------------------------------------------------------
# Per-stage prefill (mirrors fp8_native_hierarchy_mqa_logits)
# ---------------------------------------------------------------------------

def prefill_stage1(k_fp8, k_scales, K):
    if K < 64:
        return fp8_native_block_mean_pooling_grouped_interface(k_fp8, k_scales, K)
    return fp8_native_block_mean_pooling_interface(k_fp8, k_scales, K)


def prefill_stage2(q_fp8, blocked_k_fp8, blocked_k_scale, K, weights, cu_ks, cu_ke):
    cu_ks_blk = cu_ks // K
    cu_ke_blk = (cu_ke + K - 1) // K
    return pool_mqa_attn_return_logits_fp8_interface(
        q_fp8=q_fp8,
        blocked_kv_fp8=blocked_k_fp8,
        blocked_kv_scale=blocked_k_scale,
        kv_block_size=K,
        weights_f32=weights,
        cu_seqlen_blocked_ks=cu_ks_blk,
        cu_seqlen_blocked_ke=cu_ke_blk,
    )


def prefill_stage3(score, block_topk):
    topk_actual = min(block_topk, score.shape[-1])
    return torch.topk(
        score.bfloat16(), k=topk_actual, dim=-1, sorted=False,
    ).indices


def prefill_stage4(q_fp8, k_fp8, k_scales, topk_idx, K, weights, cu_ks, cu_ke):
    return block_sparse_mqa_triton(
        q_fp8=q_fp8, k_fp8=k_fp8, k_scale=k_scales,
        topk_block_index=topk_idx,
        kv_block_size=K, weights=weights,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )


# ---------------------------------------------------------------------------
# Per-stage decode (mirrors fp8_native_hierarchy_paged_mqa_logits, post-#1)
# ---------------------------------------------------------------------------

def decode_stage2(q_fp8, pool_k_pages, pool_page_tables, weights,
                  num_pool_blocks_per_req):
    return batch_decode_pool_mqa_triton(
        q_fp8=q_fp8,
        pool_k_pages=pool_k_pages,
        pool_page_tables=pool_page_tables,
        weights_f32=weights,
        context_lens_pool=num_pool_blocks_per_req,
        pool_page_size=POOL_PAGE,
    )


def decode_stage3(score, block_topk):
    return torch.topk(
        score, k=min(block_topk, score.shape[-1]), dim=-1, sorted=False,
    ).indices


def decode_stage4(q_fp8, kv_cache_fp8, topk_idx, K, weights,
                  context_lens, block_tables):
    return sparse_paged_mqa_triton(
        q_fp8=q_fp8, kv_cache_fp8=kv_cache_fp8,
        topk_block_index=topk_idx,
        kv_block_size=K, weights=weights,
        context_lens=context_lens, block_tables=block_tables,
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_prefill_inputs(sq, skv):
    torch.manual_seed(0)
    q = torch.randn(sq, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(skv, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_scale = 0.05 + 0.02 * torch.rand(skv, device=DEVICE, dtype=torch.float32)
    weights = torch.randn(sq, H, device=DEVICE, dtype=torch.float32)
    cu_ks = torch.zeros(sq, device=DEVICE, dtype=torch.int32)
    cu_ke = torch.linspace(skv // 4, skv, sq, device=DEVICE).to(torch.int32)
    return q, k_fp8, k_scale, weights, cu_ks, cu_ke


def make_decode_inputs(K, B, ctx):
    torch.manual_seed(1)
    num_kv_blocks_per_req = (ctx + PAGED - 1) // PAGED
    num_phys = B * num_kv_blocks_per_req + 16
    num_pool_blocks_per_req_v = (ctx + K - 1) // K
    num_pool_pages_per_req = (
        num_pool_blocks_per_req_v + POOL_PAGE - 1
    ) // POOL_PAGE
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
    block_tables = torch.stack([
        torch.arange(
            b * num_kv_blocks_per_req, (b + 1) * num_kv_blocks_per_req,
            device=DEVICE, dtype=torch.int32,
        ) for b in range(B)
    ])
    pool_page_tables = torch.stack([
        torch.arange(
            b * num_pool_pages_per_req, (b + 1) * num_pool_pages_per_req,
            device=DEVICE, dtype=torch.int32,
        ) for b in range(B)
    ])
    num_pool_blocks_per_req = torch.full(
        (B,), num_pool_blocks_per_req_v, device=DEVICE, dtype=torch.int32,
    )
    return (q, kv_cache, pool_k_pages, pool_page_tables, weights,
            context_lens, block_tables, num_pool_blocks_per_req)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

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
    return s.elapsed_time(e) * 1e3 / iters  # μs


# ---------------------------------------------------------------------------
# Per-stage timing per chunk / step
# ---------------------------------------------------------------------------

def profile_prefill_chunk(K, sq, skv):
    """Return (s1, s2, s3, s4) μs for a single chunk."""
    q, k_fp8, k_scale, w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)

    # Run stage 1 once to get its outputs (needed for stage 2/4 inputs).
    blocked_k_fp8, blocked_k_scale = prefill_stage1(k_fp8, k_scale, K)
    s2_score = prefill_stage2(q, blocked_k_fp8, blocked_k_scale, K, w, cu_ks, cu_ke)
    s3_topk_idx = prefill_stage3(s2_score, BLOCK_TOPK)

    s1 = cuda_bench(lambda: prefill_stage1(k_fp8, k_scale, K))
    s2 = cuda_bench(lambda: prefill_stage2(
        q, blocked_k_fp8, blocked_k_scale, K, w, cu_ks, cu_ke,
    ))
    s3 = cuda_bench(lambda: prefill_stage3(s2_score, BLOCK_TOPK))
    s4 = cuda_bench(lambda: prefill_stage4(
        q, k_fp8, k_scale, s3_topk_idx, K, w, cu_ks, cu_ke,
    ))

    del q, k_fp8, k_scale, w, cu_ks, cu_ke
    del blocked_k_fp8, blocked_k_scale, s2_score, s3_topk_idx
    torch.cuda.empty_cache()
    return s1, s2, s3, s4


def profile_decode_step(K, B, ctx):
    inp = make_decode_inputs(K, B, ctx)
    q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req = inp

    # Stage 1 (tail) is skipped per #1 — measure 0.
    s1 = 0.0
    s2_score = decode_stage2(q, pp, ppt, w, num_pool_per_req)
    s3_topk_idx = decode_stage3(s2_score, BLOCK_TOPK)

    s2 = cuda_bench(lambda: decode_stage2(q, pp, ppt, w, num_pool_per_req))
    s3 = cuda_bench(lambda: decode_stage3(s2_score, BLOCK_TOPK))
    s4 = cuda_bench(lambda: decode_stage4(q, kv, s3_topk_idx, K, w, ctxl, bt))

    del q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req, s2_score, s3_topk_idx
    torch.cuda.empty_cache()
    return s1, s2, s3, s4


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def main():
    print("=" * 110)
    print(f"Per-stage profile  ctx={CTX//1024}K  prefill_chunk={PREFILL_CHUNK}  "
          f"decode_steps={N_DECODE_STEPS}  layers={LAYERS}")
    print("=" * 110)

    for K in (16, 32):
        # Profile prefill chunks at the 3 representative skv values:
        # (early=8K, mid=64K, late=128K). 16 chunks span 8K..128K.
        # Sum across all 16 chunks by interpolation: assume linear so we
        # just measure all 16 to be precise (sums fast).
        print(f"\n--- K={K}  PREFILL chunks (per-chunk μs) ---")
        print(f"{'chunk':>5} {'skv':>7} | {'stage1':>7} {'stage2':>7} "
              f"{'stage3':>7} {'stage4':>7} | {'total':>7}")
        prefill_per_chunk = []
        for i in range(N_PREFILL_CHUNKS):
            skv = (i + 1) * PREFILL_CHUNK
            s1, s2, s3, s4 = profile_prefill_chunk(K, PREFILL_CHUNK, skv)
            tot = s1 + s2 + s3 + s4
            prefill_per_chunk.append((s1, s2, s3, s4, tot))
            print(f"{i:>5} {skv:>7} | {s1:>7.0f} {s2:>7.0f} {s3:>7.0f} "
                  f"{s4:>7.0f} | {tot:>7.0f}")

        sum_s1 = sum(c[0] for c in prefill_per_chunk)
        sum_s2 = sum(c[1] for c in prefill_per_chunk)
        sum_s3 = sum(c[2] for c in prefill_per_chunk)
        sum_s4 = sum(c[3] for c in prefill_per_chunk)
        sum_tot = sum_s1 + sum_s2 + sum_s3 + sum_s4
        print(f"{'sum':>5} {'all':>7} | {sum_s1:>7.0f} {sum_s2:>7.0f} "
              f"{sum_s3:>7.0f} {sum_s4:>7.0f} | {sum_tot:>7.0f}")
        print(f"    %share        | "
              f"{sum_s1/sum_tot*100:>6.1f}% {sum_s2/sum_tot*100:>6.1f}% "
              f"{sum_s3/sum_tot*100:>6.1f}% {sum_s4/sum_tot*100:>6.1f}%")

        for B in (1, 4):
            print(f"\n--- K={K}  B={B}  DECODE step (μs) ---")
            print(f"{'stage1':>7} {'stage2':>7} {'stage3':>7} {'stage4':>7} | "
                  f"{'total':>7}")
            d1, d2, d3, d4 = profile_decode_step(K, B, ctx=CTX)
            d_tot = d1 + d2 + d3 + d4
            print(f"{d1:>7.0f} {d2:>7.0f} {d3:>7.0f} {d4:>7.0f} | {d_tot:>7.0f}")
            print(f"%share: 0%, {d2/d_tot*100:.1f}%, "
                  f"{d3/d_tot*100:.1f}%, {d4/d_tot*100:.1f}%")

            # E2E aggregate per (K, B)
            prefill_total_ms = sum_tot * LAYERS * B / 1e3
            decode_total_ms = d_tot * N_DECODE_STEPS * LAYERS / 1e3
            e2e_ms = prefill_total_ms + decode_total_ms
            print(f"e2e (K={K} B={B}): "
                  f"prefill={prefill_total_ms:.1f}ms ({prefill_total_ms/e2e_ms*100:.1f}%)  "
                  f"decode={decode_total_ms:.1f}ms ({decode_total_ms/e2e_ms*100:.1f}%)  "
                  f"total={e2e_ms:.1f}ms")


if __name__ == "__main__":
    main()
