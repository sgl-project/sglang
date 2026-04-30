"""Per-stage profile for the e2e-simulated workload.

Faithful to ``orchestrator.py`` dispatch:
  prefill stage 3: fast_topk_runtime for K∈{16,32}, torch.topk(bf16) otherwise
  prefill stage 4: tilelang vanilla for K=128, triton grouped (K_CHUNKS=16
                   persistent for K∈{16,32}) for K<128
  decode  stage 1: skipped (force_maintain in stage 2)
  decode  stage 3: torch.topk
  decode  stage 4: sparse_paged_mqa_triton

Production-shape workload: ctx=128K, prefill_chunk=8K (16 chunks),
decode_steps=256, layers=61, block_topk = 8192 // k_block_size.

Reports per-stage μs (per chunk / per step), % share, and aggregated to
e2e ms (× LAYERS × B for prefill, × N_DECODE_STEPS × LAYERS for decode).

Read-only — does not modify production kernels.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_mean_pooling_grouped_interface,
    fp8_native_block_mean_pooling_interface,
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
    pool_mqa_attn_return_logits_fp8_interface,
)
from sglang.srt.layers.attention.nsa.hisa.fast_topk_runtime import (
    MAX_TOPK as _FAST_TOPK_MAX,
    fast_topk_runtime,
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


# ---------------------------------------------------------------------------
# Stage runners (faithful to orchestrator.py dispatch)
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


def prefill_stage3(score, K, block_topk):
    """Match orchestrator: fast_topk for K∈{16,32}, torch.topk(bf16) otherwise."""
    topk = min(block_topk, score.shape[-1])
    if K in (16, 32) and topk <= _FAST_TOPK_MAX:
        return fast_topk_runtime(score, topk)
    return torch.topk(score.bfloat16(), k=topk, dim=-1, sorted=False).indices


def prefill_stage4(q_fp8, k_fp8, k_scales, topk_idx, K, weights, cu_ks, cu_ke):
    """Match orchestrator: tilelang for K=128, triton grouped (persistent
    K_CHUNKS=16 for K∈{16,32}) otherwise."""
    if K == 128:
        return fp8_native_block_sparse_mqa_attn_return_logits_interface(
            q=q_fp8, k=k_fp8, k_scale=k_scales,
            topk_block_index=topk_idx,
            kv_block_size=K, weights=weights,
            cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        )
    return block_sparse_mqa_triton(
        q_fp8=q_fp8, k_fp8=k_fp8, k_scale=k_scales,
        topk_block_index=topk_idx,
        kv_block_size=K, weights=weights,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )


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


def profile_prefill_chunk(K, sq, skv, block_topk):
    """Returns (s1, s2, s3, s4) μs for one chunk."""
    q, k_fp8, k_scale, w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)

    blocked_k_fp8, blocked_k_scale = prefill_stage1(k_fp8, k_scale, K)
    s2_score = prefill_stage2(q, blocked_k_fp8, blocked_k_scale, K, w, cu_ks, cu_ke)
    s3_topk_idx = prefill_stage3(s2_score, K, block_topk)

    s1 = cuda_bench(lambda: prefill_stage1(k_fp8, k_scale, K))
    s2 = cuda_bench(lambda: prefill_stage2(
        q, blocked_k_fp8, blocked_k_scale, K, w, cu_ks, cu_ke,
    ))
    s3 = cuda_bench(lambda: prefill_stage3(s2_score, K, block_topk))
    s4 = cuda_bench(lambda: prefill_stage4(
        q, k_fp8, k_scale, s3_topk_idx, K, w, cu_ks, cu_ke,
    ))

    del q, k_fp8, k_scale, w, cu_ks, cu_ke
    del blocked_k_fp8, blocked_k_scale, s2_score, s3_topk_idx
    del s_lengths, s_indices
    torch.cuda.empty_cache()
    return s1, s2, s3, s4


def profile_decode_step(K, B, ctx, block_topk):
    """Returns (s1, s2, s3, s4) μs for one step. s1==0 (tail skipped)."""
    inp = make_decode_inputs(K, B, ctx)
    q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req = inp

    s1 = 0.0
    s2_score = decode_stage2(q, pp, ppt, w, num_pool_per_req)
    s3_topk_idx = decode_stage3(s2_score, block_topk)

    s2 = cuda_bench(lambda: decode_stage2(q, pp, ppt, w, num_pool_per_req))
    s3 = cuda_bench(lambda: decode_stage3(s2_score, block_topk))
    s4 = cuda_bench(lambda: decode_stage4(q, kv, s3_topk_idx, K, w, ctxl, bt))

    del q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req, s2_score, s3_topk_idx
    torch.cuda.empty_cache()
    return s1, s2, s3, s4


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _row(stages_us, total_us, label_w=18):
    """Format: stage1_us (%) | stage2_us (%) | ... | total_us"""
    parts = []
    for s in stages_us:
        pct = (s / total_us * 100) if total_us > 0 else 0.0
        parts.append(f"{s:>9.1f} ({pct:>4.1f}%)")
    parts.append(f"{total_us:>9.1f}")
    return "  ".join(parts)


def main(K_list=(16, 128)):
    print("=" * 110)
    print(f"Per-stage e2e profile  ctx={CTX//1024}K  prefill_chunk={PREFILL_CHUNK}  "
          f"decode_steps={N_DECODE_STEPS}  layers={LAYERS}  block_topk=8192/K")
    print("=" * 110)

    # Single decode B for reporting (B=1 is the per-request shape).
    DECODE_B = 1

    # Collect all results first, then print compact tables.
    results = {}  # K → {prefill: (s1..s4 sum, μs), decode: (s1..s4, μs)}
    for K in K_list:
        block_topk = 8192 // K
        # ---- Prefill: sum across 16 chunks ----
        chunk_breakdowns = []
        for i in range(N_PREFILL_CHUNKS):
            skv = (i + 1) * PREFILL_CHUNK
            chunk_breakdowns.append(
                profile_prefill_chunk(K, PREFILL_CHUNK, skv, block_topk)
            )
        pf_s1 = sum(c[0] for c in chunk_breakdowns)
        pf_s2 = sum(c[1] for c in chunk_breakdowns)
        pf_s3 = sum(c[2] for c in chunk_breakdowns)
        pf_s4 = sum(c[3] for c in chunk_breakdowns)

        # ---- Decode: one step ----
        d1, d2, d3, d4 = profile_decode_step(K, DECODE_B, CTX, block_topk)

        results[K] = {
            "block_topk": block_topk,
            "prefill_chunk_sum_us": (pf_s1, pf_s2, pf_s3, pf_s4),
            "decode_step_us": (d1, d2, d3, d4),
            "chunks": chunk_breakdowns,
        }

    # ---- Report ----
    for K in K_list:
        r = results[K]
        print()
        print(f"K = {K}    block_topk = {r['block_topk']}")
        print("-" * 110)
        # Per-chunk breakdown for prefill
        print(f"PREFILL — per-chunk μs (16 chunks, skv = (i+1)*8K):")
        print(f"  {'chunk':>5} {'skv':>7} | {'stage1':>8} {'stage2':>8} "
              f"{'stage3':>8} {'stage4':>8} | {'total':>8}")
        for i, (s1, s2, s3, s4) in enumerate(r["chunks"]):
            tot = s1 + s2 + s3 + s4
            print(f"  {i:>5} {(i+1)*PREFILL_CHUNK:>7} | "
                  f"{s1:>8.1f} {s2:>8.1f} {s3:>8.1f} {s4:>8.1f} | {tot:>8.1f}")

        # Aggregated prefill (sum across 16 chunks; per-request, B=1)
        pf = r["prefill_chunk_sum_us"]
        pf_tot = sum(pf)
        print(f"\nPREFILL — sum across 16 chunks (per-request, μs):")
        print(f"  {'stage1':>14}  {'stage2':>14}  {'stage3':>14}  {'stage4':>14}  "
              f"| {'total':>9}")
        print(f"  {_row(pf, pf_tot)}")

        # Decode (one step)
        d = r["decode_step_us"]
        d_tot = sum(d)
        print(f"\nDECODE — one step (B={DECODE_B}, μs):")
        print(f"  {'stage1':>14}  {'stage2':>14}  {'stage3':>14}  {'stage4':>14}  "
              f"| {'total':>9}")
        print(f"  {_row(d, d_tot)}")

        # Aggregated to e2e ms (per-request, B=1)
        prefill_ms = pf_tot * LAYERS / 1e3
        decode_ms = d_tot * N_DECODE_STEPS * LAYERS / 1e3
        e2e_ms = prefill_ms + decode_ms
        print(f"\nE2E (per-request, B=1):")
        print(f"  prefill = {prefill_ms:>8.1f} ms  ({prefill_ms/e2e_ms*100:>4.1f}%)")
        print(f"  decode  = {decode_ms:>8.1f} ms  ({decode_ms/e2e_ms*100:>4.1f}%)")
        print(f"  total   = {e2e_ms:>8.1f} ms")
        print()

    # ---- Cross-K summary table: stage % of e2e per K ----
    print("=" * 110)
    print(f"E2E STAGE BREAKDOWN — per-request (B={DECODE_B}, layers={LAYERS}, "
          f"prefill_chunks=16, decode_steps={N_DECODE_STEPS})")
    print("=" * 110)
    print(f"{'K':>4} | "
          + "  ".join(f"{f'pf_s{i}':>7}" for i in range(1, 5))
          + "  | "
          + "  ".join(f"{f'dc_s{i}':>7}" for i in range(1, 5))
          + "  | " + f"{'e2e ms':>8}")
    print(f"{'':>4} | "
          + "  ".join("μs    %  ".rjust(7) for _ in range(4))
          + "  | "
          + "  ".join("μs    %  ".rjust(7) for _ in range(4))
          + "  | ")
    print("-" * 110)
    for K in K_list:
        r = results[K]
        pf_us = list(r["prefill_chunk_sum_us"])
        d_us = list(r["decode_step_us"])
        # Convert to e2e ms
        pf_ms = [s * LAYERS / 1e3 for s in pf_us]
        d_ms = [s * N_DECODE_STEPS * LAYERS / 1e3 for s in d_us]
        e2e = sum(pf_ms) + sum(d_ms)
        cells = []
        for ms in pf_ms:
            cells.append(f"{ms:>5.0f}({ms/e2e*100:>4.1f}%)")
        for ms in d_ms:
            cells.append(f"{ms:>5.0f}({ms/e2e*100:>4.1f}%)")
        print(f"{K:>4} | " + "  ".join(cells[:4]) + "  | "
              + "  ".join(cells[4:]) + f"  | {e2e:>8.1f}")


if __name__ == "__main__":
    main(K_list=(16, 128))
