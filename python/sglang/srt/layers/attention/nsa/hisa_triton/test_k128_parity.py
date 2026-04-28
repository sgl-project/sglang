"""K=128 triton vs tilelang parity check (correctness + speed).

Validates that the triton ports we already have can stand in for tilelang
at the K=128 production setting, BEFORE we touch any production dispatch.

Three swap candidates (= the only places where K>=64 still calls tilelang
in the hot path):
  1. update_pool_for_completed_blocks  (SK15 triton vs tilelang)
  2. tail_only_v3                      (SK16 triton vs tilelang)
  3. fp8_native_hierarchy_mqa_logits   (ragged prefill orchestrator;
                                        triton vs tilelang)

For each: identical input copies → run both → byte-equal compare →
microbench head-to-head. Decision logic at end:
  - precision: must pass at fp8-strict tolerance OR be byte-equal
  - speed:     triton within 1.5x of tilelang is acceptable (we already
               saw triton can win on tail_only at K=128)

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m \\
        sglang.srt.layers.attention.nsa.hisa_triton.test_k128_parity
"""
from __future__ import annotations

import statistics
import sys
import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_hierarchy_mqa_logits,
    fp8_native_paged_mean_pooling_completed_blocks_v3_interface,
    fp8_native_paged_mean_pooling_tail_only_v3_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    tail_only_v3_triton,
    update_pool_for_completed_blocks_triton,
)
from sglang.srt.layers.attention.nsa.hisa_triton.orchestrator import (
    fp8_native_hierarchy_mqa_logits_triton,
)
from sglang.srt.layers.attention.nsa.hisa_triton.benchmark import (
    _make_paged_kv_cache_soa,
)
from sglang.srt.layers.attention.nsa.hisa_triton.test_precision import (
    _strictest_passing_tolerance,
    _topk_iou,
)

DEVICE = torch.device("cuda")
H, D, PAGED, PP = 64, 128, 64, 64
K = 128


def _flush_l2():
    torch.empty(int(256e6 // 4), dtype=torch.int, device=DEVICE).zero_()


def cuda_bench(fn, warmups: int = 5, iters: int = 30) -> float:
    torch.cuda.synchronize()
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        _flush_l2()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


# --------------------------------------------------------------------------
# Common setup uses ``_make_paged_kv_cache_soa`` from benchmark.py — that
# helper writes the production SoA byte layout (fp8 region first, scales
# second) within each page; an AoS-per-token init would silently produce
# nonsense outputs from both kernels.
# --------------------------------------------------------------------------
_make_paged_kv = _make_paged_kv_cache_soa


# --------------------------------------------------------------------------
# Test 1 — SK15 update_pool_for_completed_blocks
# --------------------------------------------------------------------------
@torch.inference_mode()
def test_update_pool_K128(B: int, ctx_len: int):
    print(f"\n=== Test 1: update_pool_for_completed_blocks  K={K}  B={B}  ctx={ctx_len} ===")

    max_kv_blocks = (ctx_len + PAGED - 1) // PAGED
    max_pool_blocks = (ctx_len + K - 1) // K
    max_pool_pages = (max_pool_blocks + PP - 1) // PP

    num_phys = B * max_kv_blocks + 4
    num_pool_phys = B * max_pool_pages + 4
    max_req = max(B, 4)
    max_ctx = ctx_len + 64

    kv_cache = _make_paged_kv(num_phys, PAGED, D, seed=42)
    kv_flat = kv_cache.view(num_phys, -1)

    # Common metadata
    req_to_token = (
        torch.arange(max_req * max_ctx, dtype=torch.int32, device=DEVICE)
        .view(max_req, max_ctx) % (num_phys * PAGED)
    )
    pool_page_tables = (
        torch.arange(max_req * max_pool_pages, dtype=torch.int32, device=DEVICE)
        .view(max_req, max_pool_pages) % num_pool_phys
    )
    req_pool_indices = torch.arange(B, dtype=torch.int64, device=DEVICE)
    prev_seq_lens = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    new_seq_lens = torch.full((B,), ctx_len, dtype=torch.int32, device=DEVICE)

    # Two output buffers, identical initial state.
    out_a = torch.zeros(num_pool_phys, PP * (D + 4), dtype=torch.uint8, device=DEVICE)
    out_b = out_a.clone()

    # Run triton on out_a, tilelang on out_b.
    update_pool_for_completed_blocks_triton(
        kv_cache_flat=kv_flat,
        req_to_token=req_to_token,
        pool_page_tables=pool_page_tables,
        req_pool_indices=req_pool_indices,
        prev_seq_lens=prev_seq_lens,
        new_seq_lens=new_seq_lens,
        pool_k_pages=out_a,
        k_block_size=K,
        paged_block_size=PAGED,
        pool_page_size=PP,
        max_pool_per_req_grid=max_pool_blocks,
    )
    fp8_native_paged_mean_pooling_completed_blocks_v3_interface(
        kv_cache_flat=kv_flat,
        req_to_token=req_to_token,
        pool_page_tables=pool_page_tables,
        req_pool_indices=req_pool_indices,
        prev_seq_lens=prev_seq_lens,
        new_seq_lens=new_seq_lens,
        pool_k_pages=out_b,
        k_block_size=K,
        paged_block_size=PAGED,
        pool_page_size=PP,
        max_pool_per_req_grid=max_pool_blocks,
    )
    torch.cuda.synchronize()

    # Compare: only the slots actually written (the first max_pool_pages per req).
    # The rest stays zero and is tested implicitly.
    # SoA layout per page: bytes [0, PP*D) = all PP tokens' fp8 concatenated;
    # bytes [PP*D, PP*(D+4)) = all PP scales concatenated.
    a_fp8 = (
        out_a[:, : PP * D].contiguous().view(torch.float8_e4m3fn)
        .reshape(num_pool_phys, PP, D).to(torch.float32)
    )
    b_fp8 = (
        out_b[:, : PP * D].contiguous().view(torch.float8_e4m3fn)
        .reshape(num_pool_phys, PP, D).to(torch.float32)
    )
    a_scale = (
        out_a[:, PP * D :].contiguous().view(torch.float32)
        .reshape(num_pool_phys, PP)
    )
    b_scale = (
        out_b[:, PP * D :].contiguous().view(torch.float32)
        .reshape(num_pool_phys, PP)
    )

    # Per-page diff diagnostic.
    a_per_page = a_fp8.reshape(num_pool_phys, -1).abs().sum(dim=-1)
    b_per_page = b_fp8.reshape(num_pool_phys, -1).abs().sum(dim=-1)
    a_written = (a_per_page > 0)
    b_written = (b_per_page > 0)
    only_a = (a_written & ~b_written).sum().item()
    only_b = (b_written & ~a_written).sum().item()
    both = (a_written & b_written).sum().item()
    print(f"  pages with non-zero fp8 data: triton={a_written.sum().item()}  tilelang={b_written.sum().item()}  "
          f"only-triton={only_a}  only-tilelang={only_b}  both={both}")

    print(f"  fp8 data:")
    level, diag = _strictest_passing_tolerance(a_fp8, b_fp8)
    print(f"    {level:>20} | {diag}")
    byte_eq = torch.equal(out_a, out_b)
    print(f"    byte-equal? {byte_eq}")
    # Compare only pages BOTH wrote.
    if both > 0:
        common = a_written & b_written
        a_common = a_fp8[common]
        b_common = b_fp8[common]
        level, diag = _strictest_passing_tolerance(a_common, b_common)
        print(f"  fp8 data (pages both wrote, n={a_common.numel()}):")
        print(f"    {level:>20} | {diag}")
    print(f"  scales:")
    level, diag = _strictest_passing_tolerance(a_scale, b_scale)
    print(f"    {level:>20} | {diag}")

    # Bench
    pool_a = out_a.clone(); pool_b = out_b.clone()
    t_triton = cuda_bench(lambda: update_pool_for_completed_blocks_triton(
        kv_cache_flat=kv_flat, req_to_token=req_to_token,
        pool_page_tables=pool_page_tables, req_pool_indices=req_pool_indices,
        prev_seq_lens=prev_seq_lens, new_seq_lens=new_seq_lens,
        pool_k_pages=pool_a, k_block_size=K, paged_block_size=PAGED,
        pool_page_size=PP, max_pool_per_req_grid=max_pool_blocks,
    ))
    t_tilelang = cuda_bench(lambda: fp8_native_paged_mean_pooling_completed_blocks_v3_interface(
        kv_cache_flat=kv_flat, req_to_token=req_to_token,
        pool_page_tables=pool_page_tables, req_pool_indices=req_pool_indices,
        prev_seq_lens=prev_seq_lens, new_seq_lens=new_seq_lens,
        pool_k_pages=pool_b, k_block_size=K, paged_block_size=PAGED,
        pool_page_size=PP, max_pool_per_req_grid=max_pool_blocks,
    ))
    print(f"  speed (μs):  triton={t_triton*1000:.2f}  tilelang={t_tilelang*1000:.2f}  "
          f"ratio={t_triton/t_tilelang:.2f}x")
    return t_triton, t_tilelang


# --------------------------------------------------------------------------
# Test 2 — SK16 tail_only_v3
# --------------------------------------------------------------------------
@torch.inference_mode()
def test_tail_only_K128(B: int, ctx_len: int):
    print(f"\n=== Test 2: tail_only_v3  K={K}  B={B}  ctx={ctx_len} ===")

    max_kv_blocks = (ctx_len + PAGED - 1) // PAGED
    max_pool_blocks = (ctx_len + K - 1) // K
    max_pool_pages = (max_pool_blocks + PP - 1) // PP

    num_phys = B * max_kv_blocks + 4
    num_pool_phys = B * max_pool_pages + 4

    kv_cache = _make_paged_kv(num_phys, PAGED, D, seed=7)
    kv_flat = kv_cache.view(num_phys, -1)

    block_tables = (
        torch.arange(B * max_kv_blocks, dtype=torch.int32, device=DEVICE)
        .view(B, max_kv_blocks) % num_phys
    )
    pool_page_tables = (
        torch.arange(B * max_pool_pages, dtype=torch.int32, device=DEVICE)
        .view(B, max_pool_pages) % num_pool_phys
    )
    # Use a non-multiple-of-K context to exercise the partial-tail path.
    context_lens = torch.full((B,), ctx_len, dtype=torch.int32, device=DEVICE)
    context_lens[0] = ctx_len - K // 3   # partial tail

    out_a = torch.zeros(num_pool_phys, PP * (D + 4), dtype=torch.uint8, device=DEVICE)
    out_b = out_a.clone()

    tail_only_v3_triton(
        kv_cache_flat=kv_flat, context_lens=context_lens,
        block_tables=block_tables, pool_page_tables=pool_page_tables,
        pool_k_pages=out_a, k_block_size=K, paged_block_size=PAGED, pool_page_size=PP,
    )
    # tilelang interface takes 4d kv_cache (not flat), let's check signature.
    fp8_native_paged_mean_pooling_tail_only_v3_interface(
        kv_cache=kv_cache, context_lens=context_lens,
        block_tables=block_tables, pool_page_tables=pool_page_tables,
        pool_k_pages=out_b, k_block_size=K, pool_page_size=PP,
    )
    torch.cuda.synchronize()

    # SoA layout per page: bytes [0, PP*D) = all PP tokens' fp8 concatenated;
    # bytes [PP*D, PP*(D+4)) = all PP scales concatenated.
    a_fp8 = (
        out_a[:, : PP * D].contiguous().view(torch.float8_e4m3fn)
        .reshape(num_pool_phys, PP, D).to(torch.float32)
    )
    b_fp8 = (
        out_b[:, : PP * D].contiguous().view(torch.float8_e4m3fn)
        .reshape(num_pool_phys, PP, D).to(torch.float32)
    )
    a_scale = (
        out_a[:, PP * D :].contiguous().view(torch.float32)
        .reshape(num_pool_phys, PP)
    )
    b_scale = (
        out_b[:, PP * D :].contiguous().view(torch.float32)
        .reshape(num_pool_phys, PP)
    )

    a_per_page = a_fp8.reshape(num_pool_phys, -1).abs().sum(dim=-1)
    b_per_page = b_fp8.reshape(num_pool_phys, -1).abs().sum(dim=-1)
    a_written = (a_per_page > 0)
    b_written = (b_per_page > 0)
    only_a = (a_written & ~b_written).sum().item()
    only_b = (b_written & ~a_written).sum().item()
    both = (a_written & b_written).sum().item()
    print(f"  pages with non-zero fp8 data: triton={a_written.sum().item()}  tilelang={b_written.sum().item()}  "
          f"only-triton={only_a}  only-tilelang={only_b}  both={both}")

    print(f"  fp8 data:")
    level, diag = _strictest_passing_tolerance(a_fp8, b_fp8)
    print(f"    {level:>20} | {diag}")
    byte_eq = torch.equal(out_a, out_b)
    print(f"    byte-equal? {byte_eq}")
    if both > 0:
        common = a_written & b_written
        a_common = a_fp8[common]
        b_common = b_fp8[common]
        level, diag = _strictest_passing_tolerance(a_common, b_common)
        print(f"  fp8 data (pages both wrote, n={a_common.numel()}):")
        print(f"    {level:>20} | {diag}")
    print(f"  scales:")
    level, diag = _strictest_passing_tolerance(a_scale, b_scale)
    print(f"    {level:>20} | {diag}")

    pool_a = out_a.clone(); pool_b = out_b.clone()
    t_triton = cuda_bench(lambda: tail_only_v3_triton(
        kv_cache_flat=kv_flat, context_lens=context_lens,
        block_tables=block_tables, pool_page_tables=pool_page_tables,
        pool_k_pages=pool_a, k_block_size=K, paged_block_size=PAGED, pool_page_size=PP,
    ))
    t_tilelang = cuda_bench(lambda: fp8_native_paged_mean_pooling_tail_only_v3_interface(
        kv_cache=kv_cache, context_lens=context_lens,
        block_tables=block_tables, pool_page_tables=pool_page_tables,
        pool_k_pages=pool_b, k_block_size=K, pool_page_size=PP,
    ))
    print(f"  speed (μs):  triton={t_triton*1000:.2f}  tilelang={t_tilelang*1000:.2f}  "
          f"ratio={t_triton/t_tilelang:.2f}x")
    return t_triton, t_tilelang


# --------------------------------------------------------------------------
# Test 3 — fp8_native_hierarchy_mqa_logits (ragged prefill orchestrator)
# --------------------------------------------------------------------------
@torch.inference_mode()
def test_ragged_orchestrator_K128(seq_q: int, seq_kv: int):
    print(f"\n=== Test 3: ragged hierarchy_mqa_logits  K={K}  sq={seq_q}  skv={seq_kv} ===")
    block_topk = max(64, 8192 // K)  # production formula

    torch.manual_seed(123)
    q = torch.randn(seq_q, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(seq_kv, D, device=DEVICE).to(torch.float8_e4m3fn)
    scale_f32 = 0.05 + 0.02 * torch.rand(seq_kv, device=DEVICE, dtype=torch.float32)
    scale_uint8 = scale_f32.view(torch.uint8).reshape(seq_kv, 4)
    w = torch.randn(seq_q, H, device=DEVICE, dtype=torch.float32)

    # Causal-ish ragged ranges.
    cu_ks = torch.zeros(seq_q, device=DEVICE, dtype=torch.int32)
    cu_ke = torch.linspace(seq_kv // 4, seq_kv, seq_q, device=DEVICE).to(torch.int32)

    bs_t, topk_t = fp8_native_hierarchy_mqa_logits_triton(
        q, (k_fp8, scale_uint8), w, cu_ks, cu_ke,
        k_block_size=K, block_topk=block_topk,
    )
    bs_l, topk_l = fp8_native_hierarchy_mqa_logits(
        q, (k_fp8, scale_uint8), w, cu_ks, cu_ke,
        k_block_size=K, block_topk=block_topk,
    )
    torch.cuda.synchronize()

    print(f"  block_sparse_logits shapes: triton={tuple(bs_t.shape)} tilelang={tuple(bs_l.shape)}")
    print(f"  topk_block_indices  shapes: triton={tuple(topk_t.shape)} tilelang={tuple(topk_l.shape)}")

    # Direct topk_indices set-IoU (the actual chosen blocks; if these diverge,
    # downstream sparse-MQA picks different positions → final logits will
    # naturally have low induced-topk IoU even when each side is internally
    # consistent. So compare BOTH).
    rows = topk_t.shape[0]
    ious_direct = []
    for r in range(rows):
        a = set(topk_t[r].cpu().tolist())
        b = set(topk_l[r].cpu().tolist())
        ious_direct.append(len(a & b) / max(len(a | b), 1))
    import statistics as st
    print(f"  topk_indices direct IoU: mean={st.mean(ious_direct):.3f}  min={min(ious_direct):.3f}")

    level, diag = _strictest_passing_tolerance(bs_t, bs_l)
    print(f"  block_sparse_logits: {level:>20} | {diag}")
    # Logits-induced topk IoU (will be artificially low if topk_indices already diverged).
    iou_mean, iou_min, _ = _topk_iou(
        bs_t.reshape(bs_t.shape[0], -1), bs_l.reshape(bs_l.shape[0], -1), k=block_topk,
    )
    print(f"  topk-set IoU (logits-induced): mean={iou_mean:.3f}  min={iou_min:.3f}")

    t_triton = cuda_bench(lambda: fp8_native_hierarchy_mqa_logits_triton(
        q, (k_fp8, scale_uint8), w, cu_ks, cu_ke,
        k_block_size=K, block_topk=block_topk,
    ))
    t_tilelang = cuda_bench(lambda: fp8_native_hierarchy_mqa_logits(
        q, (k_fp8, scale_uint8), w, cu_ks, cu_ke,
        k_block_size=K, block_topk=block_topk,
    ))
    print(f"  speed (μs):  triton={t_triton*1000:.2f}  tilelang={t_tilelang*1000:.2f}  "
          f"ratio={t_triton/t_tilelang:.2f}x")
    return t_triton, t_tilelang


# --------------------------------------------------------------------------
def main():
    rows = []
    # Sweep production-relevant shapes
    for B, ctx in [(1, 65536), (1, 131072), (10, 65536), (32, 65536)]:
        rows.append(("update_pool", f"B={B} ctx={ctx}", *test_update_pool_K128(B, ctx)))
        rows.append(("tail_only", f"B={B} ctx={ctx}", *test_tail_only_K128(B, ctx)))
    # Sweep (sq, skv) covering chunked-prefill production shapes:
    #   - small  (warmup / short prompts): sq in {32, 64, 128}, skv ≤ 8K
    #   - medium (typical samsum / 8K context):                   skv = 8K..32K
    #   - long   (production target 65K-128K):                    skv = 65K..128K
    # chunked_prefill_size in production scripts is 8192, so sq <= 8192 always.
    SQ_SKV_GRID = [
        (32,    4096),    (64,    8192),    (128,   8192),
        (256,   8192),    (256,   32768),
        (512,   32768),   (512,   65536),
        (1024,  32768),   (1024,  65536),   (1024,  131072),
        (2048,  65536),   (2048,  131072),
        (4096,  65536),
        (8192,  65536),   (8192,  131072),
    ]
    for sq, skv in SQ_SKV_GRID:
        try:
            rows.append((
                "ragged_orchestrator", f"sq={sq:>4} skv={skv:>6}",
                *test_ragged_orchestrator_K128(sq, skv),
            ))
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at sq={sq} skv={skv}, skipping rest of sweep")
            break

    print("\n\n=== SUMMARY @ K=128 ===")
    print(f"{'kernel':<25} {'config':<18} {'triton(μs)':>12} {'tilelang(μs)':>14} {'ratio':>8}")
    print("-" * 80)
    for name, cfg, t_triton, t_tilelang in rows:
        print(f"{name:<25} {cfg:<18} {t_triton*1000:>12.2f} {t_tilelang*1000:>14.2f} {t_triton/t_tilelang:>7.2f}x")


if __name__ == "__main__":
    main()
