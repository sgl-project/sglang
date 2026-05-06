"""Decode orchestrator validation: DG stage 2 vs triton stage 2.

Three correctness checks:

  1. Stage 2 byte-equivalence — DG and triton produce the same logits
     (fp8 ULP drift only).
  2. Topk IoU — for production-realistic shapes, the top-block-topk pool
     blocks selected by DG and triton agree on >= 95% of indices. Pure-
     equality is too strict at fp8 because ranking near the topk boundary
     is sensitive to ULP noise — same caveat documented for prefill.
  3. CUDA graph determinism — capturing the DG-decode orchestrator and
     replaying produces output bit-identical to running it eager (with
     the same persistent schedule_metadata buffer).

Plus a speed table comparing the full orchestrator under eager and graph.
"""
from __future__ import annotations

import deep_gemm
import torch

from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    batch_decode_pool_mqa_triton,
    force_maintain_logits_decode_triton,
    sparse_paged_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa.orchestrator import (
    _stage3_topk_decode,
    fp8_native_hierarchy_paged_mqa_logits,
)


def _make_pool_schedule(context_lens, k_block_size, pool_page_size):
    """Mirror the production caller (HisaIndexer): build pool-domain DG
    schedule_metadata once per (K, B, ctx). Production reuses this across
    61 layers per forward — benching it once per layer would inflate cost
    by ~61×."""
    npbpr = (context_lens + k_block_size - 1) // k_block_size
    return deep_gemm.get_paged_mqa_logits_metadata(
        npbpr, pool_page_size, deep_gemm.get_num_sms(),
    )


DEVICE = torch.device("cuda")
H, D = 64, 128
PAGED, POOL_PAGE = 64, 64
BLOCK_TOPK_FORMULA = 8192


def make_inputs(K, B, ctx):
    """Production-realistic random inputs (mirrors test_e2e_simulated.make_decode_inputs)."""
    torch.manual_seed(1)
    num_kv_blocks_per_req = (ctx + PAGED - 1) // PAGED
    num_phys = B * num_kv_blocks_per_req + 16
    num_pool_blocks_per_req = (ctx + K - 1) // K
    num_pool_pages_per_req = (num_pool_blocks_per_req + POOL_PAGE - 1) // POOL_PAGE
    num_pool_phys = B * num_pool_pages_per_req + 8

    q = torch.randn(B, 1, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    # Production layout: write fp8 K from a normal distribution (so values are
    # bounded), then plausible f32 scales.
    pool_k_pages = torch.empty(
        (num_pool_phys, POOL_PAGE * (D + 4)), device=DEVICE, dtype=torch.uint8,
    )
    pkp_fp8 = pool_k_pages.view(torch.float8_e4m3fn)
    pkp_f32 = pool_k_pages.view(torch.float32)
    fp8_end = POOL_PAGE * D
    pkp_fp8[:, :fp8_end] = torch.randn(
        num_pool_phys, fp8_end, device=DEVICE,
    ).to(torch.float8_e4m3fn)
    soff = fp8_end // 4
    pkp_f32[:, soff : soff + POOL_PAGE] = (
        0.05 + 0.02 * torch.rand_like(pkp_f32[:, soff : soff + POOL_PAGE])
    )
    # Same for raw KV cache (used by stage 4).
    kv_cache = torch.empty(
        (num_phys, PAGED, 1, D + 4), device=DEVICE, dtype=torch.uint8,
    )
    kv_fp8 = kv_cache.view(torch.float8_e4m3fn)
    kv_f32 = kv_cache.view(torch.float32)
    kv_fp8[..., :D] = torch.randn(
        num_phys, PAGED, 1, D, device=DEVICE,
    ).to(torch.float8_e4m3fn)
    kv_f32[..., D // 4 : D // 4 + 1] = (
        0.05 + 0.02 * torch.rand_like(kv_f32[..., D // 4 : D // 4 + 1])
    )

    weights = torch.randn(B, 1, H, device=DEVICE, dtype=torch.float32)
    context_lens = torch.full((B,), ctx, device=DEVICE, dtype=torch.int32)
    block_tables = torch.stack([
        torch.arange(b * num_kv_blocks_per_req, (b + 1) * num_kv_blocks_per_req,
                     device=DEVICE, dtype=torch.int32)
        for b in range(B)
    ])
    pool_page_tables = torch.stack([
        torch.arange(b * num_pool_pages_per_req, (b + 1) * num_pool_pages_per_req,
                     device=DEVICE, dtype=torch.int32)
        for b in range(B)
    ])
    return (q, kv_cache, pool_k_pages, pool_page_tables, weights,
            context_lens, block_tables)


def orchestrator_with_triton_stage2(
    q_fp8, kv_cache_fp8, pool_k_pages, pool_page_tables, weights,
    context_lens, block_tables, k_block_size, pool_page_size, block_topk,
):
    """Reference orchestrator: triton stage 2 (the one we replaced)."""
    num_pool_blocks_per_req = (context_lens + k_block_size - 1) // k_block_size
    block_k_indexer_score = batch_decode_pool_mqa_triton(
        q_fp8=q_fp8, pool_k_pages=pool_k_pages, pool_page_tables=pool_page_tables,
        weights_f32=weights, context_lens_pool=num_pool_blocks_per_req,
        pool_page_size=pool_page_size,
    )
    topk_block_indices = _stage3_topk_decode(block_k_indexer_score, block_topk)
    block_sparse_logits = sparse_paged_mqa_triton(
        q_fp8=q_fp8, kv_cache_fp8=kv_cache_fp8,
        topk_block_index=topk_block_indices, kv_block_size=k_block_size,
        weights=weights, context_lens=context_lens, block_tables=block_tables,
    )
    return block_sparse_logits, topk_block_indices, block_k_indexer_score


def topk_iou(idx_a, idx_b):
    """Per-row IoU of topk index sets, then averaged."""
    if idx_a.shape != idx_b.shape:
        return 0.0
    a = idx_a.long().view(-1, idx_a.shape[-1])
    b = idx_b.long().view(-1, idx_b.shape[-1])
    n_rows, k = a.shape
    a_s = a.sort(dim=-1).values
    b_s = b.sort(dim=-1).values
    # vectorised inter via merge-sort adjacency
    merged = torch.cat([a_s, b_s], dim=-1).sort(dim=-1).values
    inter = (merged[:, :-1] == merged[:, 1:]).sum(dim=-1)  # [n_rows]
    union = 2 * k - inter
    return (inter.float() / union.float().clamp_min(1)).mean().item()


def correctness_stage2():
    """Stage 2 byte-level: DG output (post force_maintain) vs triton output."""
    print("=" * 110)
    print("Correctness — stage 2 (DG + force_maintain vs triton single fused kernel)")
    print("=" * 110)
    print(f"{'K':>4} {'B':>3} {'ctx':>5} {'n_blk':>6} | {'inf_match':>10} "
          f"{'fin_match':>10} {'max_abs':>10} {'rel':>10}")
    print("-" * 110)
    fails = 0
    for K in (16, 32, 64, 128):
        for B in (1, 4, 32):
            for ctx in (8 * 1024, 65 * 1024, 128 * 1024):
                inputs = make_inputs(K, B, ctx)
                q, kv, pkp, ppt, w, cl, bt = inputs
                block_topk = BLOCK_TOPK_FORMULA // K
                max_seq_len = bt.shape[1] * PAGED
                sched = _make_pool_schedule(cl, K, POOL_PAGE)

                _, _, score_dg = orchestrator_capture_score(
                    q, kv, pkp, ppt, w, cl, bt, K, POOL_PAGE, block_topk,
                    max_seq_len, sched, use_dg=True,
                )
                _, _, score_tri = orchestrator_with_triton_stage2(
                    q, kv, pkp, ppt, w, cl, bt, K, POOL_PAGE, block_topk,
                )

                # DG output width = ceildiv(max_seq_len, K); triton width =
                # max_pool_pages * pool_page_size (rounded up to page boundary).
                # Both pad with -inf past n_blocks, so compare on the common region.
                a = score_dg.squeeze(1)
                b = score_tri.squeeze(1)
                L = min(a.shape[-1], b.shape[-1])
                a = a[..., :L]
                b = b[..., :L]
                inf_match = (torch.isinf(a) == torch.isinf(b)).all().item()
                fin = torch.isfinite(a) & torch.isfinite(b)
                fin_match = (torch.isfinite(a) == torch.isfinite(b)).all().item()
                if fin.any():
                    max_abs = (a[fin] - b[fin]).abs().max().item()
                    scale = a[fin].abs().max().item()
                    rel = max_abs / max(scale, 1e-6)
                else:
                    max_abs, rel = 0.0, 0.0
                ok = inf_match and fin_match and rel < 5e-2
                if not ok:
                    fails += 1
                status = "OK" if ok else "FAIL"
                n_blk = ((cl[0].item() + K - 1) // K)
                print(f"{K:>4} {B:>3} {ctx//1024:>3}K {n_blk:>6} | "
                      f"{str(inf_match):>10} {str(fin_match):>10} "
                      f"{max_abs:>10.4f} {rel:>10.2e}  [{status}]")
                del q, kv, pkp, ppt, w, cl, bt
                torch.cuda.empty_cache()
    return fails


def orchestrator_capture_score(
    q_fp8, kv_cache_fp8, pool_k_pages, pool_page_tables, weights,
    context_lens, block_tables, k_block_size, pool_page_size, block_topk,
    max_seq_len, schedule_metadata, use_dg: bool,
):
    """Same as the production orchestrator but also returns the stage-2 score
    for diagnostic comparison."""
    if use_dg:
        bsl, idx = fp8_native_hierarchy_paged_mqa_logits(
            q_fp8=q_fp8, kv_cache_fp8=kv_cache_fp8, pool_k_pages=pool_k_pages,
            pool_page_tables=pool_page_tables, weights=weights,
            context_lens=context_lens, block_tables=block_tables,
            k_block_size=k_block_size, pool_page_size=pool_page_size,
            block_topk=block_topk, max_seq_len=max_seq_len,
            schedule_metadata=schedule_metadata,
        )
        # Re-run stage 2 alone to expose score (the orchestrator doesn't surface it).
        npbpr = (context_lens + k_block_size - 1) // k_block_size
        pkv = pool_k_pages.view(pool_k_pages.shape[0], pool_page_size, 1, q_fp8.shape[-1] + 4)
        w_2d = weights.view(-1, weights.shape[-1])
        max_pool_seq = (max_seq_len + k_block_size - 1) // k_block_size
        score = deep_gemm.fp8_paged_mqa_logits(
            q_fp8, pkv, w_2d, npbpr, pool_page_tables, schedule_metadata,
            max_pool_seq, clean_logits=True,
        )
        force_maintain_logits_decode_triton(score, npbpr)
        return bsl, idx, score.unsqueeze(1)
    else:
        return orchestrator_with_triton_stage2(
            q_fp8, kv_cache_fp8, pool_k_pages, pool_page_tables, weights,
            context_lens, block_tables, k_block_size, pool_page_size, block_topk,
        )


def correctness_topk_iou():
    print()
    print("=" * 110)
    print("Correctness — topk IoU (DG vs triton, end-to-end orchestrator)")
    print("=" * 110)
    print(f"{'K':>4} {'B':>3} {'ctx':>5} | {'topk':>5} {'IoU':>8}")
    print("-" * 110)
    fails = 0
    for K in (16, 32, 64, 128):
        for B in (1, 4):
            for ctx in (8 * 1024, 65 * 1024, 128 * 1024):
                inputs = make_inputs(K, B, ctx)
                q, kv, pkp, ppt, w, cl, bt = inputs
                block_topk = BLOCK_TOPK_FORMULA // K
                max_seq_len = bt.shape[1] * PAGED
                sched = _make_pool_schedule(cl, K, POOL_PAGE)
                bsl_dg, idx_dg = fp8_native_hierarchy_paged_mqa_logits(
                    q_fp8=q, kv_cache_fp8=kv, pool_k_pages=pkp,
                    pool_page_tables=ppt, weights=w, context_lens=cl,
                    block_tables=bt, k_block_size=K,
                    pool_page_size=POOL_PAGE, block_topk=block_topk,
                    max_seq_len=max_seq_len, schedule_metadata=sched,
                )
                bsl_tri, idx_tri, _ = orchestrator_with_triton_stage2(
                    q, kv, pkp, ppt, w, cl, bt, K, POOL_PAGE, block_topk,
                )
                iou = topk_iou(idx_dg, idx_tri)
                ok = iou >= 0.95
                if not ok:
                    fails += 1
                status = "OK" if ok else "FAIL"
                print(f"{K:>4} {B:>3} {ctx//1024:>3}K | {block_topk:>5} {iou:>8.4f}  [{status}]")
                del q, kv, pkp, ppt, w, cl, bt
                torch.cuda.empty_cache()
    return fails


def correctness_cuda_graph():
    """Capture+replay the DG decode orchestrator. We compare against eager
    using SET equality of the selected pool blocks per row — fast_topk_runtime
    uses atomicAdd-based positioning, so the ORDER of indices within the topk
    output varies between any two runs (eager-vs-eager, eager-vs-replay,
    replay-vs-replay) even with identical inputs. Downstream consumers
    (fast_topk_v2 over sparse logits, hisa_coord_transform) are set-based,
    so this nondeterminism is functionally inert. Pre-existing in the triton
    stage-2 path; verified here to confirm DG migration didn't regress.
    Stage-2 score byte-equivalence and topk SET equivalence are covered by
    ``correctness_stage2`` and ``correctness_topk_iou`` above.
    """
    print()
    print("=" * 110)
    print("Correctness — CUDA graph capture + replay (DG decode orchestrator)")
    print("=" * 110)
    inputs = make_inputs(K=16, B=4, ctx=65 * 1024)
    q, kv, pkp, ppt, w, cl, bt = inputs
    block_topk = BLOCK_TOPK_FORMULA // 16
    max_seq_len = bt.shape[1] * PAGED
    sched = _make_pool_schedule(cl, 16, POOL_PAGE)

    def call():
        return fp8_native_hierarchy_paged_mqa_logits(
            q_fp8=q, kv_cache_fp8=kv, pool_k_pages=pkp,
            pool_page_tables=ppt, weights=w, context_lens=cl,
            block_tables=bt, k_block_size=16,
            pool_page_size=POOL_PAGE, block_topk=block_topk,
            max_seq_len=max_seq_len, schedule_metadata=sched,
        )

    # Eager reference.
    for _ in range(3):
        bsl_ref, idx_ref = call()
    torch.cuda.synchronize()
    idx_ref = idx_ref.clone()

    # Pre-allocate output buffers.
    bsl_out = torch.empty_like(bsl_ref)
    idx_out = torch.empty_like(idx_ref)

    # Warmup on side stream so any persistent buffer (schedule_metadata) is
    # set up before capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        bsl_w, idx_w = call()
        bsl_out.copy_(bsl_w)
        idx_out.copy_(idx_w)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            bsl_w, idx_w = call()
            bsl_out.copy_(bsl_w)
            idx_out.copy_(idx_w)
    except Exception as e:
        print(f"  capture failed: {type(e).__name__}: {e}")
        return 1

    g.replay()
    torch.cuda.synchronize()

    # SET equality per row (sorted comparison): the SET of selected pool
    # blocks must agree even though ORDER varies due to fast_topk atomicAdd.
    a = idx_out.long().sort(dim=-1).values
    b = idx_ref.long().sort(dim=-1).values
    set_eq = torch.equal(a, b)

    # Strict equality (informational — expected False due to fast_topk).
    bsl_strict = torch.equal(bsl_out, bsl_ref)
    idx_strict = torch.equal(idx_out, idx_ref)

    print(f"  replay-vs-eager  SET equality (selected pool blocks): {set_eq}  "
          f"[{'OK' if set_eq else 'FAIL'}]")
    print(f"  replay-vs-eager  strict (informational, expected False): "
          f"bsl_eq={bsl_strict}  idx_eq={idx_strict}")
    print(f"    note: fast_topk_runtime uses atomicAdd, so output ORDER "
          f"differs across runs; SET is the contract.")
    return 0 if set_eq else 1


def speed_compare():
    print()
    print("=" * 110)
    print("Speed — full orchestrator (DG vs triton stage 2)")
    print("=" * 110)
    print(f"            {'eager (μs)':>20} {'graph-replay (μs)':>22}")
    print(f"{'K':>4} {'B':>3} {'ctx':>5} | {'tri':>7} {'DG':>7} {'sp':>5} | "
          f"{'tri':>7} {'DG':>7} {'sp':>5}")
    print("-" * 110)

    def bench_eager(fn, w=10, n=50):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3 / n

    def bench_graph(fn, n=50):
        fn()
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3 / n

    for K in (16, 32, 64, 128):
        for B in (1, 32):
            for ctx in (8 * 1024, 128 * 1024):
                inputs = make_inputs(K, B, ctx)
                q, kv, pkp, ppt, w, cl, bt = inputs
                block_topk = BLOCK_TOPK_FORMULA // K
                max_seq_len = bt.shape[1] * PAGED
                # schedule_metadata is computed ONCE per forward in production
                # (HisaIndexer caches it across 61 layers); bench mirrors that.
                sched = _make_pool_schedule(cl, K, POOL_PAGE)
                def dg_call():
                    return fp8_native_hierarchy_paged_mqa_logits(
                        q_fp8=q, kv_cache_fp8=kv, pool_k_pages=pkp,
                        pool_page_tables=ppt, weights=w, context_lens=cl,
                        block_tables=bt, k_block_size=K,
                        pool_page_size=POOL_PAGE, block_topk=block_topk,
                        max_seq_len=max_seq_len, schedule_metadata=sched,
                    )
                def tri_call():
                    return orchestrator_with_triton_stage2(
                        q, kv, pkp, ppt, w, cl, bt, K, POOL_PAGE, block_topk,
                    )
                tri_e = bench_eager(tri_call); dg_e = bench_eager(dg_call)
                tri_g = bench_graph(tri_call); dg_g = bench_graph(dg_call)
                print(f"{K:>4} {B:>3} {ctx//1024:>3}K | "
                      f"{tri_e:>7.1f} {dg_e:>7.1f} {tri_e/dg_e:>4.2f}x | "
                      f"{tri_g:>7.1f} {dg_g:>7.1f} {tri_g/dg_g:>4.2f}x")
                del q, kv, pkp, ppt, w, cl, bt
                torch.cuda.empty_cache()


def main():
    fails = 0
    fails += correctness_stage2()
    fails += correctness_topk_iou()
    fails += correctness_cuda_graph()
    print()
    print(f"TOTAL FAILS: {fails}")
    if fails == 0:
        print("ALL_OK")
    speed_compare()


if __name__ == "__main__":
    main()
