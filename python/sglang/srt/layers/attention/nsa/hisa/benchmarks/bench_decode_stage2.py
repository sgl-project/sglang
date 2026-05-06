"""Decode stage-2 ONLY: triton ``batch_decode_pool_mqa_triton`` vs DG
``fp8_paged_mqa_logits`` + ``force_maintain_logits_decode_triton``.

Same input generator as ``test_dg_decode.py`` (production-realistic
pool_k_pages SoA layout, paged_block_size=64, pool_page_size=64).
``schedule_metadata`` is precomputed ONCE per (K, B, ctx) outside the
timed region — matches production where HisaIndexer caches it across
61 layers. Times the kernel only (no stage-1 tail-pool, no stage-3 topk,
no stage-4 sparse-paged).

Both eager and graph-replay reported. Eager numbers are dominated by
launch latency (triton single fused kernel vs DG-then-force_maintain
two-kernel chain) — graph replay is the production-relevant column.
"""
from __future__ import annotations

import deep_gemm
import torch

from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    batch_decode_pool_mqa_triton,
    force_maintain_logits_decode_triton,
)
from sglang.srt.layers.attention.nsa.hisa.tests.test_dg_decode import (
    BLOCK_TOPK_FORMULA, D, H, PAGED, POOL_PAGE, make_inputs,
)


DEVICE = torch.device("cuda")


def make_pool_schedule(context_lens, k_block_size, pool_page_size):
    npbpr = (context_lens + k_block_size - 1) // k_block_size
    return deep_gemm.get_paged_mqa_logits_metadata(
        npbpr, pool_page_size, deep_gemm.get_num_sms(),
    )


def stage2_triton(q, pool_k_pages, pool_page_tables, weights, npbpr):
    return batch_decode_pool_mqa_triton(
        q_fp8=q,
        pool_k_pages=pool_k_pages,
        pool_page_tables=pool_page_tables,
        weights_f32=weights,
        context_lens_pool=npbpr,
        pool_page_size=POOL_PAGE,
    )


def stage2_dg(q, pool_k_pages, pool_page_tables, weights, npbpr,
              schedule_metadata, max_pool_seq_len):
    pkv = pool_k_pages.view(
        pool_k_pages.shape[0], POOL_PAGE, 1, q.shape[-1] + 4,
    )
    w_2d = weights.view(-1, weights.shape[-1])
    score = deep_gemm.fp8_paged_mqa_logits(
        q, pkv, w_2d, npbpr, pool_page_tables, schedule_metadata,
        max_pool_seq_len, clean_logits=True,
    )
    force_maintain_logits_decode_triton(score, npbpr)
    return score


def stage2_dg_no_fm(q, pool_k_pages, pool_page_tables, weights, npbpr,
                    schedule_metadata, max_pool_seq_len):
    """DG only — no force_maintain. Useful to isolate the DG kernel cost."""
    pkv = pool_k_pages.view(
        pool_k_pages.shape[0], POOL_PAGE, 1, q.shape[-1] + 4,
    )
    w_2d = weights.view(-1, weights.shape[-1])
    return deep_gemm.fp8_paged_mqa_logits(
        q, pkv, w_2d, npbpr, pool_page_tables, schedule_metadata,
        max_pool_seq_len, clean_logits=True,
    )


def bench_eager(fn, w=20, n=200):
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


def bench_graph(fn, n=200):
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
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / n


def main():
    print("=" * 110)
    print("Decode stage-2 kernel-level bench: triton vs DG (+force_maintain)")
    print(f"H={H}  D={D}  PAGED={PAGED}  POOL_PAGE={POOL_PAGE}")
    print("=" * 110)
    print(f"{'K':>4} {'B':>3} {'ctx':>5} {'n_blk':>6} | "
          f"{'eager (μs)':>32} | {'graph-replay (μs)':>32}")
    print(f"{'':>21} | {'tri':>7} {'DG':>7} {'DG-fm':>7} {'sp':>5} | "
          f"{'tri':>7} {'DG':>7} {'DG-fm':>7} {'sp':>5}")
    print("-" * 110)

    for K in (16, 32, 64, 128):
        for B in (1, 4, 16, 32):
            for ctx in (8 * 1024, 65 * 1024, 128 * 1024):
                inputs = make_inputs(K, B, ctx)
                q, _, pkp, ppt, w, cl, bt = inputs
                npbpr = (cl + K - 1) // K
                max_seq_len = bt.shape[1] * PAGED
                max_pool_seq_len = (max_seq_len + K - 1) // K
                sched = make_pool_schedule(cl, K, POOL_PAGE)

                def tri_call():
                    return stage2_triton(q, pkp, ppt, w, npbpr)

                def dg_call():
                    return stage2_dg(q, pkp, ppt, w, npbpr, sched, max_pool_seq_len)

                def dg_only_call():
                    return stage2_dg_no_fm(q, pkp, ppt, w, npbpr, sched, max_pool_seq_len)

                tri_e = bench_eager(tri_call)
                dg_e = bench_eager(dg_call)
                dg_only_e = bench_eager(dg_only_call)
                tri_g = bench_graph(tri_call)
                dg_g = bench_graph(dg_call)
                dg_only_g = bench_graph(dg_only_call)

                n_blk = npbpr.max().item()
                print(f"{K:>4} {B:>3} {ctx//1024:>3}K {n_blk:>6} | "
                      f"{tri_e:>7.1f} {dg_e:>7.1f} {dg_only_e:>7.1f} "
                      f"{tri_e/dg_e:>4.2f}x | "
                      f"{tri_g:>7.1f} {dg_g:>7.1f} {dg_only_g:>7.1f} "
                      f"{tri_g/dg_g:>4.2f}x")
                del q, pkp, ppt, w, cl, bt
                torch.cuda.empty_cache()
        print("-" * 110)


if __name__ == "__main__":
    main()
