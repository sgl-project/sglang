"""CUDA-graph A/B for fast_topk vs torch.topk on the decode hot path.

Hypothesis (from bench_decode_topk_ab.py + analysis): fast_topk's kernel-
level win on decode (~22 μs) is masked by CPU dispatch latency between
stage 3 and stage 4 — fast_topk's stage 3 GPU time (9 μs) is shorter than
the CPU work needed to launch stage 4, so the GPU sits idle waiting.

If true, capturing the whole decode step into a CUDA graph and replaying
should fix it: replay launches the entire pipeline with one `cudaGraphLaunch`
(no per-kernel CPU dispatch in the steady state), so fast_topk's 22 μs
GPU savings should fully translate to wall-clock.

Run: CUDA_VISIBLE_DEVICES=0 python bench_decode_topk_cudagraph.py
"""
from __future__ import annotations

import statistics

import torch

from sglang.srt.layers.attention.nsa.hisa.fast_topk_runtime import fast_topk_runtime
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    batch_decode_pool_mqa_triton,
    sparse_paged_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa.benchmarks.profile_e2e_stages import (
    POOL_PAGE,
    make_decode_inputs,
)


def decode_step_torch_topk(q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req,
                           K, block_topk):
    score = batch_decode_pool_mqa_triton(
        q_fp8=q, pool_k_pages=pp, pool_page_tables=ppt,
        weights_f32=w, context_lens_pool=num_pool_per_req,
        pool_page_size=POOL_PAGE,
    )
    idx = torch.topk(
        score, k=min(block_topk, score.shape[-1]), dim=-1, sorted=False,
    ).indices
    return sparse_paged_mqa_triton(
        q_fp8=q, kv_cache_fp8=kv, topk_block_index=idx,
        kv_block_size=K, weights=w,
        context_lens=ctxl, block_tables=bt,
    )


def decode_step_fast_topk(q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req,
                          K, block_topk):
    score = batch_decode_pool_mqa_triton(
        q_fp8=q, pool_k_pages=pp, pool_page_tables=ppt,
        weights_f32=w, context_lens_pool=num_pool_per_req,
        pool_page_size=POOL_PAGE,
    )
    B, S, L = score.shape
    topk = min(block_topk, L)
    idx = fast_topk_runtime(score.view(B, L), topk).view(B, S, topk)
    return sparse_paged_mqa_triton(
        q_fp8=q, kv_cache_fp8=kv, topk_block_index=idx,
        kv_block_size=K, weights=w,
        context_lens=ctxl, block_tables=bt,
    )


def capture(fn, *args):
    """Warmup on side stream, then capture into a CUDAGraph. Returns the
    graph (replay with ``g.replay()``)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(11):
            fn(*args)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(*args)
    return g


def bench_replay(g: torch.cuda.CUDAGraph, n_blocks=30, iters_per_block=200):
    # Warmup the replay path
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_blocks):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters_per_block):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3 / iters_per_block)
    return times


def bench_eager(fn, args, n_blocks=30, iters_per_block=200):
    """Same A/B harness but without graph capture, for direct comparison."""
    for _ in range(50):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_blocks):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters_per_block):
            fn(*args)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3 / iters_per_block)
    return times


def fmt(label, vals):
    return (f"{label:>16}  min={min(vals):>6.2f}  med={statistics.median(vals):>6.2f}"
            f"  mean={statistics.mean(vals):>6.2f}  std={statistics.stdev(vals):>5.2f}")


def main():
    cases = [
        (16, 1, 128 * 1024),
        (16, 4, 128 * 1024),
        (32, 1, 128 * 1024),
    ]
    print("=" * 110)
    print("CUDA-graph A/B: fast_topk vs torch.topk on decode  "
          "(30 blocks × 200 iters)")
    print("=" * 110)

    for K, B, ctx in cases:
        block_topk = 8192 // K
        inputs = make_decode_inputs(K, B, ctx)
        q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req = inputs
        args = (q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req, K, block_topk)

        print(f"\n--- K={K} B={B} ctx={ctx//1024}K block_topk={block_topk} ---")

        # Eager (no graph) — for comparison
        eager_torch = bench_eager(decode_step_torch_topk, args)
        eager_fast = bench_eager(decode_step_fast_topk, args)
        print(f"  EAGER:")
        print(f"     {fmt('torch.topk', eager_torch)}  μs/step")
        print(f"     {fmt('fast_topk',  eager_fast)}  μs/step")
        eager_delta = statistics.median(eager_fast) - statistics.median(eager_torch)
        print(f"     Δ (fast − torch) median = {eager_delta:+.2f} μs/step")

        # CUDA graph
        g_torch = capture(decode_step_torch_topk, *args)
        g_fast = capture(decode_step_fast_topk, *args)
        graph_torch = bench_replay(g_torch)
        graph_fast = bench_replay(g_fast)
        print(f"  CUDA GRAPH:")
        print(f"     {fmt('torch.topk', graph_torch)}  μs/step")
        print(f"     {fmt('fast_topk',  graph_fast)}  μs/step")
        graph_delta = statistics.median(graph_fast) - statistics.median(graph_torch)
        print(f"     Δ (fast − torch) median = {graph_delta:+.2f} μs/step")

        # Did the graph reveal the kernel-level win?
        print(f"  ANALYSIS:")
        print(f"     eager     fast−torch = {eager_delta:+6.2f} μs  ({'fast slower' if eager_delta>0 else 'fast faster'})")
        print(f"     graph     fast−torch = {graph_delta:+6.2f} μs  ({'fast slower' if graph_delta>0 else 'fast faster'})")
        print(f"     graph reveals extra savings of {eager_delta - graph_delta:+.2f} μs/step")


if __name__ == "__main__":
    main()
