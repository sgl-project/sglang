"""Stage 4 isolated bench: prefill ``block_sparse_mqa_triton`` and decode
``sparse_paged_mqa_triton``. Goal: quantify BW utilization vs H100 HBM3
peak (~3.35 TB/s) to decide where the headroom is.

For each (K, shape):
  - measure kernel time (eager + graph-replay)
  - estimate bytes read (Q + sparse K + scales + weights + topk + tables)
  - report BW achieved + utilization %

Bytes calculation (single call):
  Prefill block_sparse_mqa (sq queries, each picks block_topk pool blocks):
    Q     : sq * H * D fp8 (1 byte/elem)
    K     : sq * block_topk * K * D fp8       (worst case — no cross-query reuse
                                               at this kernel level; the kernel
                                               reads each query's sparse K subset)
    Scale : sq * block_topk * K * 4
    W     : sq * H * 4
    topk  : sq * block_topk * 8 (i64)
    cu    : (sq+1) * 4 * 2

  Decode sparse_paged_mqa (B reqs, each picks block_topk pool blocks):
    Q     : B * H * D
    K     : B * block_topk * K * (D+4) bytes (interleaved fp8+scale via paged)
    W     : B * H * 4
    topk  : B * block_topk * 8
    bt    : B * (max_kv_blocks_per_req) * 4
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    block_sparse_mqa_triton,
    sparse_paged_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa_triton.test_e2e_simulated import (
    BLOCK_TOPK_FORMULA, D, H, PAGED, POOL_PAGE,
    make_decode_inputs, make_prefill_inputs,
)


# H100 SXM5 HBM3 peak = 3350 GB/s. PCIe variant is ~2.0 TB/s.
HBM_PEAK_GB_S = 3350.0


def bench_eager(fn, w=20, n=100):
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
    return s.elapsed_time(e) * 1e3 / n  # μs


def bench_graph(fn, n=100):
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
    return s.elapsed_time(e) * 1e3 / n  # μs


def prefill_bytes(sq, K, block_topk):
    q = sq * H * D                         # fp8
    k = sq * block_topk * K * D            # fp8
    s = sq * block_topk * K * 4            # f32 scale
    w = sq * H * 4                         # f32
    tk = sq * block_topk * 8               # i64
    cu = (sq + 1) * 4 * 2                  # ks + ke
    return q + k + s + w + tk + cu


def decode_bytes(B, K, block_topk, max_kv_blocks):
    q = B * H * D                          # fp8
    # paged kv-cache stores fp8 + 4B scale per token interleaved per page;
    # sparse_paged_mqa loads (D+4) bytes per token in selected blocks.
    k = B * block_topk * K * (D + 4)
    w = B * H * 4
    tk = B * block_topk * 8
    bt = B * max_kv_blocks * 4
    cl = B * 4
    return q + k + w + tk + bt + cl


def bw_pct(bytes_, time_us):
    gbs = bytes_ / time_us * 1e-3          # GB/s = bytes / μs * 1e-3
    return gbs, gbs / HBM_PEAK_GB_S * 100.0


def bench_prefill_stage4():
    print("=" * 120)
    print("Prefill stage 4 — block_sparse_mqa_triton")
    print(f"{'K':>4} {'sq':>5} {'skv':>6} {'topk':>5} {'bytes':>9} | "
          f"{'eager(μs)':>10} {'GB/s':>8} {'%peak':>6} | "
          f"{'graph(μs)':>10} {'GB/s':>8} {'%peak':>6}")
    print("-" * 120)
    sq = 8192
    for K in (16, 32, 64, 128):
        block_topk = BLOCK_TOPK_FORMULA // K
        for skv in (32 * 1024, 65 * 1024, 128 * 1024):
            q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)
            # Build a topk index that respects per-row [ks, ke).
            n_blocks_max = (skv + K - 1) // K
            # Random valid pool block indices per row, capped by ke[row]//K.
            ke_blocks = (cu_ke + K - 1) // K
            topk_idx = torch.randint(
                0, n_blocks_max, (sq, block_topk),
                device=q.device, dtype=torch.int64,
            )
            # Clamp to per-row valid range (ke_blocks - 1).
            topk_idx = torch.minimum(topk_idx, (ke_blocks - 1).unsqueeze(1).long())

            cu_ks_blk = (cu_ks // K).to(torch.int32)
            cu_ke_blk = ke_blocks.to(torch.int32)

            def call():
                return block_sparse_mqa_triton(
                    q_fp8=q, k_fp8=k_fp8, k_scale=k_scale,
                    topk_block_index=topk_idx, kv_block_size=K,
                    weights=w, cu_seqlen_ks=cu_ks_blk, cu_seqlen_ke=cu_ke_blk,
                )

            t_e = bench_eager(call)
            t_g = bench_graph(call)
            B = prefill_bytes(sq, K, block_topk)
            ge, pe = bw_pct(B, t_e)
            gg, pg = bw_pct(B, t_g)
            print(f"{K:>4} {sq:>5} {skv//1024:>4}K {block_topk:>5} "
                  f"{B/1e9:>8.2f}G | "
                  f"{t_e:>10.1f} {ge:>8.0f} {pe:>5.1f}% | "
                  f"{t_g:>10.1f} {gg:>8.0f} {pg:>5.1f}%")
            del q, k_fp8, k_scale, w, cu_ks, cu_ke, cu_ks_blk, cu_ke_blk, topk_idx
            torch.cuda.empty_cache()
        print("-" * 120)


def bench_decode_stage4():
    print()
    print("=" * 120)
    print("Decode stage 4 — sparse_paged_mqa_triton")
    print(f"{'K':>4} {'B':>3} {'ctx':>5} {'topk':>5} {'bytes':>9} | "
          f"{'eager(μs)':>10} {'GB/s':>8} {'%peak':>6} | "
          f"{'graph(μs)':>10} {'GB/s':>8} {'%peak':>6}")
    print("-" * 120)
    for K in (16, 32, 64, 128):
        block_topk = BLOCK_TOPK_FORMULA // K
        for B in (1, 4, 16, 32):
            for ctx in (8 * 1024, 65 * 1024, 128 * 1024):
                inputs = make_decode_inputs(K, B, ctx)
                q, kv, pkp, ppt, w, cl, bt = inputs

                # Build random valid topk pool block indices in [0, ctx//K).
                n_pool = (ctx + K - 1) // K
                topk_idx = torch.randint(
                    0, n_pool, (B, 1, block_topk),
                    device=q.device, dtype=torch.int64,
                )

                def call():
                    return sparse_paged_mqa_triton(
                        q_fp8=q, kv_cache_fp8=kv,
                        topk_block_index=topk_idx, kv_block_size=K,
                        weights=w, context_lens=cl, block_tables=bt,
                    )

                t_e = bench_eager(call)
                t_g = bench_graph(call)
                bytes_ = decode_bytes(B, K, block_topk, bt.shape[1])
                ge, pe = bw_pct(bytes_, t_e)
                gg, pg = bw_pct(bytes_, t_g)
                print(f"{K:>4} {B:>3} {ctx//1024:>3}K {block_topk:>5} "
                      f"{bytes_/1e6:>8.1f}M | "
                      f"{t_e:>10.1f} {ge:>8.0f} {pe:>5.1f}% | "
                      f"{t_g:>10.1f} {gg:>8.0f} {pg:>5.1f}%")
                del q, kv, pkp, ppt, w, cl, bt, topk_idx
                torch.cuda.empty_cache()
        print("-" * 120)


def main():
    print(f"H100 HBM3 peak = {HBM_PEAK_GB_S:.0f} GB/s")
    bench_prefill_stage4()
    bench_decode_stage4()


if __name__ == "__main__":
    main()
