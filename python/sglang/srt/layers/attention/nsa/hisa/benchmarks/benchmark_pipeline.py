"""HISA per-stage decode pipeline benchmark — K=16 vs K=128 at long context.

Times the 4 stages of the decode-side hisa pipeline (the v4 orchestrator
+ the per-layer K-cache write hook update_pool) on identical inputs at
K=16 vs K=128, so we can attribute the observed e2e gap to (a) algorithmic
block-count growth (K=16 has 8x more pool blocks) vs (b) kernel
implementation differences (triton-vs-tilelang on stage 0/1).

Stages timed (per layer, B configurable, ctx fixed):
  0. update_pool          — SK15 triton (K<64) / tilelang (K>=64)
  1. tail-refresh         — SK16 triton (K<64) / tilelang (K>=64)
  2. block-MQA            — batch_decode_pool_mqa_triton (both K)
  3. torch.topk           — same kernel (compare grid only)
  4. sparse-paged-MQA     — sparse_paged_mqa_triton (grouped G=4 / G=1)

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m \\
        sglang.srt.layers.attention.nsa.hisa.benchmarks.benchmark_pipeline \\
        --batch 1 --ctx 131072 --block-topk 64

The script prints two tables:
  - per-stage median latency (ms) for K=16 and K=128
  - "ratio" column = K16/K128 (>1 means K=16 is slower at this stage)
"""
from __future__ import annotations

import argparse
import statistics

import torch

from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
    fp8_native_paged_mean_pooling_completed_blocks_interface,
    fp8_native_paged_mean_pooling_tail_only_interface,
)
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    batch_decode_pool_mqa_triton,
    sparse_paged_mqa_triton,
    tail_only_triton,
    update_pool_for_completed_blocks_triton,
)

DEVICE = torch.device("cuda")
H, D, PAGED, PP = 64, 128, 64, 64           # DeepSeek-V3 hisa indexer dims


def _flush_l2():
    torch.empty(int(256e6 // 4), dtype=torch.int, device=DEVICE).zero_()


@torch.inference_mode()
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
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


# --------------------------------------------------------------------------
def make_inputs(B: int, ctx_len: int, K: int):
    """Allocate buffers for the v4 decode pipeline at (B, ctx_len, K)."""
    max_kv_blocks = (ctx_len + PAGED - 1) // PAGED
    max_pool_blocks = (ctx_len + K - 1) // K
    max_pool_pages = (max_pool_blocks + PP - 1) // PP

    num_phys = B * max_kv_blocks
    num_pool_phys = B * max_pool_pages

    kv_cache = torch.zeros(num_phys, PAGED, 1, D + 4, dtype=torch.uint8, device=DEVICE)
    pool_pages = torch.zeros(num_pool_phys, PP * (D + 4), dtype=torch.uint8, device=DEVICE)
    block_tables = torch.arange(B * max_kv_blocks, dtype=torch.int32, device=DEVICE).view(B, max_kv_blocks)
    pool_page_tables = torch.arange(B * max_pool_pages, dtype=torch.int32, device=DEVICE).view(B, max_pool_pages)
    context_lens = torch.full((B,), ctx_len, dtype=torch.int32, device=DEVICE)

    q_fp8 = torch.randn(B, 1, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    weights = torch.rand(B, 1, H, device=DEVICE)

    # update_pool inputs
    max_req = max(B, 4)
    max_ctx = ctx_len + 64
    req_to_token = torch.arange(max_req * max_ctx, dtype=torch.int32, device=DEVICE).view(max_req, max_ctx)
    req_to_token = req_to_token % (num_phys * PAGED)
    pool_page_tables_v3 = torch.arange(max_req * max_pool_pages, dtype=torch.int32, device=DEVICE).view(max_req, max_pool_pages) % num_pool_phys
    req_pool_indices = torch.arange(B, dtype=torch.int64, device=DEVICE)
    prev_seq_lens = torch.full((B,), ctx_len - 1, dtype=torch.int32, device=DEVICE)
    new_seq_lens = torch.full((B,), ctx_len, dtype=torch.int32, device=DEVICE)

    return dict(
        kv_cache=kv_cache, pool_pages=pool_pages,
        block_tables=block_tables, pool_page_tables=pool_page_tables,
        pool_page_tables_v3=pool_page_tables_v3,
        context_lens=context_lens, q_fp8=q_fp8, weights=weights,
        req_to_token=req_to_token, req_pool_indices=req_pool_indices,
        prev_seq_lens=prev_seq_lens, new_seq_lens=new_seq_lens,
        max_pool_pages=max_pool_pages, max_pool_blocks=max_pool_blocks,
        num_pool_blocks_per_req=(context_lens + K - 1) // K,
    )


# --------------------------------------------------------------------------
def bench_stage_update_pool(K: int, x: dict, force: str = "auto") -> float:
    """Stage 0: update_pool_for_completed_blocks.
    force = 'auto'    → production dispatch (K<64 triton SK15, K>=64 tilelang)
          = 'triton'  → always SK15 triton (works for all K)
          = 'tilelang'→ always tilelang (asserts pooling%paged==0; errors for K<64)
    """
    kv_flat = x["kv_cache"].view(x["kv_cache"].shape[0], -1)
    use_triton = (force == "triton") or (force == "auto" and K < PAGED)
    if use_triton:
        fn = lambda: update_pool_for_completed_blocks_triton(
            kv_cache_flat=kv_flat,
            req_to_token=x["req_to_token"],
            pool_page_tables=x["pool_page_tables_v3"],
            req_pool_indices=x["req_pool_indices"],
            prev_seq_lens=x["prev_seq_lens"],
            new_seq_lens=x["new_seq_lens"],
            pool_k_pages=x["pool_pages"],
            k_block_size=K,
            paged_block_size=PAGED,
            pool_page_size=PP,
            max_pool_per_req_grid=x["max_pool_blocks"],
        )
    else:
        fn = lambda: fp8_native_paged_mean_pooling_completed_blocks_interface(
            kv_cache_flat=kv_flat,
            req_to_token=x["req_to_token"],
            pool_page_tables=x["pool_page_tables_v3"],
            req_pool_indices=x["req_pool_indices"],
            prev_seq_lens=x["prev_seq_lens"],
            new_seq_lens=x["new_seq_lens"],
            pool_k_pages=x["pool_pages"],
            k_block_size=K,
            paged_block_size=PAGED,
            pool_page_size=PP,
            max_pool_per_req_grid=x["max_pool_blocks"],
        )
    return cuda_bench(fn)


def bench_stage_tail(K: int, x: dict, force: str = "auto") -> float:
    kv_flat = x["kv_cache"].view(x["kv_cache"].shape[0], -1)
    use_triton = (force == "triton") or (force == "auto" and K < PAGED)
    if use_triton:
        fn = lambda: tail_only_triton(
            kv_cache_flat=kv_flat,
            context_lens=x["context_lens"],
            block_tables=x["block_tables"],
            pool_page_tables=x["pool_page_tables"],
            pool_k_pages=x["pool_pages"],
            k_block_size=K, paged_block_size=PAGED, pool_page_size=PP,
        )
    else:
        fn = lambda: fp8_native_paged_mean_pooling_tail_only_interface(
            kv_cache=x["kv_cache"],
            context_lens=x["context_lens"],
            block_tables=x["block_tables"],
            pool_page_tables=x["pool_page_tables"],
            pool_k_pages=x["pool_pages"],
            k_block_size=K, pool_page_size=PP,
        )
    return cuda_bench(fn)


def bench_stage_block_mqa(K: int, x: dict) -> float:
    fn = lambda: batch_decode_pool_mqa_triton(
        q_fp8=x["q_fp8"],
        pool_k_pages=x["pool_pages"],
        pool_page_tables=x["pool_page_tables"],
        weights_f32=x["weights"],
        context_lens_pool=x["num_pool_blocks_per_req"],
        pool_page_size=PP,
    )
    return cuda_bench(fn)


def bench_stage_topk(K: int, x: dict, block_topk: int) -> tuple[float, torch.Tensor]:
    nb = x["max_pool_pages"] * PP
    scores = torch.randn(x["q_fp8"].shape[0], 1, nb, device=DEVICE)
    fn = lambda: torch.topk(scores, k=min(block_topk, nb), dim=-1, sorted=False).indices
    return cuda_bench(fn), fn()


def bench_stage_sparse_mqa(K: int, x: dict, topk_idx: torch.Tensor) -> float:
    fn = lambda: sparse_paged_mqa_triton(
        q_fp8=x["q_fp8"],
        kv_cache_fp8=x["kv_cache"],
        topk_block_index=topk_idx,
        kv_block_size=K,
        weights=x["weights"],
        context_lens=x["context_lens"],
        block_tables=x["block_tables"],
    )
    return cuda_bench(fn)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=131072, help="seq_len_kv (default 128K)")
    ap.add_argument("--block-topk", type=int, default=-1,
                    help="if -1 (default): auto = 8192 // K (production formula, fixed token coverage)")
    ap.add_argument("--topk-tokens", type=int, default=8192,
                    help="fixed token coverage when --block-topk=-1 (default 8192)")
    ap.add_argument("--Ks", type=int, nargs="+", default=[16, 128])
    ap.add_argument("--force", choices=["auto", "triton", "tilelang"], default="auto",
                    help="override dispatch for stages 0/1 (update_pool, tail-refresh)")
    args = ap.parse_args()

    btk_label = (f"block_topk={args.block_topk}" if args.block_topk > 0
                 else f"block_topk=auto({args.topk_tokens}/K)")
    print(f"\nHISA decode-pipeline bench  B={args.batch}  ctx={args.ctx}  {btk_label}\n")

    rows = []      # K → list of (stage_name, ms, kernel_label)
    for K in args.Ks:
        x = make_inputs(args.batch, args.ctx, K)
        n_pool = x["max_pool_blocks"]
        n_pool_pages = x["max_pool_pages"]
        block_topk = args.block_topk if args.block_topk > 0 else max(1, args.topk_tokens // K)
        print(f"[K={K}]  pool_blocks={n_pool}  pool_pages={n_pool_pages}  block_topk={block_topk}")

        try:
            t_upd = bench_stage_update_pool(K, x, force=args.force)
        except Exception as e:
            print(f"  update_pool({args.force}) failed for K={K}: {type(e).__name__}: {str(e)[:160]}")
            t_upd = float("nan")
        try:
            t_tail = bench_stage_tail(K, x, force=args.force)
        except Exception as e:
            print(f"  tail-refresh({args.force}) failed for K={K}: {type(e).__name__}: {str(e)[:160]}")
            t_tail = float("nan")
        t_blk = bench_stage_block_mqa(K, x)
        t_top, topk_idx = bench_stage_topk(K, x, block_topk)
        # broadcast topk to int64 [B, 1, topk]
        topk_idx = topk_idx.to(torch.int64)
        t_spm = bench_stage_sparse_mqa(K, x, topk_idx)

        if args.force == "auto":
            impl_upd = "TRITON SK15" if K < PAGED else "TILELANG"
            impl_tail = "TRITON SK16" if K < PAGED else "TILELANG"
        else:
            impl_upd = f"forced:{args.force}"
            impl_tail = f"forced:{args.force}"
        impl_blk = "TRITON v3"
        impl_spm = (f"TRITON grouped G={PAGED // K}, TILE={PAGED}"
                    if K < PAGED
                    else ("TRITON legacy" if K == PAGED else f"TRITON grouped G=1, TILE={K}"))
        rows.append((K, [
            ("0. update_pool",      t_upd, impl_upd),
            ("1. tail-refresh",     t_tail, impl_tail),
            ("2. block-MQA",        t_blk, impl_blk),
            ("3. torch.topk",       t_top, "torch"),
            ("4. sparse-paged-MQA", t_spm, impl_spm),
        ]))

    # --- Tables ---
    K_LIST = [r[0] for r in rows]
    print(f"\n{'stage':<22}", end="")
    for K in K_LIST:
        print(f"{'K=' + str(K) + ' (ms)':>20}", end="")
    if len(K_LIST) == 2:
        print(f"{'ratio K' + str(K_LIST[0]) + '/K' + str(K_LIST[1]):>16}", end="")
    print()
    print("-" * (22 + 20 * len(K_LIST) + (16 if len(K_LIST) == 2 else 0)))

    n_stages = len(rows[0][1])
    totals = {K: 0.0 for K in K_LIST}
    for s in range(n_stages):
        name = rows[0][1][s][0]
        print(f"{name:<22}", end="")
        for K, stages in rows:
            t = stages[s][1]
            print(f"{t:>20.4f}", end="")
            totals[K] += t
        if len(K_LIST) == 2:
            t0 = rows[0][1][s][1]; t1 = rows[1][1][s][1]
            print(f"{(t0 / t1):>16.2f}", end="")
        print()

    print("-" * (22 + 20 * len(K_LIST) + (16 if len(K_LIST) == 2 else 0)))
    print(f"{'TOTAL':<22}", end="")
    for K in K_LIST:
        print(f"{totals[K]:>20.4f}", end="")
    if len(K_LIST) == 2:
        print(f"{(totals[K_LIST[0]] / totals[K_LIST[1]]):>16.2f}", end="")
    print()

    # Kernel-impl labels
    print(f"\n{'stage':<22}", end="")
    for K in K_LIST:
        print(f"{'K=' + str(K) + ' impl':>30}", end="")
    print()
    for s in range(n_stages):
        name = rows[0][1][s][0]
        print(f"{name:<22}", end="")
        for K, stages in rows:
            print(f"{stages[s][2]:>30}", end="")
        print()


if __name__ == "__main__":
    main()
