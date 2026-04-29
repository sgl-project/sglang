"""Profile every stage of the HisaIndexer pipeline AND compare to the
baseline deep_gemm pipeline on the same shapes.

Stages profiled:

  Hisa (prefill):
    mean_pool       (fp8_native_block_mean_pooling_interface)
    pool_mqa        (pool_mqa_attn_return_logits_fp8_interface)
    topk            (fast_topk_v2 on block_sparse_logits)
    coord_transform (gather + abs_block*k_block_size + relevant%k_block_size
                     + ks-subtract + causal mask — post-process our topk output
                     to match fast_topk_v2(row_starts=ks) semantics)
    sparse_mqa      (fp8_native_block_sparse_mqa_attn_return_logits_interface)

  Hisa (decode): same except paged variants, no ks-subtract in coord_transform.

  Baseline (for comparison):
    fp8_mqa_logits / fp8_paged_mqa_logits   (single dense kernel)
    fast_topk_v2                             (with row_starts=ks for prefill)

Prefill shapes mirror *chunked-prefill* on the real server (chunked_prefill_size=
8192): every call has Q=8192 new tokens extending a growing K pool. We sweep
K_total ∈ {8K, 16K, 32K, 64K} which is what each chunk of a 64K prompt sees.
"""
from __future__ import annotations

import argparse
import statistics
from typing import Callable

import deep_gemm
import torch
from sgl_kernel import fast_topk_v2

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    # decode stages
    fp8_native_paged_mean_pooling_interface,
    batch_pool_mqa_attn_return_logits_fp8_legacy_interface,
    fp8_native_paged_block_sparse_mqa_attn_return_logits_interface,
    # prefill stages
    fp8_native_block_mean_pooling_interface,
    pool_mqa_attn_return_logits_fp8_interface,
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
)


DEVICE = torch.device("cuda")


# --------------------------------------------------------------------------
# Timing helper
# --------------------------------------------------------------------------

def _flush_l2() -> None:
    torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda").zero_()


@torch.inference_mode()
def cuda_bench(fn: Callable, warmups: int = 5, iters: int = 20) -> tuple[float, float]:
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
    return statistics.median(times), statistics.stdev(times) if len(times) > 1 else 0.0


def _fmt(t: float, std: float, total: float) -> str:
    pct = 100.0 * t / total if total > 0 else 0.0
    return f"{t:7.3f}±{std:5.3f} ({pct:4.1f}%)"


# --------------------------------------------------------------------------
# Input builders
# --------------------------------------------------------------------------

H, D = 64, 128
INDEX_TOPK = 2048          # fast_topk_v2 hardcodes topk=2048
K_BLOCK_SIZE = 128
BLOCK_TOPK = 64


def make_prefill_inputs(q_len: int, k_total: int) -> dict:
    """Last-chunk scenario: Q=q_len NEW tokens at the end of a k_total-token seq.
    ks[i]=0, ke[i]=(k_total-q_len)+i+1   (causal within the full K range).
    """
    q = torch.randn(q_len, H, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

    k = torch.randn(k_total, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k_scale_f32 = 0.1 + 0.01 * torch.rand(k_total, device=DEVICE, dtype=torch.float32)
    k_scale_uint8 = k_scale_f32.view(torch.uint8).clone().reshape(k_total, 4)

    weights = torch.randn(q_len, H, device=DEVICE, dtype=torch.float32)

    prefix = k_total - q_len
    ks = torch.zeros(q_len, device=DEVICE, dtype=torch.int32)
    ke = (torch.arange(q_len, device=DEVICE, dtype=torch.int32) + prefix + 1)

    return dict(
        q=q, k=k, k_scale_f32=k_scale_f32, k_scale_uint8=k_scale_uint8,
        weights=weights, ks=ks, ke=ke, q_len=q_len, k_total=k_total,
    )


def make_decode_inputs(B: int, ctx: int, pbs: int = 64, num_sms: int = 132) -> dict:
    max_blocks_per_seq = (ctx + pbs - 1) // pbs
    total_blocks = max_blocks_per_seq * B + 4

    q = torch.randn(B, 1, H, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    kv_cache = torch.empty(total_blocks, pbs, 1, D + 4, device=DEVICE, dtype=torch.uint8)
    kv_cache[..., :D].copy_(
        torch.randn(total_blocks, pbs, 1, D, device=DEVICE, dtype=torch.bfloat16)
        .to(torch.float8_e4m3fn).view(torch.uint8)
    )
    scales = 0.1 + 0.01 * torch.rand(total_blocks, pbs, 1, 1, device=DEVICE, dtype=torch.float32)
    kv_cache[..., D:].copy_(scales.view(torch.uint8).reshape(total_blocks, pbs, 1, 4))

    weights = torch.randn(B, H, device=DEVICE, dtype=torch.float32)
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=DEVICE)
    block_tables = torch.arange(
        max_blocks_per_seq * B, device=DEVICE, dtype=torch.int32
    ).reshape(B, max_blocks_per_seq)
    schedule = deep_gemm.get_paged_mqa_logits_metadata(seq_lens, pbs, num_sms)
    max_seq_len = block_tables.shape[1] * pbs

    return dict(
        q=q, kv_cache=kv_cache, weights=weights, seq_lens=seq_lens,
        block_tables=block_tables, schedule=schedule, max_seq_len=max_seq_len,
        B=B, ctx=ctx,
    )


# --------------------------------------------------------------------------
# Prefill stage benchmarks
# --------------------------------------------------------------------------

def bench_prefill(q_len: int, k_total: int, warmups: int, iters: int) -> dict:
    inp = make_prefill_inputs(q_len, k_total)
    q, k, k_scale_f32, k_scale_uint8 = inp["q"], inp["k"], inp["k_scale_f32"], inp["k_scale_uint8"]
    weights, ks, ke = inp["weights"], inp["ks"], inp["ke"]

    # --- hisa stages ---
    cu_ks_blk = ks // K_BLOCK_SIZE
    cu_ke_blk = (ke + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE

    def h_mean_pool():
        return fp8_native_block_mean_pooling_interface(k, k_scale_f32, K_BLOCK_SIZE)
    blocked_k, blocked_k_scale = h_mean_pool()
    t_mp, s_mp = cuda_bench(h_mean_pool, warmups, iters)

    def h_pool_mqa():
        return pool_mqa_attn_return_logits_fp8_interface(
            q_fp8=q, blocked_kv_fp8=blocked_k, blocked_kv_scale=blocked_k_scale,
            kv_block_size=K_BLOCK_SIZE, weights_f32=weights,
            cu_seqlen_blocked_ks=cu_ks_blk, cu_seqlen_blocked_ke=cu_ke_blk,
        )
    block_logits = h_pool_mqa()
    t_pm, s_pm = cuda_bench(h_pool_mqa, warmups, iters)

    topk_k = min(BLOCK_TOPK, block_logits.shape[-1])
    def h_topk_blk():
        return torch.topk(block_logits, k=topk_k, dim=-1, sorted=False).indices
    topk_block_indices = h_topk_blk()
    # (topk of block_logits is part of hisa's compose; we roll it into pool_mqa
    #  timing effectively — skip separate measurement)

    # Run sparse_mqa so we have block_sparse_logits to topk over.
    def h_sparse_mqa():
        return fp8_native_block_sparse_mqa_attn_return_logits_interface(
            q=q, k=k, k_scale=k_scale_f32, topk_block_index=topk_block_indices,
            kv_block_size=K_BLOCK_SIZE, weights=weights,
            cu_seqlen_ks=ks, cu_seqlen_ke=ke,
        )
    block_sparse_logits = h_sparse_mqa()
    t_sm, s_sm = cuda_bench(h_sparse_mqa, warmups, iters)

    # Now the POST-PROCESSING stages in HisaIndexer._get_topk_ragged.
    sparse_len = block_sparse_logits.shape[-1]
    full_lens = torch.full((q_len,), sparse_len, dtype=torch.int32, device=DEVICE)

    def h_fast_topk():
        return fast_topk_v2(block_sparse_logits, full_lens, INDEX_TOPK)
    relevant = h_fast_topk()
    t_tk, s_tk = cuda_bench(h_fast_topk, warmups, iters)

    def h_coord_transform():
        rs = relevant.clamp(min=0)
        abs_block = torch.gather(
            topk_block_indices.to(torch.int64), -1,
            (rs // K_BLOCK_SIZE).to(torch.int64),
        )
        raw = abs_block * K_BLOCK_SIZE + (rs % K_BLOCK_SIZE)
        raw = raw - ks[:, None]
        valid = (raw >= 0) & (raw < (ke - ks)[:, None])
        return raw.masked_fill(~valid | (relevant == -1), -1).to(torch.int32)
    t_ct, s_ct = cuda_bench(h_coord_transform, warmups, iters)

    hisa_total = t_mp + t_pm + t_sm + t_tk + t_ct

    # --- baseline: deep_gemm.fp8_mqa_logits + fast_topk_v2 with row_starts=ks ---
    def b_dense_logits():
        return deep_gemm.fp8_mqa_logits(
            q, (k, k_scale_f32), weights, ks, ke, clean_logits=False,
        )
    dense_logits = b_dense_logits()
    t_bd, s_bd = cuda_bench(b_dense_logits, warmups, iters)

    seqlens_topk = (ke - ks).to(torch.int32)
    def b_fast_topk():
        return fast_topk_v2(dense_logits, seqlens_topk, INDEX_TOPK, row_starts=ks)
    t_bt, s_bt = cuda_bench(b_fast_topk, warmups, iters)

    baseline_total = t_bd + t_bt

    return dict(
        hisa=dict(
            mean_pool=(t_mp, s_mp), pool_mqa=(t_pm, s_pm),
            sparse_mqa=(t_sm, s_sm), fast_topk=(t_tk, s_tk),
            coord=(t_ct, s_ct), total=hisa_total,
        ),
        baseline=dict(
            dense=(t_bd, s_bd), fast_topk=(t_bt, s_bt),
            total=baseline_total,
        ),
        speedup=baseline_total / hisa_total if hisa_total > 0 else float("nan"),
    )


# --------------------------------------------------------------------------
# Decode stage benchmarks
# --------------------------------------------------------------------------

def bench_decode(B: int, ctx: int, warmups: int, iters: int) -> dict:
    inp = make_decode_inputs(B, ctx)
    q, kv_cache = inp["q"], inp["kv_cache"]
    weights, seq_lens = inp["weights"], inp["seq_lens"]
    block_tables, schedule = inp["block_tables"], inp["schedule"]
    max_seq_len = inp["max_seq_len"]

    max_nb = (ctx + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE

    # --- hisa ---
    def h_mean_pool():
        return fp8_native_paged_mean_pooling_interface(
            max_nb, kv_cache, seq_lens, block_tables, K_BLOCK_SIZE,
        )
    blocked_k, blocked_k_scale, num_pool = h_mean_pool()
    t_mp, s_mp = cuda_bench(h_mean_pool, warmups, iters)

    def h_pool_mqa():
        return batch_pool_mqa_attn_return_logits_fp8_legacy_interface(
            q_fp8=q, blocked_kv_fp8=blocked_k, blocked_kv_scale=blocked_k_scale,
            weights_f32=weights, context_lens=num_pool, kv_block_size=K_BLOCK_SIZE,
        )
    block_logits = h_pool_mqa()
    t_pm, s_pm = cuda_bench(h_pool_mqa, warmups, iters)

    topk_k = min(BLOCK_TOPK, block_logits.shape[-1])
    topk_block_indices = torch.topk(block_logits, k=topk_k, dim=-1, sorted=False).indices

    def h_sparse_mqa():
        return fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(
            q_fp8=q, kv_cache_fp8=kv_cache, topk_block_index=topk_block_indices,
            kv_block_size=K_BLOCK_SIZE, weights=weights,
            context_lens=seq_lens, block_tables=block_tables,
        )
    block_sparse_logits = h_sparse_mqa()
    t_sm, s_sm = cuda_bench(h_sparse_mqa, warmups, iters)

    # Squeeze for HisaIndexer compatibility (paged output is [B, 1, ...]).
    if block_sparse_logits.ndim == 3:
        block_sparse_logits_2d = block_sparse_logits.squeeze(1)
        topk_block_indices_2d = topk_block_indices.squeeze(1) if topk_block_indices.ndim == 3 else topk_block_indices
    else:
        block_sparse_logits_2d = block_sparse_logits
        topk_block_indices_2d = topk_block_indices

    sparse_len = block_sparse_logits_2d.shape[-1]
    full_lens = torch.full((B,), sparse_len, dtype=torch.int32, device=DEVICE)

    def h_fast_topk():
        return fast_topk_v2(block_sparse_logits_2d, full_lens, INDEX_TOPK)
    relevant = h_fast_topk()
    t_tk, s_tk = cuda_bench(h_fast_topk, warmups, iters)

    def h_coord_transform():
        rs = relevant.clamp(min=0)
        abs_block = torch.gather(
            topk_block_indices_2d.to(torch.int64), -1,
            (rs // K_BLOCK_SIZE).to(torch.int64),
        )
        raw = abs_block * K_BLOCK_SIZE + (rs % K_BLOCK_SIZE)
        valid = raw < seq_lens[:, None]
        return raw.masked_fill(~valid | (relevant == -1), -1).to(torch.int32)
    t_ct, s_ct = cuda_bench(h_coord_transform, warmups, iters)

    hisa_total = t_mp + t_pm + t_sm + t_tk + t_ct

    # --- baseline: deep_gemm.fp8_paged_mqa_logits + fast_topk_v2 ---
    kv_reshape = kv_cache.view(-1, 64, 1, 132)  # [num_blocks, pbs=64, 1, D+4]
    def b_paged():
        return deep_gemm.fp8_paged_mqa_logits(
            q, kv_reshape, weights.squeeze(-1) if weights.ndim == 3 else weights,
            seq_lens, block_tables, schedule,
            max_context_len=max_seq_len, clean_logits=False,
        )
    dense = b_paged()
    t_bd, s_bd = cuda_bench(b_paged, warmups, iters)

    # For baseline paged: fast_topk_v2(logits, seq_lens, topk). logits shape is
    # [B, max_context_len] — same across the batch. No row_starts for paged.
    def b_fast_topk():
        return fast_topk_v2(dense, seq_lens, INDEX_TOPK)
    t_bt, s_bt = cuda_bench(b_fast_topk, warmups, iters)

    baseline_total = t_bd + t_bt

    return dict(
        hisa=dict(
            mean_pool=(t_mp, s_mp), pool_mqa=(t_pm, s_pm),
            sparse_mqa=(t_sm, s_sm), fast_topk=(t_tk, s_tk),
            coord=(t_ct, s_ct), total=hisa_total,
        ),
        baseline=dict(
            dense=(t_bd, s_bd), fast_topk=(t_bt, s_bt),
            total=baseline_total,
        ),
        speedup=baseline_total / hisa_total if hisa_total > 0 else float("nan"),
    )


# --------------------------------------------------------------------------
# Pretty-print
# --------------------------------------------------------------------------

def print_prefill_table(rows: list[tuple[int, int, dict]]) -> None:
    print("\n" + "=" * 118)
    print(f"PREFILL stage profile  (Q=8192 fixed; K sweeps; index_topk={INDEX_TOPK}, "
          f"k_block_size={K_BLOCK_SIZE}, block_topk={BLOCK_TOPK})")
    print("=" * 118)
    print(f"{'K_total':>8} | {'mean_pool':>19} {'pool_mqa':>19} {'sparse_mqa':>19} "
          f"{'fast_topk':>19} {'coord':>19} | {'TOTAL':>8}")
    print("-" * 118)
    for q_len, k_total, r in rows:
        h = r["hisa"]
        tot = h["total"]
        print(
            f"{k_total:>8} | "
            f"{_fmt(*h['mean_pool'], tot):>19} "
            f"{_fmt(*h['pool_mqa'], tot):>19} "
            f"{_fmt(*h['sparse_mqa'], tot):>19} "
            f"{_fmt(*h['fast_topk'], tot):>19} "
            f"{_fmt(*h['coord'], tot):>19} | "
            f"{tot:>7.3f}"
        )
    print()
    print(f"{'K_total':>8} | {'baseline dense':>22} {'baseline fast_topk':>22} | "
          f"{'baseline TOTAL':>15} | {'hisa TOTAL':>11} | {'speedup':>8}")
    print("-" * 118)
    for q_len, k_total, r in rows:
        b = r["baseline"]
        btot = b["total"]
        htot = r["hisa"]["total"]
        print(
            f"{k_total:>8} | "
            f"{b['dense'][0]:>10.3f}±{b['dense'][1]:5.3f}    "
            f"{b['fast_topk'][0]:>10.3f}±{b['fast_topk'][1]:5.3f}    | "
            f"{btot:>14.3f} | {htot:>10.3f} | {r['speedup']:>7.2f}x"
        )


def print_decode_table(rows: list[tuple[int, int, dict]]) -> None:
    print("\n" + "=" * 118)
    print(f"DECODE stage profile  (paged, next_n=1; index_topk={INDEX_TOPK}, "
          f"k_block_size={K_BLOCK_SIZE}, block_topk={BLOCK_TOPK})")
    print("=" * 118)
    print(f"{'B':>3} {'ctx':>6} | {'mean_pool':>19} {'pool_mqa':>19} {'sparse_mqa':>19} "
          f"{'fast_topk':>19} {'coord':>19} | {'TOTAL':>8}")
    print("-" * 118)
    for B, ctx, r in rows:
        h = r["hisa"]
        tot = h["total"]
        print(
            f"{B:>3} {ctx:>6} | "
            f"{_fmt(*h['mean_pool'], tot):>19} "
            f"{_fmt(*h['pool_mqa'], tot):>19} "
            f"{_fmt(*h['sparse_mqa'], tot):>19} "
            f"{_fmt(*h['fast_topk'], tot):>19} "
            f"{_fmt(*h['coord'], tot):>19} | "
            f"{tot:>7.3f}"
        )
    print()
    print(f"{'B':>3} {'ctx':>6} | {'baseline dense':>22} {'baseline fast_topk':>22} | "
          f"{'baseline TOTAL':>15} | {'hisa TOTAL':>11} | {'speedup':>8}")
    print("-" * 118)
    for B, ctx, r in rows:
        b = r["baseline"]
        btot = b["total"]
        htot = r["hisa"]["total"]
        print(
            f"{B:>3} {ctx:>6} | "
            f"{b['dense'][0]:>10.3f}±{b['dense'][1]:5.3f}    "
            f"{b['fast_topk'][0]:>10.3f}±{b['fast_topk'][1]:5.3f}    | "
            f"{btot:>14.3f} | {htot:>10.3f} | {r['speedup']:>7.2f}x"
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["both", "prefill", "decode"], default="both")
    p.add_argument("--prefill-q", type=int, default=8192)
    p.add_argument("--prefill-k-totals", type=int, nargs="+",
                   default=[8192, 16384, 32768, 65536])
    p.add_argument("--decode-batches", type=int, nargs="+", default=[1, 8, 32, 64])
    p.add_argument("--decode-ctxs", type=int, nargs="+", default=[4096, 16384, 65536])
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Timing: {args.warmups} warmups + {args.iters} iters (median ± stdev, ms)")

    if args.mode in ("both", "prefill"):
        rows = []
        for k in args.prefill_k_totals:
            try:
                r = bench_prefill(args.prefill_q, k, args.warmups, args.iters)
                rows.append((args.prefill_q, k, r))
            except Exception as e:
                print(f"[FAIL] prefill q={args.prefill_q} k={k}: {e}")
        if rows:
            print_prefill_table(rows)

    if args.mode in ("both", "decode"):
        rows = []
        for B in args.decode_batches:
            for ctx in args.decode_ctxs:
                try:
                    r = bench_decode(B, ctx, args.warmups, args.iters)
                    rows.append((B, ctx, r))
                except Exception as e:
                    print(f"[FAIL] decode B={B} ctx={ctx}: {e}")
        if rows:
            print_decode_table(rows)


if __name__ == "__main__":
    main()
