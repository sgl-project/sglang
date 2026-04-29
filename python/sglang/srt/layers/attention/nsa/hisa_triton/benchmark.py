"""Kernel-level triton vs tilelang comparison for HISA kernels.

Usage:
    python -m sglang.srt.layers.attention.nsa.hisa_triton.benchmark \\
        --kernel batch_pool_mqa \\
        --batch-sizes 1 8 32 64 \\
        --num-pool 128 512 1024

For each (B, nb) config it times both the triton and the tilelang
implementations on identical inputs, then prints a side-by-side table
with correctness + latency.
"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    batch_pool_mqa_attn_return_logits_fp8_legacy_interface,
    batch_pool_mqa_attn_return_logits_fp8_interface,
    fp8_native_block_mean_pooling_interface,
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
    fp8_native_paged_block_sparse_mqa_attn_return_logits_interface,
    fp8_native_paged_mean_pooling_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    batch_decode_pool_mqa_triton,
    batch_pool_mqa_triton,
    block_mean_pooling_triton,
    block_sparse_mqa_triton,
    paged_mean_pooling_triton,
    sparse_paged_mqa_triton,
)


DEVICE = torch.device("cuda")


# =============================================================================
# Timing helper (same pattern as hisa/benchmark/benchmark_indexer.py)
# =============================================================================


def _flush_l2() -> None:
    torch.empty(int(256e6 // 4), dtype=torch.int, device=DEVICE).zero_()


@torch.inference_mode()
def cuda_bench(fn, num_warmups: int = 5, num_iters: int = 30) -> tuple[float, float]:
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(num_iters):
        _flush_l2()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), statistics.stdev(times) if len(times) > 1 else 0.0


# =============================================================================
# batch_pool_mqa bench
# =============================================================================


@dataclass
class MqaDims:
    heads: int = 64
    dim: int = 128


def _make_paged_kv_cache_soa(
    num_phys: int, paged_block_size: int, D: int, seed: int = 0,
) -> torch.Tensor:
    """Build a paged KV cache with the production SoA byte layout per page.

    Bytes within one page (paged_block_size * (D+4) total):
      [0, paged_block_size * D)             : token-major fp8 (t0_d0..D-1, t1_d0..D-1, ...)
      [paged_block_size * D, page_bytes)    : f32 per-token scales (t0_s, t1_s, ...)

    The triton/tilelang kernels read per this SoA layout. Returned as 4D
    ``[num_phys, paged_block_size, 1, D+4]`` for caller-signature compat —
    the 4D shape is decorative; the actual bytes are SoA, NOT
    token-AoS-with-tail-scale-bytes.
    """
    torch.manual_seed(seed)
    page_bytes = paged_block_size * (D + 4)
    kv2d = torch.empty(num_phys, page_bytes, device=DEVICE, dtype=torch.uint8)

    # SoA fp8 region.
    fp8 = (
        torch.randn(num_phys, paged_block_size, D, device=DEVICE, dtype=torch.bfloat16)
        .to(torch.float8_e4m3fn)
    )
    kv2d[:, : paged_block_size * D].copy_(
        fp8.view(torch.uint8).reshape(num_phys, paged_block_size * D)
    )

    # SoA scale region.
    scale = 0.05 + 0.02 * torch.rand(
        num_phys, paged_block_size, device=DEVICE, dtype=torch.float32,
    )
    kv2d[:, paged_block_size * D :].copy_(
        scale.view(torch.uint8).reshape(num_phys, paged_block_size * 4)
    )

    return kv2d.view(num_phys, paged_block_size, 1, D + 4)


def _make_batch_pool_mqa_inputs(
    B: int, nb: int, dims: MqaDims, seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    H, D = dims.heads, dims.dim

    q = torch.randn(B, 1, H, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    bk = torch.randn(B, nb, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    # Positive scales (per-K-block).
    bks = 0.05 + 0.02 * torch.rand(B, nb, device=DEVICE, dtype=torch.float32)
    weights = torch.randn(B, H, device=DEVICE, dtype=torch.float32)
    # Random context lens in (nb / 2, nb] so the mask actually matters.
    context_lens = torch.randint(
        max(1, nb // 2), nb + 1, (B,), device=DEVICE, dtype=torch.int32,
    )
    return dict(
        q_fp8=q, blocked_k_fp8=bk, blocked_k_scale=bks,
        weights=weights, context_lens=context_lens,
    )


def _check_correctness(out_tilelang, out_triton, atol=1e-3, rtol=5e-3):
    """Per-position element-wise check on the finite entries. fp8 rounding can
    give small deltas; we allow a generous tolerance but flag if either side
    has +/-inf where the other has a finite value.
    """
    tl_finite = torch.isfinite(out_tilelang)
    tr_finite = torch.isfinite(out_triton)
    inf_match = (tl_finite == tr_finite).all().item()
    if not inf_match:
        n_tl_inf = int((~tl_finite).sum().item())
        n_tr_inf = int((~tr_finite).sum().item())
        return False, f"inf/finite positions differ: tilelang={n_tl_inf} triton={n_tr_inf}"
    both_finite = tl_finite & tr_finite
    if not both_finite.any():
        return True, "(no finite positions to compare)"
    a = out_tilelang[both_finite]
    b = out_triton[both_finite]
    abs_d = (a - b).abs()
    rel_d = abs_d / (a.abs() + 1e-6)
    max_abs = abs_d.max().item()
    max_rel = rel_d.max().item()
    ok = max_abs < atol or max_rel < rtol
    msg = f"max|abs|={max_abs:.3e}, max|rel|={max_rel:.3e}"
    return ok, msg


def bench_batch_pool_mqa(
    batch_sizes: list[int],
    num_pools: list[int],
    dims: MqaDims,
    num_warmups: int = 5,
    num_iters: int = 30,
) -> None:
    print("=" * 100)
    print(f"batch_pool_mqa  (heads={dims.heads}  dim={dims.dim})")
    print("=" * 100)
    hdr = (
        f"{'B':>4} | {'nb':>6} | {'tilelang (ms)':>15} | {'triton (ms)':>13} | "
        f"{'speedup':>8} | correctness"
    )
    print(hdr)
    print("-" * len(hdr))
    for B in batch_sizes:
        for nb in num_pools:
            inp = _make_batch_pool_mqa_inputs(B, nb, dims)

            # Warm both and check correctness on warm-up output.
            out_tl = batch_pool_mqa_attn_return_logits_fp8_legacy_interface(
                q_fp8=inp["q_fp8"],
                blocked_kv_fp8=inp["blocked_k_fp8"],
                blocked_kv_scale=inp["blocked_k_scale"],
                weights_f32=inp["weights"],
                context_lens=inp["context_lens"],
                kv_block_size=128,  # unused by the kernel; placeholder
            )
            out_tr = batch_pool_mqa_triton(
                q_fp8=inp["q_fp8"],
                blocked_k_fp8=inp["blocked_k_fp8"],
                blocked_k_scale=inp["blocked_k_scale"],
                weights_f32=inp["weights"],
                context_lens=inp["context_lens"],
            )
            ok, msg = _check_correctness(out_tl.squeeze(1), out_tr.squeeze(1))

            # Time both.
            def fn_tl():
                batch_pool_mqa_attn_return_logits_fp8_legacy_interface(
                    q_fp8=inp["q_fp8"],
                    blocked_kv_fp8=inp["blocked_k_fp8"],
                    blocked_kv_scale=inp["blocked_k_scale"],
                    weights_f32=inp["weights"],
                    context_lens=inp["context_lens"],
                    kv_block_size=128,
                )

            def fn_tr():
                batch_pool_mqa_triton(
                    q_fp8=inp["q_fp8"],
                    blocked_k_fp8=inp["blocked_k_fp8"],
                    blocked_k_scale=inp["blocked_k_scale"],
                    weights_f32=inp["weights"],
                    context_lens=inp["context_lens"],
                )

            try:
                tl_ms, tl_std = cuda_bench(fn_tl, num_warmups, num_iters)
            except Exception as e:
                tl_ms, tl_std = float("nan"), 0.0
                print(f"  [tilelang error @ B={B}, nb={nb}] {e}")
                continue
            try:
                tr_ms, tr_std = cuda_bench(fn_tr, num_warmups, num_iters)
            except Exception as e:
                tr_ms, tr_std = float("nan"), 0.0
                print(f"  [triton error @ B={B}, nb={nb}] {e}")
                continue

            speedup = tl_ms / tr_ms if tr_ms > 0 else float("nan")
            status = "OK" if ok else "❌"
            print(
                f"{B:>4} | {nb:>6} | {tl_ms:>9.3f} ±{tl_std:>4.2f} | "
                f"{tr_ms:>7.3f} ±{tr_std:>4.2f} | {speedup:>7.2f}x | "
                f"{status}  {msg}"
            )


# =============================================================================
# sparse_paged_mqa bench (the 80% hotspot)
# =============================================================================


def _make_sparse_paged_mqa_inputs(
    B: int, ctx_len: int, topk: int, dims: MqaDims,
    paged_block_size: int = 64, kv_block_size: int = 128, seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    H, D = dims.heads, dims.dim
    seq_len = 1  # decode

    q = torch.randn(B, seq_len, H, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

    # Paged KV cache: SoA byte layout (matches production).
    max_logical_blocks = (ctx_len + paged_block_size - 1) // paged_block_size
    num_phys = max_logical_blocks * B + 8  # small slack
    kv = _make_paged_kv_cache_soa(num_phys, paged_block_size, D, seed=seed)

    # Block tables: each request's logical blocks map to a disjoint set.
    block_tables = torch.arange(
        max_logical_blocks * B, device=DEVICE, dtype=torch.int32,
    ).reshape(B, max_logical_blocks)

    # Weights.
    weights = torch.randn(B, seq_len, H, device=DEVICE, dtype=torch.float32)

    # Context lens.
    context_lens = torch.full((B,), ctx_len, device=DEVICE, dtype=torch.int32)

    # TopK block indices — random valid pool-block IDs in [0, ctx_len / kv_block_size).
    num_pool_blocks = ctx_len // kv_block_size
    topk_idx = torch.stack([
        torch.randperm(num_pool_blocks, device=DEVICE)[:topk]
        for _ in range(B * seq_len)
    ]).view(B, seq_len, topk).to(torch.int64)

    return dict(
        q_fp8=q, kv_cache_fp8=kv, topk_block_index=topk_idx,
        weights=weights, context_lens=context_lens, block_tables=block_tables,
        kv_block_size=kv_block_size, paged_block_size=paged_block_size,
    )


def bench_sparse_paged_mqa(
    batch_sizes: list[int],
    context_lens: list[int],
    topk: int,
    dims: MqaDims,
    num_warmups: int = 5,
    num_iters: int = 30,
) -> None:
    print("=" * 110)
    print(f"sparse_paged_mqa  (heads={dims.heads}  dim={dims.dim}  topk={topk})")
    print("=" * 110)
    hdr = (
        f"{'B':>4} | {'ctx':>6} | {'tilelang (ms)':>15} | {'triton (ms)':>13} | "
        f"{'speedup':>8} | correctness"
    )
    print(hdr)
    print("-" * len(hdr))
    for B in batch_sizes:
        for ctx in context_lens:
            if ctx % 128 != 0 or ctx < 128:
                continue  # simplify: multiples of k_block_size=128
            inp = _make_sparse_paged_mqa_inputs(B, ctx, topk, dims)

            # Warm both + correctness check.
            out_tl = fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(
                q_fp8=inp["q_fp8"], kv_cache_fp8=inp["kv_cache_fp8"],
                topk_block_index=inp["topk_block_index"],
                kv_block_size=inp["kv_block_size"],
                weights=inp["weights"], context_lens=inp["context_lens"],
                block_tables=inp["block_tables"],
            )
            out_tr = sparse_paged_mqa_triton(
                q_fp8=inp["q_fp8"], kv_cache_fp8=inp["kv_cache_fp8"],
                topk_block_index=inp["topk_block_index"],
                kv_block_size=inp["kv_block_size"],
                weights=inp["weights"], context_lens=inp["context_lens"],
                block_tables=inp["block_tables"],
            )
            ok, msg = _check_correctness(out_tl, out_tr)

            def fn_tl():
                fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(
                    q_fp8=inp["q_fp8"], kv_cache_fp8=inp["kv_cache_fp8"],
                    topk_block_index=inp["topk_block_index"],
                    kv_block_size=inp["kv_block_size"],
                    weights=inp["weights"], context_lens=inp["context_lens"],
                    block_tables=inp["block_tables"],
                )

            def fn_tr():
                sparse_paged_mqa_triton(
                    q_fp8=inp["q_fp8"], kv_cache_fp8=inp["kv_cache_fp8"],
                    topk_block_index=inp["topk_block_index"],
                    kv_block_size=inp["kv_block_size"],
                    weights=inp["weights"], context_lens=inp["context_lens"],
                    block_tables=inp["block_tables"],
                )

            try:
                tl_ms, tl_std = cuda_bench(fn_tl, num_warmups, num_iters)
            except Exception as e:
                print(f"  [tilelang error @ B={B}, ctx={ctx}] {e}")
                continue
            try:
                tr_ms, tr_std = cuda_bench(fn_tr, num_warmups, num_iters)
            except Exception as e:
                print(f"  [triton error @ B={B}, ctx={ctx}] {e}")
                continue

            speedup = tl_ms / tr_ms if tr_ms > 0 else float("nan")
            status = "OK" if ok else "❌"
            print(
                f"{B:>4} | {ctx:>6} | {tl_ms:>9.3f} ±{tl_std:>4.2f} | "
                f"{tr_ms:>7.3f} ±{tr_std:>4.2f} | {speedup:>7.2f}x | "
                f"{status}  {msg}"
            )


# =============================================================================
# CLI
# =============================================================================


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--kernel",
        choices=[
            "batch_pool_mqa", "sparse_paged_mqa",
            "block_sparse_mqa", "batch_decode_pool_mqa_v3",
            "paged_mean_pooling", "block_mean_pooling",
            "all",
        ],
        default="all",
    )
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    # batch_pool_mqa
    p.add_argument("--num-pool", type=int, nargs="+", default=[128, 512, 1024])
    # sparse_paged_mqa
    p.add_argument("--context-lens", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    p.add_argument("--topk", type=int, default=64, help="block_topk for sparse_paged_mqa")
    p.add_argument(
        "--k-block-sizes", type=int, nargs="+", default=[128],
        help="hisa k_block_size values to bench (8/16/32/64/128)",
    )
    p.add_argument("--num-warmups", type=int, default=5)
    p.add_argument("--num-iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = _parse_args()
    torch.manual_seed(args.seed)
    dims = MqaDims(heads=args.heads, dim=args.dim)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Timing: {args.num_warmups} warmups + {args.num_iters} iters (median ± stdev ms)")

    if args.kernel in ("batch_pool_mqa", "all"):
        bench_batch_pool_mqa(
            args.batch_sizes, args.num_pool, dims,
            num_warmups=args.num_warmups, num_iters=args.num_iters,
        )
    if args.kernel in ("sparse_paged_mqa", "all"):
        bench_sparse_paged_mqa(
            args.batch_sizes, args.context_lens, args.topk, dims,
            num_warmups=args.num_warmups, num_iters=args.num_iters,
        )
    if args.kernel in ("block_sparse_mqa", "all"):
        bench_block_sparse_mqa(
            seq_qs=[1024, 4096, 16384], seq_kvs=[4096, 16384, 65536],
            topk=args.topk, dims=dims,
            num_warmups=args.num_warmups, num_iters=args.num_iters,
        )
    if args.kernel in ("batch_decode_pool_mqa_v3", "all"):
        bench_batch_decode_pool_mqa_v3(
            args.batch_sizes, num_pools=[128, 512, 1024], dims=dims,
            num_warmups=args.num_warmups, num_iters=args.num_iters,
        )
    if args.kernel in ("paged_mean_pooling", "all"):
        bench_paged_mean_pooling(
            args.batch_sizes, ctx_lens=[16384, 65536], dims=dims,
            k_block_sizes=args.k_block_sizes,
            num_warmups=args.num_warmups, num_iters=args.num_iters,
        )
    if args.kernel in ("block_mean_pooling", "all"):
        bench_block_mean_pooling(
            seq_kvs=[4096, 16384, 65536], dims=dims,
            k_block_sizes=args.k_block_sizes,
            num_warmups=args.num_warmups, num_iters=args.num_iters,
        )


# =============================================================================
# block_sparse_mqa (ragged prefill)
# =============================================================================


def _make_block_sparse_mqa_inputs(
    seq_q: int, seq_kv: int, topk: int, dims: MqaDims,
    kv_block_size: int = 128, seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    H, D = dims.heads, dims.dim
    q = torch.randn(seq_q, H, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k = torch.randn(seq_kv, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    ks = 0.05 + 0.02 * torch.rand(seq_kv, device=DEVICE, dtype=torch.float32)
    w = torch.randn(seq_q, H, device=DEVICE, dtype=torch.float32)
    cu_ks = torch.zeros(seq_q, device=DEVICE, dtype=torch.int32)
    cu_ke = torch.arange(1, seq_q + 1, device=DEVICE, dtype=torch.int32)
    num_pool_blocks = seq_kv // kv_block_size
    topk_idx = torch.stack([
        torch.randperm(num_pool_blocks, device=DEVICE)[:topk]
        for _ in range(seq_q)
    ]).to(torch.int64)
    return dict(
        q=q, k=k, k_scale=ks, topk=topk_idx,
        kv_block_size=kv_block_size,
        weights=w, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )


def bench_block_sparse_mqa(
    seq_qs: list[int], seq_kvs: list[int], topk: int, dims: MqaDims,
    num_warmups: int = 5, num_iters: int = 20,
) -> None:
    print("\n" + "=" * 110)
    print(f"block_sparse_mqa (ragged)  heads={dims.heads} dim={dims.dim} topk={topk}")
    print("=" * 110)
    hdr = f"{'seq_q':>6} | {'seq_kv':>7} | {'tilelang (ms)':>15} | {'triton (ms)':>13} | {'speedup':>8} | correctness"
    print(hdr); print("-" * len(hdr))
    for sq in seq_qs:
        for skv in seq_kvs:
            inp = _make_block_sparse_mqa_inputs(sq, skv, topk, dims)
            out_tl = fp8_native_block_sparse_mqa_attn_return_logits_interface(
                q=inp["q"], k=inp["k"], k_scale=inp["k_scale"],
                topk_block_index=inp["topk"], kv_block_size=inp["kv_block_size"],
                weights=inp["weights"],
                cu_seqlen_ks=inp["cu_seqlen_ks"], cu_seqlen_ke=inp["cu_seqlen_ke"],
            )
            out_tr = block_sparse_mqa_triton(
                q_fp8=inp["q"], k_fp8=inp["k"], k_scale=inp["k_scale"],
                topk_block_index=inp["topk"], kv_block_size=inp["kv_block_size"],
                weights=inp["weights"],
                cu_seqlen_ks=inp["cu_seqlen_ks"], cu_seqlen_ke=inp["cu_seqlen_ke"],
            )
            ok, msg = _check_correctness(out_tl, out_tr)
            def fn_tl():
                fp8_native_block_sparse_mqa_attn_return_logits_interface(
                    q=inp["q"], k=inp["k"], k_scale=inp["k_scale"],
                    topk_block_index=inp["topk"], kv_block_size=inp["kv_block_size"],
                    weights=inp["weights"],
                    cu_seqlen_ks=inp["cu_seqlen_ks"], cu_seqlen_ke=inp["cu_seqlen_ke"],
                )
            def fn_tr():
                block_sparse_mqa_triton(
                    q_fp8=inp["q"], k_fp8=inp["k"], k_scale=inp["k_scale"],
                    topk_block_index=inp["topk"], kv_block_size=inp["kv_block_size"],
                    weights=inp["weights"],
                    cu_seqlen_ks=inp["cu_seqlen_ks"], cu_seqlen_ke=inp["cu_seqlen_ke"],
                )
            tl_ms, tl_std = cuda_bench(fn_tl, num_warmups, num_iters)
            tr_ms, tr_std = cuda_bench(fn_tr, num_warmups, num_iters)
            speedup = tl_ms / tr_ms if tr_ms > 0 else float("nan")
            print(f"{sq:>6} | {skv:>7} | {tl_ms:>9.3f} ±{tl_std:>4.2f} | {tr_ms:>7.3f} ±{tr_std:>4.2f} | {speedup:>7.2f}x | {'OK' if ok else '⚠'}  {msg}")


# =============================================================================
# batch_decode_pool_mqa_v3 (v3 paged pool block_mqa)
# =============================================================================


def _make_batch_pool_mqa_v3_inputs(
    B: int, num_pool: int, dims: MqaDims,
    pool_page_size: int = 64, seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    H, D = dims.heads, dims.dim
    q = torch.randn(B, 1, H, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = torch.randn(B, H, device=DEVICE, dtype=torch.float32)
    max_pp_per_req = (num_pool + pool_page_size - 1) // pool_page_size
    N_pool_pages = B * max_pp_per_req + 8
    page_bytes = pool_page_size * (D + 4)
    pool_k_pages = torch.randint(
        0, 256, (N_pool_pages, page_bytes), device=DEVICE, dtype=torch.uint8,
    )
    # Ensure scales are positive realistic values.
    pool_k_pages_f32 = pool_k_pages.view(torch.float32)
    scale_offset_f32 = pool_page_size * D // 4
    pool_k_pages_f32[:, scale_offset_f32:scale_offset_f32 + pool_page_size] = (
        0.05 + 0.02 * torch.rand(
            N_pool_pages, pool_page_size, device=DEVICE, dtype=torch.float32,
        )
    )
    pool_page_tables = torch.arange(
        B * max_pp_per_req, device=DEVICE, dtype=torch.int32,
    ).reshape(B, max_pp_per_req)
    ctx_pool = torch.full((B,), num_pool, dtype=torch.int32, device=DEVICE)
    return dict(
        q=q, pool_k_pages=pool_k_pages, pool_page_tables=pool_page_tables,
        weights=w, context_lens_pool=ctx_pool, pool_page_size=pool_page_size,
    )


def bench_batch_decode_pool_mqa_v3(
    batch_sizes: list[int], num_pools: list[int], dims: MqaDims,
    num_warmups: int = 5, num_iters: int = 20,
) -> None:
    print("\n" + "=" * 110)
    print(f"batch_decode_pool_mqa_v3 (paged pool_k_pages)  heads={dims.heads} dim={dims.dim}")
    print("=" * 110)
    hdr = f"{'B':>4} | {'num_pool':>8} | {'tilelang (ms)':>15} | {'triton (ms)':>13} | {'speedup':>8} | correctness"
    print(hdr); print("-" * len(hdr))
    for B in batch_sizes:
        for np_ in num_pools:
            inp = _make_batch_pool_mqa_v3_inputs(B, np_, dims)
            out_tl = batch_pool_mqa_attn_return_logits_fp8_interface(
                q_fp8=inp["q"], pool_k_pages=inp["pool_k_pages"],
                pool_page_tables=inp["pool_page_tables"],
                weights_f32=inp["weights"],
                context_lens_pool=inp["context_lens_pool"],
                pool_page_size=inp["pool_page_size"],
            )
            out_tr = batch_decode_pool_mqa_triton(
                q_fp8=inp["q"], pool_k_pages=inp["pool_k_pages"],
                pool_page_tables=inp["pool_page_tables"],
                weights_f32=inp["weights"],
                context_lens_pool=inp["context_lens_pool"],
                pool_page_size=inp["pool_page_size"],
            )
            ok, msg = _check_correctness(out_tl.squeeze(1), out_tr.squeeze(1))
            def fn_tl():
                batch_pool_mqa_attn_return_logits_fp8_interface(
                    q_fp8=inp["q"], pool_k_pages=inp["pool_k_pages"],
                    pool_page_tables=inp["pool_page_tables"],
                    weights_f32=inp["weights"],
                    context_lens_pool=inp["context_lens_pool"],
                    pool_page_size=inp["pool_page_size"],
                )
            def fn_tr():
                batch_decode_pool_mqa_triton(
                    q_fp8=inp["q"], pool_k_pages=inp["pool_k_pages"],
                    pool_page_tables=inp["pool_page_tables"],
                    weights_f32=inp["weights"],
                    context_lens_pool=inp["context_lens_pool"],
                    pool_page_size=inp["pool_page_size"],
                )
            tl_ms, tl_std = cuda_bench(fn_tl, num_warmups, num_iters)
            tr_ms, tr_std = cuda_bench(fn_tr, num_warmups, num_iters)
            speedup = tl_ms / tr_ms if tr_ms > 0 else float("nan")
            print(f"{B:>4} | {np_:>8} | {tl_ms:>9.3f} ±{tl_std:>4.2f} | {tr_ms:>7.3f} ±{tr_std:>4.2f} | {speedup:>7.2f}x | {'OK' if ok else '⚠'}  {msg}")


# =============================================================================
# paged_mean_pooling (v1 full)
# =============================================================================


def _make_paged_mean_pool_inputs(
    B: int, ctx_len: int, dims: MqaDims,
    k_block_size: int = 128, paged_block_size: int = 64, seed: int = 0,
) -> dict:
    D = dims.dim
    max_logical = (ctx_len + paged_block_size - 1) // paged_block_size
    num_phys = max_logical * B + 4
    kv = _make_paged_kv_cache_soa(num_phys, paged_block_size, D, seed=seed)
    block_tables = torch.arange(
        max_logical * B, device=DEVICE, dtype=torch.int32,
    ).reshape(B, max_logical)
    ctx_lens = torch.full((B,), ctx_len, dtype=torch.int32, device=DEVICE)
    max_num_pool = (ctx_len + k_block_size - 1) // k_block_size
    return dict(
        max_num_pooling_blocks=max_num_pool,
        kv_cache=kv, context_lens=ctx_lens, block_tables=block_tables,
        k_block_size=k_block_size, paged_block_size=paged_block_size,
    )


def _paged_mean_pool_torch_ref(inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for paged mean-pool (any k_block_size).

    NOTE: the main KV cache and pool_k_pages share the same SoA byte
    layout per page: first ``paged_block_size * D`` bytes are all tokens'
    fp8 data concatenated (token-major × dim-minor), then
    ``paged_block_size * 4`` bytes of f32 scales (one per token). The
    triton/tilelang kernels read it that way; the 4D view shape
    ``[num_phys, paged_block_size, 1, D+4]`` is just for caller
    convenience and is NOT the actual byte layout.
    """
    kv = inp["kv_cache"]
    block_tables = inp["block_tables"]
    ctx_lens = inp["context_lens"]
    k_block = inp["k_block_size"]
    paged = inp["paged_block_size"]
    max_num_pool = inp["max_num_pooling_blocks"]
    B, max_log = block_tables.shape
    num_phys, _, _, DPlus4 = kv.shape
    D = DPlus4 - 4

    # SoA reinterpretation matching what the kernels read.
    kv_flat = kv.reshape(num_phys, paged * DPlus4).contiguous()
    kv_fp8 = (
        kv_flat[:, : paged * D]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(num_phys, paged, D)
    )
    kv_scale = (
        kv_flat[:, paged * D :]
        .contiguous()
        .view(torch.float32)
        .reshape(num_phys, paged)
    )

    out_k = torch.zeros(B, max_num_pool, D, dtype=torch.float8_e4m3fn, device=kv.device)
    out_s = torch.zeros(B, max_num_pool, dtype=torch.float32, device=kv.device)
    for b in range(B):
        seq = int(ctx_lens[b].item())
        n_pool = (seq + k_block - 1) // k_block
        for p in range(n_pool):
            t_s = p * k_block
            t_e = min(t_s + k_block, seq)
            acc = torch.zeros(D, dtype=torch.float32, device=kv.device)
            cnt = 0
            for t in range(t_s, t_e):
                logical = t // paged
                row = t % paged
                phys = int(block_tables[b, logical].item())
                acc += kv_fp8[phys, row].to(torch.float32) * kv_scale[phys, row].item()
                cnt += 1
            mean = acc / cnt
            mx = mean.abs().max()
            scale = max(mx.item() * (1.0 / 448.0), 1e-10)
            out_k[b, p] = (mean / scale).to(torch.float8_e4m3fn)
            out_s[b, p] = scale
    return out_k, out_s


def bench_paged_mean_pooling(
    batch_sizes: list[int], ctx_lens: list[int], dims: MqaDims,
    k_block_sizes: list[int] | None = None,
    num_warmups: int = 5, num_iters: int = 20,
) -> None:
    if k_block_sizes is None:
        k_block_sizes = [128]
    print("\n" + "=" * 130)
    print(f"paged_mean_pooling (v1 full)  dim={dims.dim}  k_block_sizes={k_block_sizes}")
    print("=" * 130)
    hdr = (
        f"{'k':>4} | {'B':>4} | {'ctx':>6} | {'ref (ms)':>15} | {'triton (ms)':>13} | "
        f"{'speedup':>8} | path | corr"
    )
    print(hdr); print("-" * len(hdr))
    for k_block in k_block_sizes:
        is_grouped = k_block < 64
        path = "grouped" if is_grouped else "legacy"
        ref_label = "torch ref" if is_grouped else "tilelang"
        for B in batch_sizes:
            for ctx in ctx_lens:
                if ctx % k_block != 0:
                    continue
                inp = _make_paged_mean_pool_inputs(B, ctx, dims, k_block_size=k_block)
                if is_grouped:
                    # Tilelang doesn't support k<paged_block_size; use a
                    # CPU-bound torch reference (small B/ctx only).
                    if B * ctx > 8 * 4096:
                        ref_msg = "ref too slow (skipped)"
                        ref_ms = float("nan"); ref_std = 0.0
                        bk_tr, bks_tr, _ = paged_mean_pooling_triton(
                            max_num_pooling_blocks=inp["max_num_pooling_blocks"],
                            kv_cache=inp["kv_cache"], context_lens=inp["context_lens"],
                            block_tables=inp["block_tables"], k_block_size=inp["k_block_size"],
                        )
                        def fn_tr(_inp=inp):
                            paged_mean_pooling_triton(
                                max_num_pooling_blocks=_inp["max_num_pooling_blocks"],
                                kv_cache=_inp["kv_cache"], context_lens=_inp["context_lens"],
                                block_tables=_inp["block_tables"], k_block_size=_inp["k_block_size"],
                            )
                        tr_ms, tr_std = cuda_bench(fn_tr, num_warmups, num_iters)
                        speedup = float("nan")
                        print(
                            f"{k_block:>4} | {B:>4} | {ctx:>6} | "
                            f"{'(skipped)':>15} | {tr_ms:>7.3f} ±{tr_std:>4.2f} | "
                            f"{'n/a':>8} | {path:>7} | {ref_msg}"
                        )
                        continue
                    bk_ref, bks_ref = _paged_mean_pool_torch_ref(inp)
                else:
                    bk_ref, bks_ref, _ = fp8_native_paged_mean_pooling_interface(
                        max_num_pooling_blocks=inp["max_num_pooling_blocks"],
                        kv_cache=inp["kv_cache"], context_lens=inp["context_lens"],
                        block_tables=inp["block_tables"], k_block_size=inp["k_block_size"],
                    )

                bk_tr, bks_tr, _ = paged_mean_pooling_triton(
                    max_num_pooling_blocks=inp["max_num_pooling_blocks"],
                    kv_cache=inp["kv_cache"], context_lens=inp["context_lens"],
                    block_tables=inp["block_tables"], k_block_size=inp["k_block_size"],
                )
                byte_diff = (bk_ref.view(torch.uint8).to(torch.int32)
                             - bk_tr.view(torch.uint8).to(torch.int32)).abs().max().item()
                scale_rel = ((bks_ref - bks_tr).abs() / (bks_ref.abs() + 1e-9)).max().item()
                corr_msg = f"max|Δbyte|={byte_diff} scale_rel={scale_rel:.2e}"

                def fn_tr(_inp=inp):
                    paged_mean_pooling_triton(
                        max_num_pooling_blocks=_inp["max_num_pooling_blocks"],
                        kv_cache=_inp["kv_cache"], context_lens=_inp["context_lens"],
                        block_tables=_inp["block_tables"], k_block_size=_inp["k_block_size"],
                    )
                tr_ms, tr_std = cuda_bench(fn_tr, num_warmups, num_iters)

                if is_grouped:
                    ref_ms, ref_std = float("nan"), 0.0
                    speedup_str = "n/a"
                    ref_str = "torch ref"
                else:
                    def fn_tl(_inp=inp):
                        fp8_native_paged_mean_pooling_interface(
                            max_num_pooling_blocks=_inp["max_num_pooling_blocks"],
                            kv_cache=_inp["kv_cache"], context_lens=_inp["context_lens"],
                            block_tables=_inp["block_tables"], k_block_size=_inp["k_block_size"],
                        )
                    ref_ms, ref_std = cuda_bench(fn_tl, num_warmups, num_iters)
                    speedup_str = f"{ref_ms / tr_ms:>7.2f}x" if tr_ms > 0 else "nan"
                    ref_str = f"{ref_ms:>9.3f} ±{ref_std:>4.2f}"

                print(
                    f"{k_block:>4} | {B:>4} | {ctx:>6} | {ref_str:>15} | "
                    f"{tr_ms:>7.3f} ±{tr_std:>4.2f} | "
                    f"{speedup_str:>8} | {path:>7} | {corr_msg}"
                )


# =============================================================================
# block_mean_pooling (ragged)
# =============================================================================


def bench_block_mean_pooling(
    seq_kvs: list[int], dims: MqaDims,
    k_block_sizes: list[int] | None = None,
    num_warmups: int = 5, num_iters: int = 20,
) -> None:
    if k_block_sizes is None:
        k_block_sizes = [128]
    D = dims.dim
    print("\n" + "=" * 120)
    print(f"block_mean_pooling (ragged)  dim={D}  k_block_sizes={k_block_sizes}")
    print("=" * 120)
    hdr = (
        f"{'k':>4} | {'seq_kv':>8} | {'tilelang (ms)':>15} | {'triton (ms)':>13} | "
        f"{'speedup':>8} | path | fp8 corr"
    )
    print(hdr); print("-" * len(hdr))
    for k_block in k_block_sizes:
        path = "grouped" if k_block < 64 else "legacy"
        for skv in seq_kvs:
            if skv % k_block != 0:
                continue
            torch.manual_seed(0)
            k = torch.randn(skv, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            ks = 0.05 + 0.02 * torch.rand(skv, device=DEVICE, dtype=torch.float32)

            bk_tl, bks_tl = fp8_native_block_mean_pooling_interface(k, ks, k_block)
            bk_tr, bks_tr = block_mean_pooling_triton(k, ks, k_block)
            byte_diff = (bk_tl.view(torch.uint8).to(torch.int32)
                         - bk_tr.view(torch.uint8).to(torch.int32)).abs().max().item()
            scale_rel = ((bks_tl - bks_tr).abs() / (bks_tl.abs() + 1e-9)).max().item()
            corr_msg = f"max|byte Δ|={byte_diff} scale_rel={scale_rel:.2e}"

            def fn_tl(): fp8_native_block_mean_pooling_interface(k, ks, k_block)
            def fn_tr(): block_mean_pooling_triton(k, ks, k_block)
            tl_ms, tl_std = cuda_bench(fn_tl, num_warmups, num_iters)
            tr_ms, tr_std = cuda_bench(fn_tr, num_warmups, num_iters)
            speedup = tl_ms / tr_ms if tr_ms > 0 else float("nan")
            print(
                f"{k_block:>4} | {skv:>8} | {tl_ms:>9.3f} ±{tl_std:>4.2f} | "
                f"{tr_ms:>7.3f} ±{tr_std:>4.2f} | {speedup:>7.2f}x | {path:>7} | {corr_msg}"
            )

if __name__ == "__main__":
    main()
