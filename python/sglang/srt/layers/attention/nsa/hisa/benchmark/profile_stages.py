"""Per-stage profiler for the hierarchical indexer pipeline.

Each stage is timed individually with CUDA events so Python dispatch overhead
is included.  This reveals where time is actually spent vs. the end-to-end
benchmark number.

Usage::

    python profile_stages.py                  # both decode + prefill
    python profile_stages.py --mode decode
    python profile_stages.py --mode prefill
    python profile_stages.py --batch-sizes 1 64 --context-lens 4096 65536
"""

from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch

from deep_gemm import get_paged_mqa_logits_metadata

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    # decode stages
    fp8_native_paged_mean_pooling_interface,
    batch_pool_mqa_attn_return_logits_fp8_interface,
    fp8_native_paged_block_sparse_mqa_attn_return_logits_interface,
    # prefill stages
    fp8_native_block_mean_pooling_interface,
    pool_mqa_attn_return_logits_fp8_interface,
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa.benchmark.benchmark_indexer import (
    IndexerDims,
    _make_decode_inputs,
    _make_prefill_inputs,
)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _flush_l2() -> None:
    torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda").zero_()


@torch.inference_mode()
def cuda_bench(fn: Callable, warmups: int = 5, iters: int = 20) -> tuple[float, float]:
    """Return (median_ms, stdev_ms) including Python dispatch overhead."""
    torch.cuda.synchronize()
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        _flush_l2()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    med = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return med, std


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _fmt_cell(t_ms: float, std_ms: float, total_ms: float) -> str:
    pct = 100.0 * t_ms / total_ms if total_ms > 0 else 0.0
    return f"{t_ms:.3f}±{std_ms:.3f} ({pct:4.1f}%)"


def _print_table(rows: list[dict], stage_names: list[str]) -> None:
    col_w = 22
    key_cols = list(rows[0].keys()) if rows else []
    key_w = [max(len(k), 6) for k in key_cols[: -len(stage_names) - 1]]

    header_keys = "  ".join(f"{k:>{w}}" for k, w in zip(key_cols, key_w))
    header_stages = "  ".join(f"{s:>{col_w}}" for s in stage_names + ["TOTAL"])
    print(f"{header_keys}  {header_stages}")
    print("-" * (sum(key_w) + len(key_w) * 2 + (len(stage_names) + 1) * (col_w + 2)))

    for row in rows:
        key_part = "  ".join(
            f"{str(v):>{w}}" for (k, v), w in zip(list(row.items())[: len(key_w)], key_w)
        )
        stage_part = "  ".join(
            f"{row[s]:>{col_w}}" for s in stage_names + ["TOTAL"]
        )
        print(f"{key_part}  {stage_part}")


# ---------------------------------------------------------------------------
# Decode profiling
# ---------------------------------------------------------------------------

@torch.inference_mode()
def profile_decode(
    batch_sizes: list[int],
    context_lens: list[int],
    k_block_size: int,
    block_topk: int,
    paged_block_size: int,
    max_model_len: int,
    num_sms: int,
    dims: IndexerDims,
    device: torch.device,
    warmups: int,
    iters: int,
) -> None:
    STAGES = ["mean_pool", "pool_mqa", "topk+cast", "sparse_mqa"]

    print("\n" + "=" * 110)
    print("DECODE PER-STAGE PROFILE  (all times in ms, includes Python dispatch)")
    print(f"  k_block_size={k_block_size}  block_topk={block_topk}  "
          f"paged_block_size={paged_block_size}  num_sms={num_sms}")
    print("=" * 110)

    rows = []
    for B in batch_sizes:
        for ctx in context_lens:
            if ctx > max_model_len:
                continue

            inp = _make_decode_inputs(B, ctx, dims, device, paged_block_size, num_sms)
            q_fp8        = inp["q_fp8"]
            kv_cache     = inp["kv_cache"]
            weights      = inp["weights"]
            seq_lens     = inp["seq_lens"]
            block_tables = inp["block_tables"]

            max_nb = (ctx + k_block_size - 1) // k_block_size

            # ---- stage 1: mean_pool (returns fp8 + per-block scale + lengths) ----
            def s1():
                return fp8_native_paged_mean_pooling_interface(
                    max_nb, kv_cache, seq_lens, block_tables, k_block_size,
                )

            blocked_k_fp8, blocked_k_scale, num_pooling_blocks = s1()
            t1, std1 = cuda_bench(s1, warmups, iters)

            # ---- stage 2: pool_mqa (fp8; matches what hierarchy now uses) ----
            def s2():
                return batch_pool_mqa_attn_return_logits_fp8_interface(
                    q_fp8=q_fp8,
                    blocked_kv_fp8=blocked_k_fp8,
                    blocked_kv_scale=blocked_k_scale,
                    weights_f32=weights,
                    context_lens=num_pooling_blocks,
                    kv_block_size=k_block_size,
                )

            block_logits = s2()
            t2, std2 = cuda_bench(s2, warmups, iters)

            # ---- stage 3: topk (int64 native; sparse_mqa now consumes int64) -
            topk_k = min(block_topk, block_logits.shape[-1])

            def s3():
                return torch.topk(block_logits, k=topk_k, dim=-1, sorted=False).indices

            topk_idx = s3()
            t3, std3 = cuda_bench(s3, warmups, iters)

            # ---- stage 4: sparse_mqa ----------------------------------------
            def s4():
                return fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(
                    q_fp8=q_fp8,
                    kv_cache_fp8=kv_cache,
                    topk_block_index=topk_idx,
                    kv_block_size=k_block_size,
                    weights=weights,
                    context_lens=seq_lens,
                    block_tables=block_tables,
                )

            t4, std4 = cuda_bench(s4, warmups, iters)

            total = t1 + t2 + t3 + t4
            row = {
                "B": B, "ctx": ctx,
                "mean_pool": _fmt_cell(t1, std1, total),
                "pool_mqa": _fmt_cell(t2, std2, total),
                "topk+cast": _fmt_cell(t3, std3, total),
                "sparse_mqa": _fmt_cell(t4, std4, total),
                "TOTAL": f"{total:.3f}",
            }
            rows.append(row)

    _print_table(rows, STAGES)


# ---------------------------------------------------------------------------
# Prefill profiling
# ---------------------------------------------------------------------------

@torch.inference_mode()
def profile_prefill(
    seq_lens: list[int],
    k_block_size: int,
    block_topk: int,
    dims: IndexerDims,
    device: torch.device,
    warmups: int,
    iters: int,
) -> None:
    STAGES = ["mean_pool", "pool_mqa", "topk+cast", "sparse_mqa"]

    print("\n" + "=" * 110)
    print("PREFILL PER-STAGE PROFILE  (all times in ms, includes Python dispatch)")
    print(f"  k_block_size={k_block_size}  block_topk={block_topk}")
    print("=" * 110)

    rows = []
    for seq_len in seq_lens:
        inp = _make_prefill_inputs(seq_len, dims, device)
        q_fp8         = inp["q_fp8"]
        k_fp8         = inp["k_fp8"]
        k_scale_f32   = inp["k_scale_f32_flat"]
        weights       = inp["weights"]
        cu_ks         = inp["cu_seqlen_ks"]
        cu_ke         = inp["cu_seqlen_ke"]

        cu_ks_blk = cu_ks // k_block_size
        cu_ke_blk = (cu_ke + k_block_size - 1) // k_block_size

        # ---- stage 1: mean_pool -------------------------------------------
        def s1():
            return fp8_native_block_mean_pooling_interface(k_fp8, k_scale_f32, k_block_size)

        blocked_k, blocked_k_scale = s1()
        t1, std1 = cuda_bench(s1, warmups, iters)

        # ---- stage 2: pool_mqa (fp8) ------------------------------------
        def s2():
            return pool_mqa_attn_return_logits_fp8_interface(
                q_fp8=q_fp8,
                blocked_kv_fp8=blocked_k,
                blocked_kv_scale=blocked_k_scale,
                kv_block_size=k_block_size,
                weights_f32=weights,
                cu_seqlen_blocked_ks=cu_ks_blk,
                cu_seqlen_blocked_ke=cu_ke_blk,
            )

        block_logits = s2()
        t2, std2 = cuda_bench(s2, warmups, iters)

        # ---- stage 3: topk (int64 native; sparse_mqa now consumes int64) ---
        topk_k = min(block_topk, block_logits.shape[-1])

        def s3():
            return torch.topk(block_logits, k=topk_k, dim=-1, sorted=False).indices

        topk_idx = s3()
        t3, std3 = cuda_bench(s3, warmups, iters)

        # ---- stage 4: sparse_mqa ------------------------------------------
        def s4():
            return fp8_native_block_sparse_mqa_attn_return_logits_interface(
                q=q_fp8,
                k=k_fp8,
                k_scale=k_scale_f32,
                topk_block_index=topk_idx,
                kv_block_size=k_block_size,
                weights=weights,
                cu_seqlen_ks=cu_ks,
                cu_seqlen_ke=cu_ke,
            )

        t4, std4 = cuda_bench(s4, warmups, iters)

        total = t1 + t2 + t3 + t4
        row = {
            "seq_len": seq_len,
            "mean_pool": _fmt_cell(t1, std1, total),
            "pool_mqa": _fmt_cell(t2, std2, total),
            "topk+cast": _fmt_cell(t3, std3, total),
            "sparse_mqa": _fmt_cell(t4, std4, total),
            "TOTAL": f"{total:.3f}",
        }
        rows.append(row)

    _print_table(rows, STAGES)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["both", "decode", "prefill"], default="both")
    p.add_argument("--n-head", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    # --block-configs: space-separated "k_block_size:block_topk" pairs.
    # Default sweeps all three configs with k_block_size * block_topk == 8192.
    p.add_argument("--block-configs", type=str, nargs="+",
                   default=["64:128", "128:64", "256:32"],
                   help="e.g. '64:128 128:64 256:32'")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 16384, 65536])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    p.add_argument("--context-lens", type=int, nargs="+", default=[4096, 16384, 65536])
    p.add_argument("--paged-block-size", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--num-sms", type=int, default=132)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = _parse()
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(0)
    dims = IndexerDims(n_head=args.n_head, head_dim=args.head_dim)

    block_configs = [tuple(int(x) for x in cfg.split(":")) for cfg in args.block_configs]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Timing: {args.warmups} warmups + {args.iters} iters (median ± stdev, ms)")

    for k_block_size, block_topk in block_configs:
        if args.mode in ("both", "decode"):
            profile_decode(
                batch_sizes=args.batch_sizes,
                context_lens=args.context_lens,
                k_block_size=k_block_size,
                block_topk=block_topk,
                paged_block_size=args.paged_block_size,
                max_model_len=args.max_model_len,
                num_sms=args.num_sms,
                dims=dims,
                device=device,
                warmups=args.warmups,
                iters=args.iters,
            )

        if args.mode in ("both", "prefill"):
            profile_prefill(
                seq_lens=args.seq_lens,
                k_block_size=k_block_size,
                block_topk=block_topk,
                dims=dims,
                device=device,
                warmups=args.warmups,
                iters=args.iters,
            )


if __name__ == "__main__":
    main()
