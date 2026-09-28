"""Benchmark gfx950 TileLang DSA sparse attention at GLM-5.3 production shapes.

Example:
  python benchmark/kernels/attention/bench_tilelang_dsa_sparse.py \
    --block-i 64 --inner-iter 33 --threads 256 --num-stages 1
"""

import argparse
import json
import math
import statistics

import torch

from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
    sparse_mla_fwd_decode_combine,
    sparse_mla_fwd_decode_partial,
)

HEADS = 16
DIM = 512
TOPK = 2112
LIVE_TOPK = 2051
KV_LEN = 13_299_712
PRODUCTION_M = (8192, 16384)


def _time_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _build_kernels(
    block_i: int,
    inner_iter: int,
    threads: int,
    num_stages: int,
):
    assert TOPK % (block_i * inner_iter) == 0
    groups = TOPK // (block_i * inner_iter)
    partial = sparse_mla_fwd_decode_partial(
        HEADS,
        DIM,
        0,
        TOPK,
        sm_scale=1.0 / math.sqrt(256),
        block_I=block_i,
        inner_iter=inner_iter,
        threads=threads,
        num_stages=num_stages,
    )
    combine = sparse_mla_fwd_decode_combine(
        HEADS,
        DIM,
        groups * block_i,
        head_per_block=4,
        block_I=block_i,
        threads=threads,
    )
    return partial, combine, groups


def _run_cell(
    kv: torch.Tensor,
    m: int,
    block_i: int,
    inner_iter: int,
    threads: int,
    num_stages: int,
    warmup: int,
    repeats: int,
    index_pattern: str,
):
    q = torch.randn((m, HEADS, DIM), device="cuda", dtype=torch.bfloat16) * 0.01
    if index_pattern == "shared":
        indices = (
            torch.randint(0, KV_LEN, (1, 1, TOPK), device="cuda", dtype=torch.int32)
            .expand(m, -1, -1)
            .contiguous()
        )
    else:
        indices = torch.randint(
            0, KV_LEN, (m, 1, TOPK), device="cuda", dtype=torch.int32
        )
    indices[..., LIVE_TOPK:] = -1
    q4 = q.unsqueeze(0)
    kv4 = kv.unsqueeze(0)
    indices4 = indices.unsqueeze(0)
    partial, combine, groups = _build_kernels(block_i, inner_iter, threads, num_stages)

    def run_partial():
        return partial(q4, kv4, indices4)

    partial_o, partial_lse = run_partial()

    def run_combine():
        return combine(partial_o, partial_lse)

    def run_complete():
        po, pl = run_partial()
        return combine(po, pl)

    actual = run_combine()
    torch.cuda.synchronize()
    partial_ms = _time_ms(run_partial, warmup, repeats)
    combine_ms = _time_ms(run_combine, warmup, repeats)
    complete_ms = _time_ms(run_complete, warmup, repeats)
    return actual, {
        "m": m,
        "block_i": block_i,
        "inner_iter": inner_iter,
        "groups": groups,
        "threads": threads,
        "num_stages": num_stages,
        "partial_ms": partial_ms,
        "combine_ms": combine_ms,
        "complete_ms": complete_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-i", type=int, required=True)
    parser.add_argument("--inner-iter", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--num-stages", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--index-pattern", choices=("shared", "random"), default="shared"
    )
    args = parser.parse_args()

    torch.manual_seed(20260928)
    kv = torch.zeros((KV_LEN, 1, DIM), device="cuda", dtype=torch.bfloat16)
    for m in PRODUCTION_M:
        torch.manual_seed(20260928 + m)
        baseline, baseline_row = _run_cell(
            kv,
            m,
            64,
            33,
            256,
            1,
            args.warmup,
            args.repeats,
            args.index_pattern,
        )
        torch.manual_seed(20260928 + m)
        candidate, candidate_row = _run_cell(
            kv,
            m,
            args.block_i,
            args.inner_iter,
            args.threads,
            args.num_stages,
            args.warmup,
            args.repeats,
            args.index_pattern,
        )
        delta = candidate.float() - baseline.float()
        candidate_row.update(
            {
                "baseline_complete_ms": baseline_row["complete_ms"],
                "complete_delta_pct": (
                    candidate_row["complete_ms"] / baseline_row["complete_ms"] - 1
                )
                * 100,
                "finite": bool(torch.isfinite(candidate).all()),
                "index_pattern": args.index_pattern,
                "max_abs": delta.abs().max().item(),
                "mean_abs": delta.abs().mean().item(),
            }
        )
        print(json.dumps(candidate_row), flush=True)
        del baseline, candidate, delta


if __name__ == "__main__":
    main()
