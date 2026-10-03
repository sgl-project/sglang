"""Benchmark gfx950 DSA sparse attention at GLM-5.3 production shapes.

Example:
  python benchmark/kernels/attention/bench_tilelang_dsa_sparse.py \
    --candidate-backend triton
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

DIM = 512
TOPK = 2112
DEFAULT_LIVE_TOPK = 2051
DEFAULT_KV_LEN = 13_299_712
DEFAULT_M = (128, 512, 2048, 4096, 8192, 12288, 16384)


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
    heads: int,
    block_i: int,
    inner_iter: int,
    threads: int,
    num_stages: int,
):
    assert TOPK % (block_i * inner_iter) == 0
    groups = TOPK // (block_i * inner_iter)
    partial = sparse_mla_fwd_decode_partial(
        heads,
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
        heads,
        DIM,
        groups * block_i,
        head_per_block=4,
        block_I=block_i,
        threads=threads,
    )
    return partial, combine, groups


def _run_cell(
    kv: torch.Tensor,
    heads: int,
    m: int,
    live_topk: int,
    kv_len: int,
    block_i: int,
    inner_iter: int,
    threads: int,
    num_stages: int,
    warmup: int,
    repeats: int,
    index_pattern: str,
    backend: str,
):
    q = torch.randn((m, heads, DIM), device="cuda", dtype=torch.bfloat16) * 0.01
    if index_pattern == "shared":
        indices = (
            torch.randint(0, kv_len, (1, 1, TOPK), device="cuda", dtype=torch.int32)
            .expand(m, -1, -1)
            .contiguous()
        )
    else:
        indices = torch.randint(
            0, kv_len, (m, 1, TOPK), device="cuda", dtype=torch.int32
        )
    indices[..., live_topk:] = -1
    if backend == "triton":
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
            triton_sparse_mla_fwd,
        )

        def run_complete():
            return triton_sparse_mla_fwd(
                q,
                q[..., DIM:],
                kv,
                indices,
                1.0 / math.sqrt(256),
                d_v=DIM,
            )

        actual = run_complete()
        torch.cuda.synchronize()
        complete_ms = _time_ms(run_complete, warmup, repeats)
        return actual, {
            "backend": backend,
            "heads": heads,
            "m": m,
            "live_topk": live_topk,
            "complete_ms": complete_ms,
        }

    if backend == "triton-decode":
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
            triton_sparse_mla_decode_splitk,
        )

        workspace = []

        def run_complete():
            return triton_sparse_mla_decode_splitk(
                q,
                q[..., DIM:],
                kv,
                indices,
                1.0 / math.sqrt(256),
                d_v=DIM,
                workspace=workspace,
            )

        actual = run_complete()
        torch.cuda.synchronize()
        complete_ms = _time_ms(run_complete, warmup, repeats)
        return actual, {
            "backend": backend,
            "heads": heads,
            "m": m,
            "live_topk": live_topk,
            "complete_ms": complete_ms,
        }

    q4 = q.unsqueeze(0)
    kv4 = kv.unsqueeze(0)
    indices4 = indices.unsqueeze(0)
    partial, combine, groups = _build_kernels(
        heads, block_i, inner_iter, threads, num_stages
    )

    def run_partial():
        return partial(q4, kv4, indices4)

    partial_o, partial_lse = run_partial()

    def run_combine():
        return combine(partial_o, partial_lse)

    def run_complete():
        return combine(*run_partial())

    actual = run_combine()
    torch.cuda.synchronize()
    partial_ms = _time_ms(run_partial, warmup, repeats)
    combine_ms = _time_ms(run_combine, warmup, repeats)
    complete_ms = _time_ms(run_complete, warmup, repeats)
    return actual, {
        "backend": backend,
        "heads": heads,
        "m": m,
        "live_topk": live_topk,
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
    parser.add_argument(
        "--candidate-backend",
        choices=("tilelang", "triton", "triton-decode"),
        default="tilelang",
    )
    parser.add_argument("--heads", type=int, choices=(8, 16), default=16)
    parser.add_argument("--m", type=int, nargs="+", default=DEFAULT_M)
    parser.add_argument("--live-topk", type=int, default=DEFAULT_LIVE_TOPK)
    parser.add_argument("--kv-len", type=int, default=DEFAULT_KV_LEN)
    parser.add_argument("--block-i", type=int, default=64)
    parser.add_argument("--inner-iter", type=int, default=33)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--index-pattern", choices=("shared", "random"), default="shared"
    )
    args = parser.parse_args()
    if not 0 <= args.live_topk <= TOPK:
        parser.error(f"--live-topk must be in [0, {TOPK}]")

    torch.manual_seed(20260928)
    kv = torch.zeros((args.kv_len, 1, DIM), device="cuda", dtype=torch.bfloat16)
    for m in args.m:
        torch.manual_seed(20260928 + m)
        baseline, baseline_row = _run_cell(
            kv,
            args.heads,
            m,
            args.live_topk,
            args.kv_len,
            64,
            33,
            256,
            1,
            args.warmup,
            args.repeats,
            args.index_pattern,
            "tilelang",
        )
        torch.manual_seed(20260928 + m)
        candidate, candidate_row = _run_cell(
            kv,
            args.heads,
            m,
            args.live_topk,
            args.kv_len,
            args.block_i,
            args.inner_iter,
            args.threads,
            args.num_stages,
            args.warmup,
            args.repeats,
            args.index_pattern,
            args.candidate_backend,
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
