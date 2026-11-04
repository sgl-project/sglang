"""Build a SGLANG_OPT_BF16_GEMM_TABLE for this GPU and cuBLASLt version.

For every (N, K) weight shape and every M, the tuner times all split-K-free
cuBLASLt algorithms (and the heuristic's own top choices) on the packed
problem, takes the fastest few, keeps those that accumulate K in the same order
as torch.mm (bitwise equal on ``order_check_operands``, where another order
differs with overwhelming probability), and times them against torch.mm with a
cold L2 under CUDA graph replay. An (N, K, M) entry is written when the best
exact algorithm beats torch.mm by at least --min-speedup. The table is marked
``"order_checked": true``; the server ignores tables without the marker.

Shapes are the unquantized linears of the served model after tensor-parallel
sharding (N = output features per rank, K = input features per rank). M is the
token count per forward, e.g. the decode CUDA graph batch sizes times the
tokens per request (1, or the draft tokens + 1 under speculative decoding).

Example (Qwen3.5-9B, TP1, decode batch sizes 1-64):
    python benchmark/kernels/pinned_bf16_gemm/tune_pinned_bf16_gemm.py \
        --shapes 12352:4096 4096:4096 10240:4096 24576:4096 4096:12288 248320:4096 \
        --m 1 2 4 8 12 16 24 32 40 48 56 64 --output bf16_gemm_table.json
    SGLANG_OPT_BF16_GEMM_TABLE=bf16_gemm_table.json sglang serve --model-path Qwen/Qwen3.5-9B ...
"""

import argparse
import json
import os

import torch

from sglang.kernels.jit.benchmark.marker import do_bench
from sglang.kernels.ops.gemm.pinned_bf16_gemm import (
    ORDER_CHECKED_KEY,
    cublaslt_version,
    enumerate_algos,
    gives_torch_bits,
    order_check_operands,
    run_algo,
    table_device_name,
)


def _bench_us(fn, *args) -> float:
    return do_bench(fn, input_args=args, disable_log_bandwidth=True).times[0] * 1e6


def _quick_us(fn, reps: int) -> float:
    fn()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / reps


def tune_one(n: int, k: int, m: int, args) -> dict:
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    x0 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    checks = [order_check_operands(m, n, k, w.device, seed) for seed in (1, 2)]
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    rows = enumerate_algos(m, n, k, args.max_workspace_mb << 20)
    quick = []
    for row in rows:
        algo, ws = row[:8].clone(), int(row[8])
        try:
            us = _quick_us(lambda: run_algo(x0, w, algo, ws, out=out), args.quick_reps)
        except RuntimeError:
            continue
        quick.append((us, algo, ws, int(row[9])))
    quick.sort(key=lambda r: r[0])

    best = None
    num_exact = 0
    for _, algo, ws, splits in quick[: args.top]:

        def runner(x_, w_, out_, algo=algo, ws=ws):
            return run_algo(x_, w_, algo, ws, out=out_)

        if not all(gives_torch_bits(runner, x_, w_) for x_, w_ in checks):
            continue
        num_exact += 1
        us = _bench_us(lambda x_, w_: run_algo(x_, w_, algo, ws), x0, w)
        if best is None or us < best["us"]:
            best = {"us": us, "algo": algo.tolist(), "ws": ws, "splits": splits}

    torch_us = _bench_us(lambda x_, w_: torch.mm(x_, w_.t()), x0, w)
    return {
        "n": n,
        "k": k,
        "m": m,
        "torch_us": torch_us,
        "num_candidates": len(rows),
        "num_exact_of_top": num_exact,
        "best": best,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shapes", nargs="+", required=True, help="N:K pairs")
    parser.add_argument("--m", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument("--quick-reps", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=0.03)
    parser.add_argument("--max-workspace-mb", type=int, default=32)
    args = parser.parse_args()

    results, entries = [], []
    for shape in args.shapes:
        n, k = (int(v) for v in shape.split(":"))
        for m in args.m:
            r = tune_one(n, k, m, args)
            results.append(r)
            best = r["best"]
            gain = 1 - best["us"] / r["torch_us"] if best else 0.0
            print(
                f"N={n:6d} K={k:6d} M={m:5d} torch {r['torch_us']:8.2f} us  "
                f"best exact {best['us'] if best else float('nan'):8.2f} us  "
                f"({r['num_exact_of_top']}/{min(args.top, r['num_candidates'])} exact "
                f"of {r['num_candidates']})  gain {gain * 100:5.1f}%",
                flush=True,
            )
            if best and gain >= args.min_speedup:
                entries.append(
                    {
                        "n": n,
                        "k": k,
                        "m": m,
                        "cublaslt_algo": best["algo"],
                        "workspace_bytes": best["ws"],
                        "torch_us": round(r["torch_us"], 3),
                        "pinned_us": round(best["us"], 3),
                    }
                )

    table = {
        "device_name": table_device_name(),
        "cublaslt_version": cublaslt_version(),
        ORDER_CHECKED_KEY: True,
        "entries": entries,
    }
    with open(args.output, "w") as f:
        json.dump(table, f, indent=1)
    with open(os.path.splitext(args.output)[0] + ".rows.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
