"""Compare the original SGLang sparse decode kernel with the FlyDSL adapter.

Use an unmodified minimax_sparse kernel directory from the pinned SGLang
revision. Includes page-table construction and attention; excludes indexer,
KV writes, model layers, communication, and serving overhead.
"""

import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path

import torch
from validate_adapter import caches, check, load_adapter, reference


def paired_graph_latency(functions, rounds=7, replays=100, calls_per_graph=16):
    graphs = {}
    samples = {name: [] for name in functions}
    for name, fn in functions.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls_per_graph):
                fn()
        graphs[name] = graph
    for round_idx in range(rounds):
        # Alternate ordering to reduce systematic warmup/clock drift effects.
        names = list(graphs)
        if round_idx % 2:
            names.reverse()
        for name in names:
            graph = graphs[name]
            for _ in range(5):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(
                start.elapsed_time(end) * 1000 / (replays * calls_per_graph)
            )
    return {
        name: {"median_us": statistics.median(values), "rounds_us": values}
        for name, values in samples.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--baseline-kernels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Import the original kernel package without importing the serving stack.
    sys.path.insert(0, str(args.baseline_kernels.resolve().parent))
    baseline = importlib.import_module(
        f"{args.baseline_kernels.name}.decode.topk_sparse"
    ).flash_decode_with_gqa_share_sparse
    adapter = load_adapter(args.adapter)
    torch.manual_seed(20260927)
    report = {
        "scope": "isolated SGLang sparse decode including FlyDSL page-table construction",
        "sglang_base": "fc9e1c8d296216ff1e216dfbe7286ef392448d28",
        "shape": {
            "q_heads_per_rank": 16,
            "kv_heads_per_rank": 1,
            "head_dim": 128,
            "page_size": 16,
            "sparse_block_size": 128,
            "topk_blocks": 16,
        },
        "checks": [],
        "benchmarks": [],
    }
    try:
        for batch in (1, 2, 10, 15, 20):
            for distribution in ("uniform", "mixed"):
                context = (
                    [32768] * batch
                    if distribution == "uniform"
                    else [32768] + [1024] * (batch - 1)
                )
                k, v, sk, sv, _, slots = caches(batch, max(context), 16)
                nk, nv = k.flatten(0, 1), v.flatten(0, 1)
                lengths = torch.tensor(context, device="cuda", dtype=torch.int64)
                requests = torch.arange(batch, device="cuda", dtype=torch.int64)
                q = torch.randn(batch, 16, 128, device="cuda", dtype=torch.bfloat16)
                topk = torch.full((1, batch, 16), -1, device="cuda", dtype=torch.int32)
                for row, length in enumerate(context):
                    count = min(16, (length + 127) // 128)
                    selected = (
                        torch.randperm((length + 127) // 128, device="cuda")[:count]
                        .sort()
                        .values
                    )
                    topk[0, row, :count] = selected
                scale = torch.ones(1, device="cuda", dtype=torch.float32)

                def original():
                    return baseline(
                        q, None, nk, nv, slots, lengths, requests, 128, topk
                    )

                def candidate():
                    return adapter.sparse_decode(
                        q,
                        sk,
                        sv,
                        topk,
                        slots,
                        requests,
                        lengths,
                        128,
                        None,
                        scale,
                        scale,
                    )

                expected = reference(
                    q, k, v, slots, requests, lengths, scale, scale, topk
                )
                label = f"batch{batch}-{distribution}"
                check(
                    f"sglang-baseline-{label}",
                    original(),
                    expected,
                    report["checks"],
                    fail=True,
                )
                check(
                    f"flydsl-candidate-{label}",
                    candidate(),
                    expected,
                    report["checks"],
                    fail=True,
                )
                timing = paired_graph_latency(
                    {"sglang_triton": original, "flydsl_adapter": candidate}
                )
                old = timing["sglang_triton"]["median_us"]
                new = timing["flydsl_adapter"]["median_us"]
                result = {
                    "batch": batch,
                    "distribution": distribution,
                    "timings": timing,
                    "latency_reduction_pct": 100 * (1 - new / old),
                    "kernel_speedup": old / new,
                }
                report["benchmarks"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
