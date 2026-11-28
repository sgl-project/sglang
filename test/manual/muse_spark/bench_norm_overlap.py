"""Paired CUDA scheduling microbenchmark; NOT a serving-performance benchmark.

Compare production host methods from two source snapshots using the same
weights/inputs. The fixture substitutes torch operators for the pooler, router
and experts, so timings do not represent a loaded Muse Spark model or TP/EP.

    git show BASE:python/sglang/srt/models/muse_spark.py > /tmp/muse_base.py
    python test/manual/muse_spark/bench_norm_overlap.py \
        --baseline-source /tmp/muse_base.py --tokens 1 16 128 --graph

Run real serving benchmarks separately with identical checkpoint, server
arguments, prompts and hardware on both revisions before claiming a speedup.
"""

import argparse
import json
import statistics

import torch
from test_norm_overlap import Fixture, load_methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--expert-hidden", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required; no CPU timing substitute")
    if min(args.tokens + [args.pairs, args.iterations]) <= 0:
        parser.error("tokens, pairs and iterations must be positive")
    torch.manual_seed(args.seed)
    f = Fixture(
        "cuda",
        hidden=args.hidden,
        expert_hidden=args.expert_hidden,
        num_experts=args.experts,
        top_k=args.top_k,
    )
    f.weight = f.weight.bfloat16()
    f.model.pre_expert_proj.weight = f.weight
    f.model.router_in_fp32 = True
    baseline = load_methods(dict(f.scope), args.baseline_source)["_forward_experts"]
    candidate = f.scope["_forward_experts"]
    report = dict(
        config=vars(args),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        scope="weight-free scheduling surrogate, warm weights, no TP/EP",
        rows=[],
    )
    for tokens in args.tokens:
        x = torch.randn(tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
        methods = [baseline, candidate]
        runs = [lambda method=m: method(f.model, x) for m in methods]
        for _ in range(10):
            for run in runs:
                torch.testing.assert_close(run(), f.reference(x))
        torch.cuda.synchronize()
        graphs, outputs = [], []
        if args.graph:
            for run in runs:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = run()
                graphs.append(graph)
                outputs.append(output)
            runs = [g.replay for g in graphs]
            # Correctness is checked after changing captured input data too.
            for _ in range(3):
                x.normal_()
                for run, output in zip(runs, outputs):
                    run()
                    torch.testing.assert_close(output, f.reference(x))
        raw = []
        for pair in range(args.pairs):
            measurements = [None, None]
            for idx in [0, 1] if pair % 2 == 0 else [1, 0]:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(args.iterations):
                    runs[idx]()
                end.record()
                end.synchronize()
                measurements[idx] = start.elapsed_time(end) * 1000 / args.iterations
                f.events.clear()
            raw.append(measurements)
        report["rows"].append(
            dict(
                tokens=tokens,
                baseline_candidate_us=raw,
                median_baseline_us=statistics.median(p[0] for p in raw),
                median_candidate_us=statistics.median(p[1] for p in raw),
            )
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
