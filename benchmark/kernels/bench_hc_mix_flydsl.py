"""Compare HC implementations on gfx950 with graph replay and rotating weights.

Example: python benchmark/kernels/bench_hc_mix_flydsl.py --weights 64 --json /tmp/hc.json
Packing, compilation and allocation are excluded from GPU replay timing.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from aiter.ops.flydsl.hc_mix import hc_mix, pack_hc_weights
from sglang.srt.layers.hc_mix_triton import fused_hc_mix


def eager(x, down, up):
    t = F.silu(F.linear(x, down) / 4)
    return (F.linear(t, up).sigmoid() * x).view(x.shape[0], 4, -1).mean(1)


def measure(fn, calls, repeats=9):
    for i in range(calls):
        fn(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(i)
    for _ in range(3):
        graph.replay()
    times = []
    for _ in range(repeats):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return {"median_us": statistics.median(times), "min_us": min(times), "max_us": max(times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=int, default=64)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.weights < 1:
        parser.error("--weights must be positive")
    torch.manual_seed(123)
    compiled = torch.compile(eager)
    results = {"device": str(torch.cuda.get_device_properties()), "torch": torch.__version__, "weight_sets": args.weights, "rows": {}}
    for m in args.rows:
        data = []
        for _ in range(args.weights):
            x = torch.randn(m, 10240, device="cuda", dtype=torch.bfloat16)
            d = torch.randn(320, 10240, device="cuda", dtype=x.dtype) * 0.02
            u = torch.randn(10240, 320, device="cuda", dtype=x.dtype) * 0.02
            data.append((x, d, u, *pack_hc_weights(d, u)))
        def fly(i):
            x, _, _, d, u = data[i % len(data)]
            return hc_mix(x, d, u)
        def triton(i):
            x, d, u, _, _ = data[i % len(data)]
            return fused_hc_mix(x, d, u, 4, 2560)
        def pytorch(i):
            return compiled(*data[i % len(data)][:3])
        row = {}
        for name, fn in [("flydsl", fly), ("persistent_triton", triton), ("torch_compile", pytorch)]:
            row[name] = measure(fn, max(128, args.weights))
            print(m, name, row[name], flush=True)
        results["rows"][m] = row
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
