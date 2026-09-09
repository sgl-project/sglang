"""Reproduce the SM120 BA microbenchmark; this does not measure serving TPS."""

import argparse
import json
import random
import statistics

import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.sm120_ba_gemm import sm120_ba_linear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() != (12, 0):
        parser.error("this benchmark requires SM120")
    torch.manual_seed(20260909)
    stream = torch.cuda.Stream(device=args.device)
    results = []
    for m in (1, 2, 4, 8):
        x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(48, 5120, device="cuda", dtype=torch.bfloat16) * 0.01
        graphs = {}
        for name, fn in (("torch", F.linear), ("candidate", sm120_ba_linear)):
            for _ in range(5):
                fn(x, w)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = [fn(x, w) for _ in range(100)]
            graphs[name] = (graph, outputs)
        samples = {name: [] for name in graphs}
        rng = random.Random(20260909)
        for _ in range(args.repeats):
            names = list(graphs)
            rng.shuffle(names)
            for name in names:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name][0].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 10)
        results.append(
            {
                "m": m,
                "n": 48,
                "k": 5120,
                "median_us": {
                    name: statistics.median(v) for name, v in samples.items()
                },
                "samples_us": samples,
            }
        )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "method": "100 calls per graph replay, randomized paired order; M1 falls back",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
