"""Compare separate and packed BF16 Beam projections on one CUDA GPU.

Uses local TP shard shapes; this is not a distributed or serving benchmark.
Example: python bench_beam_packed_projection.py --tp 8 --output results.json
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F


def measure(functions, repeats, cold_l2):
    graphs = {}
    # Warm libraries and capture each arm outside the measured interval.
    for name, function in functions.items():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(10):
                function()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = function()
        graphs[name] = (graph, output)
    eviction = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    samples = {name: [] for name in functions}
    names = list(functions)
    for repeat in range(repeats):
        for name in names if repeat % 2 == 0 else names[::-1]:
            if cold_l2:
                eviction.zero_()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64, 256, 1024, 4096]
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(tokens < 1 for tokens in args.tokens):
        parser.error("tokens and repeats must be positive")
    torch.manual_seed(42)
    widths = [48 * 128 // args.tp, 8 * 128 // args.tp, 8 * 128 // args.tp]
    qkv_weight = torch.randn(sum(widths), 5120, device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.randn(widths[0], 5120, device="cuda", dtype=torch.bfloat16)
    packed_weight = torch.cat([qkv_weight, gate_weight])
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tp_geometry": args.tp,
        "scope": "Single-GPU local projection only; no attention or collectives",
        "results": [],
    }
    for tokens in args.tokens:
        x = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)

        def separate(x=x):
            return F.linear(x, qkv_weight), F.linear(x, gate_weight)

        def packed(x=x):
            return F.linear(x, packed_weight)

        baseline = torch.cat(separate(), dim=-1)
        candidate = packed()
        # FP32 matmul with TF32 disabled is an independent numerical reference.
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = F.linear(x.float(), packed_weight.float())
        scale = reference.norm().clamp_min(1e-12)
        baseline_error = ((baseline.float() - reference).norm() / scale).item()
        candidate_error = ((candidate.float() - reference).norm() / scale).item()
        if candidate_error > max(0.004, baseline_error * 1.1):
            raise AssertionError(
                f"Packed projection relative L2 error: {candidate_error}"
            )
        for cold_l2 in (False, True):
            samples = measure(
                {"separate": separate, "packed": packed}, args.repeats, cold_l2
            )
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            row = {
                "tokens": tokens,
                "cold_l2": cold_l2,
                "median_us": medians,
                "speedup": medians["separate"] / medians["packed"],
                "baseline_relative_l2": baseline_error,
                "packed_relative_l2": candidate_error,
                "samples_us": samples,
            }
            report["results"].append(row)
            print(
                json.dumps(
                    {key: value for key, value in row.items() if key != "samples_us"}
                ),
                flush=True,
            )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
