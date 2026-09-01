#!/usr/bin/env python3
"""Benchmark the PR #36865 CuTe kernel on additional Qwen model shapes.

The benchmark uses the ModelOpt NVFP4 ABI and compares the candidate against
FlashInfer in multi-weight CUDA graphs. Rotating across distinct weights keeps
the measurement representative of a decoder stack instead of repeatedly
serving one weight tensor from L2.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from flashinfer import fp4_quantize, mm_fp4

try:
    from sglang.kernels.kda_kernels.qwen38_nvfp4_gemm_sm120 import (
        kda_nvfp4_gemm,
    )
except ModuleNotFoundError:
    # Allows the benchmark and the PR kernel package to be copied together
    # into an isolated GPU-broker workspace without replacing its SGLang tree.
    from qwen38_nvfp4_gemm_sm120 import kda_nvfp4_gemm


MODEL_SHAPES = {
    "qwen3.5-4b": {
        "gate_up": (2560, 18432),
        "down": (9216, 2560),
    },
    "qwen3.5-9b": {
        "gate_up": (4096, 24576),
        "down": (12288, 4096),
    },
    "qwen3.6-27b": {
        "gate_up": (5120, 34816),
        "down": (17408, 5120),
    },
}


def make_inputs(m: int, k: int, n: int):
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    input_global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    x, x_sf = fp4_quantize(x_bf16, input_global_scale)
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda").T
    weight_sf = torch.ones((n, k // 16), dtype=torch.float8_e4m3fn, device="cuda").T
    alpha = torch.tensor(0.03125, dtype=torch.float32, device="cuda")
    return x, weight, x_sf, weight_sf, alpha


def run_reference(args):
    return mm_fp4(*args, torch.bfloat16, backend="auto")


def run_candidate(args, role: str, m: int, tag: str):
    del role, m, tag
    return kda_nvfp4_gemm(*args, torch.bfloat16, args[1].shape[1])


def capture_graph(fn, arg_sets):
    for args in arg_sets:
        fn(args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for args in arg_sets:
            fn(args)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def time_graph(graph, calls_per_replay: int, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (repeats * calls_per_replay)


def benchmark_row(model, role, m, k, n, num_weights, warmup, repeats, trials):
    torch.manual_seed(1234)
    arg_sets = [make_inputs(m, k, n) for _ in range(num_weights)]

    expected = run_reference(arg_sets[0])
    actual = run_candidate(arg_sets[0], role, m, f"multimodel_{model}_{role}_m{m}")
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
    del expected, actual

    reference_graph = capture_graph(run_reference, arg_sets)
    candidate_graph = capture_graph(
        lambda args: run_candidate(args, role, m, f"multimodel_{model}_{role}_m{m}"),
        arg_sets,
    )

    reference_us = []
    candidate_us = []
    for trial in range(trials):
        order = (
            (("reference", reference_graph), ("candidate", candidate_graph))
            if trial % 2 == 0
            else (("candidate", candidate_graph), ("reference", reference_graph))
        )
        timings = {}
        for name, graph in order:
            timings[name] = time_graph(graph, num_weights, warmup, repeats)
        reference_us.append(timings["reference"])
        candidate_us.append(timings["candidate"])

    ref_median = statistics.median(reference_us)
    cand_median = statistics.median(candidate_us)
    result = {
        "model": model,
        "role": role,
        "m": m,
        "k": k,
        "n": n,
        "weights": num_weights,
        "reference_us": reference_us,
        "candidate_us": candidate_us,
        "reference_median_us": ref_median,
        "candidate_median_us": cand_median,
        "speedup": ref_median / cand_median,
        "correct": True,
    }
    print(json.dumps(result), flush=True)
    del reference_graph, candidate_graph, arg_sets
    torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_SHAPES, default=list(MODEL_SHAPES)
    )
    parser.add_argument("--rows", nargs="+", type=int, choices=(1, 9), default=[1, 9])
    parser.add_argument("--weights", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark requires an SM120 GPU")
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "models": args.models,
                "rows": args.rows,
                "weights": args.weights,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "trials": args.trials,
            }
        ),
        flush=True,
    )
    results = []
    for model in args.models:
        for role, (k, n) in MODEL_SHAPES[model].items():
            for m in args.rows:
                results.append(
                    benchmark_row(
                        model,
                        role,
                        m,
                        k,
                        n,
                        args.weights,
                        args.warmup,
                        args.repeats,
                        args.trials,
                    )
                )
    print(
        json.dumps(
            {
                "summary": {
                    "geomean_speedup": statistics.geometric_mean(
                        result["speedup"] for result in results
                    ),
                    "min_speedup": min(result["speedup"] for result in results),
                    "max_speedup": max(result["speedup"] for result in results),
                    "rows": len(results),
                }
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
