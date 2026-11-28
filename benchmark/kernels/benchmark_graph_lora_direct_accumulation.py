"""Benchmark direct accumulation for the torch-native graph LoRA-B path.

Examples:
    python3 benchmark/kernels/benchmark_graph_lora_direct_accumulation.py
    python3 benchmark/kernels/benchmark_graph_lora_direct_accumulation.py \
        --repetitions 1000 --output results.csv
    python3 benchmark/kernels/benchmark_graph_lora_direct_accumulation.py --eager
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Callable, NamedTuple

import torch


class Case(NamedTuple):
    name: str
    tokens: int
    rank: int
    width: int
    adapters: int
    dtype: torch.dtype
    packed: bool


CASES = (
    Case("control-1", 1, 8, 4096, 1, torch.float16, False),
    Case("control-3", 8, 16, 4096, 3, torch.bfloat16, True),
    Case("decode", 32, 16, 4096, 4, torch.float16, True),
    Case("batch", 128, 32, 4096, 4, torch.bfloat16, False),
    Case("qkv", 32, 32, 12288, 4, torch.float16, True),
    Case("wide", 512, 64, 11008, 8, torch.bfloat16, True),
)


def baseline(
    output: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
) -> None:
    for adapter_idx in range(weights.shape[0]):
        masked = torch.where((weight_indices == adapter_idx).unsqueeze(1), inputs, 0)
        output.add_(torch.mm(masked, weights[adapter_idx].t()))


def candidate(
    output: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
) -> None:
    if weights.shape[0] < 4:
        baseline(output, inputs, weights, weight_indices)
        return
    for adapter_idx in range(weights.shape[0]):
        masked = torch.where((weight_indices == adapter_idx).unsqueeze(1), inputs, 0)
        output.addmm_(masked, weights[adapter_idx].t())


def make_tensors(case: Case):
    padding = 64 if case.packed else 0
    generator = torch.Generator(device="cuda").manual_seed(20260901)
    inputs = torch.randn(
        case.tokens,
        case.rank,
        dtype=case.dtype,
        device="cuda",
        generator=generator,
    )
    weights = torch.randn(
        case.adapters,
        case.width,
        case.rank,
        dtype=case.dtype,
        device="cuda",
        generator=generator,
    )
    weight_indices = torch.arange(
        case.tokens, dtype=torch.int32, device="cuda"
    ).remainder(case.adapters + 1)
    weight_indices[weight_indices == case.adapters] = -1
    base = torch.randn(
        case.tokens,
        case.width + 2 * padding,
        dtype=case.dtype,
        device="cuda",
        generator=generator,
    )
    output_slice = slice(padding, padding + case.width)
    return inputs, weights, weight_indices, base, output_slice


def run(
    operation: Callable,
    backing: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    base: torch.Tensor,
    output_slice: slice,
) -> None:
    backing.copy_(base)
    operation(backing[:, output_slice], inputs, weights, weight_indices)


def capture(
    operation: Callable, backing: torch.Tensor, tensors
) -> torch.cuda.CUDAGraph:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        run(operation, backing, *tensors)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(operation, backing, *tensors)
    return graph


def time_operation(
    operation: Callable,
    backing: torch.Tensor,
    tensors,
    eager: bool,
    warmups: int,
    repetitions: int,
) -> list[float]:
    if eager:
        replay = lambda: run(operation, backing, *tensors)
    else:
        replay = capture(operation, backing, tensors).replay

    for _ in range(warmups):
        replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def percentiles(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    return tuple(
        ordered[round(percentile * (len(ordered) - 1))]
        for percentile in (0.10, 0.50, 0.90)
    )


def extra_peak_bytes(operation: Callable, backing: torch.Tensor, tensors) -> int:
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    run(operation, backing, *tensors)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before


def benchmark_case(case: Case, args: argparse.Namespace) -> dict:
    tensors = make_tensors(case)
    base = tensors[-2]
    baseline_backing = base.clone()
    candidate_backing = base.clone()
    with torch.inference_mode():
        run(baseline, baseline_backing, *tensors)
        run(candidate, candidate_backing, *tensors)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            candidate_backing, baseline_backing, rtol=3e-2, atol=3e-2
        )
        baseline_times = percentiles(
            time_operation(
                baseline,
                baseline_backing,
                tensors,
                args.eager,
                args.warmups,
                args.repetitions,
            )
        )
        candidate_times = percentiles(
            time_operation(
                candidate,
                candidate_backing,
                tensors,
                args.eager,
                args.warmups,
                args.repetitions,
            )
        )
        baseline_peak = extra_peak_bytes(baseline, baseline_backing, tensors)
        candidate_peak = extra_peak_bytes(candidate, candidate_backing, tensors)

    return {
        "case": case.name,
        "tokens": case.tokens,
        "rank": case.rank,
        "width": case.width,
        "adapters": case.adapters,
        "dtype": str(case.dtype).removeprefix("torch."),
        "layout": "packed" if case.packed else "whole",
        "execution": "eager" if args.eager else "cudagraph",
        "baseline_p10_us": baseline_times[0],
        "baseline_p50_us": baseline_times[1],
        "baseline_p90_us": baseline_times[2],
        "candidate_p10_us": candidate_times[0],
        "candidate_p50_us": candidate_times[1],
        "candidate_p90_us": candidate_times[2],
        "speedup": baseline_times[1] / candidate_times[1],
        "baseline_extra_peak_bytes": baseline_peak,
        "candidate_extra_peak_bytes": candidate_peak,
    }


def print_results(results: list[dict], repetitions: int) -> None:
    capability = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (SM{capability[0]}{capability[1]})")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"Execution: {results[0]['execution']}; repetitions: {repetitions}\n")
    print(
        "| case | dtype | layout | baseline p10/p50/p90 (us) | "
        "candidate p10/p50/p90 (us) | speedup | extra peak bytes (base -> cand) |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result['case']} | {result['dtype']} | {result['layout']} | "
            f"{result['baseline_p10_us']:.3f}/{result['baseline_p50_us']:.3f}/"
            f"{result['baseline_p90_us']:.3f} | "
            f"{result['candidate_p10_us']:.3f}/{result['candidate_p50_us']:.3f}/"
            f"{result['candidate_p90_us']:.3f} | {result['speedup']:.3f}x | "
            f"{result['baseline_extra_peak_bytes']} -> "
            f"{result['candidate_extra_peak_bytes']} |"
        )
    speedups = [result["speedup"] for result in results if result["adapters"] >= 4]
    geomean = math.exp(statistics.fmean(math.log(value) for value in speedups))
    print(f"\nDirect-path p50 geometric-mean speedup: {geomean:.3f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--warmups", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, help="optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmups < 1 or args.repetitions < 1:
        raise ValueError("warmups and repetitions must be positive")

    results = [benchmark_case(case, args) for case in CASES]
    print_results(results, repetitions=args.repetitions)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=results[0])
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
