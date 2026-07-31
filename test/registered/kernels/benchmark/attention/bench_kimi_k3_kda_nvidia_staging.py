"""Benchmark the complete Kimi K3 NVIDIA KDA prefill boundary.

This benchmark charges all boundary work around the persistent vendor kernel:

    input normalization/repack + state gather/transpose + vendor kernel
    + output unpack + state scatter/transpose

It compares that forced path with Triton and with the production crossover
selection. Both uniform and ragged request batches are supported. Example:

    python bench_kimi_k3_kda_nvidia_staging.py \
        --cells 2x128,4x1024,8x2048,4096+1 \
        --first triton --output /tmp/kda-prefill.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from sglang.srt.layers.attention.linear.kernels import kda_nvidia
from sglang.srt.layers.attention.linear.kernels.kda_nvidia import NvidiaKDAKernel
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

_HEADS = 24
_D = 128
_LOWER_BOUND = -5.0


@dataclass
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    state: torch.Tensor
    slots: torch.Tensor
    cu_seqlens: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    lengths: list[int]


def make_inputs(lengths: list[int], seed: int) -> Inputs:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(lengths)
    total = sum(lengths)
    slots_count = batch + 4

    def randn(*shape, dtype=torch.bfloat16, scale=0.1):
        return (
            torch.randn(
                *shape,
                dtype=dtype,
                device="cuda",
                generator=generator,
            )
            * scale
        )

    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    return Inputs(
        q=randn(1, total, _HEADS, _D),
        k=randn(1, total, _HEADS, _D),
        v=randn(1, total, _HEADS, _D),
        g=randn(1, total, _HEADS, _D),
        beta=torch.sigmoid(randn(1, total, _HEADS, dtype=torch.float32)),
        state=randn(
            slots_count,
            _HEADS,
            _D,
            _D,
            dtype=torch.float32,
            scale=0.01,
        ),
        slots=torch.randperm(slots_count, device="cuda", generator=generator)[:batch]
        .to(torch.int32)
        .contiguous(),
        cu_seqlens=cu_seqlens,
        a_log=randn(1, 1, _HEADS, 1, dtype=torch.float32, scale=0.5) - 1.5,
        dt_bias=randn(_HEADS * _D, dtype=torch.float32),
        lengths=lengths,
    )


def clone_inputs(source: Inputs) -> Inputs:
    return Inputs(
        q=source.q.clone(),
        k=source.k.clone(),
        v=source.v.clone(),
        g=source.g.clone(),
        beta=source.beta.clone(),
        state=source.state.clone(),
        slots=source.slots,
        cu_seqlens=source.cu_seqlens,
        a_log=source.a_log,
        dt_bias=source.dt_bias,
        lengths=source.lengths,
    )


def common_kwargs(inputs: Inputs) -> dict:
    return {
        "ssm_states": inputs.state,
        "cache_indices": inputs.slots,
        "query_start_loc": inputs.cu_seqlens,
        "A_log": inputs.a_log,
        "dt_bias": inputs.dt_bias,
        "lower_bound": _LOWER_BOUND,
        "extend_seq_lens_cpu": inputs.lengths,
        "is_spec_decode": False,
        "return_intermediate_states": False,
    }


def triton(kernel: TritonKDAKernel, inputs: Inputs) -> torch.Tensor:
    return kernel.extend(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        **common_kwargs(inputs),
    )


def nvidia(kernel: NvidiaKDAKernel, inputs: Inputs) -> torch.Tensor:
    return kernel.extend(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        **common_kwargs(inputs),
    )


@contextmanager
def forced_nvidia_gate(enabled: bool):
    original = kda_nvidia._nvidia_kda_wins_staging_gate
    if enabled:
        kda_nvidia._nvidia_kda_wins_staging_gate = lambda _lengths: True
    try:
        yield
    finally:
        kda_nvidia._nvidia_kda_wins_staging_gate = original


def correctness(source: Inputs) -> dict[str, float]:
    triton_inputs = clone_inputs(source)
    nvidia_inputs = clone_inputs(source)
    expected = triton(TritonKDAKernel(), triton_inputs)
    with forced_nvidia_gate(True):
        actual = nvidia(NvidiaKDAKernel(), nvidia_inputs)
    torch.cuda.synchronize()

    expected_state = triton_inputs.state[triton_inputs.slots.long()]
    actual_state = nvidia_inputs.state[nvidia_inputs.slots.long()]
    output_cos = torch.nn.functional.cosine_similarity(
        expected.float().flatten(),
        actual.float().flatten(),
        dim=0,
    )
    state_cos = torch.nn.functional.cosine_similarity(
        expected_state.flatten(),
        actual_state.flatten(),
        dim=0,
    )
    return {
        "output_cosine": float(output_cos),
        "output_max_abs": float((expected.float() - actual.float()).abs().max()),
        "state_cosine": float(state_cos),
        "state_max_abs": float((expected_state - actual_state).abs().max()),
    }


def time_sample(fn, iterations: int) -> dict[str, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    wall_end = time.perf_counter_ns()
    return {
        "gpu_us": start.elapsed_time(end) * 1000.0 / iterations,
        "wall_us": (wall_end - wall_start) / 1000.0 / iterations,
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def benchmark_cell(
    lengths: list[int],
    first: str,
    samples: int,
    iterations: int,
    seed: int,
) -> dict:
    source = make_inputs(lengths, seed)
    errors = correctness(source)
    inputs = {
        name: clone_inputs(source) for name in ("triton", "forced_nvidia", "selected")
    }
    triton_kernel = TritonKDAKernel()
    forced_kernel = NvidiaKDAKernel()
    selected_kernel = NvidiaKDAKernel()

    def run_triton():
        return triton(triton_kernel, inputs["triton"])

    def run_forced_nvidia():
        with forced_nvidia_gate(True):
            return nvidia(forced_kernel, inputs["forced_nvidia"])

    def run_selected():
        return nvidia(selected_kernel, inputs["selected"])

    functions = {
        "triton": run_triton,
        "forced_nvidia": run_forced_nvidia,
        "selected": run_selected,
    }
    for _ in range(3):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()

    names = list(functions)
    first_index = names.index(first)
    initial_order = names[first_index:] + names[:first_index]
    timings = {name: [] for name in names}
    for sample in range(samples):
        order = initial_order if sample % 2 == 0 else list(reversed(initial_order))
        for provider in order:
            timings[provider].append(time_sample(functions[provider], iterations))

    def summary(provider: str) -> dict[str, float]:
        gpu = [item["gpu_us"] for item in timings[provider]]
        wall = [item["wall_us"] for item in timings[provider]]
        return {
            "gpu_p50_us": statistics.median(gpu),
            "gpu_p95_us": percentile(gpu, 0.95),
            "wall_p50_us": statistics.median(wall),
            "wall_p95_us": percentile(wall, 0.95),
        }

    medians = {name: summary(name) for name in names}
    baseline = medians["triton"]["gpu_p50_us"]
    for provider in ("forced_nvidia", "selected"):
        candidate = medians[provider]["gpu_p50_us"]
        medians[provider]["saved_vs_triton_us"] = baseline - candidate
        medians[provider]["saved_vs_triton_percent"] = (
            100.0 * (baseline - candidate) / baseline
        )

    bucket = next(value for value in kda_nvidia._BUCKETS if value >= max(lengths))
    return {
        "lengths": lengths,
        "batch": len(lengths),
        "tokens": sum(lengths),
        "bucket": bucket,
        "padding_tokens": len(lengths) * bucket - sum(lengths),
        "padding_fraction": 1.0 - sum(lengths) / (len(lengths) * bucket),
        "production_selects_nvidia": (
            kda_nvidia._nvidia_kda_wins_staging_gate(lengths)
        ),
        "samples": samples,
        "iterations": iterations,
        "correctness": errors,
        "raw": timings,
        "median": medians,
    }


def parse_cell(value: str) -> list[int]:
    if "x" in value:
        batch, length = value.split("x", maxsplit=1)
        return [int(length)] * int(batch)
    return [int(length) for length in value.split("+")]


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells",
        default=(
            "2x128,2x512,2x1024,2x2048,"
            "4x128,4x512,4x1024,4x2048,"
            "8x128,8x512,8x1024,8x2048,"
            "4096+1,4096+1024,2048+128+128+128"
        ),
    )
    parser.add_argument(
        "--first",
        choices=("triton", "forced_nvidia", "selected"),
        required=True,
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=32541)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("NVIDIA KDA prefill requires datacenter Blackwell")

    cells = [parse_cell(value) for value in args.cells.split(",")]
    result = {
        "metadata": {
            "git_sha": git_sha(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "command": vars(args) | {"output": str(args.output)},
        },
        "cells": [],
    }
    for index, lengths in enumerate(cells):
        cell = benchmark_cell(
            lengths,
            args.first,
            args.samples,
            args.iterations,
            args.seed + index,
        )
        result["cells"].append(cell)
        forced = cell["median"]["forced_nvidia"]
        selected = cell["median"]["selected"]
        print(
            f"lengths={lengths}: Triton "
            f"{cell['median']['triton']['gpu_p50_us']:.3f} us, "
            f"NVIDIA {forced['gpu_p50_us']:.3f} us "
            f"({forced['saved_vs_triton_percent']:+.2f}%), "
            f"selected {selected['gpu_p50_us']:.3f} us "
            f"({selected['saved_vs_triton_percent']:+.2f}%)",
            flush=True,
        )
        del cell
        torch.cuda.empty_cache()

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        temporary.replace(args.output)
    print(payload)


if __name__ == "__main__":
    main()
