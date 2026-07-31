"""Benchmark copy-free Kimi K3 KDA prefix restore at the TP4 state shape.

The full-boundary comparison includes the exact 69-layer FP32 temporal state,
the BF16 convolution window, and the first KDA update:

    baseline:  copy checkpoint -> working slot, then update working slot
    candidate: copy conv only, read temporal checkpoint, write working slot

The packed-index overhead comparison measures ordinary KDA prefill with one
ownership-resolution kernel against the pre-change int32 index path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    prepare_kda_state_io_indices,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda

_LAYERS = 69
_HEADS = 24
_D = 128
_CONV_ELEMENTS_PER_LAYER = 3 * 3 * _HEADS * _D


@dataclass
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    baseline_v: torch.Tensor
    candidate_v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    cu_seqlens: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    source_map: torch.Tensor
    baseline_state_io_indices: torch.Tensor
    candidate_state_io_indices: torch.Tensor
    baseline_temporal: torch.Tensor
    candidate_temporal: torch.Tensor
    baseline_conv: torch.Tensor
    candidate_conv: torch.Tensor


def make_inputs(batch: int, tokens: int, seed: int) -> Inputs:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    slots = 2 * batch
    shape = (1, batch * tokens, _HEADS, _D)

    def randn(*tensor_shape, dtype=torch.bfloat16, scale=0.1):
        return (
            torch.randn(
                *tensor_shape,
                dtype=dtype,
                device="cuda",
                generator=generator,
            )
            * scale
        )

    baseline_temporal = randn(
        _LAYERS,
        slots,
        _HEADS,
        _D,
        _D,
        dtype=torch.float32,
        scale=0.01,
    )
    candidate_temporal = baseline_temporal.clone()
    baseline_conv = randn(
        _LAYERS,
        slots,
        _CONV_ELEMENTS_PER_LAYER,
    )
    candidate_conv = baseline_conv.clone()
    baseline_v = randn(_LAYERS, *shape)
    src = torch.arange(batch, device="cuda", dtype=torch.int32)
    dst = torch.arange(batch, 2 * batch, device="cuda", dtype=torch.int32)
    source_map = torch.arange(slots, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.arange(
        0,
        (batch + 1) * tokens,
        tokens,
        device="cuda",
        dtype=torch.int32,
    )
    return Inputs(
        q=randn(*shape),
        k=randn(*shape),
        baseline_v=baseline_v,
        candidate_v=baseline_v.clone(),
        g=randn(*shape, dtype=torch.float32) - 1.0,
        beta=torch.sigmoid(randn(1, batch * tokens, _HEADS, dtype=torch.float32)),
        cu_seqlens=cu_seqlens,
        src=src,
        dst=dst,
        source_map=source_map,
        baseline_state_io_indices=torch.empty(
            batch, device="cuda", dtype=torch.int32
        ),
        candidate_state_io_indices=torch.empty(
            batch, device="cuda", dtype=torch.int32
        ),
        baseline_temporal=baseline_temporal,
        candidate_temporal=candidate_temporal,
        baseline_conv=baseline_conv,
        candidate_conv=candidate_conv,
    )


def run_layers(
    inputs: Inputs,
    temporal: torch.Tensor,
    values: torch.Tensor,
    state_io_indices: torch.Tensor | None,
) -> torch.Tensor:
    output = None
    for layer in range(_LAYERS):
        output = chunk_kda(
            q=inputs.q,
            k=inputs.k,
            v=values[layer],
            g=inputs.g,
            beta=inputs.beta,
            initial_state=temporal[layer],
            initial_state_indices=inputs.dst,
            initial_state_io_indices=state_io_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=inputs.cu_seqlens,
        )
    return output


def baseline_prepare(inputs: Inputs) -> None:
    src = inputs.src.long()
    dst = inputs.dst.long()
    inputs.baseline_conv[:, dst] = inputs.baseline_conv[:, src]
    inputs.baseline_temporal[:, dst] = inputs.baseline_temporal[:, src]
    inputs.source_map[dst] = inputs.dst
    prepare_kda_state_io_indices(
        inputs.dst,
        inputs.source_map,
        inputs.baseline_state_io_indices,
    )


def baseline_run(inputs: Inputs) -> torch.Tensor:
    return run_layers(
        inputs,
        inputs.baseline_temporal,
        inputs.baseline_v,
        inputs.baseline_state_io_indices,
    )


def baseline(inputs: Inputs) -> torch.Tensor:
    baseline_prepare(inputs)
    return baseline_run(inputs)


def candidate_prepare(inputs: Inputs) -> None:
    src = inputs.src.long()
    dst = inputs.dst.long()
    inputs.candidate_conv[:, dst] = inputs.candidate_conv[:, src]
    inputs.source_map[dst] = inputs.src
    prepare_kda_state_io_indices(
        inputs.dst,
        inputs.source_map,
        inputs.candidate_state_io_indices,
    )


def candidate_run(inputs: Inputs) -> torch.Tensor:
    return run_layers(
        inputs,
        inputs.candidate_temporal,
        inputs.candidate_v,
        inputs.candidate_state_io_indices,
    )


def candidate(inputs: Inputs) -> torch.Tensor:
    candidate_prepare(inputs)
    return candidate_run(inputs)


def no_map(inputs: Inputs) -> torch.Tensor:
    return run_layers(inputs, inputs.baseline_temporal, inputs.baseline_v, None)


def packed_indices_prepare(inputs: Inputs) -> None:
    prepare_kda_state_io_indices(
        inputs.dst,
        inputs.source_map,
        inputs.candidate_state_io_indices,
    )


def packed_indices_run(inputs: Inputs) -> torch.Tensor:
    return run_layers(
        inputs,
        inputs.candidate_temporal,
        inputs.candidate_v,
        inputs.candidate_state_io_indices,
    )


def packed_indices(inputs: Inputs) -> torch.Tensor:
    packed_indices_prepare(inputs)
    return packed_indices_run(inputs)


def correctness(inputs: Inputs) -> dict[str, float]:
    expected = baseline(inputs)
    actual = candidate(inputs)
    torch.cuda.synchronize()
    dst = inputs.dst.long()
    src = inputs.src.long()
    return {
        "output_max_abs": float((expected.float() - actual.float()).abs().max()),
        "destination_state_max_abs": float(
            (inputs.baseline_temporal[:, dst] - inputs.candidate_temporal[:, dst])
            .abs()
            .max()
        ),
        "source_state_max_abs": float(
            (inputs.baseline_temporal[:, src] - inputs.candidate_temporal[:, src])
            .abs()
            .max()
        ),
        "conv_max_abs": float(
            (
                inputs.baseline_conv[:, dst].float()
                - inputs.candidate_conv[:, dst].float()
            )
            .abs()
            .max()
        ),
    }


def graph_replay(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()

    def replay():
        graph.replay()
        return output

    return replay


def graph_replay_with_prepare(prepare, run):
    prepare()
    run()
    torch.cuda.synchronize()
    prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()

    def replay():
        prepare()
        graph.replay()
        return output

    return replay


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


def benchmark_pair(
    functions: dict[str, object],
    first: str,
    samples: int,
    iterations: int,
) -> dict:
    for _ in range(2):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()

    timings = {name: [] for name in functions}
    names = tuple(functions)
    initial_order = names if first == names[0] else names[::-1]
    for sample in range(samples):
        order = initial_order if sample % 2 == 0 else initial_order[::-1]
        for provider in order:
            timings[provider].append(time_sample(functions[provider], iterations))

    def median(provider: str, metric: str) -> float:
        return statistics.median(x[metric] for x in timings[provider])

    before, after = names
    before_gpu = median(before, "gpu_us")
    after_gpu = median(after, "gpu_us")
    return {
        "raw": timings,
        "median": {
            "before": before,
            "after": after,
            "before_gpu_us": before_gpu,
            "after_gpu_us": after_gpu,
            "saved_gpu_us": before_gpu - after_gpu,
            "saved_gpu_percent": 100.0 * (before_gpu - after_gpu) / before_gpu,
            "before_wall_us": median(before, "wall_us"),
            "after_wall_us": median(after, "wall_us"),
        },
    }


def benchmark_cell(
    batch: int,
    tokens: int,
    mode: str,
    comparison: str,
    first: str,
    samples: int,
    iterations: int,
    seed: int,
) -> dict:
    inputs = make_inputs(batch, tokens, seed)
    errors = correctness(inputs)
    if comparison == "boundary":
        if mode == "graph":
            functions = {
                "baseline": graph_replay_with_prepare(
                    lambda: baseline_prepare(inputs),
                    lambda: baseline_run(inputs),
                ),
                "candidate": graph_replay_with_prepare(
                    lambda: candidate_prepare(inputs),
                    lambda: candidate_run(inputs),
                ),
            }
        else:
            functions = {
                "baseline": lambda: baseline(inputs),
                "candidate": lambda: candidate(inputs),
            }
    else:
        # Both pools hold identical destination state after correctness().
        if mode == "graph":
            functions = {
                "no_map": graph_replay(lambda: no_map(inputs)),
                "packed_indices": graph_replay_with_prepare(
                    lambda: packed_indices_prepare(inputs),
                    lambda: packed_indices_run(inputs),
                ),
            }
        else:
            functions = {
                "no_map": lambda: no_map(inputs),
                "packed_indices": lambda: packed_indices(inputs),
            }

    result = benchmark_pair(functions, first, samples, iterations)
    return {
        "batch": batch,
        "tokens": tokens,
        "mode": mode,
        "comparison": comparison,
        "first": first,
        "samples": samples,
        "iterations": iterations,
        "state_bytes_per_request": {
            "temporal": _LAYERS * _HEADS * _D * _D * 4,
            "conv": _LAYERS * _CONV_ELEMENTS_PER_LAYER * 2,
            "source_map": 4,
            "packed_state_indices": 4,
        },
        "correctness": errors,
        **result,
    }


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,4,16,32")
    parser.add_argument("--tokens", default="1,17")
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument(
        "--comparison",
        choices=("boundary", "map-overhead"),
        default="boundary",
    )
    parser.add_argument(
        "--first",
        choices=("baseline", "candidate", "no_map", "packed_indices"),
        required=True,
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=32541)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    expected_first = (
        {"baseline", "candidate"}
        if args.comparison == "boundary"
        else {"no_map", "packed_indices"}
    )
    if args.first not in expected_first:
        raise ValueError(
            f"--first {args.first!r} is invalid for {args.comparison}; "
            f"choose one of {sorted(expected_first)}"
        )

    result = {
        "metadata": {
            "git_sha": git_sha(),
            "dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "command": vars(args) | {"output": str(args.output)},
        },
        "cells": [],
    }
    for mode in args.modes.split(","):
        for tokens in [int(value) for value in args.tokens.split(",")]:
            for batch in [int(value) for value in args.batches.split(",")]:
                cell = benchmark_cell(
                    batch,
                    tokens,
                    mode,
                    args.comparison,
                    args.first,
                    args.samples,
                    args.iterations,
                    args.seed + 1000 * tokens + batch,
                )
                result["cells"].append(cell)
                median = cell["median"]
                print(
                    f"{args.comparison} B={batch} T={tokens} {mode}: "
                    f"{median['before_gpu_us']:.3f} -> "
                    f"{median['after_gpu_us']:.3f} us, "
                    f"save {median['saved_gpu_us']:.3f} us "
                    f"({median['saved_gpu_percent']:.2f}%)",
                    flush=True,
                )
                torch.cuda.empty_cache()

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        temporary.replace(args.output)
    else:
        print(payload)


if __name__ == "__main__":
    main()
