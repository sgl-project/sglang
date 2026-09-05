#!/usr/bin/env python3
"""Benchmark Kimi-K3's unfused DEP16 KDA projection prologue.

This reproduces the five BF16 GEMMs selected when full DP attention makes
``attn_tp_size=1`` while the global TP/EP size is 16::

    qkv = qkv_proj(x)             # 7168 -> 36864
    beta = b_proj(x)              # 7168 -> 96
    fa = f_a_proj(x)              # 7168 -> 128
    forget = f_b_proj(fa)         # 128  -> 12288
    gate = g_proj(x)              # 7168 -> 12288

The primary result is CUDA-graph replay time for the complete chain. Multiple
weight rotations can be captured to prevent an unrealistically warm weight
cache; reported latency is divided by the rotation count.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

HIDDEN = 7168
NUM_HEADS = 96
HEAD_DIM = 128
PROJECTION = NUM_HEADS * HEAD_DIM
SHAPES = {
    "qkv": (3 * PROJECTION, HIDDEN),
    "beta": (NUM_HEADS, HIDDEN),
    "f_a": (HEAD_DIM, HIDDEN),
    "f_b": (PROJECTION, HEAD_DIM),
    "gate": (PROJECTION, HIDDEN),
}


@dataclass
class Result:
    m: int
    rotations: int
    warmup_replays: int
    batches: int
    replays_per_batch: int
    batch_us: list[float]
    median_us: float
    min_us: float
    max_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m",
        default="32",
        help="Comma-separated local token counts. DEP16 at global batch 512 uses M=32.",
    )
    parser.add_argument("--rotations", type=int, default=2)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--replays-per-batch", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--nvtx-replays",
        type=int,
        default=0,
        help="After timing, replay this many times in an NVTX range for nsys.",
    )
    return parser.parse_args()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def allocate_weights(rotations: int) -> list[dict[str, torch.Tensor]]:
    weights = []
    for _ in range(rotations):
        current = {
            name: torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            for name, shape in SHAPES.items()
        }
        # Materialize every page before timing. Values do not affect GEMM dispatch.
        for weight in current.values():
            weight.fill_(0.01)
        weights.append(current)
    torch.cuda.synchronize()
    return weights


def unfused_chain(
    x: torch.Tensor, weights: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    qkv = F.linear(x, weights["qkv"])
    beta = F.linear(x, weights["beta"])
    f_a = F.linear(x, weights["f_a"])
    forget = F.linear(f_a, weights["f_b"])
    gate = F.linear(x, weights["gate"])
    return qkv, beta, forget, gate


def capture_graph(
    x: torch.Tensor, weight_sets: list[dict[str, torch.Tensor]]
) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...]]:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            for weights in weight_sets:
                outputs = unfused_chain(x, weights)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for weights in weight_sets:
            outputs = unfused_chain(x, weights)
    return graph, outputs


def benchmark_m(args: argparse.Namespace, m: int, weight_sets) -> Result:
    x = torch.randn((m, HIDDEN), device="cuda", dtype=torch.bfloat16)
    graph, outputs = capture_graph(x, weight_sets)

    # Keep graph outputs alive and make accidental removal obvious.
    assert len(outputs) == 4 and outputs[0].shape == (m, 3 * PROJECTION)

    for _ in range(args.warmup_replays):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(args.batches):
        start.record()
        for _ in range(args.replays_per_batch):
            graph.replay()
        end.record()
        end.synchronize()
        per_chain_us = (
            start.elapsed_time(end) * 1000.0 / args.replays_per_batch / args.rotations
        )
        samples.append(per_chain_us)

    if args.nvtx_replays:
        # This also provides a narrow capture range for:
        #   nsys profile --capture-range=cudaProfilerApi ...
        cudart = torch.cuda.cudart()
        cudart.cudaProfilerStart()
        torch.cuda.nvtx.range_push(f"k3_dep16_unfused_m{m}")
        for _ in range(args.nvtx_replays):
            graph.replay()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()

    return Result(
        m=m,
        rotations=args.rotations,
        warmup_replays=args.warmup_replays,
        batches=args.batches,
        replays_per_batch=args.replays_per_batch,
        batch_us=samples,
        median_us=statistics.median(samples),
        min_us=min(samples),
        max_us=max(samples),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rotations < 1:
        raise ValueError("--rotations must be positive")

    ms = [int(value) for value in args.m.split(",")]
    if any(m <= 0 for m in ms):
        raise ValueError("all M values must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    metadata = {
        "benchmark": "kimi_k3_kda_dep16_unfused_projections",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": git_revision() or os.environ.get("K3_BENCH_GIT_REV"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": props.name,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "device_memory_bytes": props.total_memory,
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
        },
        "shapes": {name: list(shape) for name, shape in SHAPES.items()},
    }

    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Allocating {args.rotations} rotating weight sets...", flush=True)
    weight_sets = allocate_weights(args.rotations)

    results = []
    for m in ms:
        result = benchmark_m(args, m, weight_sets)
        results.append(result)
        print(
            f"M={m:4d}: median={result.median_us:9.3f} us "
            f"min={result.min_us:9.3f} us max={result.max_us:9.3f} us "
            f"samples={','.join(f'{v:.3f}' for v in result.batch_us)}",
            flush=True,
        )

    document = {**metadata, "results": [asdict(result) for result in results]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
