"""Benchmark the full Kimi K3 KDA decode boundary at TP4/TP8 shapes.

The baseline is the production unfused chain:

    causal_conv1d_update -> Triton packed KDA -> sigmoid-gated RMSNorm

The candidate is ``kda_fused_decode``. The benchmark records raw balanced
samples for both eager launch and CUDA graph replay. Run each order in at
least three fresh processes:

    python bench_kimi_k3_kda_fused_decode_tp4.py --first baseline
    python bench_kimi_k3_kda_fused_decode_tp4.py --first candidate
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
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.kda_fused_decode import kda_fused_decode
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

_D = 128
_LOWER_BOUND = -5.0
_EPS = 1e-6


@dataclass
class Inputs:
    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    onorm_g: torch.Tensor
    conv_states: torch.Tensor
    ssm_states: torch.Tensor
    cache_indices: torch.Tensor
    conv_weight: torch.Tensor
    conv_weight_t_q: torch.Tensor
    conv_weight_t_k: torch.Tensor
    conv_weight_t_v: torch.Tensor
    conv_bias: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    onorm_weight: torch.Tensor


def make_inputs(heads: int, batch: int, seed: int) -> Inputs:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seg = heads * _D
    conv_dim = 3 * seg
    slots = batch + 8

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

    conv_weight = randn(
        conv_dim, 4, dtype=torch.float32, scale=0.05
    ).contiguous()
    conv_weight_t = conv_weight.t().contiguous()
    return Inputs(
        mixed_qkv=randn(batch, conv_dim).contiguous(),
        a=randn(batch, seg).contiguous(),
        b=randn(batch, heads).contiguous(),
        onorm_g=randn(batch, seg).contiguous(),
        conv_states=randn(slots, 3, conv_dim).contiguous(),
        ssm_states=randn(
            slots, heads, _D, _D, dtype=torch.float32, scale=0.01
        ).contiguous(),
        cache_indices=torch.randperm(
            slots, device="cuda", generator=generator
        )[:batch]
        .to(torch.int32)
        .contiguous(),
        conv_weight=conv_weight,
        conv_weight_t_q=conv_weight_t[:, :seg].contiguous(),
        conv_weight_t_k=conv_weight_t[:, seg : 2 * seg].contiguous(),
        conv_weight_t_v=conv_weight_t[:, 2 * seg :].contiguous(),
        conv_bias=randn(conv_dim, dtype=torch.float32, scale=0.01).contiguous(),
        a_log=randn(heads, dtype=torch.float32).contiguous(),
        dt_bias=randn(seg, dtype=torch.float32).contiguous(),
        onorm_weight=(
            1.0 + randn(_D, dtype=torch.float32, scale=0.1)
        ).contiguous(),
    )


def clone_inputs(inputs: Inputs) -> Inputs:
    values = vars(inputs).copy()
    values["conv_states"] = inputs.conv_states.clone()
    values["ssm_states"] = inputs.ssm_states.clone()
    return Inputs(**values)


def baseline(inputs: Inputs) -> torch.Tensor:
    heads = inputs.ssm_states.shape[-3]
    qkv = causal_conv1d_update(
        inputs.mixed_qkv,
        inputs.conv_states.transpose(-1, -2),
        inputs.conv_weight,
        inputs.conv_bias,
        activation="silu",
        conv_state_indices=inputs.cache_indices,
    )
    out = TritonKDAKernel().packed_decode(
        qkv,
        inputs.a,
        inputs.b,
        A_log=inputs.a_log,
        dt_bias=inputs.dt_bias,
        scale=_D**-0.5,
        ssm_states=inputs.ssm_states,
        cache_indices=inputs.cache_indices,
        num_v_heads=heads,
        head_v_dim=_D,
        lower_bound=_LOWER_BOUND,
    )
    return rms_norm_gated(
        x=out,
        g=inputs.onorm_g.view(1, -1, heads, _D),
        weight=inputs.onorm_weight,
        bias=None,
        activation="sigmoid",
        eps=_EPS,
    )


def candidate(inputs: Inputs) -> torch.Tensor:
    return kda_fused_decode(
        inputs.mixed_qkv,
        inputs.a,
        inputs.b,
        inputs.conv_states,
        inputs.conv_weight_t_q,
        inputs.conv_weight_t_k,
        inputs.conv_weight_t_v,
        inputs.conv_bias,
        inputs.a_log,
        inputs.dt_bias,
        inputs.onorm_g,
        inputs.onorm_weight,
        inputs.ssm_states,
        inputs.cache_indices,
        scale=_D**-0.5,
        onorm_eps=_EPS,
        lower_bound=_LOWER_BOUND,
    )


def correctness(source: Inputs) -> dict[str, float]:
    baseline_inputs = clone_inputs(source)
    candidate_inputs = clone_inputs(source)
    baseline_out = baseline(baseline_inputs)
    candidate_out = candidate(candidate_inputs)
    torch.cuda.synchronize()
    return {
        "output_max_abs": float(
            (baseline_out.float() - candidate_out.float()).abs().max()
        ),
        "state_max_abs": float(
            (baseline_inputs.ssm_states - candidate_inputs.ssm_states).abs().max()
        ),
        "conv_max_abs": float(
            (
                baseline_inputs.conv_states.float()
                - candidate_inputs.conv_states.float()
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
        fn()
    return graph.replay


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


def benchmark_cell(
    heads: int,
    batch: int,
    mode: str,
    first: str,
    samples: int,
    iterations: int,
    seed: int,
) -> dict:
    source = make_inputs(heads, batch, seed)
    errors = correctness(source)
    baseline_inputs = clone_inputs(source)
    candidate_inputs = clone_inputs(source)

    def baseline_fn():
        return baseline(baseline_inputs)

    def candidate_fn():
        return candidate(candidate_inputs)

    for _ in range(5):
        baseline_fn()
        candidate_fn()
    torch.cuda.synchronize()

    if mode == "graph":
        baseline_fn = graph_replay(baseline_fn)
        candidate_fn = graph_replay(candidate_fn)

    timings = {"baseline": [], "candidate": []}
    initial_order = (
        ("baseline", "candidate")
        if first == "baseline"
        else ("candidate", "baseline")
    )
    functions = {"baseline": baseline_fn, "candidate": candidate_fn}
    for sample in range(samples):
        order = initial_order if sample % 2 == 0 else initial_order[::-1]
        for provider in order:
            timings[provider].append(time_sample(functions[provider], iterations))

    def med(provider: str, metric: str) -> float:
        return statistics.median(x[metric] for x in timings[provider])

    baseline_gpu = med("baseline", "gpu_us")
    candidate_gpu = med("candidate", "gpu_us")
    result = {
        "heads": heads,
        "batch": batch,
        "mode": mode,
        "first": first,
        "samples": samples,
        "iterations": iterations,
        "correctness": errors,
        "raw": timings,
        "median": {
            "baseline_gpu_us": baseline_gpu,
            "candidate_gpu_us": candidate_gpu,
            "saved_gpu_us": baseline_gpu - candidate_gpu,
            "saved_gpu_percent": 100.0
            * (baseline_gpu - candidate_gpu)
            / baseline_gpu,
            "baseline_wall_us": med("baseline", "wall_us"),
            "candidate_wall_us": med("candidate", "wall_us"),
        },
    }
    torch.cuda.empty_cache()
    return result


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=24, choices=(12, 24))
    parser.add_argument("--batches", default="1,8,32,64,128")
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--first", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=32541)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("The fused KDA decode kernel requires Blackwell")

    batches = [int(value) for value in args.batches.split(",")]
    modes = args.modes.split(",")
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
    for mode in modes:
        for batch in batches:
            cell = benchmark_cell(
                args.heads,
                batch,
                mode,
                args.first,
                args.samples,
                args.iterations,
                args.seed + batch,
            )
            result["cells"].append(cell)
            median = cell["median"]
            print(
                f"H={args.heads} B={batch} {mode}: "
                f"{median['baseline_gpu_us']:.3f} -> "
                f"{median['candidate_gpu_us']:.3f} us, "
                f"save {median['saved_gpu_us']:.3f} us "
                f"({median['saved_gpu_percent']:.2f}%)",
                flush=True,
            )

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        temporary.replace(args.output)
    print(payload)


if __name__ == "__main__":
    main()
