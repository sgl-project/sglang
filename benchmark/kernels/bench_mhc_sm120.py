"""Compare SM120 mHC tile policies on identical inputs and CUDA graphs.

Run from the candidate checkout, without installing over a serving environment::

    PYTHONPATH=python python benchmark/kernels/bench_mhc_sm120.py \
        --output /tmp/mhc-sm120.json

The baseline uses BLOCK_K=64 everywhere. original_pr uses 32 everywhere, as
PR #45264 originally did on SM120. updated uses the production dispatch: only
the fused split-K/Sinkhorn wrapper selects 32, with row tiles of at least 16.
bf16x3 retains its own 64 tile.
Cold measurements zero a 256 MiB buffer before each timed graph replay; the
flush is excluded from timing. Warm and cold both replay one wrapper call per
graph. Small-M timings include CUDA event/graph scheduling overhead.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
import triton

import sglang.kernels.ops.layernorm.mhc as mhc

PROFILES = ("baseline", "original_pr", "updated")


def make_inputs(m, k, seed):
    torch.manual_seed(seed)
    weight = torch.randn(24, k, device="cuda", dtype=torch.float32) / k**0.5
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda") * 0.1
    return x, weight, scale, base


def make_runner(entry, inputs):
    x, weight, scale, base = inputs
    if entry == "stats":
        return lambda: (mhc.hc_mix_stats(x, weight, 1e-20),)
    if entry == "sinkhorn":
        return lambda: mhc.hc_mix_stats_sinkhorn(
            x, weight, scale, base, 4, 20, 1e-20, 1e-6
        )
    parts = mhc.split_bf16_hc_weight(weight)
    return lambda: mhc.hc_mix_stats_sinkhorn_bf16x3(
        x, parts, scale, base, 20, 1e-20, 1e-6
    )


def capture(runner, profile, cache, condition):
    with ExitStack() as stack:
        if profile != "updated":
            block_k = 64 if profile == "baseline" else 32
            stack.enter_context(patch.object(mhc, "_HC_MIX_BLOCK_K", block_k))
            stack.enter_context(patch.object(mhc, "_HC_MIX_BF16X3_BLOCK_K", block_k))
            stack.enter_context(
                patch.object(
                    mhc, "_hc_mix_stats_sinkhorn_block_k", return_value=block_k
                )
            )
        if profile == "original_pr":
            stack.enter_context(
                patch.object(
                    mhc,
                    "_hc_mix_stats_sinkhorn_block_m",
                    side_effect=lambda m, block_k: mhc._block_m_for(m),
                )
            )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                outputs = runner()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        reference_outputs = tuple(output.cpu() for output in outputs)
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph, stream=stream):
            if condition == "cold":
                cache.zero_()
            start.record()
            graph_outputs = runner()
            end.record()
        # Keep output allocations alive for the lifetime of the captured graph.
        return graph, graph_outputs, reference_outputs, start, end


def sample(captured, repetitions):
    # Capture external timing events too, so the host cannot leave a scheduling
    # gap between the start event and these short kernel launches.
    graph, _, _, start, end = captured
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        graph.replay()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return samples


def errors(outputs, reference):
    result = []
    for output, expected in zip(outputs, reference):
        delta = (output - expected).abs()
        result.append(
            {
                "max_abs": delta.max().item(),
                "relative_l2": (
                    torch.linalg.vector_norm(delta.double())
                    / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
                ).item(),
                "different_elements": int((output != expected).sum()),
            }
        )
    return result


def environment(args):
    source = Path(mhc.__file__)
    try:
        gpu_status = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,"
                "utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu_status = None
    try:
        sglang_version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        sglang_version = None
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "sglang_installed_version": sglang_version,
        "module": str(source.resolve()),
        "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "nvidia_smi": gpu_status,
        "arguments": vars(args),
        "protocol": {
            "graph_calls": 1,
            "timing_events": "external CUDA events captured around the wrapper graph",
            "flush": "uint8 buffer zeroed on the timing stream before start event",
            "flush_mib": args.flush_mib,
            "rms_eps": 1e-20,
            "hc_eps": 1e-6,
            "sinkhorn_iters": 20,
            "hc_mult": 4,
            "mix": 24,
            "partial_num_warps": mhc._HC_MIX_NUM_WARPS,
            "partial_num_stages": mhc._HC_MIX_NUM_STAGES,
            "dot_precision": mhc._HC_MIX_DOT_PRECISION,
            "bf16x3_num_warps": 4,
            "bf16x3_num_stages": 3,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--k", type=int, default=20480)
    parser.add_argument(
        "--m", type=int, nargs="+", default=[1, 8, 256, 1024, 4096, 8192, 16384]
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        choices=("stats", "sinkhorn", "bf16x3"),
        default=["stats", "sinkhorn", "bf16x3"],
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=("cold", "warm"), default=["cold", "warm"]
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.k, args.trials, args.repetitions, args.flush_mib, *args.m) <= 0:
        parser.error("shape, trial, repetition and flush sizes must be positive")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() != (12, 0):
        parser.error("this experiment requires SM120; SM121 is unmeasured")
    result = {"environment": environment(args), "rows": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    cache = torch.empty(args.flush_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    order = random.Random(args.seed)
    for m in args.m:
        inputs = make_inputs(m, args.k, args.seed)
        for entry in args.entries:
            if entry == "bf16x3" and (not 4096 <= m <= 65536 or args.k % 1024):
                continue
            runner = make_runner(entry, inputs)
            for condition in args.conditions:
                graphs = {
                    profile: capture(runner, profile, cache, condition)
                    for profile in PROFILES
                }
                reference = graphs["baseline"][2]
                samples = {profile: [] for profile in PROFILES}
                trial_medians = {profile: [] for profile in PROFILES}
                for _ in range(args.trials):
                    profiles = list(PROFILES)
                    order.shuffle(profiles)
                    for profile in profiles:
                        measured = sample(graphs[profile], args.repetitions)
                        samples[profile].extend(measured)
                        trial_medians[profile].append(statistics.median(measured))
                row = {
                    "entry": entry,
                    "m": m,
                    "k": args.k,
                    "condition": condition,
                    "profiles": {},
                }
                for profile in PROFILES:
                    row["profiles"][profile] = {
                        "median_us": statistics.median(samples[profile]),
                        "trial_medians_us": trial_medians[profile],
                        "samples_us": samples[profile],
                        "errors_vs_baseline": errors(graphs[profile][2], reference),
                    }
                baseline = row["profiles"]["baseline"]["median_us"]
                updated = row["profiles"]["updated"]["median_us"]
                row["speedup"] = baseline / updated
                result["rows"].append(row)
                output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    f"{entry:8} M={m:6} {condition:4}: {baseline:.2f} -> {updated:.2f} us ({baseline / updated:.3f}x)",
                    flush=True,
                )
            del graphs, runner
        del inputs


if __name__ == "__main__":
    main()
