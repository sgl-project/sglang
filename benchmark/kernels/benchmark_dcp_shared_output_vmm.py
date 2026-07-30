"""Benchmark DCP Output/LSE reduction: eager AG+RS versus eager CUDA VMM.

Run with:

    PYTHONPATH=python python -m torch.distributed.run --standalone \
      --nproc_per_node=4 benchmark/kernels/benchmark_dcp_shared_output_vmm.py

The default shape models GLM-5.x at TP4/DCP4: 64 partial-output heads, with
16 destination-local heads per rank and a 512-wide latent value dimension.

This standalone harness deliberately measures eager execution. The registered
NCCL collectives used by SGLang's AG+RS path require server graph-runner capture
coordination and cannot be captured by a raw ``torch.cuda.CUDAGraph`` here.
Full-graph performance must therefore be established with paired server runs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.srt.layers.dcp import cp_lse_ag_out_rs_mla
from sglang.srt.layers.dcp.shared_output import (
    create_dcp_output_vmm_workspace,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,8,16,32,64,128,256,512")
    parser.add_argument("--total-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_dcp_shared_output_vmm.json"),
    )
    return parser.parse_args()


def _time_call(
    fn: Callable[[], torch.Tensor],
    *,
    iterations: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
) -> list[float]:
    dist.barrier(group=device_group)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) * 1000.0 / iterations

    rank_times: list[float | None] = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(rank_times, local_us, group=cpu_group)
    return [float(value) for value in rank_times]


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "stdev_us": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_us": min(samples),
        "max_us": max(samples),
        "cv_pct": (
            statistics.stdev(samples) / statistics.fmean(samples) * 100.0
            if len(samples) > 1
            else 0.0
        ),
    }


def _worker(args: argparse.Namespace) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coordinator = ps.init_world_group(
        ranks=list(range(dist.get_world_size())),
        local_rank=rank,
        backend="nccl",
    )
    world_size = coordinator.world_size
    if args.total_heads % world_size:
        raise ValueError(
            f"total_heads={args.total_heads} must be divisible by "
            f"world_size={world_size}"
        )

    rows_list = [int(value) for value in args.rows.split(",")]
    workspace = create_dcp_output_vmm_workspace(
        max_rows=max(rows_list),
        total_heads=args.total_heads,
        head_dim=args.head_dim,
        group=coordinator,
    )
    results: list[dict] = []

    for rows in rows_list:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260729 + rank * 97 + rows)
        partial_output = torch.randn(
            rows,
            args.total_heads,
            args.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        partial_lse = torch.randn(
            rows,
            args.total_heads,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )

        baseline_fn = (
            lambda output=partial_output, lse=partial_lse: cp_lse_ag_out_rs_mla(
                output,
                lse,
                coordinator,
            )
        )
        vmm_fn = lambda output=partial_output, lse=partial_lse: workspace.merge(
            output,
            lse,
            is_lse_base_on_e=False,
        )

        for _ in range(args.warmup_iterations):
            baseline_fn()
            vmm_fn()
        torch.cuda.synchronize()
        dist.barrier(group=coordinator.cpu_group)

        baseline_output_hbd = baseline_fn()
        vmm_output = vmm_fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            vmm_output,
            baseline_output_hbd.transpose(0, 1),
            atol=2e-2,
            rtol=2e-2,
        )

        trial_samples = {"ag_rs": [], "vmm": []}
        raw_trials = []
        for trial in range(args.trials):
            order = (
                (("ag_rs", baseline_fn), ("vmm", vmm_fn))
                if trial % 2 == 0
                else (("vmm", vmm_fn), ("ag_rs", baseline_fn))
            )
            trial_result = {"trial": trial, "order": [name for name, _ in order]}
            for name, fn in order:
                rank_times = _time_call(
                    fn,
                    iterations=args.iterations,
                    device_group=coordinator.device_group,
                    cpu_group=coordinator.cpu_group,
                )
                critical_us = max(rank_times)
                trial_samples[name].append(critical_us)
                trial_result[name] = {
                    "critical_us": critical_us,
                    "rank_us": rank_times,
                }
            raw_trials.append(trial_result)

        if rank == 0:
            baseline = _summary(trial_samples["ag_rs"])
            vmm = _summary(trial_samples["vmm"])
            paired_savings = [
                baseline_us - vmm_us
                for baseline_us, vmm_us in zip(
                    trial_samples["ag_rs"], trial_samples["vmm"]
                )
            ]
            mean_savings = statistics.fmean(paired_savings)
            result = {
                "rows": rows,
                "ag_rs": baseline,
                "vmm": vmm,
                "paired_savings": {
                    **_summary(paired_savings),
                    "speedup_pct": mean_savings / baseline["mean_us"] * 100.0,
                    "ratio": baseline["mean_us"] / vmm["mean_us"],
                },
                "raw_trials": raw_trials,
            }
            results.append(result)
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "raw_trials"}
                )
            )

        torch.cuda.synchronize()
        dist.barrier(group=coordinator.cpu_group)

    if rank == 0:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        payload = {
            "environment": {
                "commit": commit,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "world_size": world_size,
                "execution": "eager",
                "total_heads": args.total_heads,
                "local_heads": args.total_heads // world_size,
                "head_dim": args.head_dim,
                "warmup_iterations": args.warmup_iterations,
                "iterations_per_trial": args.iterations,
                "trials": args.trials,
                "timing_statistic": "maximum CUDA-event time across ranks",
            },
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.output}")

    workspace.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    _worker(_parse_args())
