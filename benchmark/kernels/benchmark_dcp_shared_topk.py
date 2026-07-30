"""Benchmark owner-sharded DCP candidate merge: AllGather versus direct VMM.

Local logit scoring and local Top-K are common to both transports and are
prepared outside the timed region. End-to-end serving measurements separately
attribute the replicated-to-owner-sharded scoring change.
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
from sglang.srt.layers.dcp.shared_topk import (
    merge_owner_topk_allgather,
)
from sglang.srt.layers.dcp.shared_topk_vmm import (
    create_dcp_topk_vmm_workspace,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,8,16,32,64,128,256,512")
    parser.add_argument("--local-context", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_dcp_shared_topk.json"),
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
    rows_list = [int(value) for value in args.rows.split(",")]
    workspace = create_dcp_topk_vmm_workspace(
        max_rows=max(rows_list),
        local_candidates=args.topk,
        group=coordinator,
    )
    pipelined_workspaces = [
        create_dcp_topk_vmm_workspace(
            max_rows=max(rows_list),
            local_candidates=args.topk,
            group=coordinator,
        )
        for _ in range(2)
    ]
    results: list[dict] = []

    for rows in rows_list:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260729 + rank * 97 + rows)
        logits = torch.randn(
            rows,
            args.local_context,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        logits[:, :4] = 10.0
        local_indices = torch.topk(
            logits,
            args.topk,
            dim=1,
            sorted=False,
        ).indices.to(torch.int32)

        def allgather_fn(logits=logits, local_indices=local_indices):
            return merge_owner_topk_allgather(
                logits,
                local_indices,
                args.topk,
                dcp_rank=rank,
                dcp_size=world_size,
                group=coordinator,
            )

        def vmm_fn(logits=logits, local_indices=local_indices):
            return workspace.merge(
                logits,
                local_indices,
                args.topk,
                dcp_rank=rank,
                dcp_size=world_size,
            )

        pipelined_step = 0

        def vmm_pipelined_fn(logits=logits, local_indices=local_indices):
            nonlocal pipelined_step
            output = pipelined_workspaces[pipelined_step % 2].merge(
                logits,
                local_indices,
                args.topk,
                dcp_rank=rank,
                dcp_size=world_size,
                pipelined=True,
            )
            pipelined_step += 1
            return output

        expected = allgather_fn()
        actual = vmm_fn()
        pipelined_actual = vmm_pipelined_fn()
        torch.cuda.synchronize()
        for candidate in (actual, pipelined_actual):
            torch.testing.assert_close(
                candidate.sort(dim=1).values,
                expected.sort(dim=1).values,
                rtol=0,
                atol=0,
            )

        functions = {
            "candidate_allgather": allgather_fn,
            "vmm_direct": vmm_fn,
            "vmm_direct_pipelined": vmm_pipelined_fn,
        }
        for _ in range(args.warmup_iterations):
            for fn in functions.values():
                fn()
        torch.cuda.synchronize()
        dist.barrier(group=coordinator.cpu_group)

        samples = {name: [] for name in functions}
        raw_trials = []
        for trial in range(args.trials):
            order = tuple(functions) if trial % 2 == 0 else tuple(reversed(functions))
            trial_result = {"trial": trial, "order": list(order)}
            for name in order:
                rank_times = _time_call(
                    functions[name],
                    iterations=args.iterations,
                    device_group=coordinator.device_group,
                    cpu_group=coordinator.cpu_group,
                )
                critical_us = max(rank_times)
                samples[name].append(critical_us)
                trial_result[name] = {
                    "critical_us": critical_us,
                    "rank_us": rank_times,
                }
            raw_trials.append(trial_result)

        if rank == 0:
            allgather = _summary(samples["candidate_allgather"])
            vmm = _summary(samples["vmm_direct"])
            vmm_pipelined = _summary(samples["vmm_direct_pipelined"])
            result = {
                "rows": rows,
                "candidate_allgather": allgather,
                "vmm_direct": vmm,
                "vmm_direct_pipelined": vmm_pipelined,
                "vmm_delta": {
                    "absolute_us": vmm["mean_us"] - allgather["mean_us"],
                    "relative_pct": (vmm["mean_us"] / allgather["mean_us"] - 1.0)
                    * 100.0,
                },
                "vmm_pipelined_delta": {
                    "absolute_us": vmm_pipelined["mean_us"] - allgather["mean_us"],
                    "relative_pct": (
                        vmm_pipelined["mean_us"] / allgather["mean_us"] - 1.0
                    )
                    * 100.0,
                },
                "raw_trials": raw_trials,
            }
            results.append(result)
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "raw_trials"}
                )
            )

    if rank == 0:
        payload = {
            "environment": {
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "world_size": world_size,
                "execution": "eager",
                "local_context": args.local_context,
                "global_context": args.local_context * world_size,
                "topk": args.topk,
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
    for pipelined_workspace in pipelined_workspaces:
        pipelined_workspace.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    _worker(_parse_args())
