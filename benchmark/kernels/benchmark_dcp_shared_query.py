"""Benchmark DCP Query transport: AllGather versus consumer-direct peer loads.

This measures the full FP8 Query/K preparation boundary after the absorbed
Query BMM. The AllGather control uses a preallocated combined BF16 buffer, so
the synthetic benchmark does not charge it for a q_nope concatenation that
SGLang #31821 already avoids in the serving path.
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
from sglang.kernels.ops.attention.utils import mla_quantize_and_rope_for_fp8
from sglang.srt.layers.dcp.shared_query_direct import (
    create_dcp_query_direct_vmm_workspace,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,8,16,32,64,128,256,512")
    parser.add_argument("--local-heads", type=int, default=16)
    parser.add_argument("--nope-dim", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_dcp_shared_query.json"),
    )
    return parser.parse_args()


def _time_call(
    fn: Callable[[], object],
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


def _assert_fp8_equal(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor.view(torch.uint8),
            expected_tensor.view(torch.uint8),
            rtol=0,
            atol=0,
        )


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
    consumer_workspace = create_dcp_query_direct_vmm_workspace(
        max_rows=max(rows_list),
        local_heads=args.local_heads,
        nope_dim=args.nope_dim,
        rope_dim=args.rope_dim,
        group=coordinator,
    )
    pipelined_consumer_workspaces = [
        create_dcp_query_direct_vmm_workspace(
            max_rows=max(rows_list),
            local_heads=args.local_heads,
            nope_dim=args.nope_dim,
            rope_dim=args.rope_dim,
            group=coordinator,
        )
        for _ in range(2)
    ]
    results: list[dict] = []

    for rows in rows_list:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260729 + rank * 97 + rows)
        q_nope = torch.randn(
            rows,
            args.local_heads,
            args.nope_dim,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        q_rope = torch.randn(
            rows,
            args.local_heads,
            args.rope_dim,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        k_nope = torch.randn(
            rows,
            args.nope_dim,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        k_rope = torch.randn(
            rows,
            args.rope_dim,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        positions = torch.arange(rows, dtype=torch.int64, device="cuda") * 3 + 1
        frequency = torch.arange(args.rope_dim // 2, device="cuda", dtype=torch.float32)
        frequency = 1.0 / (10_000 ** (frequency / (args.rope_dim // 2)))
        angles = torch.outer(
            torch.arange(rows * 3 + 2, device="cuda", dtype=torch.float32),
            frequency,
        )
        cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(
            torch.bfloat16
        )

        # Model the #31821 fused-combine contract: q_nope is already the
        # preallocated buffer's tail; only q_rope is copied per call.
        combined = torch.empty(
            args.local_heads,
            rows,
            args.rope_dim + args.nope_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        combined[..., args.rope_dim :].copy_(q_nope.transpose(0, 1))

        def allgather_fn(
            combined=combined,
            q_rope=q_rope,
            k_nope=k_nope,
            k_rope=k_rope,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
        ):
            combined[..., : args.rope_dim].copy_(q_rope.transpose(0, 1))
            gathered = coordinator.all_gather(combined, dim=0)
            gathered_rope, gathered_nope = gathered.split(
                [args.rope_dim, args.nope_dim], dim=-1
            )
            return mla_quantize_and_rope_for_fp8(
                gathered_nope.transpose(0, 1),
                gathered_rope.transpose(0, 1),
                k_nope,
                k_rope,
                positions,
                cos_sin_cache,
                True,
                args.nope_dim,
                args.rope_dim,
            )

        def consumer_fn(
            q_nope=q_nope,
            q_rope=q_rope,
            k_nope=k_nope,
            k_rope=k_rope,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
        ):
            return consumer_workspace.quantize_remote(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                positions,
                cos_sin_cache,
                is_neox=True,
            )

        pipelined_step = 0

        def consumer_pipelined_fn(
            q_nope=q_nope,
            q_rope=q_rope,
            k_nope=k_nope,
            k_rope=k_rope,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
        ):
            nonlocal pipelined_step
            output = pipelined_consumer_workspaces[
                pipelined_step % len(pipelined_consumer_workspaces)
            ].quantize_remote(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                positions,
                cos_sin_cache,
                is_neox=True,
                pipelined=True,
            )
            pipelined_step += 1
            return output

        expected = allgather_fn()
        _assert_fp8_equal(consumer_fn(), expected)
        _assert_fp8_equal(consumer_pipelined_fn(), expected)

        functions = {
            "allgather": allgather_fn,
            "consumer_direct": consumer_fn,
            "consumer_direct_pipelined": consumer_pipelined_fn,
        }
        for _ in range(args.warmup_iterations):
            for fn in functions.values():
                fn()
        torch.cuda.synchronize()
        dist.barrier(group=coordinator.cpu_group)

        samples = {name: [] for name in functions}
        raw_trials = []
        names = tuple(functions)
        for trial in range(args.trials):
            rotation = trial % len(names)
            order = names[rotation:] + names[:rotation]
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
            summaries = {name: _summary(values) for name, values in samples.items()}
            result = {
                "rows": rows,
                **summaries,
                "delta_vs_allgather": {
                    name: {
                        "absolute_us": (
                            summaries[name]["mean_us"]
                            - summaries["allgather"]["mean_us"]
                        ),
                        "relative_pct": (
                            summaries[name]["mean_us"]
                            / summaries["allgather"]["mean_us"]
                            - 1.0
                        )
                        * 100.0,
                    }
                    for name in (
                        "consumer_direct",
                        "consumer_direct_pipelined",
                    )
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
                "local_heads": args.local_heads,
                "total_heads": args.local_heads * world_size,
                "nope_dim": args.nope_dim,
                "rope_dim": args.rope_dim,
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

    consumer_workspace.close()
    for pipelined_consumer_workspace in pipelined_consumer_workspaces:
        pipelined_consumer_workspace.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    _worker(_parse_args())
