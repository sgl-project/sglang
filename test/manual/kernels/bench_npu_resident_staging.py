#!/usr/bin/env python3
"""Compare two resident staging implementations using one NPU kernel wheel.

Install the paired kernel wheel and place the candidate checkout's ``python``
directory first in PYTHONPATH, preserving the existing path. The source-path
arguments do not change how imports inside those files resolve.

Example (paths must point to distinct framework checkouts):
  python bench_npu_resident_staging.py --baseline /path/base/.../npu_kv_residency.py \
      --candidate /path/candidate/.../npu_kv_residency.py --output /path/result.json
The shell should redirect both stdout and stderr to a persistent log.
"""

import argparse
import importlib.util
import json
import os
import platform
import statistics

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
    create_fused_shm_tensor as create_shm_tensor,
)
from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
    free_fused_shm as free_shm,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def run_case(
    modules, mode, case, warmup, repetitions, device_id, validate_only, layout
):
    device = torch.device("npu", device_id)
    batch, width, capacity, context, features = 16, 4, 12288, 67584, 576
    long_prefix = 64000
    prefix, host_stride = 1024, 2048
    host, _, host_pointer = create_shm_tensor(
        shape=(1, (batch + 1) * host_stride, 1, features),
        dtype=torch.bfloat16,
        device_id=device_id,
        name=f"resident_ab_{os.getpid()}_{case}_{mode}",
    )
    # Exact BF16 integers, deterministic and dependent on both coordinates.
    feature = torch.arange(features, dtype=torch.int32)
    for start in range(0, (batch + 1) * host_stride, 512):
        token = torch.arange(start, start + 512, dtype=torch.int32)[:, None]
        host[0, start : start + 512, 0].copy_(
            ((token * 17 + feature * 13) % 251).to(torch.bfloat16)
        )
    table = (
        torch.arange(batch + 1, dtype=torch.int32, device=device)[:, None] * host_stride
        + torch.arange(context, dtype=torch.int32, device=device)[None, :] % host_stride
    )
    index_dtype = torch.int32 if layout == "int32-stride1" else torch.int64
    requests = torch.arange(1, batch + 1, dtype=index_dtype, device=device)
    is_short = torch.tensor(
        [case != "long" and (case != "mixed" or i % 2 == 0) for i in range(batch)],
        dtype=torch.bool,
        device=device,
    )
    lengths = torch.where(is_short, prefix + width, long_prefix + width).to(index_dtype)
    first = torch.where(is_short, prefix, long_prefix).to(index_dtype)
    positions = first[:, None] + torch.arange(width, dtype=torch.int32, device=device)
    if layout == "int64-stride4":
        first = positions[:, 0]
        assert first.stride(0) == width
    token_requests = requests[:, None].expand(batch, width).reshape(-1).contiguous()
    positions = positions.reshape(-1).contiguous()
    token_valid = torch.ones(batch * width, dtype=torch.bool, device=device)
    current_cpu = (
        (torch.arange(batch * width, dtype=torch.int32)[:, None] * 29 + feature * 7)
        % 251
    ).to(torch.bfloat16)[:, None, :]
    current = current_cpu.to(device)
    source = torch.arange(batch * width, dtype=torch.int64, device=device)
    history = (
        torch.empty(batch * capacity, dtype=torch.int64, device=device),
        torch.empty(batch * capacity, dtype=torch.int64, device=device),
        torch.empty(batch * capacity, dtype=torch.bool, device=device),
    )
    current_workspace = (
        torch.empty(batch * width, dtype=torch.int64, device=device),
        torch.empty(batch * width, dtype=torch.bool, device=device),
    )
    gap = case in ("short_gap", "mixed")
    initial_end = prefix // 2 if gap else prefix
    results, snapshots = {}, {}
    try:
        for name, module in modules.items():
            cache = torch.full(
                (batch + 1, capacity, 1, features),
                -7,
                dtype=torch.bfloat16,
                device=device,
            )
            old_begin = torch.full((batch + 1,), -1, dtype=torch.int32, device=device)
            old_end = old_begin.clone()
            next_extents = tuple(old_begin.clone() for _ in range(3))
            for i in range(batch):
                if case != "long" and (case != "mixed" or i % 2 == 0):
                    row = i + 1
                    cache[row, :initial_end].copy_(
                        host[0, row * host_stride : row * host_stride + initial_end]
                    )
                    old_begin[row] = 0
                    old_end[row] = initial_end
            reset_end = old_end.clone()

            def stage():
                module.stage_resident_kv(
                    current=current,
                    source_indices=source,
                    host=host,
                    host_address=host_pointer,
                    cache=cache,
                    req_to_token=table,
                    requests=requests,
                    lengths=lengths,
                    first_positions=first,
                    token_requests=token_requests,
                    token_positions=positions,
                    token_valid=token_valid,
                    old_begin=old_begin,
                    old_end=old_end,
                    next_extents=next_extents,
                    history_workspace=history,
                    current_workspace=current_workspace,
                    capacity=capacity,
                    context_len=context,
                )

            def iteration():
                # Gap cases deliberately pay for this identical device reset in
                # both implementations so each iteration restores the same gap.
                if gap:
                    old_end.copy_(reset_end)
                stage()

            iteration()
            torch.npu.synchronize()
            for i in range(batch):
                row = i + 1
                short = case != "long" and (case != "mixed" or i % 2 == 0)
                if short:
                    torch.testing.assert_close(
                        cache[row, :prefix].cpu(),
                        host[0, row * host_stride : row * host_stride + prefix],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        cache[row, prefix : prefix + width].cpu(),
                        current_cpu[i * width : (i + 1) * width],
                        rtol=0,
                        atol=0,
                    )
                    assert (
                        old_begin[row].item() == 0
                        and old_end[row].item() == prefix + width
                    )
                else:
                    assert torch.all(cache[row] == -7).item()
                    assert old_begin[row].item() == old_end[row].item() == -1
            snapshots[name] = (cache.cpu(), old_begin.cpu(), old_end.cpu())
            if mode == "graph":
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    iteration()
                execute = graph.replay
            else:
                execute = iteration
            # Validate the captured execution before measuring it as well.
            execute()
            torch.npu.synchronize()
            for actual, expected in zip(
                (cache.cpu(), old_begin.cpu(), old_end.cpu()), snapshots[name]
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if not validate_only:
                for _ in range(warmup):
                    execute()
                torch.npu.synchronize()
                samples = []
                for _ in range(repetitions):
                    start, end = (
                        torch.npu.Event(enable_timing=True),
                        torch.npu.Event(enable_timing=True),
                    )
                    start.record()
                    execute()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                results[name] = {
                    "median_ms": statistics.median(samples),
                    "p90_ms": percentile(samples, 0.90),
                    "min_ms": min(samples),
                    "samples_ms": samples,
                }
            execute = None
            if mode == "graph":
                graph = None
            cache = old_begin = old_end = next_extents = None
        for a, b in zip(snapshots["baseline"], snapshots["candidate"]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        results["numeric"] = (
            "pass: independent expected values and baseline/candidate exact match"
        )
        if not validate_only:
            results["candidate_over_baseline"] = (
                results["candidate"]["median_ms"] / results["baseline"]["median_ms"]
            )
        return results
    finally:
        torch.npu.synchronize()
        free_shm(device_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--layout",
        choices=("int32-stride1", "int64-stride4"),
        default="int32-stride1",
        help="Request/length index dtype and first_positions view layout",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 1:
        parser.error("warmup must be nonnegative and repetitions positive")
    modules = {
        name: load_module(name, path)
        for name, path in [("baseline", args.baseline), ("candidate", args.candidate)]
    }
    device_id = torch.npu.current_device()
    cases = (
        ("long",)
        if args.layout == "int64-stride4"
        else ("long", "short_no_gap", "short_gap", "mixed")
    )
    report = {
        "configuration": {
            "batch": 16,
            "width": 4,
            "capacity": 12288,
            "context": 67584,
            "long_prefix": 64000,
            "long_length": 64004,
            "features": 576,
            "dtype": "bfloat16",
            "request_length_position_dtype": (
                "int32" if args.layout == "int32-stride1" else "int64"
            ),
            "first_positions_stride": (1 if args.layout == "int32-stride1" else 4),
            "layout": args.layout,
            "prefix": 1024,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "seed": "deterministic integer coordinate formulas",
            "device_id": device_id,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "baseline": os.path.abspath(args.baseline),
            "candidate": os.path.abspath(args.candidate),
            "gap_timing": "includes old_end.copy_ reset in both implementations",
        },
        "cases": {},
        "status": "running",
    }
    try:
        report["validation"] = {}
        for mode in ("eager", "graph"):
            for case in cases:
                key = f"{mode}/{case}"
                print(f"VALIDATE {key}", flush=True)
                report["validation"][key] = run_case(
                    modules,
                    mode,
                    case,
                    args.warmup,
                    args.repetitions,
                    device_id,
                    True,
                    args.layout,
                )
        for mode in ("eager", "graph"):
            for case in cases:
                key = f"{mode}/{case}"
                print(f"TIME {key}", flush=True)
                report["cases"][key] = run_case(
                    modules,
                    mode,
                    case,
                    args.warmup,
                    args.repetitions,
                    device_id,
                    False,
                    args.layout,
                )
                print(json.dumps({key: report["cases"][key]}, indent=2), flush=True)
        report["status"] = "pass"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
        print(f"JSON {args.output} status={report['status']}", flush=True)


if __name__ == "__main__":
    main()
