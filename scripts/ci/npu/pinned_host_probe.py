"""Probe torch_npu pinned Host allocations without loading model weights.

Examples:
    python3 scripts/ci/npu/pinned_host_probe.py --iterations 2000
    python3 scripts/ci/npu/pinned_host_probe.py --workers 4 --hold-gb 0.5
    python3 scripts/ci/npu/pinned_host_probe.py --workers 16 --copy-from-npu

Each worker holds its own ``--hold-gb`` allocation. The probe only tests the
Host allocator and an optional small NPU-to-Host copy; it cannot reproduce the
full model's CPU, NPU, or distributed-memory footprint.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import sys
import traceback
from datetime import datetime, timezone


def _read_file(path: str) -> str | None:
    try:
        with open(path) as file:
            return file.read().strip()
    except OSError:
        return None


def _proc_field(key: str) -> str | None:
    status = _read_file("/proc/self/status")
    if status is None:
        return None
    for line in status.splitlines():
        if line.startswith(f"{key}:"):
            return line.partition(":")[2].strip()
    return None


def _snapshot(torch_npu) -> dict[str, object]:
    result: dict[str, object] = {
        "cgroup_current_bytes": _read_file("/sys/fs/cgroup/memory.current"),
        "cgroup_max_bytes": _read_file("/sys/fs/cgroup/memory.max"),
        "mem_available": None,
        "process_rss": _proc_field("VmRSS"),
        "process_locked": _proc_field("VmLck"),
        "cgroup_peak_bytes": _read_file("/sys/fs/cgroup/memory.peak"),
    }
    meminfo = _read_file("/proc/meminfo")
    if meminfo is not None:
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                result["mem_available"] = line.partition(":")[2].strip()
                break
    try:
        stats = torch_npu.npu.host_memory_stats()
        result.update(
            {
                "pinned_active_bytes": stats.get("active_bytes.current"),
                "pinned_reserved_bytes": stats.get("allocated_bytes.current"),
                "pinned_active_peak_bytes": stats.get("active_bytes.peak"),
                "pinned_reserved_peak_bytes": stats.get("allocated_bytes.peak"),
                "pinned_active_requests": stats.get("active_requests.current"),
                "pinned_owned_blocks": stats.get("allocations.current"),
                "pinned_host_alloc_calls": stats.get("num_host_alloc"),
                "pinned_host_free_calls": stats.get("num_host_free"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["pinned_stats_error"] = repr(exc)
    return result


def _emit(rank: int, event: str, torch_npu=None, **details: object) -> None:
    record: dict[str, object] = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "worker": rank,
        "event": event,
        **details,
    }
    if torch_npu is not None:
        record.update(_snapshot(torch_npu))
    print(json.dumps(record, sort_keys=True), flush=True)


def _worker(rank: int, args: argparse.Namespace, barrier) -> None:
    torch_npu = None
    held = None
    pending = []
    try:
        import torch
        import torch_npu

        device_count = torch.npu.device_count()
        if device_count < 1:
            raise RuntimeError("No NPU is visible to torch_npu")
        device = rank % device_count
        torch.npu.set_device(device)
        _emit(
            rank,
            "start",
            torch_npu,
            device=device,
            visible_devices=device_count,
            torch_version=torch.__version__,
            torch_npu_version=getattr(torch_npu, "__version__", "unknown"),
        )

        hold_bytes = round(args.hold_gb * 1_000_000_000)
        if hold_bytes:
            _emit(rank, "before_hold", torch_npu, requested_bytes=hold_bytes)
            held = torch.empty((hold_bytes,), dtype=torch.uint8, pin_memory=True)
            held.fill_(0)
            _emit(rank, "after_hold", torch_npu, requested_bytes=hold_bytes)

        # Align the small-allocation phase so large blocks stay live together.
        if barrier is not None:
            barrier.wait(timeout=300)

        source = None
        if args.copy_from_npu:
            source = torch.zeros(
                (args.token_count,), dtype=torch.int64, device=f"npu:{device}"
            )
        small_bytes = args.token_count * 8
        _emit(rank, "before_small", torch_npu, requested_bytes=small_bytes)
        for iteration in range(1, args.iterations + 1):
            host = torch.empty((args.token_count,), dtype=torch.int64, pin_memory=True)
            if source is not None:
                host.copy_(source, non_blocking=True)
            pending.append(host)
            if len(pending) == args.burst:
                if source is not None:
                    torch.npu.synchronize()
                pending.clear()
            if iteration % args.log_every == 0:
                _emit(rank, "small_progress", torch_npu, iteration=iteration)
        if source is not None:
            torch.npu.synchronize()
        _emit(rank, "after_small", torch_npu, iterations=args.iterations)

        pending.clear()
        host = None
        held = None
        gc.collect()
        _emit(rank, "after_release", torch_npu)
        if hasattr(torch_npu.npu, "host_empty_cache"):
            torch_npu.npu.host_empty_cache()
            _emit(rank, "after_empty_cache", torch_npu)
        else:
            _emit(rank, "cache_clear_unavailable", torch_npu)
    except Exception as exc:  # noqa: BLE001
        if barrier is not None:
            barrier.abort()
        _emit(
            rank,
            "failure",
            torch_npu,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe torch_npu pinned Host allocations without model weights."
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--hold-gb",
        type=float,
        default=0,
        help="Decimal GB of pinned Host memory held per worker (default: 0)",
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--burst", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--copy-from-npu",
        action="store_true",
        help="Also copy the small NPU tensor into each pinned Host tensor",
    )
    args = parser.parse_args()
    if (
        args.workers < 1
        or not math.isfinite(args.hold_gb)
        or args.hold_gb < 0
        or args.iterations < 0
        or args.token_count < 1
        or args.burst < 1
        or args.log_every < 1
    ):
        parser.error(
            "workers, token-count, burst, log-every must be positive; hold-gb and iterations must be nonnegative"
        )
    return args


def main() -> int:
    args = _parse_args()
    print(
        json.dumps(
            {
                "event": "plan",
                "workers": args.workers,
                "hold_gb_per_worker": args.hold_gb,
                "total_requested_hold_gb": args.workers * args.hold_gb,
                "small_bytes_per_allocation": args.token_count * 8,
                "iterations_per_worker": args.iterations,
                "copy_from_npu": args.copy_from_npu,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.workers) if args.workers > 1 else None
    workers = [
        context.Process(target=_worker, args=(rank, args, barrier))
        for rank in range(args.workers)
    ]
    for process in workers:
        process.start()
    for process in workers:
        process.join()
    return 0 if all(process.exitcode == 0 for process in workers) else 1


if __name__ == "__main__":
    sys.exit(main())
