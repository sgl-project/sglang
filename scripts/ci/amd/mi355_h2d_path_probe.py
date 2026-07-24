#!/usr/bin/env python3
"""Compare per-GPU H2D paths without loading a model."""

import argparse
import json
import mmap
import multiprocessing as mp
import os
import queue
import statistics
import time
import warnings
from pathlib import Path

import torch

MODEL_CACHE = Path(
    "/sgl-data/hf-cache/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots"
)


def _find_weka_source(size: int) -> Path | None:
    if not MODEL_CACHE.is_dir():
        return None
    candidates = [
        path
        for path in MODEL_CACHE.glob("*/*.safetensors")
        if path.is_file() and path.stat().st_size >= size
    ]
    return max(candidates, key=lambda path: path.stat().st_size, default=None)


def _warm_weka_segment(path: str, size: int, rank: int, world_size: int) -> None:
    start = rank * size // world_size
    end = (rank + 1) * size // world_size
    with open(path, "rb", buffering=0) as file:
        file.seek(start)
        remaining = end - start
        while remaining:
            data = file.read(min(32 * 1024 * 1024, remaining))
            if not data:
                raise RuntimeError(f"Unexpected EOF while warming {path}")
            remaining -= len(data)


def _make_source(case: str, size: int, rank: int, world_size: int, weka_path: str):
    resources = []
    if case == "pageable":
        source = torch.empty(size, dtype=torch.uint8)
        source.fill_(rank)
    elif case == "pinned":
        source = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        source.fill_(rank)
    elif case in ("weka_mmap", "weka_small_sync", "weka_small_pinned_batch"):
        _warm_weka_segment(weka_path, size, rank, world_size)
        file = open(weka_path, "rb", buffering=0)
        mapping = mmap.mmap(file.fileno(), length=size, access=mmap.ACCESS_READ)
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        source = torch.frombuffer(mapping, dtype=torch.uint8, count=size)
        resources.extend([mapping, file])
    else:
        raise ValueError(f"Unknown source case: {case}")
    return source, resources


def _time_copy(destination: torch.Tensor, source: torch.Tensor, gpu_id: int) -> float:
    torch.cuda.synchronize(gpu_id)
    start = time.perf_counter()
    destination.copy_(source, non_blocking=False)
    torch.cuda.synchronize(gpu_id)
    return time.perf_counter() - start


def _time_small_copy_round(
    destination: torch.Tensor,
    source: torch.Tensor,
    gpu_id: int,
    chunk_size: int,
    copy_count: int,
    round_index: int,
    pinned_batch: bool,
) -> float:
    max_chunks = source.numel() // chunk_size
    torch.cuda.synchronize(gpu_id)
    start = time.perf_counter()
    pinned_sources = []
    for copy_index in range(copy_count):
        chunk_index = (round_index * copy_count + copy_index) % max_chunks
        source_chunk = source.narrow(0, chunk_index * chunk_size, chunk_size)
        if pinned_batch:
            source_chunk = source_chunk.pin_memory()
            pinned_sources.append(source_chunk)
            destination.copy_(source_chunk, non_blocking=True)
        else:
            destination.copy_(source_chunk, non_blocking=False)
    torch.cuda.synchronize(gpu_id)
    pinned_sources.clear()
    return time.perf_counter() - start


def _worker(
    rank: int,
    world_size: int,
    case: str,
    size: int,
    iterations: int,
    small_copy_size: int,
    small_copy_count: int,
    weka_path: str,
    barrier,
    result_queue,
) -> None:
    resources = []
    try:
        allowed_cpus = sorted(os.sched_getaffinity(0))
        cores_per_rank = len(allowed_cpus) // world_size
        rank_cpus = allowed_cpus[rank * cores_per_rank : (rank + 1) * cores_per_rank]
        os.sched_setaffinity(0, rank_cpus)
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)

        properties = torch.cuda.get_device_properties(rank)
        source, resources = _make_source(case, size, rank, world_size, weka_path)
        is_small_copy = case in ("weka_small_sync", "weka_small_pinned_batch")
        destination_size = small_copy_size if is_small_copy else size
        destination = torch.empty(
            destination_size, dtype=torch.uint8, device=f"cuda:{rank}"
        )

        barrier.wait()
        if is_small_copy:
            _time_small_copy_round(
                destination,
                source,
                rank,
                small_copy_size,
                1,
                0,
                case == "weka_small_pinned_batch",
            )
        else:
            _time_copy(destination, source, rank)
        barrier.wait()

        concurrent_samples = []
        for iteration in range(iterations):
            barrier.wait()
            if is_small_copy:
                elapsed = _time_small_copy_round(
                    destination,
                    source,
                    rank,
                    small_copy_size,
                    small_copy_count,
                    iteration,
                    case == "weka_small_pinned_batch",
                )
            else:
                elapsed = _time_copy(destination, source, rank)
            concurrent_samples.append(elapsed)
            barrier.wait()

        isolated_samples = []
        for active_rank in range(world_size):
            barrier.wait()
            if rank == active_rank:
                for iteration in range(iterations):
                    if is_small_copy:
                        elapsed = _time_small_copy_round(
                            destination,
                            source,
                            rank,
                            small_copy_size,
                            small_copy_count,
                            iteration,
                            case == "weka_small_pinned_batch",
                        )
                    else:
                        elapsed = _time_copy(destination, source, rank)
                    isolated_samples.append(elapsed)
            barrier.wait()

        bytes_per_sample = small_copy_size * small_copy_count if is_small_copy else size
        gib = bytes_per_sample / (1024**3)
        result_queue.put(
            {
                "rank": rank,
                "case": case,
                "cpus": rank_cpus,
                "device_name": properties.name,
                "uuid": str(getattr(properties, "uuid", "unknown")),
                "copy_count": small_copy_count if is_small_copy else 1,
                "copy_size": small_copy_size if is_small_copy else size,
                "concurrent_gib_s": [gib / elapsed for elapsed in concurrent_samples],
                "isolated_gib_s": [gib / elapsed for elapsed in isolated_samples],
            }
        )
    except Exception as error:
        result_queue.put(
            {
                "rank": rank,
                "case": case,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        try:
            barrier.abort()
        except Exception:
            pass
    finally:
        if "source" in locals():
            del source
        if "destination" in locals():
            del destination
        for resource in resources:
            resource.close()


def _run_case(
    case: str,
    size: int,
    iterations: int,
    small_copy_size: int,
    small_copy_count: int,
    weka_path: str,
    world_size: int,
) -> list[dict]:
    context = mp.get_context("spawn")
    barrier = context.Barrier(world_size)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_worker,
            args=(
                rank,
                world_size,
                case,
                size,
                iterations,
                small_copy_size,
                small_copy_count,
                weka_path,
                barrier,
                result_queue,
            ),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    results = []
    deadline = time.monotonic() + 300
    while len(results) < world_size and time.monotonic() < deadline:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            if all(not process.is_alive() for process in processes):
                break

    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    if len(results) != world_size:
        raise RuntimeError(
            f"{case}: expected {world_size} worker results, got {len(results)}"
        )
    return sorted(results, key=lambda result: result["rank"])


def _write_summary(payload: dict) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a") as summary:
        summary.write("## MI355 H2D path probe\n\n")
        summary.write(
            f"Buffer: {payload['buffer_mib']} MiB; "
            f"iterations: {payload['iterations']}; "
            f"small copies: {payload['small_copy_count']} x "
            f"{payload['small_copy_kib']} KiB; "
            f"Weka source: `{payload['weka_file'] or 'unavailable'}`\n\n"
        )
        for case, results in payload["cases"].items():
            summary.write(f"### {case}\n\n")
            summary.write(
                "| GPU | concurrent median GiB/s | isolated median GiB/s | CPUs |\n"
            )
            summary.write("| --- | ---: | ---: | --- |\n")
            for result in results:
                if "error" in result:
                    summary.write(
                        f"| {result['rank']} | error | error | "
                        f"`{result['error']}` |\n"
                    )
                    continue
                concurrent = statistics.median(result["concurrent_gib_s"])
                isolated = statistics.median(result["isolated_gib_s"])
                summary.write(
                    f"| {result['rank']} | {concurrent:.2f} | {isolated:.2f} | "
                    f"`{result['cpus']}` |\n"
                )
            summary.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-mib", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--small-copy-kib", type=int, default=1536)
    parser.add_argument("--small-copy-count", type=int, default=16)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size != 8:
        raise RuntimeError(f"Expected 8 GPUs, found {world_size}")

    size = args.buffer_mib * 1024 * 1024
    small_copy_size = args.small_copy_kib * 1024
    weka_source = _find_weka_source(size)
    cases = ["pageable", "pinned"]
    if weka_source is not None:
        cases.extend(["weka_mmap", "weka_small_sync", "weka_small_pinned_batch"])

    payload = {
        "buffer_mib": args.buffer_mib,
        "iterations": args.iterations,
        "small_copy_kib": args.small_copy_kib,
        "small_copy_count": args.small_copy_count,
        "weka_file": str(weka_source) if weka_source else None,
        "cases": {},
    }
    for case in cases:
        print(f"MI355_H2D_PROBE_BEGIN case={case}", flush=True)
        payload["cases"][case] = _run_case(
            case,
            size,
            args.iterations,
            small_copy_size,
            args.small_copy_count,
            str(weka_source or ""),
            world_size,
        )
        print(f"MI355_H2D_PROBE_END case={case}", flush=True)

    print(f"MI355_H2D_PROBE_RESULT {json.dumps(payload, sort_keys=True)}")
    _write_summary(payload)

    errors = [
        result
        for results in payload["cases"].values()
        for result in results
        if "error" in result
    ]
    if errors:
        raise RuntimeError(f"Probe workers failed: {errors}")


if __name__ == "__main__":
    main()
