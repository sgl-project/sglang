# SPDX-License-Identifier: Apache-2.0

"""Paired CUDA benchmark for chunked diffusion LoRA weight merging.

Example:
    python bench_lora_merge_addmm.py --output /tmp/results.csv
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

CHUNK_BYTES = 32 * 1024 * 1024
CASES = [
    (320, 320, 4, 1.0),
    (320, 320, 8, 0.25),
    (1280, 1280, 8, 1.0),
    (1280, 1280, 16, -0.5),
    (3072, 3072, 16, 0.25),
    (3072, 3072, 32, 1.0),
    (4096, 4096, 32, -0.5),
    (4096, 4096, 64, 1.0),
    (12288, 4096, 4, 0.25),
    (12288, 4096, 64, 1.0),
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def baseline(dst, lora_b, lora_a, scale, chunk_rows):
    for start in range(0, dst.shape[0], chunk_rows):
        end = min(start + chunk_rows, dst.shape[0])
        delta = lora_b[start:end] @ lora_a
        dst[start:end].add_(delta, alpha=scale)


def candidate(dst, lora_b, lora_a, scale, chunk_rows):
    for start in range(0, dst.shape[0], chunk_rows):
        end = min(start + chunk_rows, dst.shape[0])
        dst[start:end].addmm_(lora_b[start:end], lora_a, alpha=scale)


def time_variant(
    fn: Callable,
    initial: torch.Tensor,
    lora_b: torch.Tensor,
    lora_a: torch.Tensor,
    scale: float,
    chunk_rows: int,
    repetitions: int,
) -> float:
    dst = initial.clone()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn(dst, lora_b, lora_a, scale, chunk_rows)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repetitions


def peak_extra(
    fn: Callable,
    initial: torch.Tensor,
    lora_b: torch.Tensor,
    lora_a: torch.Tensor,
    scale: float,
    chunk_rows: int,
) -> int:
    dst = initial.clone()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn(dst, lora_b, lora_a, scale, chunk_rows)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before


def validate(initial, lora_b, lora_a, scale, chunk_rows) -> float:
    expected = initial.clone()
    actual = initial.clone()
    baseline(expected, lora_b, lora_a, scale, chunk_rows)
    candidate(actual, lora_b, lora_a, scale, chunk_rows)
    atol = 1e-5 if initial.dtype is torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)
    return (actual.float() - expected.float()).abs().max().item()


def profile_variant(name, fn, initial, lora_b, lora_a, scale, chunk_rows):
    dst = initial.clone()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(dst, lora_b, lora_a, scale, chunk_rows)
    torch.cuda.synchronize()
    return (
        name
        + "\n"
        + prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    rows_out = []
    representative = None
    free_bytes, _ = torch.cuda.mem_get_info()

    for dtype in DTYPES:
        for rows, columns, rank, scale in CASES:
            element_size = torch.empty((), dtype=dtype).element_size()
            estimate = (
                element_size * (2 * rows * columns + rows * rank + rank * columns)
                + CHUNK_BYTES
            )
            if estimate > int(0.8 * free_bytes):
                print(f"skip dtype={dtype} shape=({rows},{columns}) rank={rank}")
                continue
            chunk_rows = max(1, CHUNK_BYTES // (columns * element_size))
            initial = torch.randn(rows, columns, device="cuda", dtype=dtype) * 1e-3
            lora_b = torch.randn(rows, rank, device="cuda", dtype=dtype) * 1e-3
            lora_a = torch.randn(rank, columns, device="cuda", dtype=dtype) * 1e-3
            max_abs = validate(initial, lora_b, lora_a, scale, chunk_rows)
            warm = initial.clone()
            for _ in range(args.warmups):
                baseline(warm, lora_b, lora_a, scale, chunk_rows)
                candidate(warm, lora_b, lora_a, scale, chunk_rows)
            torch.cuda.synchronize()

            peaks = {
                "baseline": peak_extra(
                    baseline, initial, lora_b, lora_a, scale, chunk_rows
                ),
                "candidate": peak_extra(
                    candidate, initial, lora_b, lora_a, scale, chunk_rows
                ),
            }
            variants = {"baseline": baseline, "candidate": candidate}
            for round_idx in range(7):
                order = (
                    ["baseline", "candidate"]
                    if round_idx % 2 == 0
                    else ["candidate", "baseline"]
                )
                for variant in order:
                    latency = time_variant(
                        variants[variant],
                        initial,
                        lora_b,
                        lora_a,
                        scale,
                        chunk_rows,
                        args.repetitions,
                    )
                    rows_out.append(
                        {
                            "dtype": str(dtype).removeprefix("torch."),
                            "rows": rows,
                            "columns": columns,
                            "rank": rank,
                            "scale": scale,
                            "variant": variant,
                            "round": round_idx,
                            "latency_us": f"{latency:.6f}",
                            "peak_extra_bytes": peaks[variant],
                            "max_abs": f"{max_abs:.9g}",
                        }
                    )
            if dtype is torch.bfloat16 and (rows, columns, rank) == (4096, 4096, 32):
                representative = (initial, lora_b, lora_a, scale, chunk_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    groups = defaultdict(lambda: defaultdict(list))
    for row in rows_out:
        key = (
            row["dtype"],
            row["rows"],
            row["columns"],
            row["rank"],
            row["scale"],
        )
        groups[key][row["variant"]].append(float(row["latency_us"]))
    speedups = []
    print("dtype shape rank scale baseline_us candidate_us speedup max_abs")
    for key in sorted(groups):
        baseline_us = statistics.median(groups[key]["baseline"])
        candidate_us = statistics.median(groups[key]["candidate"])
        speedup = baseline_us / candidate_us
        speedups.append(speedup)
        dtype, rows, columns, rank, scale = key
        max_abs = next(
            row["max_abs"]
            for row in rows_out
            if (row["dtype"], row["rows"], row["columns"], row["rank"], row["scale"])
            == key
        )
        print(
            f"{dtype} {rows}x{columns} {rank} {scale:g} "
            f"{baseline_us:.3f} {candidate_us:.3f} {speedup:.3f}x {max_abs}"
        )
    geomean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    print(f"geomean_speedup={geomean:.3f}x minimum_speedup={min(speedups):.3f}x")

    if args.profile_output is not None:
        if representative is None:
            raise RuntimeError("representative profiler case was skipped")
        initial, lora_b, lora_a, scale, chunk_rows = representative
        tables = [
            profile_variant(
                "baseline", baseline, initial, lora_b, lora_a, scale, chunk_rows
            ),
            profile_variant(
                "candidate", candidate, initial, lora_b, lora_a, scale, chunk_rows
            ),
        ]
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text("\n\n".join(tables))


if __name__ == "__main__":
    main()
