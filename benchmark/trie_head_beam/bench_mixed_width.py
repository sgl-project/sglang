#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from common import require_single_visible_gpu


class ConstraintValidator:
    def __init__(
        self,
        valid_keys_path: Path,
        *,
        token_start: int,
        codebook_size: int,
        sid_length: int,
    ) -> None:
        self.valid_keys = np.load(valid_keys_path, mmap_mode="r")
        self.token_start = token_start
        self.codebook_size = codebook_size
        self.sid_length = sid_length

    def validate(self, rows: list[list[int]], expected: int) -> dict[str, int]:
        wrong_result_count = int(len(rows) != expected)
        length_invalid = sum(len(row) != self.sid_length for row in rows)
        complete_rows = [row for row in rows if len(row) == self.sid_length]
        if not complete_rows:
            return {
                "result_count": len(rows),
                "invalid_sid_count": length_invalid,
                "duplicate_sid_count": 0,
                "wrong_result_count": wrong_result_count,
            }

        tokens = np.asarray(complete_rows, dtype=np.int64)
        offsets = self.token_start + np.arange(self.sid_length) * self.codebook_size
        codes = tokens - offsets[None, :]
        in_range = np.all((codes >= 0) & (codes < self.codebook_size), axis=1)
        keys = np.zeros(codes.shape[0], dtype=np.uint64)
        for column in codes.T:
            keys = keys * self.codebook_size + column.astype(np.uint64)
        positions = np.searchsorted(self.valid_keys, keys)
        bounded = positions < self.valid_keys.size
        valid = np.zeros(keys.size, dtype=np.bool_)
        valid[bounded] = self.valid_keys[positions[bounded]] == keys[bounded]
        valid &= in_range
        return {
            "result_count": len(rows),
            "invalid_sid_count": length_invalid + int((~valid).sum()),
            "duplicate_sid_count": int(keys.size - np.unique(keys).size),
            "wrong_result_count": wrong_result_count,
        }


class GpuMonitor:
    def __init__(self, physical_gpu: int) -> None:
        self.physical_gpu = physical_gpu
        self.samples: list[int] = []
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=6)
        if self._thread.is_alive():
            raise RuntimeError("GPU memory monitor did not stop")
        if self._error is not None:
            raise RuntimeError("GPU memory monitoring failed") from self._error

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.physical_gpu}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.samples.append(int(result.stdout.strip().splitlines()[0]))
            except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
                self._error = exc
                return
            self._stop.wait(0.1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "request_index": len(rows),
                    "source_dataset_index": row.get("source_dataset_index"),
                    "prompt_sha256": row.get("prompt_sha256"),
                    "input_ids": [int(token) for token in row["input_ids"]],
                }
            )
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError(f"Expected {limit} inputs in {path}, found {len(rows)}")
    return rows


def extract_sid_tokens(raw: object) -> list[list[int]]:
    if not isinstance(raw, dict):
        raise TypeError("Generation response must be an object")
    beams = raw.get("meta_info", {}).get("beam_results")
    if not isinstance(beams, list):
        raise ValueError("Generation response must contain meta_info.beam_results")
    return [
        [
            int(token)
            for token in (
                beam.get("output_ids")
                or beam.get("meta_info", {}).get("output_ids")
                or []
            )
        ]
        for beam in beams
    ]


def latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percent / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(percentile(50), 3),
        "p95": round(percentile(95), 3),
        "p99": round(percentile(99), 3),
        "max": round(max(values), 3),
    }


def with_widths(
    rows: list[dict[str, Any]],
    widths: list[int],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "beam_width": widths[index % len(widths)],
        }
        for index, row in enumerate(rows)
    ]


def run(args: argparse.Namespace) -> None:
    from sglang import Engine

    validator = ConstraintValidator(
        args.valid_sid_keys,
        token_start=args.token_start,
        codebook_size=args.codebook_size,
        sid_length=args.sid_length,
    )
    correctness_rows = load_input_rows(
        args.correctness_inputs,
        args.correctness_requests,
    )
    performance_rows = load_input_rows(
        args.performance_inputs,
        args.performance_requests,
    )
    engine_kwargs = {
        "model_path": str(args.model),
        "tokenizer_path": str(args.model),
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running_requests,
        "context_length": args.context_length,
        "tp_size": 1,
        "dp_size": 1,
        "attention_backend": "flashinfer",
        "schedule_policy": "fcfs",
        "random_seed": 0,
        "disable_cuda_graph": True,
    }
    print(json.dumps({"engine_kwargs": engine_kwargs}, indent=2), flush=True)
    startup_started = time.perf_counter()
    engine = Engine(**engine_kwargs)
    engine_init_seconds = time.perf_counter() - startup_started
    print(f"Engine initialized in {engine_init_seconds:.3f}s", flush=True)

    async def generate(row: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        beam_width = int(row["beam_width"])
        try:
            raw = await engine.async_generate(
                input_ids=row["input_ids"],
                sampling_params={
                    "beam_width": beam_width,
                    "max_new_tokens": args.sid_length,
                    "n": beam_width,
                },
            )
            return {
                **row,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "sid_tokens": extract_sid_tokens(raw),
            }
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            return {
                **row,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def execute(
        rows: list[dict[str, Any]],
        concurrency: int,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(concurrency)

        async def limited(row: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await generate(row)

        return await asyncio.gather(*(limited(row) for row in rows))

    def run_point(
        label: str,
        rows: list[dict[str, Any]],
        *,
        concurrency: int,
        warmup_count: int,
    ) -> dict[str, Any]:
        warmup_rows = [dict(rows[index % len(rows)]) for index in range(warmup_count)]
        warmup_results = engine.loop.run_until_complete(
            execute(warmup_rows, concurrency)
        )
        warmup_failures = [row for row in warmup_results if "error" in row]
        if warmup_failures:
            raise RuntimeError(f"Warmup failed: {warmup_failures[0]['error']}")
        engine.freeze_gc()

        monitor = GpuMonitor(args.physical_gpu)
        monitor.start()
        started = time.perf_counter()
        output_rows = engine.loop.run_until_complete(execute(rows, concurrency))
        wall_seconds = time.perf_counter() - started
        monitor.stop()

        invalid_sid_count = 0
        duplicate_sid_count = 0
        wrong_result_count = 0
        successful_latencies = []
        successful_by_width: Counter[int] = Counter()
        errors_by_width: Counter[int] = Counter()
        for row in output_rows:
            beam_width = int(row["beam_width"])
            if "error" in row:
                errors_by_width[beam_width] += 1
                continue
            validity = validator.validate(row["sid_tokens"], beam_width)
            row.update(validity)
            invalid_sid_count += validity["invalid_sid_count"]
            duplicate_sid_count += validity["duplicate_sid_count"]
            wrong_result_count += validity["wrong_result_count"]
            successful_latencies.append(float(row["latency_ms"]))
            successful_by_width[beam_width] += 1

        raw_path = args.output_dir / f"{label}.jsonl.gz"
        with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

        successful = len(successful_latencies)
        summary = {
            "label": label,
            "requests": len(rows),
            "successful_requests": successful,
            "concurrency": concurrency,
            "beam_width_counts": dict(
                sorted(Counter(int(row["beam_width"]) for row in rows).items())
            ),
            "successful_by_width": dict(sorted(successful_by_width.items())),
            "errors_by_width": dict(sorted(errors_by_width.items())),
            "wall_seconds": round(wall_seconds, 3),
            "completion_qps": round(successful / wall_seconds, 4),
            "latency_ms": (
                latency_summary(successful_latencies) if successful_latencies else None
            ),
            "error_count": len(rows) - successful,
            "invalid_sid_count": invalid_sid_count,
            "duplicate_sid_count": duplicate_sid_count,
            "requests_with_wrong_result_count": wrong_result_count,
            "gpu_memory_used_mib": {
                "max_observed": max(monitor.samples) if monitor.samples else None,
                "samples": len(monitor.samples),
            },
            "raw_output": str(raw_path),
        }
        (args.output_dir / f"{label}.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    try:
        for width in args.widths:
            summaries.append(
                run_point(
                    f"{args.label}-w{width}-correctness-n{len(correctness_rows)}",
                    with_widths(correctness_rows, [width]),
                    concurrency=args.homogeneous_correctness_concurrency,
                    warmup_count=args.correctness_warmup,
                )
            )
            for concurrency in args.concurrency:
                summaries.append(
                    run_point(
                        (
                            f"{args.label}-w{width}-c{concurrency}"
                            f"-n{len(performance_rows)}"
                        ),
                        with_widths(performance_rows, [width]),
                        concurrency=concurrency,
                        warmup_count=args.performance_warmup,
                    )
                )

        if args.include_mixed:
            mixed_widths = list(args.widths)
            summaries.append(
                run_point(
                    (
                        f"{args.label}-mixed-{'-'.join(map(str, mixed_widths))}"
                        f"-correctness-n{len(correctness_rows)}"
                    ),
                    with_widths(correctness_rows, mixed_widths),
                    concurrency=args.mixed_correctness_concurrency,
                    warmup_count=args.correctness_warmup,
                )
            )
            for concurrency in args.mixed_concurrency:
                summaries.append(
                    run_point(
                        (
                            f"{args.label}-mixed-{'-'.join(map(str, mixed_widths))}"
                            f"-c{concurrency}-n{len(performance_rows)}"
                        ),
                        with_widths(performance_rows, mixed_widths),
                        concurrency=concurrency,
                        warmup_count=args.performance_warmup,
                    )
                )
    finally:
        engine.shutdown()

    compact = {
        "pid": os.getpid(),
        "engine_init_seconds": round(engine_init_seconds, 3),
        "engine_kwargs": engine_kwargs,
        "benchmark": {
            "widths": args.widths,
            "concurrency": args.concurrency,
            "include_mixed": args.include_mixed,
            "mixed_concurrency": args.mixed_concurrency,
            "correctness_warmup": args.correctness_warmup,
            "performance_warmup": args.performance_warmup,
        },
        "source": {
            "model": str(args.model),
            "valid_sid_keys": str(args.valid_sid_keys),
            "correctness_inputs": str(args.correctness_inputs),
            "correctness_inputs_sha256": sha256(args.correctness_inputs),
            "performance_inputs": str(args.performance_inputs),
            "performance_inputs_sha256": sha256(args.performance_inputs),
        },
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(compact, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark homogeneous and mixed trie-head beam widths."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--valid-sid-keys", type=Path, required=True)
    parser.add_argument("--correctness-inputs", type=Path, required=True)
    parser.add_argument("--performance-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="trie-head")
    parser.add_argument("--correctness-requests", type=int, default=200)
    parser.add_argument("--performance-requests", type=int, default=1000)
    parser.add_argument("--correctness-warmup", type=int, default=50)
    parser.add_argument("--performance-warmup", type=int, default=50)
    parser.add_argument("--widths", type=int, nargs="+", default=[1000, 2000])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 10])
    parser.add_argument("--homogeneous-correctness-concurrency", type=int, default=1)
    parser.add_argument("--include-mixed", action="store_true")
    parser.add_argument("--mixed-correctness-concurrency", type=int, default=4)
    parser.add_argument("--mixed-concurrency", type=int, nargs="+", default=[4, 10])
    parser.add_argument("--max-running-requests", type=int, default=20020)
    parser.add_argument("--context-length", type=int, default=10000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.55)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--token-start", type=int, default=151669)
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--sid-length", type=int, default=3)
    args = parser.parse_args()
    require_single_visible_gpu(args.physical_gpu)
    run(args)


if __name__ == "__main__":
    main()
